"""FastAPI inference service for the deployable TF-IDF + LogReg detector.

Loads the trained artifact once at startup and exposes a prediction endpoint.
Run locally:
    uvicorn app.main:app --reload
Then POST to /predict with {"text": "..."} or {"texts": ["...", "..."]}.

This service depends ONLY on the lightweight artifact — no torch/tensorflow,
so the container stays small and starts in milliseconds.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Config, load_config
from src.data import clean_text
from src.logging_utils import configure_logging, get_logger
from src.models.tfidf import TfidfLogRegModel

logger = get_logger(__name__)

# Module-level state, loaded once at startup. Typed explicitly so the request
# handlers stay fully type-checked (no `# type: ignore` needed).
_cfg: Optional[Config] = None
_model: Optional[TfidfLogRegModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _model
    _cfg = load_config()
    configure_logging(_cfg.logging.level, _cfg.logging.format)
    logger.info("Loading model artifacts...")
    _model = TfidfLogRegModel.load(_cfg)
    logger.info("Service ready.")
    yield
    _model = None
    _cfg = None


app = FastAPI(
    title="AI vs Human Text Detector",
    description="Classifies text as human-written (0) or AI-generated (1).",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: Optional[str] = Field(None, description="A single text to classify.")
    texts: Optional[list[str]] = Field(None, description="A batch of texts.")


class Prediction(BaseModel):
    label: int = Field(..., description="0 = human, 1 = AI")
    ai_probability: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(_model is not None)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None or _cfg is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if req.texts is not None:
        raw_texts: list[str] = req.texts
    elif req.text:
        raw_texts = [req.text]
    else:
        raise HTTPException(status_code=422, detail="Provide 'text' or 'texts'.")

    cleaned = [clean_text(t, _cfg) for t in raw_texts]
    labels = _model.predict(cleaned)
    probs = _model.predict_proba(cleaned)

    return PredictResponse(
        predictions=[
            Prediction(label=int(lbl), ai_probability=round(float(p), 4))
            for lbl, p in zip(labels, probs)
        ]
    )
