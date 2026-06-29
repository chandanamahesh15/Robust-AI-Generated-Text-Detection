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

from src.config import load_config
from src.data import clean_text
from src.logging_utils import configure_logging, get_logger
from src.models.tfidf import TfidfLogRegModel

logger = get_logger(__name__)

# Populated at startup; kept in module state so we load the model only once.
_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    configure_logging(cfg.logging.level, cfg.logging.format)
    logger.info("Loading model artifacts...")
    _state["cfg"] = cfg
    _state["model"] = TfidfLogRegModel.load(cfg)
    logger.info("Service ready.")
    yield
    _state.clear()


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
    return {"status": "ok", "model_loaded": str("model" in _state)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    model = _state.get("model")
    cfg = _state.get("cfg")
    if model is None or cfg is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    raw_texts = req.texts if req.texts is not None else ([req.text] if req.text else None)
    if not raw_texts:
        raise HTTPException(status_code=422, detail="Provide 'text' or 'texts'.")

    cleaned = [clean_text(t, cfg) for t in raw_texts]  # type: ignore[arg-type]
    labels = model.predict(cleaned)  # type: ignore[attr-defined]
    probs = model.predict_proba(cleaned)  # type: ignore[attr-defined]

    return PredictResponse(
        predictions=[
            Prediction(label=int(lbl), ai_probability=round(float(p), 4))
            for lbl, p in zip(labels, probs)
        ]
    )
