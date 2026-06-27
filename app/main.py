import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from src.data_preprocessing import clean_text


app = FastAPI(
    title="AI-Generated Text Detection API",
    description="Predicts whether text is human-written or AI-generated.",
    version="1.0.0",
)


class TextInput(BaseModel):
    text: str


vectorizer = joblib.load("artifacts/tfidf_vectorizer.joblib")
model = joblib.load("artifacts/tfidf_logreg_model.joblib")


@app.get("/")
def home():
    return {
        "message": "AI-Generated Text Detection API is running.",
        "model": "TF-IDF + Logistic Regression",
    }


@app.post("/predict")
def predict(input_data: TextInput):
    cleaned_text = clean_text(input_data.text)
    features = vectorizer.transform([cleaned_text])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].max()

    label = "AI-generated" if prediction == 1 else "Human-written"

    return {
        "input_text": input_data.text,
        "prediction": label,
        "confidence": round(float(probability), 4),
    }