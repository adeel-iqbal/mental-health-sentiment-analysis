from fastapi import FastAPI
from pydantic import BaseModel
from app.predictor import predict_text

app = FastAPI(title="Mental Health Sentiment Analysis API")

# Request model
class PredictRequest(BaseModel):
    text: str

# Endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    result = predict_text(request.text)
    return result

# Optional: health check
@app.get("/")
def root():
    return {"message": "ML Service is running..."}
