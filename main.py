from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
from sentence_transformers import SentenceTransformer
import sys

# Load MiniLM embedder
embedder = SentenceTransformer("minilm_embedder")

# Load Logistic Regression model with error handling
def load_model():
    try:
        print("Attempting to load with joblib...")
        model = joblib.load("email_classifier.pkl")
        print("Model loaded successfully with joblib")
        return model
    except Exception as e:
        print(f"Joblib loading failed: {e}")

model = load_model()

# FastAPI app
app = FastAPI(title="Email Classifier API", version="1.0")

# Request schema
class EmailRequest(BaseModel):
    subject: str
    body: str
    sender: str

@app.get("/")
def root():
    return {"message": "Email Classifier API is running"}

@app.post("/predict")
def predict_email(data: EmailRequest):
    try:
        text = f"{data.subject} {data.body} {data.sender}"
        embedding = embedder.encode([text])  # Shape: (1, embedding_dim)
        prediction = model.predict(embedding)[0]
        return {"category": prediction}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)