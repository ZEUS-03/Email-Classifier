from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
embedder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, embedder
    try:
        logger.info("Loading models...")
        
        # Load sentence transformer
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load classifier
        if os.path.exists("email_classifier.pkl"):
            model = joblib.load("email_classifier.pkl")
            logger.info("✅ Models loaded successfully!")
        else:
            logger.error("❌ email_classifier.pkl not found!")
            
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
    
    yield

# FastAPI app
app = FastAPI(
    title="Email Classifier API",
    description="Railway-deployed email classification service",
    version="1.0.0",
    lifespan=lifespan
)

class EmailRequest(BaseModel):
    subject: str
    body: str

@app.get("/")
async def root():
    return {
        "message": "Email Classifier API running on Railway! 🚂",
        "status": "healthy" if model and embedder else "loading",
        "port": os.environ.get("PORT", "8000")
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "embedder_loaded": embedder is not None
    }

@app.post("/predict")
async def predict_email(data: EmailRequest):
    if not model or not embedder:
        raise HTTPException(status_code=503, detail="Models still loading")
    
    try:
        text = f"{data.subject} {data.body}"
        embedding = embedder.encode([text])
        prediction = model.predict(embedding)[0]
        
        return {
            "category": str(prediction),
            "confidence": "calculated" if hasattr(model, 'predict_proba') else "not_available"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)