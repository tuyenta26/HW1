import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from app.ml.age_predictor import AgePredictor
import uvicorn
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Age Prediction API",
    description="A simple API server for predicting age from uploaded images.",
    version="1.0.0"
)

# Load the ML model at startup
logger.info("Loading age prediction model...")
predictor = AgePredictor()
logger.info("Model loaded successfully.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Age Prediction API. Send a POST request with an image to /predict_age."}

@app.post("/predict_age")
async def predict_age(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type uploaded. Expected image, got {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        # Read image to memory
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Predict
        predicted_age = predictor.predict(image)
        logger.info(f"Successfully predicted age range {predicted_age} for image {file.filename}")
        
        return JSONResponse(content={
            "filename": file.filename,
            "predicted_age_range": predicted_age
        })
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the image.")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
