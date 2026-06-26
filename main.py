from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Face Mask Detection API", description="API for predicting face masks from images.")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Load the model robustly (handling potential version mismatch errors)
try:
    # Use compile=False to avoid issues with custom optimizers or quantization metrics
    model = tf.keras.models.load_model("face_mask_model.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def prepare_image(image_bytes):
    """Preprocesses the image to fit the model's expected input."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Change to the size your model expects (e.g., 224x224 or 128x128)
    image = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serves the frontend HTML."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint to receive an image and return the prediction."""
    if model is None:
        return {"error": "Model failed to load on the server."}
        
    image_bytes = await file.read()
    processed_image = prepare_image(image_bytes)
    
    # Predict
    prediction = model.predict(processed_image)[0][0]
    
    # Adjust logic based on your model's specific training (0 = Mask, 1 = No Mask usually)
    has_mask = bool(prediction < 0.5)
    confidence = float(1 - prediction if has_mask else prediction)
    
    return {
        "filename": file.filename,
        "prediction": "With Mask" if has_mask else "Without Mask",
        "confidence": round(confidence * 100, 2),
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)