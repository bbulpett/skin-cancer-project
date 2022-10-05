from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Create app as an instance of FastAPI
app = FastAPI()

# Define allowed request origins for localhost environment
origins = [
    "http://localhost",
    "http://localhost:3000"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

MODEL_FILENAME = 'Model@2022-05-17::08:34:06.h5'
MODEL = tf.keras.models.load_model(f'saved_models/{MODEL_FILENAME}')
CLASS_NAMES = [
    'actinic keratosis',
    'basal cell carcinoma',
    'dermatofibroma',
    'melanoma',
    'nevus',
    'pigmented benign keratosis',
    'seborrheic keratosis',
    'squamous cell carcinoma',
    'vascular lesion'
]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((600, 450)))
    
    return image

# POST to call predict function
@app.post("/predict")

# Data type is UploadFile
async def predict(
    file: UploadFile = File(...)
):
    # Convert file to numpy array (tensor) so that model can do predictions
    # (asynchronously to avoid blocking requests)
    bytes = await file.read()
    image = read_file_as_image(bytes)
    image_batch = np.expand_dims(image, 0)  # Convert to 2D array

    # Call the model to get predictions
    predictions = MODEL.predict(image_batch)
    
    # Get max value of predictions (most accurate) and corresponding class name
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max([predictions[0]])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
