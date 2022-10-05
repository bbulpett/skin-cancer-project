# Code for Google Cloud deployment
import numpy as np
import tensorflow as tf
from google.cloud import storage
from PIL import Image

BUCKET_NAME = 'bbulpett-skin-cancer-tf-models'
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
MODEL_FILENAME = 'Model@2022-05-17::08:34:06.h5'

model = None

# Separate GCP server will be downloading model from storage bucket
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

# Primary function to execute prediction with request input
def predict(request):
    # Set CORS headers for the preflight request
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST'
        }

        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    global model

    if model is None:
        model_source = f"models/{MODEL_FILENAME}"
        model_destination = f"/tmp/{MODEL_FILENAME}"

    download_blob(BUCKET_NAME, model_source, model_destination)

    model = tf.keras.models.load_model(model_destination)
    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((600, 450)))
    image_batch = np.expand_dims(image, 0)  # Convert to 2D array
    predictions = model.predict(image_batch)

    print(predictions)

    # Get max value of predictions (most accurate) and corresponding class name
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max([predictions[0]])

    return (
        { 'class': predicted_class, 'confidence': float(confidence) },
        200,
        headers
    )
