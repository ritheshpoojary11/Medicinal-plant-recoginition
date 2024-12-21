from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
import uvicorn
import logging
import os

# Initialize FastAPI app
app = FastAPI()

# CORS configuration to allow requests from your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Setup logging for better debugging
logging.basicConfig(level=logging.INFO)

# Load the saved plant recognition model (ensure the model path is correct)
model_path = 'M_plant_recognition_model_mobilenet.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)
logging.info("Model loaded successfully")

# List of plant class labels
class_labels = ['Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avacado', 'Bamboo', 'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel']

# Predict plant type based on uploaded image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and save the image temporarily
        image_data = await file.read()
        with open("temp_image.jpg", "wb") as temp_image:
            temp_image.write(image_data)
        logging.info("Image saved temporarily for processing")

        # Load and preprocess the image
        img = load_img("temp_image.jpg", target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions using the model
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_labels[predicted_class_index]
        logging.info(f"Predicted plant class: {predicted_class_label}")

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Image processing or prediction failed")

    # Clean up temporary image
    os.remove("temp_image.jpg")

    # Return the prediction result
    return {"prediction": predicted_class_label, "details": f"Information on {predicted_class_label} could be added here."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
