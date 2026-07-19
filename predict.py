from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from pathlib import Path

MODEL_PATH = Path("model/best_model.keras")

model = load_model(MODEL_PATH)

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]
def predict_disease(image_path):

    img = load_img(image_path, target_size=(224, 224))

    img_array = img_to_array(img)

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)
    
    return CLASS_NAMES[predicted_class], confidence

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    disease, confidence = predict_disease(image_path)

    print("\nPrediction Result")
    print("-------------------------")
    print("Disease :", disease)
    print(f"Confidence : {confidence*100:.2f}%")