from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from skimage.feature import local_binary_pattern
import uvicorn
import numpy as np
import pickle
import cv2
import tensorflow as tf
import os

# Paths

FEATURE_EXTRACTOR_PATH = "cnn_feature_extractor.h5"
CLASSIFIER_PATH = "model_cnn.h5"
PCA_PATH = "pca.pkl"
MASK_PATH = "mask.npy"
THRESHOLD_PATH = "threshold.npy"
'''
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_EXTRACTOR_PATH = os.path.join(BASE_DIR, "cnn_feature_extractor.h5")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "model_cnn.h5")
PCA_PATH = os.path.join(BASE_DIR, "pca.pkl")
MASK_PATH = os.path.join(BASE_DIR, "mask.npy")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.npy")
'''
# Load models
print("Loading CNN from:", FEATURE_EXTRACTOR_PATH) #Debug Files
print("Exists?", os.path.exists(FEATURE_EXTRACTOR_PATH)) #Debug Files
feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH, compile=False)
classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)
threshold = np.load(THRESHOLD_PATH)

# Load PCA & mask
with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)
mask = np.load(MASK_PATH)

IMG_SIZE = (128, 128)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file")
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0

    lbp_features = compute_lbp(img)
    img_cnn = np.expand_dims(img, axis=-1)

    return img_cnn, lbp_features

def compute_lbp(image, P=8, R=1):
    image = (image * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        img, lbp_features = preprocess_image(image_bytes)

        # Extract CNN features
        cnn_features = feature_extractor.predict(np.expand_dims(img, axis=0), verbose=0)

        # Combine CNN + LBP features
        combined_features = np.hstack((cnn_features[0], lbp_features))

        # Apply PCA and mask
        features_pca = pca.transform([combined_features])
        selected_features = features_pca[:, mask]

        # Predict
        proba = classifier.predict(selected_features, verbose=0)[0][0]
        accuracy = proba * 100 if proba > threshold else (1 - proba) * 100
        disease = "Unhealthy" if proba > threshold else "Healthy"

        return {
            "status": "Success",
            "disease": disease,
            "accuracy": float(accuracy)
        }
    except Exception as e:
        return {
            "status": "Error",
            "error": str(e),
            "disease": "-",
            "accuracy": "-"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
