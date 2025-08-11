'''
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pickle
import cv2
import tensorflow as tf

# Paths
FEATURE_EXTRACTOR_PATH = "cnn_feature_extractor.h5"
CLASSIFIER_PATH = "model_cnn.h5"
PCA_PATH = "pca.pkl"
MASK_PATH = "mask.npy"

# Load models
feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)
classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

# Load PCA & mask
with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)
mask = np.load(MASK_PATH)

IMG_SIZE = (128, 128)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(image_bytes):
    """Preprocess uploaded image for prediction"""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        # Dummy LBP features (replace with actual LBP if used in training)
        lbp_features = np.zeros(10)

        # Extract CNN features
        cnn_features = feature_extractor.predict(np.expand_dims(img, axis=0), verbose=0)

        # Combine CNN + LBP features
        combined_features = np.hstack((cnn_features[0], lbp_features))

        # Apply PCA and mask
        features_pca = pca.transform([combined_features])
        selected_features = features_pca[:, mask]

        # Predict
        proba = classifier.predict(selected_features, verbose=0)[0][0]
        accuracy = proba * 100 if proba > 0.5 else (1 - proba) * 100
        disease = "Unhealthy" if proba > 0.5 else "Healthy"

        return {
            "status": "Success",
            "disease": disease,
            "accuracy": float(accuracy)
        }
    except Exception as e:
        return {
            "status": "Error",
            "disease": "-",
            "accuracy": "-"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

'''
'''
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import pickle
import cv2
import tensorflow as tf
from skimage.feature import local_binary_pattern
import os

# Paths
FEATURE_EXTRACTOR_PATH = "cnn_feature_extractor.h5"
CLASSIFIER_PATH = "model_cnn.h5"
PCA_PATH = "pca.pkl"
MASK_PATH = "mask.npy"

# Load models
feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)
classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

OPTIMAL_THRESHOLD = 0.37

# Load PCA & mask
with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)
mask = np.load(MASK_PATH)

IMG_SIZE = (128, 128)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def compute_lbp(image, P=8, R=1):
    """Compute LBP histogram features"""
    image = (image * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # normalize
    return hist

def preprocess_image(image_bytes):
    """Preprocess uploaded image for prediction"""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    img_gray = cv2.resize(img_gray, IMG_SIZE)
    img_gray = img_gray / 255.0

    lbp_features = compute_lbp(img_gray)
    img_cnn = np.expand_dims(img_gray, axis=-1)  # (H, W, 1)
    return img_cnn, lbp_features

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess
        image_bytes = await file.read()
        img_cnn, lbp_features = preprocess_image(image_bytes)

        # CNN features
        cnn_features = feature_extractor.predict(np.expand_dims(img_cnn, axis=0), verbose=0)

        # Combine CNN + LBP features
        combined_features = np.hstack((cnn_features[0], lbp_features))

        # Apply PCA and mask
        features_pca = pca.transform([combined_features])
        selected_features = features_pca[:, mask]

        """
        # Predict
        proba = classifier.predict(selected_features, verbose=0)[0][0]
        accuracy = proba * 100 if proba > 0.5 else (1 - proba) * 100
        disease = "Unhealthy" if proba > 0.5 else "Healthy"
        """
        # Predict
        proba = classifier.predict(selected_features, verbose=0)[0][0]
        accuracy = proba * 100 if proba > OPTIMAL_THRESHOLD else (1 - proba) * 100
        disease = "Unhealthy" if proba > OPTIMAL_THRESHOLD else "Healthy"

        return {
            "status": "Success",
            "disease": disease,
            "accuracy": round(float(accuracy), 2)
        }
    except Exception as e:
        return {
            "status": "Error",
            "disease": "-",
            "accuracy": "-",
            "error": str(e)
        }
'''
'''
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload = False)
'''
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
import pickle
import cv2
import tensorflow as tf
import os

# Paths
FEATURE_EXTRACTOR_PATH = "Backend/cnn_feature_extractor.h5"
CLASSIFIER_PATH = "Backend/model_cnn.h5"
PCA_PATH = "Backend/pca.pkl"
MASK_PATH = "Backend/mask.npy"

# Load models
feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)
classifier = tf.keras.models.load_model(CLASSIFIER_PATH)

# Load PCA & mask
with open(PCA_PATH, "rb") as f:
    pca = pickle.load(f)
mask = np.load(MASK_PATH)

IMG_SIZE = (128, 128)

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("frontend/index.html")

def preprocess_image(image_bytes):
    """Preprocess uploaded image for prediction"""
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W, 1)
    return img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        # Dummy LBP features (replace with actual LBP if used in training)
        lbp_features = np.zeros(10)

        # Extract CNN features
        cnn_features = feature_extractor.predict(np.expand_dims(img, axis=0), verbose=0)

        # Combine CNN + LBP features
        combined_features = np.hstack((cnn_features[0], lbp_features))

        # Apply PCA and mask
        features_pca = pca.transform([combined_features])
        selected_features = features_pca[:, mask]

        # Predict
        proba = classifier.predict(selected_features, verbose=0)[0][0]
        accuracy = proba * 100 if proba > 0.5 else (1 - proba) * 100
        disease = "Unhealthy" if proba > 0.5 else "Healthy"

        return {
            "status": "Success",
            "disease": disease,
            "accuracy": float(accuracy)
        }
    except Exception:
        return {
            "status": "Error",
            "disease": "-",
            "accuracy": "-"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload = False)
