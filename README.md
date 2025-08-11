# Plant Disease Prediction Deployment

## Overview
This project is an end-to-end system for detecting plant diseases from leaf images using a hybrid approach of **Convolutional Neural Networks (CNN)**, **Local Binary Patterns (LBP)**, **Principal Component Analysis (PCA)**, and **Elephant Herding Optimization (EHO)** for feature selection.  
It includes both the **backend API** (FastAPI) and the **frontend interface** (HTML/CSS/JavaScript) for user interaction.

Users can upload or capture a photo of a leaf, and the system will classify it as **Healthy** or **Unhealthy**, providing the prediction accuracy.

---

## Features
- **Image Upload**: Upload a leaf image for analysis.
- **Camera Capture**: Take a live photo for instant prediction.
- **CNN + LBP Feature Extraction**: Combines deep learning with texture features.
- **PCA Dimensionality Reduction**: Reduces feature size while retaining critical information.
- **EHO Feature Selection**: Optimizes selected features for classification.
- **FastAPI Backend**: High-performance, scalable API for inference.
- **Frontend Integration**: User-friendly interface served directly from the backend.
- **Deployable on Render/Cloud**: Configured for cloud deployment.

---

## Project Structure
```
Plant_Disease_Prediction_Deployment/
│
├── Backend/
│   ├── main.py                  # FastAPI application
│   ├── cnn_feature_extractor.h5 # Trained CNN feature extractor
│   ├── model_cnn.h5              # Final trained classifier
│   ├── pca.pkl                   # PCA transformation file
│   ├── mask.npy                  # EHO-selected feature mask
│   ├── cure_suggestions.py       # Diseases Cures (To be Implemented) 
│   ├── start.sh                  # Startup script for deployment
│   ├── requirements.txt              # Python dependencies
│
├── frontend/
│   ├── index.html                # Home page
│   ├── upload.html               # Upload/capture page
│   ├── styles.css                # Styling
│   ├── upload.js                 # Frontend JS logic
│   ├── results.html              # Results Page
│   ├── script.js                 # Backend JS logic
│
└── README.md                     # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Plant_Disease_Prediction_Deployment.git
cd Plant_Disease_Prediction_Deployment
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
cd Backend
uvicorn main:app --reload
```
Access the app at:
```
https://plant-disease-prediction-deployment.onrender.com/
```

---

## Deployment
The project is configured for deployment on **Render**:
- Backend is served via FastAPI.
- Frontend static files are served from `/frontend` through FastAPI.
- `start.sh` launches the server in production.

To deploy:
1. Push your repository to GitHub.
2. Create a new **Web Service** on Render.
3. Connect the GitHub repo and set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `bash start.sh`
4. Deploy.

---

## Model Training
The model was trained using:
- **CNN** for feature extraction.
- **LBP** for texture analysis.
- **PCA** for dimensionality reduction.
- **EHO** for optimal feature selection.
- **SMOTE** for class imbalance handling.
- **Custom decision thresholding** to improve class precision.

Training is performed using the `cnn_eho_gpu_updated.py` script. The output model files are stored in `Backend/` for inference.

---

## API Endpoints
### `POST /predict`
**Request**:
- `file`: Image file (JPG, PNG, etc.)

**Response**:
```json
{
    "status": "Success",
    "disease": "Healthy",
    "accuracy": 95.23
}
```

---

## Future Improvements
- Support for multi-class disease classification.
- Integration with a mobile application.
- Dataset expansion for better generalization.
- GPU-enabled cloud deployment.

---

## License
This project is licensed under the MIT License.
