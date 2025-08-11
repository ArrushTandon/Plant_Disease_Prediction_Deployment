# Dictionary mapping disease name -> cure suggestion
CURE_MAP = {
    "Apple___Apple_scab": "Remove and destroy infected leaves. Use a fungicide spray.",
    "Apple___Black_rot": "Prune infected branches. Apply copper-based fungicides.",
    "Corn___Common_rust": "Plant rust-resistant varieties. Apply fungicide if needed.",
    "Grape___Black_rot": "Prune away infected leaves. Use protective fungicides.",
    "Pepper__bell___Bacterial_spot": "Use copper-based fungicides, avoid overhead watering, and remove infected leaves.",
    "Potato___Early_blight": "Apply fungicides like chlorothalonil, ensure proper crop rotation, and remove affected plants.",
    "Potato___Late_blight": "Use fungicides containing mancozeb or copper, avoid wet leaves, and destroy infected tubers.",
    "Tomato_Bacterial_spot": "Use certified seeds, apply copper sprays, and avoid working with wet plants.",
    "Tomato_Early_blight": "Remove infected leaves, use fungicides, and rotate crops yearly.",
    "Tomato_Leaf_Mold": "Increase ventilation, reduce humidity, and use fungicides.",
    "Tomato_Septoria_leaf_spot": "Remove infected leaves, avoid wetting foliage, and use fungicides.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray water to remove mites, introduce natural predators like ladybugs.",
    "Tomato__Target_Spot": "Remove affected leaves, maintain crop hygiene, and use fungicides.",
    "Tomato__Tomato_mosaic_virus": "Remove and destroy infected plants, disinfect tools, and use resistant varieties.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies, remove infected plants, and use resistant varieties.",
    "Healthy": "No action needed. Keep monitoring plant health."
}

def get_cure(disease_name: str) -> str:
    return CURE_MAP.get(disease_name, "No cure information available.")