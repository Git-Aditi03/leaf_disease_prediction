import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os

# ==============================================================
#  MODEL LOADING
# ==============================================================
MODEL_PATH = "plant_disease_model_final.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ==============================================================
#  CLASS NAMES  (38 PlantVillage classes)
# ==============================================================
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# ==============================================================
#  DISEASE INFORMATION DATABASE
# ==============================================================
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "cause": "Fungus - Venturia inaequalis",
        "region": "North India (Himachal Pradesh, J&K)",
        "season": "Spring to early summer",
        "cure": "Spray Mancozeb or Carbendazim fungicide",
        "base_days": 8,
    },
    "Apple___Black_rot": {
        "cause": "Fungus - Botryosphaeria obtusa",
        "region": "North India",
        "season": "Summer-Monsoon",
        "cure": "Spray Captan or Mancozeb; remove mummified fruits",
        "base_days": 12,
    },
    "Apple___Cedar_apple_rust": {
        "cause": "Fungus - Gymnosporangium juniperi-virginianae",
        "region": "Himalayan region",
        "season": "Spring",
        "cure": "Spray Myclobutanil; remove nearby juniper hosts",
        "base_days": 10,
    },
    "Tomato___Late_blight": {
        "cause": "Oomycete - Phytophthora infestans",
        "region": "North India",
        "season": "Monsoon",
        "cure": "Spray Metalaxyl + Mancozeb; remove infected plants",
        "base_days": 5,
    },
    "Tomato___Early_blight": {
        "cause": "Fungus - Alternaria solani",
        "region": "All regions",
        "season": "Summer-Monsoon",
        "cure": "Spray Mancozeb; ensure proper plant spacing",
        "base_days": 10,
    },
    "Potato___Late_blight": {
        "cause": "Oomycete - Phytophthora infestans",
        "region": "North India",
        "season": "Monsoon",
        "cure": "Spray Metalaxyl + Mancozeb; destroy infected tubers",
        "base_days": 5,
    },
    "Potato___Early_blight": {
        "cause": "Fungus - Alternaria solani",
        "region": "All regions",
        "season": "Summer",
        "cure": "Spray Chlorothalonil or Mancozeb",
        "base_days": 10,
    },
    "Corn_(maize)___Common_rust_": {
        "cause": "Fungus - Puccinia sorghi",
        "region": "All maize-growing regions",
        "season": "Monsoon",
        "cure": "Spray Propiconazole; use resistant varieties",
        "base_days": 10,
    },
    "Tomato___Bacterial_spot": {
        "cause": "Bacterium - Xanthomonas spp.",
        "region": "All regions",
        "season": "Warm wet season",
        "cure": "Spray copper-based bactericide; avoid overhead irrigation",
        "base_days": 7,
    },
    "default": {
        "cause": "Pathogen details not available",
        "region": "India",
        "season": "Monsoon",
        "cure": "Consult your local agriculture extension officer",
        "base_days": 10,
    },
}

IMG_SIZE = (224, 224)

# ==============================================================
#  LEAF VALIDATOR
# Uses green/brown pixel ratio to check if image is a leaf.
# Non-leaf images (screenshots, blank images, faces, etc.)
# will be rejected before prediction.
# ==============================================================
def is_leaf_image(image: Image.Image) -> bool:
    img = image.resize((100, 100)).convert("RGB")
    arr = np.array(img, dtype="float32")
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Green dominance: healthy leaves
    green_dominant = np.logical_and(G > R * 0.8, G > B * 0.8)
    green_ratio = np.sum(green_dominant) / green_dominant.size

    # Brown/yellow tones: diseased leaves
    brown_mask = np.logical_and.reduce([R > 80, G > 50, B < 100, R > B])
    brown_ratio = np.sum(brown_mask) / brown_mask.size

    leaf_score = green_ratio + (brown_ratio * 0.5)
    return leaf_score > 0.15


# ==============================================================
#  WEATHER HELPER
# ==============================================================
def get_weather(city: str, api_key: str):
    try:
        url = (
            f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
            f"timeline/{city}?unitGroup=metric&key={api_key}&contentType=json"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        today = resp.json()["days"][0]
        return {
            "temp": today["temp"],
            "humidity": today["humidity"],
            "description": today["conditions"],
            "rain": today["precip"] > 0,
        }
    except Exception:
        return None


def treatment_timing(disease_name: str, weather) -> str:
    info = DISEASE_INFO.get(disease_name, DISEASE_INFO["default"])
    base = info["base_days"]
    if weather is None:
        return f"Spray within {base} days (weather data unavailable)."
    h, rain = weather["humidity"], weather["rain"]
    if "Late_blight" in disease_name and (rain or h > 85):
        return "VERY URGENT - spray within 2-3 days (high humidity / rain detected)."
    if rain and h > 80:
        return f"Spray within {max(base - 4, 1)} days (rain + high humidity)."
    if h > 85:
        return f"Spray within {max(base - 3, 1)} days (high humidity)."
    return f"Spray within {base} days (moderate weather)."


# ==============================================================
#  PREDICTION
# ==============================================================
def predict(image: Image.Image, city: str, api_key: str):
    if image is None:
        return "Please upload a leaf image.", None

    # --- Leaf check ---
    if not is_leaf_image(image):
        return (
            "This does not appear to be a plant leaf image.\n\n"
            "Please upload a clear photo of a plant leaf (e.g., tomato, apple, potato leaf).\n"
            "Make sure the leaf is the main subject and the image is not a screenshot or unrelated photo."
        ), None

    # --- Preprocess ---
    img = image.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]

    # --- Confidence check: if model is very uncertain, likely not a leaf ---
    max_conf = float(preds[top3_idx[0]])
    if max_conf < 0.40:
        return (
            f"Low confidence ({max_conf*100:.1f}%) - the image may not be a recognizable plant leaf.\n\n"
            "Please upload a clear, close-up photo of a single plant leaf."
        ), None

    # --- Weather ---
    weather = None
    weather_line = "Weather: not fetched (no city / API key provided)."
    if city.strip() and api_key.strip():
        weather = get_weather(city.strip(), api_key.strip())
        if weather:
            weather_line = (
                f"Weather: {city} - {weather['temp']}C, "
                f"Humidity {weather['humidity']}%, {weather['description']}"
            )
        else:
            weather_line = "Weather fetch failed. Check city name or API key."

    # --- Build result ---
    lines = [weather_line, ""]
    for rank, i in enumerate(top3_idx, 1):
        name = CLASS_NAMES[i]
        conf = preds[i] * 100
        info = DISEASE_INFO.get(name, DISEASE_INFO["default"])
        timing = treatment_timing(name, weather)

        if "healthy" in name.lower():
            lines.append(f"**{rank}. {name}** - {conf:.1f}% Plant is healthy!")
        else:
            lines.append(
                f"**{rank}. {name}** - {conf:.1f}%\n"
                f"   Cause   : {info['cause']}\n"
                f"   Region  : {info['region']}\n"
                f"   Season  : {info['season']}\n"
                f"   Cure    : {info['cure']}\n"
                f"   Timing  : {timing}"
            )
        lines.append("")

    label_dict = {CLASS_NAMES[i]: float(preds[i]) for i in top3_idx}
    return "\n".join(lines), label_dict


# ==============================================================
#  GRADIO UI
# ==============================================================
with gr.Blocks(title="Leaf Disease Predictor") as demo:
    gr.Markdown(
        """
        # Plant Leaf Disease Prediction
        Upload a **plant leaf photo** to identify diseases and get weather-aware treatment advice.
        > Model: MobileNetV2 fine-tuned on PlantVillage (38 classes)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Leaf Image")
            city_input = gr.Textbox(
                label="City (for weather - optional)",
                placeholder="e.g. Patna",
            )
            api_input = gr.Textbox(
                label="Visual Crossing API Key (optional)",
                placeholder="Paste your free API key here",
                type="password",
            )
            submit_btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=1):
            text_out = gr.Markdown(label="Result")
            label_out = gr.Label(num_top_classes=3, label="Top-3 Confidence", show_label=True)

    submit_btn.click(
        fn=predict,
        inputs=[img_input, city_input, api_input],
        outputs=[text_out, label_out],
    )

    gr.Markdown(
        """
        ---
        **Note:** Please upload a clear, close-up photo of a single plant leaf.
        Screenshots, blurry images, or non-leaf images will be rejected.
        """
    )

if __name__ == "__main__":
    demo.launch()
