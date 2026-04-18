import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import requests

# ==============================================================
#  MODEL LOADING
# ==============================================================
MODEL_PATH = "plant_disease_model_final.h5"
disease_model = tf.keras.models.load_model(MODEL_PATH)

# ImageNet classifier for leaf validation
# MobileNetV2 pretrained on ImageNet - used ONLY to check if image is a plant/leaf
imagenet_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=True, input_shape=(224, 224, 3)
)

# ImageNet classes that are plant/leaf related (synset indices)
# These cover: plants, leaves, flowers, vegetables, fruits on trees, etc.
PLANT_RELATED_KEYWORDS = [
    "leaf", "plant", "tree", "flower", "herb", "fern", "moss",
    "vegetable", "cabbage", "broccoli", "cauliflower", "cucumber",
    "corn", "ear of corn", "artichoke", "mushroom", "fungus",
    "strawberry", "orange", "lemon", "fig", "pineapple", "banana",
    "apple", "grape", "pomegranate", "acorn", "hip", "buckeye",
    "rapeseed", "daisy", "yellow lady", "corn poppy",
    "pot", "garden", "bud", "petal", "stem", "vine", "shrub",
]

def is_leaf_image(image: Image.Image) -> tuple:
    """
    Uses ImageNet MobileNetV2 to check if image contains a plant/leaf.
    Returns (is_leaf: bool, reason: str)
    """
    img = image.resize((224, 224)).convert("RGB")
    arr = np.array(img, dtype="float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)

    preds = imagenet_model.predict(arr, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

    # Check if any of top-5 ImageNet predictions are plant-related
    for _, label, confidence in decoded:
        label_lower = label.lower().replace("_", " ")
        for keyword in PLANT_RELATED_KEYWORDS:
            if keyword in label_lower:
                return True, f"Detected: {label} ({confidence*100:.1f}%)"

    # Also do a green pixel ratio check as secondary signal
    arr_rgb = np.array(image.resize((100, 100)).convert("RGB"), dtype="float32")
    R, G, B = arr_rgb[:,:,0], arr_rgb[:,:,1], arr_rgb[:,:,2]
    green_mask = (G > 60) & (G > R * 0.9) & (G > B * 0.85)
    green_ratio = np.sum(green_mask) / green_mask.size

    # Very strong green signal can override (e.g., diseased yellow leaf still passes ImageNet check)
    if green_ratio > 0.35:
        return True, f"Strong green signal detected ({green_ratio*100:.0f}% green pixels)"

    top_label = decoded[0][1]
    top_conf = decoded[0][2]
    return False, f"Detected as: {top_label} ({top_conf*100:.1f}%) - not a plant"


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

    # --- Step 1: Check if image is a plant/leaf using ImageNet model ---
    leaf_ok, leaf_reason = is_leaf_image(image)
    if not leaf_ok:
        return (
            "**This is not a plant leaf image.**\n\n"
            f"The image was identified as: _{leaf_reason}_\n\n"
            "Please upload a clear, close-up photo of a plant leaf "
            "(e.g. tomato leaf, apple leaf, potato leaf)."
        ), None

    # --- Step 2: Preprocess for disease model ---
    img = image.resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    arr = np.expand_dims(arr, 0)

    preds = disease_model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]

    # --- Step 3: Low confidence = unrecognizable leaf ---
    max_conf = float(preds[top3_idx[0]])
    if max_conf < 0.35:
        return (
            f"**Low confidence ({max_conf*100:.1f}%)** - could not confidently identify the leaf disease.\n\n"
            "Please upload a clearer, well-lit photo of a single plant leaf."
        ), None

    # --- Weather ---
    weather = None
    weather_line = "Weather: not fetched (no city/API key provided)."
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
        Screenshots, faces, or unrelated images will be rejected automatically.
        """
    )

if __name__ == "__main__":
    demo.launch()
