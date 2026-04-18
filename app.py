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

# ImageNet validator
validator_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=True, input_shape=(224, 224, 3)
)

# ==============================================================
#  ImageNet words that mean plant / leaf / nature
# ==============================================================
PLANT_KEYWORDS = {
    "leaf", "plant", "tree", "flower", "herb", "fern", "moss",
    "vegetable", "cabbage", "broccoli", "cauliflower", "cucumber",
    "corn", "artichoke", "mushroom", "fungus", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "apple", "grape",
    "pomegranate", "acorn", "rapeseed", "daisy", "bud", "petal",
    "vine", "shrub", "bush", "weed", "algae", "lichen", "hip",
    "buckeye", "clover", "thistle", "bramble", "dandelion", "grass",
    "reed", "horsetail", "liverwort", "watercress", "seaweed", "kelp",
    "cactus", "succulent", "agave", "aloe", "bamboo", "palm",
    "foliage", "frond", "stalk", "stem", "sprout", "seedling",
    "sapling", "groundsel", "sorrel", "dock", "nettle", "spurge",
    "plantain", "chickweed", "bindweed", "conifer", "hardwood",
}

# ==============================================================
#  STRICT LEAF VALIDATOR
#  Only ImageNet decides. No override. Period.
#  - Run image through ImageNet MobileNetV2
#  - Check top-5 predicted labels against PLANT_KEYWORDS
#  - If NONE match → reject with what was detected
#  - If ANY match → proceed to disease model
# ==============================================================
def check_is_leaf(image: Image.Image) -> tuple:
    """Returns (is_leaf: bool, message: str)"""
    img = image.resize((224, 224)).convert("RGB")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array(img, dtype="float32")
    )
    preds = validator_model.predict(np.expand_dims(arr, 0), verbose=0)
    top5 = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
    # top5 = [(id, label, confidence), ...]

    for _, label, conf in top5:
        words = label.lower().replace("_", " ").split()
        if any(w in PLANT_KEYWORDS for w in words):
            return True, f"Plant detected: {label} ({conf*100:.1f}%)"

    # None of top-5 are plants — build a helpful rejection message
    top_label = top5[0][1].replace("_", " ")
    top_conf  = top5[0][2] * 100
    return False, (
        f"Detected as **'{top_label}'** ({top_conf:.0f}% confidence) — not a plant leaf.\n\n"
        f"Top-5 predictions: {', '.join(l.replace('_',' ') for _,l,_ in top5)}"
    )


# ==============================================================
#  CLASS NAMES  (38 PlantVillage classes)
# ==============================================================
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
    "Apple___healthy", "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

# ==============================================================
#  DISEASE INFORMATION DATABASE
# ==============================================================
DISEASE_INFO = {
    "Apple___Apple_scab": {"cause": "Fungus - Venturia inaequalis", "region": "North India (Himachal Pradesh, J&K)", "season": "Spring to early summer", "cure": "Spray Mancozeb or Carbendazim fungicide", "base_days": 8},
    "Apple___Black_rot": {"cause": "Fungus - Botryosphaeria obtusa", "region": "North India", "season": "Summer-Monsoon", "cure": "Spray Captan or Mancozeb; remove mummified fruits", "base_days": 12},
    "Apple___Cedar_apple_rust": {"cause": "Fungus - Gymnosporangium juniperi-virginianae", "region": "Himalayan region", "season": "Spring", "cure": "Spray Myclobutanil; remove nearby juniper hosts", "base_days": 10},
    "Tomato___Late_blight": {"cause": "Oomycete - Phytophthora infestans", "region": "North India", "season": "Monsoon", "cure": "Spray Metalaxyl + Mancozeb; remove infected plants", "base_days": 5},
    "Tomato___Early_blight": {"cause": "Fungus - Alternaria solani", "region": "All regions", "season": "Summer-Monsoon", "cure": "Spray Mancozeb; ensure proper plant spacing", "base_days": 10},
    "Potato___Late_blight": {"cause": "Oomycete - Phytophthora infestans", "region": "North India", "season": "Monsoon", "cure": "Spray Metalaxyl + Mancozeb; destroy infected tubers", "base_days": 5},
    "Potato___Early_blight": {"cause": "Fungus - Alternaria solani", "region": "All regions", "season": "Summer", "cure": "Spray Chlorothalonil or Mancozeb", "base_days": 10},
    "Corn_(maize)___Common_rust_": {"cause": "Fungus - Puccinia sorghi", "region": "All maize-growing regions", "season": "Monsoon", "cure": "Spray Propiconazole; use resistant varieties", "base_days": 10},
    "Tomato___Bacterial_spot": {"cause": "Bacterium - Xanthomonas spp.", "region": "All regions", "season": "Warm wet season", "cure": "Spray copper-based bactericide; avoid overhead irrigation", "base_days": 7},
    "default": {"cause": "Pathogen details not available", "region": "India", "season": "Monsoon", "cure": "Consult your local agriculture extension officer", "base_days": 10},
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
#  MAIN PREDICT FUNCTION
# ==============================================================
def predict(image: Image.Image, city: str, api_key: str):
    if image is None:
        return "Please upload or capture a leaf image.", None

    # ── Step 1: Strict ImageNet leaf check ──────────────────────
    is_leaf, msg = check_is_leaf(image)
    if not is_leaf:
        return (
            "**This is not a plant leaf image.**\n\n"
            + msg + "\n\n"
            "Please upload a **clear photo of a plant leaf** "
            "(tomato, apple, potato, corn, grape, etc.).\n"
            "You can also use the **Webcam** tab to capture a leaf live."
        ), None

    # ── Step 2: Disease prediction ───────────────────────────────
    img = image.resize(IMG_SIZE)
    arr = np.array(img.convert("RGB"), dtype="float32") / 255.0
    arr = np.expand_dims(arr, 0)
    preds = disease_model.predict(arr, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]

    # ── Step 3: Weather ──────────────────────────────────────────
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

    # ── Step 4: Build output ─────────────────────────────────────
    lines = [weather_line, ""]
    for rank, i in enumerate(top3_idx, 1):
        name   = CLASS_NAMES[i]
        conf   = preds[i] * 100
        info   = DISEASE_INFO.get(name, DISEASE_INFO["default"])
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
        Upload a **plant leaf photo** or use your **webcam** to capture one live.
        > Model: MobileNetV2 fine-tuned on PlantVillage (38 classes)
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                type="pil",
                label="Leaf Image",
                sources=["upload", "webcam", "clipboard"],
            )
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
            text_out  = gr.Markdown(label="Result")
            label_out = gr.Label(num_top_classes=3, label="Top-3 Confidence", show_label=True)

    submit_btn.click(
        fn=predict,
        inputs=[img_input, city_input, api_input],
        outputs=[text_out, label_out],
    )
    gr.Markdown(
        """
        ---
        **Tips:**
        - Use **Upload** for saved photos | **Webcam** to capture live
        - Non-leaf images (people, cars, screenshots) are automatically rejected
        - Get a free weather API key at [visualcrossing.com](https://www.visualcrossing.com/sign-up)
        """
    )

if __name__ == "__main__":
    demo.launch()
