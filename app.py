# app.py
# Main entry point — imports everything from separate modules
# This file should stay SHORT. Add features in the other files.

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from disease_info  import SIMPLE_INFO, SEVERITY_CONFIG
from weather       import get_weather, get_treatment_timing, build_weather_html
from history       import add_to_history, get_history_html
from ui_components import (CSS, JS_VOICE,
                            build_error_html, build_healthy_html, build_disease_html)

# ==============================================================
#  MODEL LOADING
# ==============================================================
MODEL_PATH = "plant_disease_model_final.h5"
disease_model = tf.keras.models.load_model(MODEL_PATH)

validator_model = tf.keras.applications.MobileNetV2(
    weights="imagenet", include_top=True, input_shape=(224, 224, 3)
)

# ==============================================================
#  CLASS NAMES
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
#  LEAF VALIDATOR
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

def check_is_leaf(image: Image.Image) -> tuple:
    img = image.resize((224, 224)).convert("RGB")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.array(img, dtype="float32")
    )
    preds = validator_model.predict(np.expand_dims(arr, 0), verbose=0)
    top5  = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
    for _, label, conf in top5:
        if any(w in PLANT_KEYWORDS for w in label.lower().replace("_", " ").split()):
            return True, label
    top_label = top5[0][1].replace("_", " ")
    top_conf  = top5[0][2] * 100
    return False, f"'{top_label}' ({top_conf:.0f}%)"

# ==============================================================
#  MAIN PREDICT FUNCTION
# ==============================================================
def predict(image: Image.Image, city: str, api_key: str):
    if image is None:
        return (build_error_html("Please upload or capture a leaf image first! 📸"),
                "", get_history_html())

    # Leaf validation
    is_leaf, reason = check_is_leaf(image)
    if not is_leaf:
        return (
            build_error_html(
                f"❌ This is not a plant leaf!\n\nDetected as: {reason}\n\n"
                "Please upload a clear photo of a plant leaf 🌿"
            ), "", get_history_html()
        )

    # Disease prediction
    arr  = np.array(image.resize((224, 224)).convert("RGB"), dtype="float32") / 255.0
    preds = disease_model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    top_idx  = int(np.argmax(preds))
    top_name = CLASS_NAMES[top_idx]
    top_conf = float(preds[top_idx]) * 100

    # Weather
    weather      = None
    weather_html = ""
    if city.strip() and api_key.strip():
        weather      = get_weather(city.strip(), api_key.strip())
        weather_html = build_weather_html(city.strip(), weather)

    is_healthy = "healthy" in top_name.lower()

    if is_healthy:
        plant_name   = top_name.split("___")[0]
        result_html  = build_healthy_html(plant_name, top_conf, weather_html)
        voice_text   = f"Good news! Your {plant_name} plant is healthy with {top_conf:.0f} percent confidence."
        emoji        = "✅"
    else:
        info         = SIMPLE_INFO.get(top_name, SIMPLE_INFO["default"])
        sev          = SEVERITY_CONFIG[info["severity"]]
        timing       = get_treatment_timing(top_name, weather, info["base_days"])
        result_html  = build_disease_html(
            top_name, top_conf, info,
            sev["label"], sev["emoji"], timing, weather_html
        )
        voice_text   = (
            f"Alert! Your plant has {info['simple_name']} with {top_conf:.0f} percent confidence. "
            f"{info['simple_cause']} "
            + " ".join(info["simple_cure_steps"])
        )
        emoji = info["emoji"]

    add_to_history(top_name, top_conf, is_healthy, emoji)
    return result_html, voice_text, get_history_html()


# ==============================================================
#  GRADIO UI
# ==============================================================
with gr.Blocks(css=CSS, title="🌿 Leaf Disease Predictor") as demo:

    gr.HTML("""
    <div class='main-header'>
        <h1>🌿 Leaf Disease Predictor</h1>
        <p>Upload a leaf photo → Get instant disease detection + simple cure guide</p>
        <p style='font-size:0.85rem;opacity:0.8;'>
            पत्ती की फोटो डालें → रोग पहचान + इलाज की जानकारी पाएं
        </p>
    </div>
    """)

    voice_state = gr.State("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='section-label'>📸 Step 1: Upload or Capture Leaf</div>")
            img_input = gr.Image(
                type="pil", label="Leaf Image",
                sources=["upload", "webcam", "clipboard"],
            )
            gr.HTML("<div class='section-label'>🌤️ Step 2: Add Weather Info (Optional)</div>")
            city_input = gr.Textbox(label="Your City Name", placeholder="e.g. Patna, Delhi, Mumbai")
            api_input  = gr.Textbox(
                label="Weather API Key (optional)",
                placeholder="Paste Visual Crossing API key for weather advice",
                type="password",
            )
            gr.HTML("<div class='section-label'>🔍 Step 3: Get Results</div>")
            submit_btn = gr.Button("🔍 Analyse My Leaf", variant="primary", elem_classes=["predict-btn"])
            voice_btn  = gr.Button("🔊 Read Result Aloud", elem_classes=["voice-btn"])

        with gr.Column(scale=1):
            gr.HTML("<div class='section-label'>📋 Result</div>")
            result_out = gr.HTML(
                value="<div class='no-history' style='padding:3rem;'>"
                      "Upload a leaf photo and click Analyse 🌿</div>"
            )
            voice_text_box = gr.Textbox(visible=False)

    with gr.Row():
        with gr.Column():
            gr.HTML("<div class='section-label'>🕐 Recent Predictions</div>")
            history_out = gr.HTML(value=get_history_html())

    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class='section-card' style='margin-top:1rem;'>
                <div class='section-title'>📖 How to use this app?</div>
                <ol class='cure-steps'>
                    <li>Take a clear photo of the <b>leaf</b> (not the whole plant)</li>
                    <li>Make sure the leaf is <b>well-lit</b> — natural sunlight is best</li>
                    <li>The leaf should <b>fill most of the photo</b></li>
                    <li>Click <b>"Analyse My Leaf"</b> button</li>
                    <li>Read the result and follow the <b>step-by-step cure guide</b></li>
                    <li>Click <b>"Read Result Aloud"</b> to hear the result spoken</li>
                </ol>
                <div class='hindi-text'>
                    पत्ती की साफ फोटो लें → "Analyse My Leaf" दबाएं → इलाज की जानकारी पाएं
                </div>
            </div>
            """)

    submit_btn.click(
        fn=predict,
        inputs=[img_input, city_input, api_input],
        outputs=[result_out, voice_text_box, history_out],
    )
    voice_btn.click(
        fn=None,
        inputs=[voice_text_box],
        js="(text) => speakResult(text)",
    )
    gr.HTML(f"<script>{JS_VOICE}</script>")

if __name__ == "__main__":
    demo.launch()
