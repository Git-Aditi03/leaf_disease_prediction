# ui_components.py
# All CSS styling and HTML card builders for the UI
# Change colours, fonts, layout here — no need to touch app.py

# ==============================================================
#  CSS — Full theme
# ==============================================================
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

* { font-family: 'Nunito', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 50%, #e0f2f1 100%) !important;
    min-height: 100vh;
}

/* Header */
.main-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #2e7d32, #1b5e20);
    border-radius: 20px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(46,125,50,0.3);
}
.main-header h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
.main-header p  { font-size: 1rem; opacity: 0.9; margin: 0.5rem 0 0; }

/* Result cards */
.result-card {
    border-radius: 16px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.healthy-card { background: linear-gradient(135deg,#f0fff4,#dcfce7); border: 2px solid #86efac; }
.disease-card { background: linear-gradient(135deg,#fff7ed,#fef3c7); border: 2px solid #fca5a5; }
.error-card   { background: linear-gradient(135deg,#fef2f2,#fee2e2); border: 2px solid #fca5a5; }

/* Badges */
.status-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-weight: 800;
    font-size: 0.85rem;
    margin-bottom: 0.8rem;
}
.healthy-badge { background: #dcfce7; color: #166534; }
.disease-badge { background: #fee2e2; color: #991b1b; }

/* Names */
.plant-name   { font-size: 1.6rem; font-weight: 800; color: #166534; margin-bottom: 0.8rem; }
.disease-name { font-size: 1.4rem; font-weight: 800; color: #991b1b; margin-bottom: 0.8rem; }

/* Confidence bar */
.confidence-row {
    display: flex; align-items: center; gap: 0.8rem;
    margin-bottom: 1rem; font-weight: 600; color: #374151;
}
.conf-bar-wrap {
    flex: 1; background: #e5e7eb; border-radius: 50px; height: 14px; overflow: hidden;
}
.conf-bar         { height:100%; border-radius:50px; background: linear-gradient(90deg,#4ade80,#16a34a); }
.conf-bar-disease { background: linear-gradient(90deg,#fb923c,#dc2626); }
.conf-num { font-weight:800; font-size:1.1rem; color:#111827; min-width:3rem; }

/* Section cards */
.section-card {
    background: white; border-radius: 12px;
    padding: 1rem 1.2rem; margin: 0.8rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.section-title { font-size:1rem; font-weight:800; color:#1f2937; margin-bottom:0.4rem; }
.section-body  { color:#374151; font-size:0.95rem; line-height:1.6; }

/* Cure steps */
.cure-steps { padding-left:1.2rem; margin:0.5rem 0 0; }
.cure-steps li { color:#374151; font-size:0.95rem; padding:0.3rem 0; line-height:1.5; font-weight:600; }

/* Info cards */
.timing-card {
    background:#eff6ff; border-radius:12px; padding:0.8rem 1rem; margin:0.8rem 0;
    color:#1e40af; font-size:0.95rem; border-left:4px solid #3b82f6;
}
.warning-card {
    background:#fffbeb; border-radius:12px; padding:0.8rem 1rem; margin:0.8rem 0;
    color:#92400e; font-size:0.95rem; border-left:4px solid #f59e0b;
}
.hindi-text { color:#6b7280; font-size:0.9rem; margin-top:0.5rem; font-style:italic; }

.healthy-message {
    background:white; border-radius:12px; padding:1rem; margin:0.8rem 0;
    line-height:1.8; color:#166534; font-size:0.95rem; font-weight:600;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
}

.weather-card {
    background:linear-gradient(135deg,#dbeafe,#eff6ff); border-radius:12px;
    padding:0.6rem 1rem; margin-bottom:0.8rem; color:#1e40af;
    font-size:0.9rem; font-weight:600; border:1px solid #bfdbfe;
}

/* Error */
.error-icon { font-size:3rem; text-align:center; margin-bottom:0.5rem; }
.error-text { font-size:1rem; color:#991b1b; font-weight:600; line-height:1.6; }
.tip-box {
    background:#fefce8; border-radius:10px; padding:0.7rem 1rem; margin-top:0.8rem;
    color:#713f12; font-size:0.9rem; font-weight:600;
}

/* History */
.history-list { display:flex; flex-direction:column; gap:0.5rem; }
.history-item {
    display:flex; align-items:center; gap:0.8rem;
    background:white; border-radius:10px; padding:0.7rem 1rem;
    box-shadow:0 2px 6px rgba(0,0,0,0.06); font-weight:600;
}
.history-emoji { font-size:1.3rem; }
.history-name  { flex:1; color:#374151; font-size:0.9rem; }
.history-conf  { color:#6b7280; font-size:0.85rem; background:#f3f4f6; padding:0.2rem 0.6rem; border-radius:50px; }
.no-history    { text-align:center; color:#9ca3af; padding:2rem; font-size:0.95rem; }

/* Buttons */
.predict-btn {
    background: linear-gradient(135deg,#16a34a,#15803d) !important;
    color:white !important; font-size:1.1rem !important; font-weight:800 !important;
    border-radius:12px !important; padding:0.8rem !important;
    box-shadow:0 4px 15px rgba(22,163,74,0.3) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
}
.predict-btn:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 6px 20px rgba(22,163,74,0.4) !important;
}
.voice-btn {
    background:linear-gradient(135deg,#7c3aed,#6d28d9) !important;
    color:white !important; border-radius:12px !important; font-weight:700 !important;
}

/* Inputs */
label { font-weight:700 !important; color:#1f2937 !important; }
input[type=text], input[type=password] {
    border-radius:10px !important; border:2px solid #d1fae5 !important; font-size:0.95rem !important;
}
input[type=text]:focus, input[type=password]:focus {
    border-color:#16a34a !important; box-shadow:0 0 0 3px rgba(22,163,74,0.1) !important;
}
.section-label {
    font-size:0.8rem; font-weight:700; color:#6b7280;
    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.3rem;
}
"""

# ==============================================================
#  JavaScript for voice output
# ==============================================================
JS_VOICE = """
function speakResult(text) {
    if (!text || text.trim() === '') return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang  = 'en-IN';
    utterance.rate  = 0.9;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
}
"""

# ==============================================================
#  HTML card builders
# ==============================================================
def build_error_html(msg: str) -> str:
    lines = msg.replace("\n", "<br>")
    return f"""
    <div class='result-card error-card'>
        <div class='error-icon'>🚫</div>
        <div class='error-text'>{lines}</div>
        <div class='tip-box'>
            💡 <b>Tip:</b> Hold the leaf flat, use good lighting,
            and make sure the leaf fills most of the photo.
        </div>
    </div>"""


def build_healthy_html(plant_name: str, conf: float, weather_html: str) -> str:
    return f"""
    <div class='result-card healthy-card'>
        {weather_html}
        <div class='status-badge healthy-badge'>🟢 HEALTHY PLANT</div>
        <div class='plant-name'>{plant_name} Plant</div>
        <div class='confidence-row'>
            <span>Confidence:</span>
            <div class='conf-bar-wrap'>
                <div class='conf-bar' style='width:{conf:.0f}%'></div>
            </div>
            <span class='conf-num'>{conf:.0f}%</span>
        </div>
        <div class='healthy-message'>
            ✅ Great news! Your plant looks healthy.<br>
            🌱 Keep watering regularly and ensure good sunlight.<br>
            🔍 Check again in 2 weeks to make sure it stays healthy.
        </div>
        <div class='hindi-text'>आपका पौधा स्वस्थ है! ऐसे ही देखभाल करते रहें। 🌱</div>
    </div>"""


def build_disease_html(disease_name: str, conf: float, info: dict,
                       severity_label: str, severity_emoji: str,
                       timing: str, weather_html: str) -> str:
    steps_html = "".join([f"<li>{s}</li>" for s in info["simple_cure_steps"]])
    return f"""
    <div class='result-card disease-card'>
        {weather_html}
        <div class='status-badge disease-badge'>{severity_emoji} {severity_label.upper()} DISEASE FOUND</div>
        <div class='disease-name'>{info['emoji']} {info['simple_name']}</div>
        <div class='confidence-row'>
            <span>Confidence:</span>
            <div class='conf-bar-wrap'>
                <div class='conf-bar conf-bar-disease' style='width:{conf:.0f}%'></div>
            </div>
            <span class='conf-num'>{conf:.0f}%</span>
        </div>
        <div class='section-card'>
            <div class='section-title'>🔬 What is this disease?</div>
            <div class='section-body'>{info['simple_cause']}</div>
            <div class='hindi-text'>{info['hindi']}</div>
        </div>
        <div class='section-card'>
            <div class='section-title'>💊 How to cure — Step by Step</div>
            <ol class='cure-steps'>{steps_html}</ol>
        </div>
        <div class='timing-card'>⏰ <b>When to spray?</b> {timing}</div>
        <div class='warning-card'>⚠️ <b>Important:</b> {info['warning']}</div>
    </div>"""