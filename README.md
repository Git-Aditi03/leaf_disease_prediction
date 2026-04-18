# 🌿 Leaf Disease Prediction

A deep-learning web app that detects **38 plant diseases** from leaf photos and provides **weather-aware treatment advice** using real-time data.

## 🔗 Live Demo
👉 [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/leaf_disease_prediction)

---

## 🧠 Model

| Item | Detail |
|------|--------|
| Architecture | MobileNetV2 (transfer learning) |
| Dataset | PlantVillage (38 classes) |
| Input size | 224 × 224 RGB |
| Output | Softmax over 38 classes |
| Framework | TensorFlow 2.15 / Keras |

---

## 📂 Repository Structure

```
leaf_disease_prediction/
│
├── app.py                          # Gradio app (Hugging Face Spaces entry point)
├── plant_disease_model_final.h5    # Trained model weights  ← upload via Git LFS
├── requirements.txt                # Python dependencies
├── training.ipynb                  # Clean training notebook (Google Colab ready)
└── README.md
```

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/leaf_disease_prediction.git
cd leaf_disease_prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
python app.py
```

Open `http://localhost:7860` in your browser.

---

## ☁️ Deploy on Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Gradio**
   - Python: **3.10**
2. Push this repo (see Git LFS note below for the `.h5` file):

```bash
# Install Git LFS once
git lfs install

# Track the model file
git lfs track "*.h5"
git add .gitattributes

git add .
git commit -m "Initial commit"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/leaf_disease_prediction
git push space main
```

---

## 🌤️ Weather Integration (optional)

The app can fetch live weather from **Visual Crossing** to give time-sensitive spray advice.

1. Get a free API key at [visualcrossing.com](https://www.visualcrossing.com/sign-up)
2. Paste it in the app's **API Key** field (it is never stored)

> **Security:** Never hard-code your API key in `app.py`. If you want to set it as a default for a private Space, use [Hugging Face Secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).

---

## 🌱 Supported Plant–Disease Classes (38)

<details>
<summary>Click to expand</summary>

- Apple: Apple Scab, Black Rot, Cedar Apple Rust, Healthy  
- Blueberry: Healthy  
- Cherry: Powdery Mildew, Healthy  
- Corn: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy  
- Grape: Black Rot, Esca, Leaf Blight, Healthy  
- Orange: Huanglongbing  
- Peach: Bacterial Spot, Healthy  
- Pepper Bell: Bacterial Spot, Healthy  
- Potato: Early Blight, Late Blight, Healthy  
- Raspberry: Healthy  
- Soybean: Healthy  
- Squash: Powdery Mildew  
- Strawberry: Leaf Scorch, Healthy  
- Tomato: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy  

</details>

---

## 📄 License

MIT License – see [LICENSE](LICENSE).
