# 🚀 Deployment Guide – GitHub + Hugging Face

Follow these steps in order.

---

## Step 1 – Create a GitHub Repository

1. Go to https://github.com/new
2. Repository name: `leaf_disease_prediction`
3. Description: `Plant leaf disease detection using MobileNetV2 + real-time weather advice`
4. Set visibility (Public or Private) → click **Create repository**

---

## Step 2 – Push the Code

Open your terminal (or Git Bash on Windows):

```bash
# Navigate to the project folder you downloaded
cd leaf_disease_prediction

# Initialise Git
git init
git branch -M main

# Install Git LFS (needed for the .h5 model file > 50 MB)
git lfs install

# Add the remote
git remote add origin https://github.com/YOUR_USERNAME/leaf_disease_prediction.git

# Stage all files
git add .
git commit -m "feat: initial leaf disease prediction app"

# Push
git push -u origin main
```

> **Note:** GitHub will ask for your username + a **Personal Access Token** (not your password).  
> Generate one at: Settings → Developer settings → Personal access tokens → Tokens (classic) → New token → tick `repo`

---

## Step 3 – Create a Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `leaf_disease_prediction`
   - **SDK:** Gradio
   - **Hardware:** CPU basic (free)
3. Click **Create Space**

---

## Step 4 – Push to Hugging Face

```bash
# Add HF remote (replace YOUR_HF_USERNAME)
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/leaf_disease_prediction

# Push (HF will use your HF token as password)
git push space main
```

Generate your HF token at: https://huggingface.co/settings/tokens  
→ New token → Role: **Write**

---

## Step 5 – Add the Model File (Git LFS)

The `.h5` file is tracked by Git LFS. Make sure it's in the folder before pushing:

```bash
# Verify LFS is tracking it
git lfs ls-files
# Should show: plant_disease_model_final.h5
```

If Hugging Face rejects the push due to file size, use **HF Hub** directly:

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="plant_disease_model_final.h5",
    path_in_repo="plant_disease_model_final.h5",
    repo_id="YOUR_HF_USERNAME/leaf_disease_prediction",
    repo_type="space",
    token="YOUR_HF_TOKEN",
)
print("Upload done!")
EOF
```

---

## Step 6 – Verify

- GitHub: `https://github.com/YOUR_USERNAME/leaf_disease_prediction`
- HF Space: `https://huggingface.co/spaces/YOUR_HF_USERNAME/leaf_disease_prediction`

The Space will build automatically (takes ~3–5 min first time).  
Watch the **Logs** tab in the Space for any errors.

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `OSError: No file or directory: plant_disease_model_final.h5` | Model file not uploaded via LFS – use `huggingface_hub` upload above |
| `ModuleNotFoundError: gradio` | Check `requirements.txt` is in root and correct |
| `KeyError: days` | Visual Crossing API key is wrong or city name is invalid |
| Space stuck on "Building" | Click **Factory reboot** in HF Space settings |
