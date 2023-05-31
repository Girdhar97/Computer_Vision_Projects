# 🧬 ThyroCheck AI — Thyroid Cancer Detection with Explainable AI

A production-ready medical diagnostic system that classifies thyroid nodules
from ultrasound images as **Benign** or **Malignant** using a custom deep
learning architecture with integrated Grad-CAM explainability.

Built by **Girdhar Saini** | [HuggingFace](https://huggingface.co/Girdhar97) | [GitHub](https://github.com/Girdhar97)

---

## 🧠 What Makes This Different

Most cancer detection projects use off-the-shelf models (ResNet, VGG).
This project implements **FibonacciNet** — a custom CNN architecture where:

- Filter counts follow the **Fibonacci sequence** → `21 → 34 → 55 → 89 → 144 → 233 → 377`
- **Partial Connection Blocks (PCB)** pass earlier feature maps to deeper layers (like skip connections, but targeted)
- **Avg2MaxPooling** (`AvgPool - 2×MaxPool`) explicitly emphasizes nodule boundaries in ultrasound images
- **DepthwiseSeparableConv** in deeper blocks for efficiency without accuracy loss

And it doesn't just predict — it **explains** via Grad-CAM heatmaps showing exactly where the model focused.

---

## 🏗️ Project Structure

```
Research_Based_Cancer_Detection_XAI/
├── src/
│   ├── app.py                     # FastAPI entry point (uvicorn server)
│   ├── streamlit_app.py           # Streamlit dashboard
│   ├── requirements.txt           # Python dependencies
│   ├── backend/
│   │   └── routes.py              # /analyze and /report API endpoints
│   ├── frontend/
│   │   ├── templates/index.html   # Drag-and-drop web UI
│   │   └── static/
│   │       ├── style.css          # Medical dark theme UI
│   │       └── app.js             # Fetch API + report download logic
│   └── utils/
│       ├── config.py              # HuggingFace repo config
│       ├── model_architecture.py  # FibonacciNet, Avg2MaxPooling, DepthwiseSeparableConv
│       ├── processing.py          # Image preprocessing pipeline
│       ├── gradcam.py             # Grad-CAM heatmap generation
│       ├── report_generator.py    # Automated DOCX clinical report
│       └── logger.py              # Application logging
└── test files/
    ├── 0-non-cancer.jpg           # Sample benign image
    └── 1-cancer.jpg               # Sample malignant image
```

---

## ⚙️ Local Setup

### 1. Clone & Navigate

```bash
git clone https://github.com/Girdhar97/Computer_Vision_Projects.git
cd Computer_Vision_Projects/Research_Based_Cancer_Detection_XAI
```

### 2. Create Virtual Environment

```bash
python3 -m venv cv_env
source cv_env/bin/activate        # Linux/Mac
# cv_env\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r src/requirements.txt
```

### 4. Run Streamlit Dashboard

```bash
cd src
streamlit run streamlit_app.py
```

Opens at → `http://localhost:8501`
> WSL users: Use Network URL shown in terminal → `http://<your-wsl-ip>:8501`

### 5. Run FastAPI Web App (separate terminal, same env)

```bash
cd src
python3 app.py
```

Opens at → `http://localhost:8000`
> WSL users: Use Network IP instead → `http://<hostname -I output>:8000`

---

## 📊 API Reference

### `POST /analyze`

Upload an ultrasound image → returns prediction + Grad-CAM heatmap

```json
{
  "label": "Malignant (Cancerous)",
  "score": 0.9123,
  "percent": 91.23,
  "class_id": 1,
  "is_malignant": true,
  "original_image": "<base64>",
  "gradcam_image": "<base64>"
}
```

### `POST /report`

Upload an ultrasound image → returns auto-generated `.docx` clinical report

---

## 🧩 System Architecture

```
User uploads Ultrasound Image
          ↓
  Streamlit / FastAPI Frontend
          ↓
  Image Preprocessing (224×224, normalize /255)
          ↓
  FibonacciNet Inference (TensorFlow/Keras)
          ↓
  ┌───────────────────────────────┐
  │  Prediction                   │  → Benign / Malignant + Confidence Score
  │  Grad-CAM Heatmap             │  → Attention overlay on original image
  │  DOCX Report                  │  → Auto-generated clinical report
  └───────────────────────────────┘
          ↓
  Model weights hosted on HuggingFace
  → Girdhar97/thyroid-cancer-model
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Custom Architecture | FibonacciNet (original implementation) |
| Explainable AI | Grad-CAM (GradientTape) |
| Backend API | FastAPI + Uvicorn |
| Frontend (Dashboard) | Streamlit |
| Frontend (Web) | HTML5, CSS3, Vanilla JS, Jinja2 |
| Report Generation | python-docx |
| Model Hosting | HuggingFace Hub (`Girdhar97/thyroid-cancer-model`) |
| Dataset | [Thyroid Ultrasound Dataset — Kaggle](https://www.kaggle.com/datasets/diveshzz/thyroid-cancer-classification-ultrasound-dataset) |

---

## 📄 License

MIT License — feel free to use and build on this.

---

*Built with curiosity and a lot of terminal errors. 🔬*
