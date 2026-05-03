# 🎭 AI Deepfake Detection System

An AI-powered deepfake detector that analyzes face images for GAN artifacts, noise patterns, and color inconsistencies to determine if an image is real or AI-generated.

## 🚀 Features
- 🎭 Detects GAN-generated and AI face-swapped images
- 👤 Automatic face detection with bounding boxes
- 📊 Confidence scores for Real vs Fake
- 🔬 Detailed artifact analysis (sharpness, noise, color)
- ⚡ Runs fully locally — no API needed

## 🛠️ Tech Stack
- Python, TensorFlow, MobileNetV2
- OpenCV for face detection
- Streamlit for dashboard
- PIL / NumPy for image processing

## ▶️ How to Run
```bash
git clone https://github.com/rohithvandadi07-ux/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
streamlit run app.py
```

## 🧪 Test It
- Upload any image from [thispersondoesnotexist.com](https://thispersondoesnotexist.com) → should be FAKE
- Upload a real photo → should be REAL

## 👤 Author
**Rohith V** | [🔗 LinkedIn](https://www.linkedin.com/in/rohithv0507/)
