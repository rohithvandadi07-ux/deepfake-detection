# 🎭 Deepfake Detection System

> Explainable AI model that detects GAN-generated and AI face-swapped images with visual artifact heatmaps, frequency domain analysis and frame-by-frame video scanning.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## 💡 Problem Statement
With the rapid rise of generative AI, deepfake images and videos have become a serious threat to digital trust and security. This system provides an explainable, real-time solution to detect AI-generated faces.

## 🚀 Features
- 🎭 Detects GAN-generated & AI face-swapped images
- 🌡️ **Visual artifact heatmap** — highlights suspicious regions
- 🔬 **Frequency domain analysis** — detects GAN spectral artifacts
- 👤 Automatic face detection with bounding boxes
- 🎥 **Video support** — frame-by-frame analysis with trend chart
- 📊 4-indicator artifact scoring system
- ⚡ Runs fully locally — no API needed

## ⚙️ Tech Stack
- Python, OpenCV, MobileNetV2, TensorFlow
- Matplotlib for heatmap & frequency visualization
- Streamlit for interactive dashboard
- PIL / NumPy for image processing

## 📊 How It Works
The system analyzes 4 artifact indicators:
1. **Surface Smoothness** — GAN faces are unnaturally smooth
2. **Noise Pattern** — Real photos have organic sensor noise
3. **Color Channel Balance** — GANs often produce RGB imbalances
4. **Frequency Spectrum** — GAN artifacts appear as rings in FFT

## ▶️ How to Run
```bash
git clone https://github.com/rohithvandadi07-ux/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
streamlit run app.py
```

## 🧪 Test It
- Upload from [thispersondoesnotexist.com](https://thispersondoesnotexist.com) → **FAKE**
- Upload a real photo → **AUTHENTIC**

## 👤 Author
**Rohith V** | [🔗 LinkedIn](https://www.linkedin.com/in/rohithv0507/)
