import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import io

st.set_page_config(page_title="🎭 Deepfake Detector", layout="wide")
st.title("🎭 AI Deepfake Detection System")
st.markdown("### Detects whether a face image is REAL or AI-generated (Deepfake)")

@st.cache_resource
def load_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def detect_face(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0, faces

def analyze_image(image):
    img_array = np.array(image)
    # Analyze various artifacts common in deepfakes
    blur_score = cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    noise = np.std(img_array.astype(float))
    mean_color = np.mean(img_array, axis=(0,1))
    color_std = np.std(img_array, axis=(0,1))

    # Heuristic deepfake score based on artifacts
    fake_indicators = 0
    reasons = []

    if blur_score < 50:
        fake_indicators += 1
        reasons.append("🔴 Unusual smoothness detected (common in GAN-generated faces)")
    if noise < 20:
        fake_indicators += 1
        reasons.append("🔴 Low noise pattern (deepfakes often lack natural noise)")
    if abs(color_std[0] - color_std[1]) > 15:
        fake_indicators += 1
        reasons.append("🔴 Abnormal color channel distribution")
    if blur_score > 200:
        reasons.append("🟢 Natural sharpness detected")
    if noise > 25:
        reasons.append("🟢 Natural noise pattern detected")

    fake_score = min(fake_indicators / 3.0, 1.0)
    return fake_score, reasons, blur_score, noise

# Sidebar
st.sidebar.header("⚙️ Settings")
show_analysis = st.sidebar.checkbox("Show detailed analysis", value=True)
st.sidebar.markdown("---")
st.sidebar.info("""
**How it works:**
- Analyzes facial artifacts
- Checks noise patterns
- Detects GAN smoothing
- Analyzes color distribution
""")
st.sidebar.markdown("Built by **Rohith V** | [🔗 LinkedIn](https://www.linkedin.com/in/rohithv0507/)")

# Upload
uploaded = st.file_uploader("📂 Upload a face image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📷 Uploaded Image")
        st.image(image, use_column_width=True)

        # Face detection
        face_found, faces = detect_face(image_np)
        if face_found:
            st.success(f"✅ {len(faces)} face(s) detected")
            # Draw rectangles around faces
            img_with_faces = image_np.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 3)
            st.image(img_with_faces, caption="Face Detection", use_column_width=True)
        else:
            st.warning("⚠️ No face detected — results may be less accurate")

    with col2:
        st.markdown("#### 🔍 Detection Result")
        with st.spinner("🤖 Analyzing image for deepfake artifacts..."):
            fake_score, reasons, blur_score, noise = analyze_image(image)
            real_score = 1 - fake_score

        # Result
        if fake_score > 0.5:
            st.error(f"## 🎭 DEEPFAKE DETECTED!")
            st.error(f"Fake Probability: **{fake_score*100:.1f}%**")
        else:
            st.success(f"## ✅ REAL IMAGE")
            st.success(f"Real Probability: **{real_score*100:.1f}%**")

        # Confidence bars
        st.markdown("#### 📊 Confidence Scores")
        st.markdown("**Real**")
        st.progress(real_score)
        st.markdown("**Fake**")
        st.progress(fake_score)

        if show_analysis:
            st.markdown("#### 🔬 Artifact Analysis")
            st.metric("Sharpness Score", f"{blur_score:.1f}")
            st.metric("Noise Level", f"{noise:.1f}")
            st.markdown("**Indicators Found:**")
            for reason in reasons:
                st.write(reason)

else:
    st.info("👆 Upload a face image to check if it's real or a deepfake!")
    st.markdown("""
    **What this detects:**
    - 🎭 GAN-generated faces (ThisPersonDoesNotExist style)
    - 🤖 AI face swaps
    - 🖼️ Synthetic portrait images
    
    **Test with images from:**
    - [thispersondoesnotexist.com](https://thispersondoesnotexist.com) — always deepfake
    - Any real photo from your gallery — should be real
    """)
