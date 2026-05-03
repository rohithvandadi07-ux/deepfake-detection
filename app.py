import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import io
import tempfile
import os

st.set_page_config(page_title="🎭 Deepfake Detector", layout="wide")

st.markdown("""
<style>
.big-title { font-size: 2.5rem; font-weight: 800; }
.fake-box { background: #ff4b4b22; border-left: 4px solid #ff4b4b; padding: 1rem; border-radius: 8px; }
.real-box { background: #00c85322; border-left: 4px solid #00c853; padding: 1rem; border-radius: 8px; }
.metric-card { background: #1e1e1e; padding: 1rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎭 Deepfake Detection System</p>', unsafe_allow_html=True)
st.markdown("#### Explainable AI model with visual artifact analysis & heatmap visualization")

# ── Core Analysis ──────────────────────────────────────────────
def detect_faces(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    return faces

def analyze_artifacts(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur_score    = cv2.Laplacian(gray, cv2.CV_64F).var()
    noise_level   = np.std(image_np.astype(float))
    color_std     = np.std(image_np, axis=(0, 1))
    freq          = np.fft.fft2(gray)
    freq_shift    = np.fft.fftshift(freq)
    magnitude     = np.log(np.abs(freq_shift) + 1)
    freq_score    = np.std(magnitude)

    indicators, reasons = 0, []

    if blur_score < 80:
        indicators += 1
        reasons.append(("🔴", "Unusual surface smoothness", "Common in GAN-generated faces"))
    else:
        reasons.append(("🟢", "Natural sharpness detected", f"Laplacian score: {blur_score:.1f}"))

    if noise_level < 22:
        indicators += 1
        reasons.append(("🔴", "Low natural noise", "Deepfakes often lack organic sensor noise"))
    else:
        reasons.append(("🟢", "Natural noise pattern", f"Noise level: {noise_level:.1f}"))

    if abs(color_std[0] - color_std[1]) > 12:
        indicators += 1
        reasons.append(("🔴", "Color channel imbalance", "Abnormal RGB distribution detected"))
    else:
        reasons.append(("🟢", "Balanced color channels", "Normal RGB distribution"))

    if freq_score < 3.5:
        indicators += 1
        reasons.append(("🔴", "Abnormal frequency pattern", "GAN artifacts in frequency domain"))
    else:
        reasons.append(("🟢", "Normal frequency spectrum", f"Freq score: {freq_score:.2f}"))

    fake_score = min(indicators / 4.0, 1.0)
    return fake_score, reasons, blur_score, noise_level, magnitude

def generate_heatmap(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    diff = cv2.absdiff(gray, blur)
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap_rgb, 0.4, 0)
    return heatmap_rgb, overlay

def draw_face_boxes(image_np, faces, label, color):
    out = image_np.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 3)
        cv2.putText(out, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out

def analyze_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores, frames_analyzed = [], []
    sample_every = max(1, total // 10)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            score, _, _, _, _ = analyze_artifacts(rgb)
            scores.append(score)
            frames_analyzed.append(frame_count)
        frame_count += 1
    cap.release()
    return scores, frames_analyzed, total

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Detection Mode", ["📸 Image", "🎥 Video"])
show_heatmap   = st.sidebar.checkbox("Show Artifact Heatmap", value=True)
show_frequency = st.sidebar.checkbox("Show Frequency Analysis", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works:**
- Analyzes 4 artifact indicators
- Checks frequency domain patterns
- Detects GAN smoothing artifacts
- Visualizes suspicious regions
""")
st.sidebar.markdown("Built by **Rohith V** | [🔗 LinkedIn](https://www.linkedin.com/in/rohithv0507/)")

# ── IMAGE MODE ─────────────────────────────────────────────────
if mode == "📸 Image":
    uploaded = st.file_uploader("📂 Upload a face image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        image    = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

        fake_score, reasons, blur, noise, magnitude = analyze_artifacts(image_np)
        real_score = 1 - fake_score
        faces      = detect_faces(image_np)
        label_col  = (255, 50, 50) if fake_score > 0.5 else (50, 255, 50)
        label_txt  = "FAKE" if fake_score > 0.5 else "REAL"
        boxed      = draw_face_boxes(image_np, faces, label_txt, label_col)
        heatmap, overlay = generate_heatmap(image_np)

        # ── Result Banner ──
        st.markdown("---")
        if fake_score > 0.5:
            st.markdown(f'<div class="fake-box"><h2>🎭 DEEPFAKE DETECTED</h2><p>Fake probability: <b>{fake_score*100:.1f}%</b> — {len(faces)} face(s) found</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="real-box"><h2>✅ AUTHENTIC IMAGE</h2><p>Real probability: <b>{real_score*100:.1f}%</b> — {len(faces)} face(s) found</p></div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Metrics ──
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎯 Fake Score",    f"{fake_score*100:.1f}%")
        col2.metric("✅ Real Score",    f"{real_score*100:.1f}%")
        col3.metric("🔍 Sharpness",     f"{blur:.1f}")
        col4.metric("📊 Noise Level",   f"{noise:.1f}")

        # ── Confidence Bars ──
        st.markdown("#### 📊 Confidence")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Real**")
            st.progress(real_score)
        with c2:
            st.markdown("**🎭 Fake**")
            st.progress(fake_score)

        st.markdown("---")

        # ── Image Views ──
        cols = ["📷 Original", "👤 Face Detection", "🌡️ Artifact Heatmap", "🔀 Overlay"]
        tabs = st.tabs(cols)
        with tabs[0]: st.image(image_np,  use_column_width=True)
        with tabs[1]: st.image(boxed,     use_column_width=True)
        with tabs[2]: st.image(heatmap,   use_column_width=True) if show_heatmap else st.info("Enable heatmap in sidebar")
        with tabs[3]: st.image(overlay,   use_column_width=True)

        # ── Frequency ──
        if show_frequency:
            st.markdown("---")
            st.markdown("#### 🔬 Frequency Domain Analysis")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(magnitude, cmap='inferno')
            ax.set_title("Frequency Spectrum (anomalies appear as rings/artifacts)")
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        # ── Artifact Report ──
        st.markdown("---")
        st.markdown("#### 🧬 Artifact Analysis Report")
        for icon, title, detail in reasons:
            st.markdown(f"**{icon} {title}** — _{detail}_")

# ── VIDEO MODE ─────────────────────────────────────────────────
elif mode == "🎥 Video":
    uploaded = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        tfile.flush()

        st.info("⚡ Analyzing 10 sampled frames for efficiency...")
        with st.spinner("🤖 Processing video..."):
            scores, frame_nums, total_frames = analyze_video_frames(tfile.name)
        os.unlink(tfile.name)

        avg_score = np.mean(scores)
        max_score = np.max(scores)
        fake_frames = sum(1 for s in scores if s > 0.5)

        st.markdown("---")
        if avg_score > 0.5:
            st.error(f"## 🎭 VIDEO LIKELY DEEPFAKE — Avg fake score: {avg_score*100:.1f}%")
        else:
            st.success(f"## ✅ VIDEO LIKELY AUTHENTIC — Avg fake score: {avg_score*100:.1f}%")

        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Avg Fake Score",   f"{avg_score*100:.1f}%")
        c2.metric("🔺 Peak Fake Score",  f"{max_score*100:.1f}%")
        c3.metric("🎭 Suspicious Frames", f"{fake_frames}/{len(scores)}")

        st.markdown("#### 📈 Frame-by-Frame Analysis")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(frame_nums, [s*100 for s in scores], marker='o', color='#ff4b4b', linewidth=2)
        ax.axhline(y=50, color='gray', linestyle='--', label='Threshold (50%)')
        ax.fill_between(frame_nums, [s*100 for s in scores], alpha=0.3, color='#ff4b4b')
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Fake Score (%)")
        ax.set_title("Deepfake Score Across Video Frames")
        ax.legend()
        st.pyplot(fig)
        plt.close()

else:
    st.info("👆 Upload an image or video to analyze!")
    st.markdown("""
    **Test with:**
    - 🌐 [thispersondoesnotexist.com](https://thispersondoesnotexist.com) → always deepfake
    - 📸 Any real photo → should be authentic
    - 🎥 Any video with faces → frame-by-frame analysis
    """)
