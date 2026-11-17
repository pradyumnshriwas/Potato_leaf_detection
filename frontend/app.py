"""Streamlit frontend: upload a potato leaf photo, get a prediction from the FastAPI backend."""
import base64
import html
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
LOW_CONFIDENCE = 0.70

# label, emoji, accent colour, advice
INFO = {
    "Potato___Early_blight": ("Early Blight", "🟠", "#e08a00",
        "Fungal disease (Alternaria solani). Remove affected leaves, rotate crops and consider a suitable fungicide."),
    "Potato___Late_blight": ("Late Blight", "🔴", "#c62828",
        "Serious and fast-spreading disease (Phytophthora infestans). Remove infected plants quickly and contact a local agriculture expert."),
    "Potato___healthy": ("Healthy Leaf", "🟢", "#2e7d32",
        "No sign of disease detected. Keep monitoring the crop regularly."),
}

st.set_page_config(page_title="Potato Leaf Doctor", page_icon="🥔", layout="wide")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg, #1b5e20 0%, #43a047 60%, #9ccc65 100%);
       color: #fff; padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.6rem;
       box-shadow: 0 8px 24px rgba(27,94,32,.25);}
.hero h1 {margin: 0; font-size: 2.3rem; color: #fff;}
.hero p {margin: .4rem 0 0; font-size: 1.05rem; opacity: .92;}
.chips span {display: inline-block; background: rgba(255,255,255,.18); padding: .25rem .8rem;
             border-radius: 999px; margin: .8rem .4rem 0 0; font-size: .85rem;}
[data-testid="stVerticalBlockBorderWrapper"] {background: #fff; border-radius: 18px !important;
       border: none !important; box-shadow: 0 2px 14px rgba(0,0,0,.08); padding: .6rem .8rem;}
.card-title {font-size: 1.1rem; font-weight: 700; color: #1b5e20; margin-bottom: .4rem;}
.preview img {width: 100%; border-radius: 14px; max-height: 380px; object-fit: contain; background: #f3f8ef;}
.muted {color: #5b6b5f; font-size: .9rem;}
.advice {border-left: 5px solid; padding: .8rem 1rem; border-radius: 8px; margin-top: .8rem;}
.bar-row {margin: .65rem 0;}
.bar-label {display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: 4px;}
.bar-bg {background: #e4efdc; border-radius: 999px; height: 12px; overflow: hidden;}
.bar-fill {height: 100%; border-radius: 999px;}
.placeholder {text-align: center; padding: 3rem 1rem; color: #5b6b5f;}
.placeholder .big {font-size: 3.5rem;}
div.stButton > button {border-radius: 12px; padding: .65rem 1rem; font-weight: 600; width: 100%;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🥔 Potato Leaf Doctor</h1>
  <p>Upload a photo of a potato leaf and our CNN will tell you whether it is healthy or diseased.</p>
  <div class="chips"><span>🧠 CNN · TensorFlow</span><span>⚡ FastAPI backend</span><span>🎈 Streamlit UI</span></div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ System status")
    try:
        ok = requests.get(f"{API_URL}/health", timeout=3).json().get("model_loaded")
        st.success("API connected · model ready" if ok else "API up, model not loaded")
    except requests.RequestException:
        st.error("API not reachable. Start the FastAPI server first.")
    st.markdown("### 🍃 Classes")
    for label, emoji, _, _ in INFO.values():
        st.markdown(f"{emoji} {label}")
    st.markdown("### 📸 Tips for best results")
    st.markdown("- One leaf, filling most of the frame\n- Good daylight, no blur\n- Plain background if possible")
    st.markdown("### 🧪 About")
    st.caption("CNN trained on PlantVillage potato images (128×128 RGB). Validation accuracy ≈ 91%. "
               "Not a substitute for expert diagnosis.")

left, right = st.columns([1, 1], gap="large")

with left:
  with st.container(border=True):
    st.markdown('<div class="card-title">1 · Upload a leaf image</div>', unsafe_allow_html=True)
    source = st.radio("Source", ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")
    file = (st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if source == "Upload" else st.camera_input("Take a photo", label_visibility="collapsed"))
    if file:
        mime = file.type or "image/jpeg"
        b64 = base64.b64encode(file.getvalue()).decode()
        st.markdown(f'<div class="preview"><img src="data:{mime};base64,{b64}" alt="leaf"></div>',
                    unsafe_allow_html=True)
        st.caption(html.escape(file.name or "camera photo"))
    analyze = st.button("🔍 Analyze leaf", type="primary", disabled=file is None)

with right:
  with st.container(border=True):
    st.markdown('<div class="card-title">2 · Result</div>', unsafe_allow_html=True)
    if file and analyze:
        with st.spinner("Analyzing the leaf..."):
            res, err = None, None
            try:
                r = requests.post(f"{API_URL}/predict",
                                  files={"file": (file.name or "photo.jpg", file.getvalue(), file.type or "image/jpeg")},
                                  timeout=30)
                if r.ok:
                    res = r.json()
                else:
                    err = r.json().get("detail", "Prediction failed.")
            except requests.RequestException:
                err = "Could not reach the backend. Is the FastAPI server running?"
        if err:
            st.error(err)
        else:
            label, emoji, color, advice = INFO[res["predicted_class"]]
            conf = res["confidence"]
            st.markdown(
                f'<div class="muted">Prediction</div>'
                f'<div style="font-size:2rem;font-weight:700;color:{color}">{emoji} {label}</div>'
                f'<div class="muted" style="margin-top:.8rem">Model score</div>'
                f'<div style="font-size:3.2rem;font-weight:800;line-height:1.1;color:{color}">{conf:.1%}</div>'
                f'<div class="advice" style="border-color:{color};background:{color}14">{advice}</div>',
                unsafe_allow_html=True,
            )
            if conf < LOW_CONFIDENCE:
                st.warning("Low confidence. Try a sharper, well-lit photo of a single leaf.")
            st.markdown("<p class='muted' style='margin-top:1.2rem'>All class probabilities</p>", unsafe_allow_html=True)
            for key, p in sorted(res["probabilities"].items(), key=lambda kv: -kv[1]):
                l, _, c, _ = INFO[key]
                st.markdown(
                    f'<div class="bar-row"><div class="bar-label"><span>{l}</span><b>{p:.1%}</b></div>'
                    f'<div class="bar-bg"><div class="bar-fill" style="width:{p*100:.1f}%;background:{c}"></div></div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div class="placeholder"><div class="big">🌿</div>'
                    '<p>Upload a leaf and press <b>Analyze</b> to see the diagnosis here.</p></div>',
                    unsafe_allow_html=True)

st.caption("Built for the CNN independent project assessment · Predictions are AI estimates, not expert diagnosis.")
