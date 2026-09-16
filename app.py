import json
import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from huggingface_hub import hf_hub_download

from cnn_model import DentalCNN
from patient_report import get_verdict, get_score_bar
from gradcam import GradCAM

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="OrthoLense",
    page_icon="🦷",
    layout="centered",
)

HF_REPO   = "LeeYL33/dental-ai"
MODEL_PATH = Path("dental_cnn_trained.pth")

# ── Load model + GradCAM (cached) ───────────────────────────────
def load_threshold() -> float:
    if Path("threshold.json").exists():
        return json.load(open("threshold.json"))["threshold"]
    return 0.5

@st.cache_resource
def load_model_and_cam():
    if not MODEL_PATH.exists():
        with st.spinner("Loading AI model... (first time only, ~43MB)"):
            hf_hub_download(
                repo_id=HF_REPO,
                filename="dental_cnn_trained.pth",
                local_dir=".",
            )
    model = DentalCNN(pretrained=False)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
    model.eval()
    cam = GradCAM(model)
    return model, cam

# ── Preprocessing ────────────────────────────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_inference(image: Image.Image, model, gradcam: GradCAM):
    tensor = PREPROCESS(image).unsqueeze(0)

    # Grad-CAM (needs gradients → outside no_grad)
    cam_map      = gradcam.generate(tensor)
    overlay_img  = gradcam.overlay(image, cam_map)

    # Score (back under no_grad)
    threshold = load_threshold()
    with torch.no_grad():
        score = torch.sigmoid(model(tensor).squeeze()).item()

    return score, overlay_img, cam_map, threshold

# ── Gauge chart ───────────────────────────────────────────────
def make_gauge(score: float):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    color = "#e74c3c" if score >= 0.6 else "#f39c12" if score >= 0.35 else "#2ecc71"
    ax.barh(0, score, color=color, height=0.5)
    ax.barh(0, 1 - score, left=score, color="#e0e0e0", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.text(score / 2, 0, f"{int(score*100)}%", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    ax.axvline(0.35, color="#f39c12", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.axvline(0.60, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7)
    plt.tight_layout(pad=0)
    return fig

# ── Colorbar legend ───────────────────────────────────────────────
def make_colorbar():
    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.patch.set_facecolor("none")
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="jet")
    ax.set_yticks([])
    ax.set_xticks([0, 128, 255])
    ax.set_xticklabels(["Normal", "Caution", "Concern"], fontsize=9)
    plt.tight_layout(pad=0.1)
    return fig


# ════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════
st.title("🦷 OrthoLense")
st.caption("Upload a single photo of your teeth and let AI give you a first read on whether orthodontic treatment may be needed.")
st.info("This result is for reference only and does not replace a professional medical diagnosis.", icon="ℹ️")
st.divider()

uploaded = st.file_uploader(
    "Upload a front-facing photo of your teeth (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Analyzing with AI..."):
        model, gradcam = load_model_and_cam()
        score, overlay_img, cam_map, threshold = run_inference(image, model, gradcam)
        verdict = get_verdict(score, threshold=threshold)

    # ── Verdict result ─────────────────────────────────────────────
    st.markdown(f"## {verdict['emoji']} {verdict['label']}")
    st.write(verdict["description"])
    st.success(f"💡 {verdict['cta']}")

    st.divider()

    # ── Original vs Grad-CAM ──────────────────────────────────────
    st.subheader("🔍 What the AI focused on")
    st.caption("Red/yellow = areas the AI flagged as a concern  |  Blue = areas that look fine")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original photo**")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("**AI analysis heatmap**")
        st.image(overlay_img, use_container_width=True)

    st.pyplot(make_colorbar())

    st.divider()

    # ── Score gauge ───────────────────────────────────────────
    st.subheader("📊 Orthodontic need score")
    col_a, col_b = st.columns([4, 1])
    with col_a:
        st.pyplot(make_gauge(score))
    with col_b:
        color = "red" if score >= 0.6 else "orange" if score >= 0.35 else "green"
        st.markdown(
            f"<p style='font-size:32px; font-weight:bold; color:{color}; text-align:center'>"
            f"{int(score*100)}%</p>",
            unsafe_allow_html=True,
        )
    st.caption(f"`{get_score_bar(score)}`  Scale: 🟢 0-35%  🟡 35-60%  🔴 60%+")

    st.divider()

    st.markdown(
        "<div style='text-align:center; color:gray; font-size:12px;'>"
        "This AI screening result is not a medical diagnosis.<br>"
        "Please see a licensed orthodontist for an accurate diagnosis."
        "</div>",
        unsafe_allow_html=True,
    )

else:
    st.markdown("""
    ### Who this is for
    - Anyone curious whether they need orthodontic treatment but hesitant to visit a clinic
    - Parents who want to check their child's dental alignment
    - Anyone who wants a preview before an orthodontic consultation

    **Photo tips 📸**
    - Open your mouth slightly and take a front-facing shot with your teeth clearly visible
    - Shoot in a well-lit area
    - A regular smartphone camera is enough
    """)
