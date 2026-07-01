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

# ── 페이지 설정 ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI 치아 교정 스크리닝",
    page_icon="🦷",
    layout="centered",
)

HF_REPO   = "LeeYL33/dental-ai"
MODEL_PATH = Path("dental_cnn_trained.pth")

# ── 모델 + GradCAM 로드 (캐시) ────────────────────────────────
def load_threshold() -> float:
    if Path("threshold.json").exists():
        return json.load(open("threshold.json"))["threshold"]
    return 0.5

@st.cache_resource
def load_model_and_cam():
    if not MODEL_PATH.exists():
        with st.spinner("AI 모델 로딩 중... (최초 1회, 약 43MB)"):
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

# ── 전처리 ────────────────────────────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_inference(image: Image.Image, model, gradcam: GradCAM):
    tensor = PREPROCESS(image).unsqueeze(0)

    # Grad-CAM (그래디언트 필요 → no_grad 밖에서)
    cam_map      = gradcam.generate(tensor)
    overlay_img  = gradcam.overlay(image, cam_map)

    # 점수 (no_grad로 다시)
    threshold = load_threshold()
    with torch.no_grad():
        score = torch.sigmoid(model(tensor).squeeze()).item()

    return score, overlay_img, cam_map, threshold

# ── 게이지 차트 ───────────────────────────────────────────────
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

# ── 컬러바 범례 ───────────────────────────────────────────────
def make_colorbar():
    fig, ax = plt.subplots(figsize=(4, 0.4))
    fig.patch.set_facecolor("none")
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect="auto", cmap="jet")
    ax.set_yticks([])
    ax.set_xticks([0, 128, 255])
    ax.set_xticklabels(["정상", "주의", "문제"], fontsize=9)
    plt.tight_layout(pad=0.1)
    return fig


# ════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════
st.title("🦷 AI 치아 교정 스크리닝")
st.caption("치아 사진 한 장으로 교정 필요 여부를 AI가 먼저 확인해드려요.")
st.info("이 결과는 참고용이며 전문 의료 진단을 대체하지 않습니다.", icon="ℹ️")
st.divider()

uploaded = st.file_uploader(
    "치아 정면 사진을 업로드하세요 (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("AI 분석 중..."):
        model, gradcam = load_model_and_cam()
        score, overlay_img, cam_map, threshold = run_inference(image, model, gradcam)
        verdict = get_verdict(score, threshold=threshold)

    # ── 판정 결과 ─────────────────────────────────────────────
    st.markdown(f"## {verdict['emoji']} {verdict['label']}")
    st.write(verdict["description"])
    st.success(f"💡 {verdict['cta']}")

    st.divider()

    # ── 원본 vs Grad-CAM ──────────────────────────────────────
    st.subheader("🔍 AI가 주목한 부분")
    st.caption("빨간색/노란색 = AI가 문제로 판단한 영역  |  파란색 = 양호한 영역")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**원본 사진**")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("**AI 분석 히트맵**")
        st.image(overlay_img, use_container_width=True)

    st.pyplot(make_colorbar())

    st.divider()

    # ── 점수 게이지 ───────────────────────────────────────────
    st.subheader("📊 교정 필요도")
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
    st.caption(f"`{get_score_bar(score)}`  기준: 🟢 0~35%  🟡 35~60%  🔴 60%+")

    st.divider()

    st.markdown(
        "<div style='text-align:center; color:gray; font-size:12px;'>"
        "본 AI 스크리닝 결과는 의료 진단이 아닙니다.<br>"
        "정확한 진단은 반드시 치과 전문의에게 받으세요."
        "</div>",
        unsafe_allow_html=True,
    )

else:
    st.markdown("""
    ### 이런 분들께 유용해요
    - 교정이 필요한지 궁금하지만 병원 가기 부담스러운 분
    - 아이의 치열을 체크하고 싶은 부모님
    - 교정 상담 전 미리 파악하고 싶은 분

    **사진 촬영 팁 📸**
    - 입을 살짝 벌리고 치아가 잘 보이게 정면으로 찍으세요
    - 밝은 곳에서 촬영하세요
    - 스마트폰 일반 카메라로 충분해요
    """)
