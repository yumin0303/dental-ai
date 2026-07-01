import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image


class GradCAM:
    """
    Grad-CAM 히트맵 생성.
    layer3(14×14) 사용 → layer4(7×7)보다 2배 정밀한 공간 해상도.
    """

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        # layer3 = backbone[6] → 출력 해상도 14×14
        # forward hook에서 activation에 직접 grad hook 등록 (동결 레이어 우회)
        model.backbone[6].register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, input, output):
        output.requires_grad_(True)
        self.activations = output
        output.register_hook(self._grad_hook)

    def _grad_hook(self, grad):
        self.gradients = grad

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """0~1 정규화된 CAM 반환 (14×14 numpy array)."""
        self.model.eval()
        with torch.enable_grad():
            output = self.model(input_tensor)
            self.model.zero_grad()
            output.backward()

        # 채널별 그래디언트 평균 × 활성화
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().detach().numpy()

        # 가우시안 스무딩으로 노이즈 제거
        cam = gaussian_filter(cam, sigma=1.0)

        # 상위 50% 활성화만 사용 (하위 노이즈 제거)
        threshold = np.percentile(cam, 50)
        cam = np.where(cam >= threshold, cam - threshold, 0)

        # 0~1 정규화
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def _tooth_mask(self, image_pil: Image.Image) -> np.ndarray:
        """
        치아가 아닌 영역을 억제하는 마스크 생성.
        - 어두운 영역(목구멍·입안): 억제
        - 붉은/분홍 계열(잇몸·입술): hue + 채도로 억제
        - 피부색(손·얼굴): 채도로 억제
        - 밝고 채도 낮은 흰/크림색(치아): 통과
        """
        rgb = np.array(image_pil.convert("RGB")).astype(float) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        brightness = (r + g + b) / 3.0
        max_rgb = np.maximum.reduce([r, g, b])
        min_rgb = np.minimum.reduce([r, g, b])
        chroma = max_rgb - min_rgb
        saturation = np.where(max_rgb > 1e-6, chroma / max_rgb, 0.0)

        # HSV hue (0~1): 붉은색=0 or 1, 황색=1/6, 녹색=2/6
        hue = np.zeros_like(brightness)
        m = chroma > 1e-6
        mask_r = m & (max_rgb == r)
        mask_g = m & (max_rgb == g)
        mask_b = m & (max_rgb == b)
        hue[mask_r] = (((g - b) / (chroma + 1e-9)) % 6)[mask_r] / 6.0
        hue[mask_g] = (((b - r) / (chroma + 1e-9)) + 2)[mask_g] / 6.0
        hue[mask_b] = (((r - g) / (chroma + 1e-9)) + 4)[mask_b] / 6.0

        # 붉은색/분홍 계열 억제 (hue < 0.08 or hue > 0.92, 즉 0° 부근)
        red_hue = np.minimum(hue, 1.0 - hue)  # 0=빨강, 클수록 빨강과 거리
        red_suppress = np.clip(red_hue / 0.08, 0.0, 1.0)  # 0.08 이내(빨강/분홍) → 억제

        # 어두운 영역 억제 (목구멍·입안): 0.20~0.40 전환
        dark_weight = np.clip((brightness - 0.20) / 0.20, 0.0, 1.0)

        # 채도 높은 영역 억제 (피부·잇몸): 0.15~0.35 전환
        sat_weight = np.clip(1.0 - (saturation - 0.15) / 0.20, 0.0, 1.0)

        return dark_weight * sat_weight * red_suppress

    def overlay(self, original: Image.Image, cam: np.ndarray) -> Image.Image:
        """
        활성화 강도에 따라 알파를 가변 적용.
        - 활성화 높은 곳: 히트맵 진하게
        - 활성화 낮은 곳: 원본 이미지 유지
        치아 영역 마스크로 목구멍·잇몸·손 영역 억제.
        """
        w, h = original.size

        # CAM을 원본 크기로 업샘플링
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_resized = np.array(cam_pil.resize((w, h), Image.BILINEAR)) / 255.0

        # 치아 마스크 적용 (비치아 영역 억제)
        mask = self._tooth_mask(original)
        cam_resized = cam_resized * mask

        # 마스크 후 재정규화 (치아 영역 내에서 대비 복원)
        if cam_resized.max() > 1e-8:
            cam_resized = cam_resized / cam_resized.max()

        # 컬러맵 적용 (파랑=낮음, 빨강=높음)
        heatmap = (plt.cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)

        # 활성화 강도에 따라 알파 가변 적용 (최대 65%)
        alpha_map = cam_resized * 0.65
        alpha_3d  = np.stack([alpha_map] * 3, axis=2)

        orig_np = np.array(original.convert("RGB")).astype(float)
        blended = (alpha_3d * heatmap + (1 - alpha_3d) * orig_np).clip(0, 255).astype(np.uint8)

        return Image.fromarray(blended)
