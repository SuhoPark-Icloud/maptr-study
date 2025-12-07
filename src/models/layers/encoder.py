"""
`MapEncoder` 클래스를 정의하며, 이는 MapTR 모델의 인코더 컴포넌트입니다.
ResNet50 백본과 DepthNet을 사용하여 다중 뷰 이미지 특징을 추출하고 깊이 분포를 예측합니다.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

_original_style_use = plt.style.use


def _patched_style_use(style):
    """
    `matplotlib.pyplot.style.use` 함수의 패치 버전입니다.
    레거시 seaborn 스타일 이름을 처리하고, 스타일 적용을 시도하며,
    스타일을 찾을 수 없는 경우에도 오류 없이 진행됩니다.
    """
    if style == "seaborn-whitegrid":
        style = "seaborn-v0_8-whitegrid"
    try:
        _original_style_use(style)
    except OSError:
        pass


plt.style.use = _patched_style_use


class MapEncoder(nn.Module):
    def __init__(self, C=64, D=59):
        """
        `MapTR` 인코더를 초기화합니다.
        ResNet50 백본과 DepthNet으로 구성되며,
        출력 특징 채널 수 (C)와 깊이 구간 수 (D)를 설정합니다.

        Args:
            C (int): 출력 특징 채널 수 (LSS로 전달될 특징의 개수).
            D (int): 깊이 구간 수 (LSS의 깊이 범위에 사용될 개수).
        """
        super().__init__()
        self.C = C
        self.D = D

        # 1. Backbone (ResNet50)
        # Pre-trained 가중치를 사용하여 학습 속도를 높입니다.
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # 우리는 'Layer 3' (Stride 16)까지의 특징만 사용합니다.
        # Layer 4 (Stride 32)는 맵핑에 쓰기에 해상도가 너무 작습니다.
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # ResNet Layer3의 출력 채널은 1024개입니다.
        backbone_dim = 1024

        # 2. DepthNet (1x1 Conv)
        # 이미지 특징에서 '깊이 분포(D)'와 '의미 특징(C)'을 동시에 예측합니다.
        # 출력 채널: D (Depth) + C (Feature)
        self.depth_net = nn.Conv2d(
            backbone_dim, self.D + self.C, kernel_size=1, padding=0
        )

    def get_depth_feat(self, x):
        """
        백본과 DepthNet을 통해 입력 이미지를 처리하여 깊이 확률과 이미지 특징을 추출합니다.

        Args:
            x (torch.Tensor): 입력 이미지 텐서 [B*N, 3, H, W].

        Returns:
            tuple:
                - depth (torch.Tensor): Softmax 처리된 깊이 확률 텐서 [B*N, D, fH, fW].
                - feat (torch.Tensor): 추출된 이미지 특징 텐서 [B*N, C, fH, fW].
        """
        # 1. Backbone Forward
        # [B*N, 3, 450, 800] -> [B*N, 1024, 29, 50] (approx /16)
        x = self.backbone(x)

        # 2. DepthNet Forward
        # [B*N, 1024, fH, fW] -> [B*N, D+C, fH, fW]
        x = self.depth_net(x)

        # 3. Split into Depth and Feature
        # depth: 앞쪽 D개 채널, feat: 뒤쪽 C개 채널
        depth = x[:, : self.D]
        feat = x[:, self.D :]

        # 4. Depth Softmax
        # 깊이 값은 확률(Probability)이어야 하므로 Softmax 적용
        depth = F.softmax(depth, dim=1)

        return depth, feat

    def forward(self, imgs):
        """
        `MapEncoder`의 포워드 패스.
        여러 카메라의 이미지 배치(`imgs`)를 입력으로 받아, 배치와 카메라 차원을 결합한 후
        `get_depth_feat`를 통해 처리합니다.
        이후 출력(`depth`, `feat`)을 다시 배치 및 카메라 차원으로 분리하여 반환합니다.

        Args:
            imgs (torch.Tensor): 입력 이미지 텐서 [B, N, 3, H, W].

        Returns:
            tuple:
                - depth (torch.Tensor): 깊이 확률 텐서 [B, N, D, fH, fW].
                - feat (torch.Tensor): 이미지 특징 텐서 [B, N, C, fH, fW].
        """
        B, N, C_in, H, W = imgs.shape

        # Combine Batch and Camera dims for efficient processing
        imgs = imgs.view(B * N, C_in, H, W)

        # Encoder Forward
        depth, feat = self.get_depth_feat(imgs)

        # Reshape back to separate B and N
        # depth: [B*N, D, fH, fW] -> [B, N, D, fH, fW]
        # feat:  [B*N, C, fH, fW] -> [B, N, C, fH, fW]
        depth = depth.view(B, N, self.D, depth.shape[2], depth.shape[3])
        feat = feat.view(B, N, self.C, feat.shape[2], feat.shape[3])

        return depth, feat


# --- Testing Block ---
if __name__ == "__main__":
    print("🧪 Testing Map Encoder...")

    # Init
    encoder = MapEncoder(C=64, D=59)  # LSS 설정과 맞춰야 함 (1~60m)
    encoder.eval()  # 테스트 모드

    # Dummy Input (Dataset 출력과 동일한 형태)
    # Batch=1, Cam=6, Channel=3, Height=450, Width=800
    dummy_imgs = torch.randn(1, 6, 3, 450, 800)

    with torch.no_grad():
        depth, feat = encoder(dummy_imgs)

    print("✅ Encoder Forward Success!")
    print(f"   Input Image: {dummy_imgs.shape}")
    print(f"   Output Depth: {depth.shape} (Expected: [1, 6, 59, 29, 50])")
    print(f"   Output Feat:  {feat.shape}  (Expected: [1, 6, 64, 29, 50])")

    # Check Softmax
    print(
        f"   Depth Sum Check: {depth[0, 0, :, 0, 0].sum().item():.4f} (Should be 1.0)"
    )
