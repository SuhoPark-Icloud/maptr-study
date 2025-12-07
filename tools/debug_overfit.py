"""
단일 데이터 배치에 대해 `MapTR` 모델을 과적합(overfitting)시켜,
모델과 학습 파이프라인이 정상적으로 동작하는지 디버깅하는 스크립트입니다.
고정된 배치로 반복 학습을 수행하고 주기적으로 예측 결과를 시각화하여 저장합니다.
"""

import os
import sys

# MPS Fallback 설정
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader

# 프로젝트 루트 경로 설정
sys.path.append(os.getcwd())

from src.datasets.dataset import MapTRDataset
from src.models.detectors.maptr import MapTR
from src.models.losses.loss import MapLoss
from src.models.losses.matcher import MapMatcher
from tools.train import maptr_collate_fn


def plot_overfit_result(pred_pts, pred_scores, gt_targets, epoch):
    """
    과적합 디버깅 중인 모델의 예측 결과와 정답을 시각화하여 이미지 파일로 저장합니다.

    Args:
        pred_pts (torch.Tensor): 모델이 예측한 포인트 좌표 텐서.
        pred_scores (torch.Tensor): 모델이 예측한 신뢰도 점수 텐서.
        gt_targets (list[dict]): 정답 타겟 리스트.
        epoch (int): 현재 에폭.
    """
    # 첫 번째 샘플만 시각화
    pred_pts = pred_pts[0].detach().cpu()
    pred_scores = pred_scores[0].detach().cpu()
    gt_pts = gt_targets[0]["points"].detach().cpu()

    plt.figure(figsize=(10, 10))

    # 보기 편하게 축 회전 (Forward=Up)
    swap_axis = True

    # 1. GT 그리기 (초록색)
    if gt_pts.numel() > 0:
        for i in range(len(gt_pts)):
            pts = gt_pts[i].numpy()
            pts_meter = np.copy(pts)

            # Denormalization
            # X (전후): 0~1 -> -30~30 (Range 60)
            real_x = pts_meter[:, 0] * 60.0 - 30.0
            # Y (좌우): 0~1 -> -15~15 (Range 30)
            real_y = pts_meter[:, 1] * 30.0 - 15.0

            if swap_axis:
                # 가로: Lateral(Y), 세로: Longitudinal(X)
                plt.plot(
                    real_y, real_x, "g-", linewidth=3, label="GT" if i == 0 else ""
                )
            else:
                plt.plot(
                    real_x, real_y, "g-", linewidth=3, label="GT" if i == 0 else ""
                )

    # 2. 예측 그리기 (빨간색)
    max_scores, _ = pred_scores.max(dim=-1)

    # 학습 초기일 수 있으므로 0.1 이상이면 그리기
    threshold = 0.1
    drawn = False

    for i in range(len(pred_pts)):
        if max_scores[i] > threshold:
            pts = pred_pts[i].numpy()
            pts_meter = np.copy(pts)

            # Denormalization
            real_x = pts_meter[:, 0] * 60.0 - 30.0
            real_y = pts_meter[:, 1] * 30.0 - 15.0

            if swap_axis:
                plt.plot(
                    real_y,
                    real_x,
                    "r--",
                    linewidth=1.5,
                    label="Pred" if not drawn else "",
                )
            else:
                plt.plot(
                    real_x,
                    real_y,
                    "r--",
                    linewidth=1.5,
                    label="Pred" if not drawn else "",
                )
            drawn = True

    plt.grid(True)

    # 축 범위 설정
    if swap_axis:
        plt.xlim(-15, 15)  # 좌우 15m
        plt.ylim(-30, 30)  # 전후 30m
        plt.xlabel("Lateral (Y)")
        plt.ylabel("Longitudinal (X)")
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    else:
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        plt.xlabel("X")
        plt.ylabel("Y")

    plt.legend()
    plt.title(f"Overfitting Debug - Epoch {epoch}")

    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/overfit_epoch_{epoch}.png")
    plt.close()


def main():
    """
    메인 과적합 디버깅 함수입니다. 전체 디버깅 과정을 설정하고 실행합니다.
    1. nuScenes 데이터셋에서 정답(GT)이 포함된 단일 배치를 고정하여 준비합니다.
    2. 모델과 학습 관련 설정을 초기화합니다.
    3. 고정된 배치에 대해 모델을 수백 에폭 동안 반복 학습시켜 과적합을 유도합니다.
    4. 주기적으로 손실을 출력하고, `plot_overfit_result`를 호출하여 학습 과정을 시각화합니다.
    """
    # 1. 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🐞 Debugging Overfit on {device}...")

    # 데이터셋 로드
    dataroot = os.path.join(os.getcwd(), "data", "nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    # batch_size=1
    dataset = MapTRDataset(nusc, is_train=True)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=maptr_collate_fn
    )

    # 2. 고정된 배치(Fixed Batch) 하나만 가져오기
    fixed_batch = None
    for batch in dataloader:
        _, _, _, targets = batch
        # GT가 있는 샘플만 선택
        if len(targets[0]["points"]) > 0:
            fixed_batch = batch
            print(f"✅ Found a sample with {len(targets[0]['points'])} GT elements.")
            break

    if fixed_batch is None:
        print("❌ No valid GT sample found.")
        return

    # 데이터를 GPU로 이동
    imgs, sensor2egos, intrinsics, targets = fixed_batch
    imgs = imgs.to(device)
    sensor2egos = sensor2egos.to(device)
    intrinsics = intrinsics.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # 3. 모델 및 학습 설정
    model = MapTR(num_classes=3).to(device)
    matcher = MapMatcher(cost_class=2.0, cost_point=5.0)
    criterion = MapLoss(num_classes=3, matcher=matcher).to(device)

    # Overfitting용 LR
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)

    # 4. 반복 학습 (Overfitting Loop)
    model.train()
    print("\n🚀 Starting Overfitting Loop (300 Epochs)...")

    for epoch in range(1, 301):
        # Forward
        outputs = model(imgs, sensor2egos, intrinsics)

        # 시각화를 위해 변수 추출
        cls_logits = outputs["pred_logits"]
        point_coords = outputs["pred_points"]

        # Loss Calculation
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())

        # Backward
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
        optimizer.step()

        # 로그 출력
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/300 | Total Loss: {losses.item():.6f} | "
                f"Class: {loss_dict['loss_ce'].item():.4f}, "
                f"BBox: {loss_dict['loss_bbox'].item():.4f}"
            )

        # 중간 결과 시각화 (50 에폭마다)
        if epoch % 50 == 0:
            plot_overfit_result(point_coords, cls_logits.sigmoid(), targets, epoch)

    print("\n✅ Debugging Complete. Check 'output/' folder.")


if __name__ == "__main__":
    main()
