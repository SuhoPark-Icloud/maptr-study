"""
`MapTR` 모델의 전체 파이프라인(데이터 로드부터 모델 추론까지)이
정상적으로 동작하는지 검증하는 스크립트입니다.
단일 데이터 샘플에 대해 전체 모델을 실행하고, 입력 및 출력의 형태(shape)를 출력하며,
최종 결과를 시각화하여 파이프라인의 무결성을 확인합니다.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader

# 모듈 경로 추가
sys.path.append(os.getcwd())

from src.datasets.dataset import MapTRDataset
from src.models.detectors.maptr import MapTR
from tools.train import maptr_collate_fn


def main():
    """
    메인 파이프라인 검증 함수입니다.
    1. 데이터셋과 모델을 로드합니다. (선택적으로 체크포인트 로드)
    2. 데이터셋에서 유효한 샘플을 하나 찾아 파이프라인을 실행합니다.
    3. 모델의 각 단계별 입출력 텐서의 형태(shape)를 출력하여 데이터 흐름을 확인합니다.
    4. 최종 예측 결과를 `plot_results`를 통해 시각화하고 저장합니다.
    """
    print("🚀 Verifying Full MapTR Pipeline with Real Data...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Device: {device}")

    # 1. 데이터셋 로드
    dataroot = os.path.join(os.getcwd(), "data", "nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    dataset = MapTRDataset(nusc, is_train=False)

    # collate_fn을 사용하여 gt 데이터를 함께 로드
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=maptr_collate_fn
    )
    print("✅ Dataset & DataLoader Ready.")

    # 2. 모델 초기화
    # MapTR 전체 모델을 사용합니다.
    model = MapTR(num_classes=3).to(device)

    # [옵션] 학습된 체크포인트가 있다면 로드해서 확인 (없으면 랜덤 가중치)
    checkpoint_path = "./checkpoints/maptr_epoch_100.pth"
    if os.path.exists(checkpoint_path):
        print(f"✅ Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("⚠️ No checkpoint found. Using random weights.")

    model.eval()  # 평가 모드
    print("✅ MapTR Model Initialized.")

    # 3. 파이프라인 실행
    print("\n🔎 Searching for a sample with Ground Truth data...")

    found_sample = False
    valid_batch = None
    for i, batch in enumerate(dataloader):
        imgs, sensor2egos, intrinsics, targets = batch
        # Check if the first item in the batch has ground truth points
        if targets and len(targets[0]["points"]) > 0:
            print(f"✅ Found sample with GT at index {i}.")
            valid_batch = batch
            found_sample = True
            break

    if not found_sample:
        print("\n❌ Could not find a sample with Ground Truth data in the dataset.")
        return

    print("\n🔄 Running Pipeline on the found sample...")
    with torch.no_grad():
        # 데이터 가져오기
        imgs, sensor2egos, intrinsics, targets = valid_batch
        imgs = imgs.to(device)
        sensor2egos = sensor2egos.to(device)
        intrinsics = intrinsics.to(device)

        print(f"   - Input Image Shape : {imgs.shape}")
        print(f"   - Input Sensor2Ego Shape: {sensor2egos.shape}")
        print(f"   - Input Intrinsics Shape: {intrinsics.shape}")

        # 모델 실행
        cls_logits, point_coords = model(imgs, sensor2egos, intrinsics)

        print("\n✅ Final Output:")
        print(f"   - Class Scores: {cls_logits.shape}")
        print(f"   - Map Points  : {point_coords.shape}")

    # 4. 결과 시각화
    print("\n🎨 Visualizing Results...")
    # 검증용이므로 threshold를 낮게(0.1) 잡음
    plot_results(point_coords.cpu(), cls_logits.cpu().sigmoid(), targets, threshold=0.1)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "verify_full_pipeline.png")
    plt.savefig(save_path)
    print(f"🖼️  Visualization saved to: {save_path}")
    print("✨ Full pipeline verification complete!")


def plot_results(pred_pts, pred_scores, gt_targets, threshold=0.3):
    """
    `MapTR` 모델의 예측 결과와 정답(Ground Truth)을 BEV(Bird's Eye View) 시점에서 시각화합니다.
    좌측에는 정답, 우측에는 예측 결과를 나란히 표시하여 비교합니다.

    Args:
        pred_pts (torch.Tensor): 모델이 예측한 포인트 좌표 텐서.
        pred_scores (torch.Tensor): 모델이 예측한 각 클래스에 대한 신뢰도 점수 텐서.
        gt_targets (list[dict]): 정답 타겟 리스트.
        threshold (float, optional): 시각화할 예측의 신뢰도 임계값. 기본값은 0.3.
    """
    pred_pts = pred_pts[0]
    pred_scores = pred_scores[0]
    gt_pts = gt_targets[0]["points"]

    plt.figure(figsize=(12, 12))  # 정사각형 비율 추천

    # [설정] 축 회전: 전후(X)를 세로축으로, 좌우(Y)를 가로축으로
    swap_axis = True

    # --- Ground Truth ---
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth (Ego Frame)")

    if gt_pts.numel() > 0:
        for i in range(len(gt_pts)):
            pts_normalized = gt_pts[i].numpy()
            pts_meter = np.copy(pts_normalized)

            # Denormalization (새로운 좌표계 반영)
            # X: 0~1 -> -30~30 (Range 60)
            real_x = pts_meter[:, 0] * 60.0 - 30.0
            # Y: 0~1 -> -15~15 (Range 30)
            real_y = pts_meter[:, 1] * 30.0 - 15.0

            if swap_axis:
                # Plot (Y, X) -> (Lateral, Forward)
                plt.plot(real_y, real_x, "g-", linewidth=2)
            else:
                plt.plot(real_x, real_y, "g-", linewidth=2)

    plt.grid(True)
    plt.axis("equal")

    if swap_axis:
        plt.xlim(-15, 15)
        plt.ylim(-30, 30)
        plt.xlabel("Lateral Y (meters)")
        plt.ylabel("Longitudinal X (meters)")
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    else:
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")

    # --- Prediction ---
    plt.subplot(1, 2, 2)
    plt.title(f"Prediction (Score > {threshold})")

    max_scores, _ = pred_scores.max(dim=-1)

    drawn_count = 0
    for i in range(len(pred_pts)):
        if max_scores[i] > threshold:
            pts_normalized = pred_pts[i].numpy()
            pts_meter = np.copy(pts_normalized)

            # Denormalization
            real_x = pts_meter[:, 0] * 60.0 - 30.0
            real_y = pts_meter[:, 1] * 30.0 - 15.0

            if swap_axis:
                plt.plot(real_y, real_x, "r-", linewidth=2)
            else:
                plt.plot(real_x, real_y, "r-", linewidth=2)
            drawn_count += 1

    print(f"   -> Drawn {drawn_count} predictions")

    plt.grid(True)
    plt.axis("equal")

    if swap_axis:
        plt.xlim(-15, 15)
        plt.ylim(-30, 30)
        plt.xlabel("Lateral Y (meters)")
        plt.ylabel("Longitudinal X (meters)")
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    else:
        plt.xlim(-30, 30)
        plt.ylim(-15, 15)
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")

    plt.suptitle("Full Pipeline Verification")


if __name__ == "__main__":
    main()
