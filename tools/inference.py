"""
사전 학습된 `MapTR` 모델을 사용하여 추론을 실행하고, 그 결과를 시각화하는 스크립트입니다.
데이터 로드, 모델 가중치 로드, 단일 샘플에 대한 추론 실행,
정답(GT)과 예측 결과를 비교하여 이미지 파일로 저장하는 과정을 포함합니다.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from src.datasets.dataset import MapTRDataset
from src.models.detectors.maptr import MapTR
from tools.train import maptr_collate_fn


def main():
    """
    메인 추론 함수입니다. 전체 추론 과정을 설정하고 실행합니다.
    1. nuScenes mini 데이터셋과 모델을 로드합니다.
    2. 지정된 경로에서 사전 학습된 모델의 체크포인트(가중치)를 로드합니다.
    3. 데이터셋에서 유효한 정답(GT) 데이터가 있는 샘플을 찾아 추론을 실행합니다.
    4. 추론 결과를 `plot_results` 함수를 통해 시각화하고 이미지 파일로 저장합니다.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🔎 Inference on {device}...")

    # 1. Load Data
    dataroot = os.path.join(os.getcwd(), "data", "nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    dataset = MapTRDataset(nusc, is_train=False)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=maptr_collate_fn
    )
    print("✅ Dataset & DataLoader Ready.")

    # 2. Load Model
    model = MapTR(num_classes=3).to(device)

    # 체크포인트 로드
    checkpoint_path = "./checkpoints/maptr_epoch_100.pth"  # 가장 최근 모델 사용 권장
    if not os.path.exists(checkpoint_path):
        # 100이 없으면 50, 10 등 순차적으로 확인 (예시)
        checkpoint_path = "./checkpoints/maptr_epoch_10.pth"

    if os.path.exists(checkpoint_path):
        print(f"✅ Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("⚠️ Checkpoint not found. Running with random weights.")

    model.eval()

    # 3. Inference
    print("\n🔎 Searching for a sample with Ground Truth data...")
    valid_batch = None
    for i, batch in enumerate(dataloader):
        imgs, sensor2egos, intrinsics, targets = batch
        if targets and len(targets[0]["points"]) > 0:
            print(f"✅ Found sample with GT at index {i}.")
            valid_batch = batch
            break

    if not valid_batch:
        print("\n❌ Could not find a sample with Ground Truth data.")
        return

    print("\n🔄 Running Inference...")
    with torch.no_grad():
        imgs, sensor2egos, intrinsics, targets = valid_batch
        imgs = imgs.to(device)
        sensor2egos = sensor2egos.to(device)
        intrinsics = intrinsics.to(device)

        cls_logits, point_coords = model(imgs, sensor2egos, intrinsics)

        print("\n✅ Inference Finished:")

        # 진단용 점수 출력
        scores = cls_logits[0].sigmoid()
        max_scores, _ = scores.max(dim=-1)
        top_scores, _ = max_scores.sort(descending=True)
        print(f"📊 Top 10 Confidence Scores: {top_scores[:10].tolist()}")

    # 4. 시각화 (Threshold 0.2로 설정)
    print("\n🎨 Visualizing...")
    plot_results(point_coords.cpu(), cls_logits.cpu().sigmoid(), targets, threshold=0.2)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "inference_result.png")
    plt.savefig(save_path)
    print(f"🖼️  Visualization saved to: {save_path}")


def plot_results(pred_pts, pred_scores, gt_targets, threshold=0.3):
    """
    모델의 예측 결과와 정답(Ground Truth)을 Matplotlib을 사용하여
    BEV(Bird's Eye View) 시점에서 시각화합니다.
    좌측에는 정답, 우측에는 예측 결과를 나란히 표시합니다.

    Args:
        pred_pts (torch.Tensor): 모델이 예측한 포인트 좌표 텐서.
        pred_scores (torch.Tensor): 모델이 예측한 각 클래스에 대한 신뢰도 점수 텐서.
        gt_targets (list[dict]): 정답 타겟 리스트.
        threshold (float, optional): 시각화할 예측의 신뢰도 임계값. 기본값은 0.3.
    """
    pred_pts = pred_pts[0]
    pred_scores = pred_scores[0]
    gt_pts = gt_targets[0]["points"]

    plt.figure(figsize=(12, 12))  # 정사각형에 가까운 비율

    # [설정] 축 회전: 전후(X)를 세로축으로, 좌우(Y)를 가로축으로
    swap_axis = True

    # --- Ground Truth ---
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth (Ego Frame)")

    if gt_pts.numel() > 0:
        for i in range(len(gt_pts)):
            pts_normalized = gt_pts[i].cpu().numpy()
            pts_meter = np.copy(pts_normalized)

            # Denormalization
            # X: 0~1 -> -30~30
            real_x = pts_meter[:, 0] * 60.0 - 30.0
            # Y: 0~1 -> -15~15
            real_y = pts_meter[:, 1] * 30.0 - 15.0

            if swap_axis:
                # Plot (Y, X) -> (Lateral, Forward)
                plt.plot(real_y, real_x, "g-", linewidth=2)
            else:
                plt.plot(real_x, real_y, "g-", linewidth=2)

    plt.grid(True)
    plt.axis("equal")

    if swap_axis:
        plt.xlim(-15, 15)  # Lateral
        plt.ylim(-30, 30)  # Longitudinal
        plt.xlabel("Lateral Y (meters)")
        plt.ylabel("Longitudinal X (meters)")
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)  # Center Line
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)  # Ego Position
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
            pts_normalized = pred_pts[i].cpu().numpy()
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

    plt.suptitle("Inference Result (Top-down View)")


if __name__ == "__main__":
    main()
