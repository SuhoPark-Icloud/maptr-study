"""
`MapTR` 모델의 학습을 위한 메인 스크립트입니다.
데이터셋 로드, 모델 및 손실 함수 초기화, 학습 루프 실행, 체크포인트 저장,
TensorBoard 로깅 등의 전체 학습 파이프라인을 관리합니다.
"""

import os

# MPS Fallback 설정 (필수)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Custom modules (src 폴더에 있는 것들)
from src.datasets.dataset import MapTRDataset
from src.models.detectors.maptr import MapTR
from src.models.losses.loss import MapLoss
from src.models.losses.matcher import MapMatcher


# --- 1. Custom Collate Function ---
def maptr_collate_fn(batch):
    """
    `DataLoader`를 위한 사용자 정의 `collate` 함수입니다.
    배치 내의 가변적인 길이의 벡터 데이터를 PyTorch 텐서로 변환하고,
    GT(Ground Truth) 딕셔너리를 생성합니다.

    Args:
        batch (list[dict]): 데이터셋에서 로드된 샘플들의 리스트.

    Returns:
        tuple: 이미지, 센서-차량 변환 행렬, 내부 파라미터, 타겟 텐서들의 튜플.
    """
    imgs = torch.stack([item["imgs"] for item in batch])
    intrinsics = torch.stack([item["intrinsics"] for item in batch])
    sensor2egos = torch.stack([item["sensor2egos"] for item in batch])

    targets = []
    for item in batch:
        labels = []
        points = []
        for cls_name, pts in item["vectors"]:
            # 클래스 매핑: divider->0, ped_crossing->1, boundary->2
            if cls_name == "divider":
                l = 0
            elif cls_name == "ped_crossing":
                l = 1
            else:
                l = 2
            labels.append(l)
            points.append(pts)

        if len(labels) > 0:
            targets.append(
                {
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "points": torch.stack(points),  # [N, 20, 2]
                }
            )
        else:
            targets.append(
                {
                    "labels": torch.empty(0, dtype=torch.long),
                    "points": torch.empty(0, 20, 2),
                }
            )

    return imgs, sensor2egos, intrinsics, targets


# --- 2. TensorBoard Visualization Function ---
def visualize_for_tensorboard(writer, epoch, step, outputs, targets):
    """
    모델의 예측 결과(BEV 맵)와 정답(Ground Truth)을 시각화하여
    TensorBoard에 이미지로 저장합니다.

    Args:
        writer (SummaryWriter): TensorBoard 로거 객체.
        epoch (int): 현재 에폭.
        step (int): 현재 글로벌 스텝.
        outputs (dict): 모델의 예측 출력.
        targets (list[dict]): 정답 타겟 리스트.
    """
    # 첫 번째 배치의 데이터만 가져옴 (CPU로 이동)
    pred_logits = outputs["pred_logits"][0].detach().cpu()  # [Q, 3]
    pred_points = outputs["pred_points"][0].detach().cpu()  # [Q, P, 2]

    # GT가 없는 경우 예외 처리
    if len(targets) > 0:
        gt_points = targets[0]["points"].detach().cpu()  # [N, P, 2]
    else:
        gt_points = torch.empty(0)

    # 점수 계산
    scores = pred_logits.sigmoid()
    max_scores, _ = scores.max(dim=-1)

    # Figure 생성
    fig, ax = plt.subplots(figsize=(10, 10))

    # 축 설정 (Forward=Up)
    swap_axis = True

    # A. Ground Truth 그리기 (초록색)
    if gt_points.numel() > 0:
        for i in range(len(gt_points)):
            pts_norm = gt_points[i].numpy()
            pts_meter = np.copy(pts_norm)

            # Denormalize
            # X (전후): 0~1 -> -30~30 (Range 60)
            real_x = pts_meter[:, 0] * 60.0 - 30.0
            # Y (좌우): 0~1 -> -15~15 (Range 30)
            real_y = pts_meter[:, 1] * 30.0 - 15.0

            if swap_axis:
                # (Lateral, Forward) -> 지도처럼 보기
                ax.plot(real_y, real_x, "g-", linewidth=2, alpha=0.7)
            else:
                ax.plot(real_x, real_y, "g-", linewidth=2, alpha=0.7)

    # B. Prediction 그리기 (빨간색)
    # 학습 초기에는 점수가 낮으므로 0.1 이상이면 그림
    threshold = 0.1
    for i in range(len(pred_points)):
        if max_scores[i] > threshold:
            pts_norm = pred_points[i].numpy()
            pts_meter = np.copy(pts_norm)

            # Denormalize
            real_x = pts_meter[:, 0] * 60.0 - 30.0
            real_y = pts_meter[:, 1] * 30.0 - 15.0

            if swap_axis:
                ax.plot(real_y, real_x, "r-", linewidth=2)
            else:
                ax.plot(real_x, real_y, "r-", linewidth=2)

    ax.grid(True)
    ax.set_aspect("equal")

    if swap_axis:
        ax.set_xlim(-15, 15)  # 좌우 15m
        ax.set_ylim(-30, 30)  # 전후 30m
        ax.set_title(f"Epoch {epoch} Step {step} (Green: GT, Red: Pred)")
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    else:
        ax.set_xlim(-30, 30)
        ax.set_ylim(-15, 15)

    # TensorBoard에 기록
    writer.add_figure("Prediction/BEV_Map", fig, global_step=step)
    plt.close(fig)  # 메모리 누수 방지


# --- 3. Main Training Loop ---
def main():
    """
    메인 학습 함수입니다. 전체 학습 과정을 설정하고 실행합니다.
    1. 학습 장치(device), TensorBoard 로거, 데이터셋 및 데이터로더를 설정합니다.
    2. 모델, 매처, 손실 함수, 옵티마이저를 초기화합니다.
    3. 체크포인트가 있으면 학습을 재개하고, 없으면 처음부터 시작합니다.
    4. 지정된 에폭 수만큼 학습 루프를 실행하며, 손실 계산, 역전파, 파라미터 업데이트를 수행합니다.
    5. 주기적으로 TensorBoard에 손실 값과 시각화 결과를 기록합니다.
    6. 주기적으로 모델 체크포인트를 저장합니다.
    """
    # 1. Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on {device}...")

    # TensorBoard Writer 초기화 (logs/maptr_exp 폴더에 저장)
    writer = SummaryWriter(log_dir="logs/maptr_exp")

    dataroot = os.path.join(os.getcwd(), "data", "nuscenes")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    dataset = MapTRDataset(nusc, is_train=True)

    # [메모리 최적화] 배치 사이즈 2 설정
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=maptr_collate_fn
    )
    print(f"✅ Total Batch Count: {len(dataloader)}")

    # 2. Model & Loss Init
    model = MapTR(num_classes=3).to(device)
    matcher = MapMatcher(cost_class=2.0, cost_point=5.0)
    criterion = MapLoss(num_classes=3, matcher=matcher).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)

    # 3. Resume Configuration
    num_epochs = 100
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    # 이어할 체크포인트 찾기 (최신순)
    if os.path.exists(os.path.join(save_dir, "maptr_epoch_100.pth")):
        resume_path = os.path.join(save_dir, "maptr_epoch_100.pth")
    elif os.path.exists(os.path.join(save_dir, "maptr_epoch_10.pth")):
        resume_path = os.path.join(save_dir, "maptr_epoch_10.pth")
    else:
        resume_path = None

    if resume_path:
        print(f"🔄 Resuming training from {resume_path}...")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"   -> Starting from Epoch {start_epoch + 1}")
        except RuntimeError as e:
            print(f"⚠️ Checkpoint load failed (Shape mismatch?): {e}")
            print("🆕 Starting from scratch due to architecture change.")
            resume_path = None
    else:
        print("🆕 No checkpoint found. Starting training from scratch.")

    model.train()
    print("🏁 Start Training Loop...")

    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch_idx, (imgs, sensor2egos, intrinsics, targets) in enumerate(
            dataloader
        ):
            # Data to Device
            imgs = imgs.to(device)
            sensor2egos = sensor2egos.to(device)
            intrinsics = intrinsics.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward
            outputs = model(imgs, sensor2egos, intrinsics)

            # Loss Calculation
            loss_dict = criterion(outputs, targets)
            losses = sum(
                loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys()
            )

            # Backward
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35)
            optimizer.step()

            # TensorBoard Logging (Scalar)
            writer.add_scalar("Loss/Total", losses.item(), global_step)
            writer.add_scalar("Loss/Class", loss_dict["loss_ce"].item(), global_step)
            writer.add_scalar("Loss/BBox", loss_dict["loss_bbox"].item(), global_step)

            total_loss += losses.item()

            # [시각화] 10 Step 마다 수행 (배치2 기준 자주 업데이트됨)
            if global_step % 10 == 0:
                visualize_for_tensorboard(writer, epoch, global_step, outputs, targets)

            # Console Logging (10 Step 마다)
            if batch_idx % 10 == 0:
                print(
                    f"   Epoch [{epoch + 1}/{num_epochs}] Step [{batch_idx}] "
                    f"Total: {losses.item():.4f}"
                )

            global_step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Complete. Avg Loss: {avg_loss:.4f}")

        # Checkpoint 저장 (5 에폭마다)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            save_path = os.path.join(save_dir, f"maptr_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                save_path,
            )
            print(f"💾 Model saved to {save_path}")

    writer.close()


if __name__ == "__main__":
    main()
