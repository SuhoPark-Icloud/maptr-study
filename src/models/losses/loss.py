"""
`MapTR` 모델의 학습 손실 함수(`MapLoss`)를 정의합니다.
분류, 포인트 회귀, 방향 예측 손실을 포함하며,
메인(One-to-One) 및 보조(One-to-Many) 매칭 브랜치를 지원합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class MapLoss(nn.Module):
    """
    `MapTR` 모델의 손실을 계산하는 모듈입니다.
    One-to-One 매칭(메인 브랜치)과 One-to-Many 매칭(보조 브랜치)을 지원하며,
    분류를 위해 Focal Loss를 사용합니다.
    """

    def __init__(
        self,
        num_classes=3,
        matcher=None,
        weight_dict=None,
        focal_alpha=0.25,
        focal_gamma=2.0,
    ):
        """
        `MapLoss` 모듈을 초기화합니다.
        클래스 개수, 매처, 손실 가중치, Focal Loss 파라미터 등을 설정합니다.

        Args:
            num_classes (int, optional): 예측할 클래스의 개수. 기본값은 3.
            matcher: 할당 매처 객체.
            weight_dict (dict, optional): 손실 가중치 딕셔너리. 기본값은 None.
            focal_alpha (float, optional): Focal Loss의 알파 파라미터. 기본값은 0.25.
            focal_gamma (float, optional): Focal Loss의 감마 파라미터. 기본값은 2.0.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        # Loss weights
        if weight_dict is None:
            self.weight_dict = {
                "loss_ce": 2.0,
                "loss_bbox": 5.0,
                "loss_dir": 0.005,
                # Auxiliary Loss weights (same as main)
                "loss_ce_aux": 2.0,
                "loss_bbox_aux": 5.0,
                "loss_dir_aux": 0.005,
            }
        else:
            self.weight_dict = weight_dict

        # Focal Loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, outputs, targets):
        """
        `MapLoss` 모듈의 포워드 패스입니다.
        모델의 출력(`outputs`)과 정답 타겟(`targets`)을 받아
        메인 브랜치와 보조 브랜치의 손실을 계산합니다.

        Args:
            outputs (dict or tuple): 모델의 예측 출력.
                - dict: 'pred_logits', 'pred_points', (선택적)'aux_outputs' 포함.
                - tuple: (pred_logits, pred_points) (추론 시 호환성)
            targets (list[dict]): 정답 타겟 리스트. 각 dict는 'labels'와 'points' 포함.

        Returns:
            dict: 계산된 손실 값을 포함하는 딕셔너리.
        """
        # Inference 시 Tuple로 들어오는 경우 호환성 처리
        if isinstance(outputs, tuple):
            outputs = {"pred_logits": outputs[0], "pred_points": outputs[1]}

        losses = {}

        # ==========================================================
        # 1. Main Branch Loss (One-to-One Matching)
        # ==========================================================
        indices = self.matcher(outputs, targets)

        # Normalize by number of boxes (across batch)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=outputs["pred_logits"].device
        )
        num_boxes = torch.clamp(num_boxes / 1, min=1).item()

        losses["loss_ce"] = self.loss_labels(outputs, targets, indices, num_boxes)
        losses["loss_bbox"] = self.loss_points(outputs, targets, indices, num_boxes)
        losses["loss_dir"] = self.loss_dir(outputs, targets, indices, num_boxes)

        # ==========================================================
        # 2. Auxiliary Branch Loss (One-to-Many Matching)
        # ==========================================================
        if "aux_outputs" in outputs:
            aux_outputs = outputs["aux_outputs"]
            k_one2many = 6  # MapTRv2 Paper Default: k=6

            # (A) Replicate Targets k times
            # targets_aux = [{'labels': [N*k], 'points': [N*k, P, 2]}, ...]
            targets_aux = []
            for t in targets:
                t_aux = {}
                if len(t["labels"]) > 0:
                    t_aux["labels"] = t["labels"].repeat(k_one2many)
                    t_aux["points"] = t["points"].repeat(k_one2many, 1, 1)
                else:
                    t_aux["labels"] = t["labels"]
                    t_aux["points"] = t["points"]
                targets_aux.append(t_aux)

            # (B) Perform Matching on Aux Outputs
            indices_aux = self.matcher(aux_outputs, targets_aux)

            # (C) Normalize factor (also scaled by k)
            num_boxes_aux = num_boxes * k_one2many

            # (D) Calculate Aux Losses
            losses["loss_ce_aux"] = self.loss_labels(
                aux_outputs, targets_aux, indices_aux, num_boxes_aux
            )
            losses["loss_bbox_aux"] = self.loss_points(
                aux_outputs, targets_aux, indices_aux, num_boxes_aux
            )
            losses["loss_dir_aux"] = self.loss_dir(
                aux_outputs, targets_aux, indices_aux, num_boxes_aux
            )

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        분류 손실(Sigmoid Focal Loss)을 계산합니다.

        Args:
            outputs (dict): 모델의 예측 출력. 'pred_logits' 포함.
            targets (list[dict]): 정답 타겟 리스트. 'labels' 포함.
            indices (list[tuple]): 매칭 결과 인덱스.
            num_boxes (int): 배치 내 정답 박스(지도 요소)의 총 개수.

        Returns:
            torch.Tensor: 계산된 분류 손실.
        """
        src_logits = outputs["pred_logits"]  # [B, Q, Num_Classes]
        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        # [Prepare One-Hot Targets for Focal Loss]
        B, Q, C = src_logits.shape
        target_classes_onehot = torch.zeros(
            (B, Q, C), dtype=src_logits.dtype, device=src_logits.device
        )

        # Set 1.0 for matched indices and classes
        target_classes_onehot[idx[0], idx[1], target_classes_o] = 1.0

        # Compute Focal Loss
        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none",
        )

        return loss_ce.sum() / num_boxes

    def loss_points(self, outputs, targets, indices, num_boxes):
        """
        포인트 회귀 손실(L1 Loss)을 계산합니다.

        Args:
            outputs (dict): 모델의 예측 출력. 'pred_points' 포함.
            targets (list[dict]): 정답 타겟 리스트. 'points' 포함.
            indices (list[tuple]): 매칭 결과 인덱스.
            num_boxes (int): 배치 내 정답 박스(지도 요소)의 총 개수.

        Returns:
            torch.Tensor: 계산된 포인트 회귀 손실.
        """
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["points"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_points, target_points, reduction="none")
        return loss_bbox.sum() / num_boxes

    def loss_dir(self, outputs, targets, indices, num_boxes):
        """
        에지 방향 손실(Cosine Similarity)을 계산합니다.

        Args:
            outputs (dict): 모델의 예측 출력. 'pred_points' 포함.
            targets (list[dict]): 정답 타겟 리스트. 'points' 포함.
            indices (list[tuple]): 매칭 결과 인덱스.
            num_boxes (int): 배치 내 정답 박스(지도 요소)의 총 개수.

        Returns:
            torch.Tensor: 계산된 방향 손실.
        """
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["points"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        # Edge Vectors: [K, P-1, 2]
        src_vecs = src_points[:, 1:] - src_points[:, :-1]
        target_vecs = target_points[:, 1:] - target_points[:, :-1]

        # Normalize
        src_vecs = F.normalize(src_vecs, p=2, dim=-1, eps=1e-6)
        target_vecs = F.normalize(target_vecs, p=2, dim=-1, eps=1e-6)

        # 1 - CosSim (Minimize this)
        loss_dir = (1 - F.cosine_similarity(src_vecs, target_vecs, dim=-1)).sum()

        return loss_dir / num_boxes

    def _get_src_permutation_idx(self, indices):
        """
        매칭된 인덱스(`indices`)를 기반으로 소스(예측) 텐서에서 값을 추출하기 위한
        배치 및 인덱스 쌍을 생성합니다.

        Args:
            indices (list[tuple]): 매칭 결과 인덱스. 각 튜플은 (소스 인덱스, 타겟 인덱스)를 포함.

        Returns:
            tuple:
                - batch_idx (torch.Tensor): 배치 인덱스 텐서.
                - src_idx (torch.Tensor): 소스 텐서 내의 인덱스 텐서.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
