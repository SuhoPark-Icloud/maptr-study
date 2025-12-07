"""
`MapTR` 모델의 예측과 정답 간의 최적 할당을 찾는 `MapMatcher` 클래스를 정의합니다.
분류 비용과 순열-등가(permutation-equivalent) 포인트 비용을 결합하고,
헝가리안 알고리즘을 사용하여 최적의 매칭을 수행합니다.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class MapMatcher(nn.Module):
    """
    예측과 정답 지도 요소 간의 최적 매칭을 수행하는 모듈입니다.
    분류 비용과 순열-등가 포인트 비용을 합산하여 최종 비용 행렬을 계산하고,
    `linear_sum_assignment`(헝가리안 알고리즘)를 통해 가장 비용이 적은 할당을 찾습니다.
    """

    def __init__(self, cost_class=2.0, cost_point=5.0, num_points=20):
        """
        `MapMatcher`를 초기화합니다.
        분류 비용, 포인트 비용의 가중치 및 포인트 개수를 설정합니다.

        Args:
            cost_class (float, optional): 분류 비용의 가중치. 기본값은 2.0.
            cost_point (float, optional): 포인트 회귀 비용의 가중치. 기본값은 5.0.
            num_points (int, optional): 폴리라인을 구성하는 포인트의 개수. 기본값은 20.
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        self.num_points = num_points

    def get_permutation_indices(self, device):
        """
        포인트 시퀀스의 순열-등가 비교를 위해 가능한 모든 순열 인덱스를 미리 생성합니다.
        순환 이동(shift)과 순서 뒤집기(flip)를 조합하여
        2 * `num_points`개의 순열을 생성합니다 (논문 Eq(1), Eq(2) 구현).

        Args:
            device: 인덱스를 생성할 PyTorch 장치 (e.g., 'cpu', 'cuda').

        Returns:
            torch.Tensor: 생성된 순열 인덱스 텐서.
        """
        # 1. 기본 인덱스: [0, 1, ..., 19]
        base_indices = torch.arange(self.num_points, device=device)

        # 2. Shift (순환 이동) 생성: [5]
        # 예: 0,1,2... / 1,2,3... / ...
        shifts = base_indices.unsqueeze(0) + base_indices.unsqueeze(1)
        shifts = shifts % self.num_points

        # 3. Flip (역방향) 생성: [5]
        # 역방향 인덱스에 대해 다시 Shift 수행
        base_indices_flip = torch.flip(base_indices, dims=(0,))
        shifts_flip = base_indices_flip.unsqueeze(0) + base_indices.unsqueeze(1)
        shifts_flip = shifts_flip % self.num_points

        # 4. 전체 순열 통합: [5, 6] (Shift 20개 + Flip 20개)
        # 모든 가능한 점의 연결 순서를 담고 있음
        perm_indices = torch.cat([shifts, shifts_flip], dim=0)

        return perm_indices

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        `MapMatcher`의 포워드 패스입니다.
        배치 내 각 샘플에 대해 모델의 예측과 정답 타겟 간의 비용 행렬을 계산하고,
        헝가리안 알고리즘을 사용하여 최적의 예측-정답 쌍을 찾습니다.

        Args:
            outputs (dict): 모델의 예측 출력. 'pred_logits'와 'pred_points' 포함.
            targets (list[dict]): 정답 타겟 리스트. 각 dict는 'labels'와 'points' 포함.

        Returns:
            list[tuple]: 각 배치의 매칭 결과 인덱스 리스트.
                         각 튜플은 (소스 인덱스, 타겟 인덱스)를 포함.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        # 미리 순열 인덱스 생성 (GPU/MPS 장치 호환)
        device = outputs["pred_points"].device
        perm_indices = self.get_permutation_indices(device)  # [5, 6]

        for i in range(bs):
            # --- 1. Classification Cost ---
            pred_prob = outputs["pred_logits"][i].softmax(-1)
            tgt_ids = targets[i]["labels"]

            # [Num_Query, Num_GT]
            # 예측 확률 중 정답 클래스에 해당하는 확률의 음수값 (확률이 높을수록 Cost 감소)
            cost_class = -pred_prob[:, tgt_ids]

            # --- 2. Permutation-Equivalent Point Cost ---
            pred_pts = outputs["pred_points"][i]  # [Num_Query, 20, 2]
            tgt_pts = targets[i]["points"]  # [Num_GT, 20, 2]

            if len(tgt_pts) == 0:
                indices.append(
                    (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                )
                continue

            # (1) Ground Truth를 모든 순열로 확장
            # tgt_pts: [Num_GT, 20, 2] -> perm_tgt: [Num_GT, 40, 20, 2]
            # 40가지(Shift+Flip) 경우의 수로 점들의 순서를 바꿈
            perm_tgt = tgt_pts[:, perm_indices, :]

            # (2) 모든 Query와 모든 GT 간의 거리 계산 (L1 Distance)
            # pred_pts: [Num_Query, 1, 1, 20, 2] (Broadcasting 준비)
            # perm_tgt: [1, Num_GT, 40, 20, 2]
            # diff: [Num_Query, Num_GT, 40, 20, 2] -> sum(-1) -> [Num_Query, Num_GT, 40, 20] (x,y 합)
            # -> mean(-1) -> [Num_Query, Num_GT, 40] (모든 점의 평균 거리)

            # *메모리 절약을 위해 반복문 사용 (Query 개수가 많을 때 OOM 방지)*
            # Cost Matrix 크기: [Num_Query, Num_GT]
            cost_point = torch.zeros(num_queries, len(tgt_pts), device=device)

            for q_idx in range(num_queries):
                # 특정 Query 하나 [5, 7, 8]와 모든 GT의 40가지 변형 [Num_GT, 40, 20, 2] 비교
                p_vec = pred_pts[q_idx].unsqueeze(0).unsqueeze(0)

                # L1 Distance: |Pred - GT_perm|
                dist = (
                    torch.abs(p_vec - perm_tgt).sum(dim=-1).mean(dim=-1)
                )  # [Num_GT, 40]

                # (3) 40가지 순열 중 최소값 선택 (Min Permutation)
                min_dist, _ = dist.min(dim=-1)  # [Num_GT]

                cost_point[q_idx] = min_dist

            # --- 3. Final Cost & Matching ---
            C = self.cost_class * cost_class + self.cost_point * cost_point

            # Hungarian Algorithm
            C_cpu = C.cpu()
            row_ind, col_ind = linear_sum_assignment(C_cpu)

            indices.append(
                (
                    torch.as_tensor(row_ind, dtype=torch.int64),
                    torch.as_tensor(col_ind, dtype=torch.int64),
                )
            )

        return indices
