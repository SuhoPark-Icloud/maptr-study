"""
`MapTR` 모델의 메인 아키텍처를 정의합니다.
인코더, LSS 변환, 디코더 서브 모듈들을 통합하여 다중 시점 이미지에서
BEV 지도 요소를 검출하는 역할을 담당합니다.
"""

import torch.nn as nn

from src.models.layers.decoder import MapDecoder
from src.models.layers.encoder import MapEncoder
from src.models.layers.lss import LSSTransform


class MapTR(nn.Module):
    """
    `MapTR` (Map TRansformer) 모델의 최상위 클래스입니다.
    인코더, LSS 변환 모듈, 디코더를 연결하여 전체 파이프라인을 구성합니다.
    """

    def __init__(self, bev_h=200, bev_w=100, d_bound=[1.0, 60.0, 1.0], num_classes=3):
        """
        `MapTR` 모델을 초기화합니다.
        BEV 맵의 높이/너비, 깊이 구간 범위, 클래스 개수 등을 설정하고
        인코더, LSS, 디코더 서브 모듈들을 생성합니다.

        Args:
            bev_h (int, optional): BEV 맵의 높이. 기본값은 200.
            bev_w (int, optional): BEV 맵의 너비. 기본값은 100.
            d_bound (list, optional): 깊이 구간 [시작, 끝, 간격]. 기본값은 [1.0, 60.0, 1.0].
            num_classes (int, optional): 예측할 클래스의 개수. 기본값은 3.
        """
        super().__init__()

        # 1. Configuration
        # LSS Grid 설정에 따라 Decoder 입력 크기가 결정됩니다.
        C = 64  # Embedding dimensions

        # Depth bin calculation (1m to 60m with 1m interval = 59 bins)
        D = int((d_bound[1] - d_bound[0]) / d_bound[2])

        # 2. Sub-modules
        self.encoder = MapEncoder(C=C, D=D)
        # LSS Grid Configuration
        # 자율주행 좌표계 표준 (Ego Frame):
        # - X축: 전후 방향 (Longitudinal). 멀리 봐야 하므로 -30m ~ 30m (Range 60m)
        # - Y축: 좌우 방향 (Lateral). 도로 폭이므로 -15m ~ 15m (Range 30m)
        grid_conf = {
            "xbound": [-30.0, 30.0, 0.15],  # 60m / 0.15 = 400 grids (X축)
            "ybound": [-15.0, 15.0, 0.15],  # 30m / 0.15 = 200 grids (Y축)
            "zbound": [-2.0, 2.0, 4.0],
            "dbound": d_bound,
        }

        # input_size는 (448, 800) 유지
        self.lss = LSSTransform(
            grid_conf=grid_conf, input_size=(448, 800), downsample=16
        )
        # Decoder 설정
        # bev_w = X축 그리드 개수 = 400
        # bev_h = Y축 그리드 개수 = 200
        self.decoder = MapDecoder(
            bev_h=200,
            bev_w=400,
            embed_dims=C,
            num_classes=num_classes,
            num_query=50,  # One-to-One (Main) Queries
        )

        # Auxiliary One-to-Many Matching Configuration
        # 논문 설정: k=6 (GT 하나를 6개의 쿼리가 맞추도록 유도)
        self.k_one2many = 6
        self.num_one2many_queries = 300  # 50 * 6 = 300

        # 보조 쿼리 임베딩 (Main 쿼리와 별도로 학습)
        self.one2many_instance_embedding = nn.Embedding(self.num_one2many_queries, C)
        # Point query는 Main Decoder와 공유

    def forward(self, imgs, sensor2egos, intrinsics):
        """
        `MapTR` 모델의 포워드 패스입니다.
        입력 이미지, 센서-차량 변환 행렬, 카메라 내부 파라미터를 사용하여
        BEV 특징 맵을 생성하고, 디코더를 통해 지도 요소를 예측합니다.
        학습 시에는 보조 브랜치를 통해 추가적인 출력을 제공합니다.

        Args:
            imgs (torch.Tensor): 입력 이미지 텐서 [B, N, 3, H, W].
            sensor2egos (torch.Tensor): 센서에서 차량 좌표계로의 변환 행렬 텐서 [B, N, 4, 4].
            intrinsics (torch.Tensor): 카메라 내부 파라미터 텐서 [B, N, 3, 3].

        Returns:
            dict or tuple:
                - dict (Train): 학습 시 예측 로짓, 포인트 좌표 및 보조 출력을 포함하는 딕셔너리.
                                 {'pred_logits': ..., 'pred_points': ..., 'aux_outputs': ...}
                - tuple (Eval): 평가 시 클래스 로짓과 포인트 좌표 튜플 (cls_logits, point_coords).
        """
        # 1. Encoder
        depth, feat = self.encoder(imgs)

        # 2. PV-to-BEV (LSS)
        bev_feat = self.lss(feat, depth, sensor2egos, intrinsics)
        # bev_feat shape: [B, C, Z, H, W] -> Squeeze Z -> [B, C, H, W]
        if bev_feat.dim() == 5:
            bev_feat = bev_feat.squeeze(2)

        # 3. Decoder (Main Branch - One-to-One)
        cls_logits, point_coords = self.decoder(bev_feat)

        # 4. Auxiliary Branch (Training Only)
        if self.training:
            B = bev_feat.shape[0]

            # (A) 보조 쿼리 준비
            # Instance Query: [Q_aux, C] -> [Q_aux, 1, C]
            one2many_queries = self.one2many_instance_embedding.weight.unsqueeze(1)
            # Point Query: [P, C] -> [1, P, C] (Main Decoder와 공유)
            point_query = self.decoder.point_embedding.weight.unsqueeze(0)

            # Combine: [Q_aux, P, C] -> Repeat Batch -> [B, Q_aux, P, C]
            aux_queries = (
                (one2many_queries + point_query).unsqueeze(0).repeat(B, 1, 1, 1)
            )

            # (B) Decoder 실행 (Main Decoder 레이어 재사용)
            aux_output = aux_queries
            for layer in self.decoder.decoder_layers:
                # query_pos로 자기 자신 사용
                aux_output = layer(
                    query=aux_output, value_map=bev_feat, query_pos=aux_queries
                )

            # (C) Prediction Head (Main Head 재사용)
            aux_instance_feat = aux_output.mean(dim=2)
            aux_cls_logits = self.decoder.cls_head(aux_instance_feat)
            aux_point_coords = self.decoder.reg_head(aux_output)

            # 딕셔너리로 반환 (Loss 계산용)
            return {
                "pred_logits": cls_logits,
                "pred_points": point_coords,
                "aux_outputs": {
                    "pred_logits": aux_cls_logits,
                    "pred_points": aux_point_coords,
                },
            }

        else:
            # 추론 시에는 Main Branch 결과만 반환 (기존 코드 호환성 유지)
            return cls_logits, point_coords
