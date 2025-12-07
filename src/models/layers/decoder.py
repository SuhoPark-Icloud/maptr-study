"""
`MapDecoder` 클래스와 `MacDeformableCrossAttention` 클래스를 정의하며,
MapTR 모델의 디코더 컴포넌트를 구현합니다.
BEV 특징 맵에서 지도 요소를 추출하고 분류/회귀하는 역할을 담당합니다.
"""

import torch.nn as nn
import torch.nn.functional as F


class MacDeformableCrossAttention(nn.Module):
    """
    MPS(Metal Performance Shaders)와 호환되는 Deformable Cross-Attention을 구현합니다.
    CUDA 커널 대신 PyTorch의 `grid_sample`을 사용하여 특징 샘플링을 수행합니다.
    """

    def __init__(self, embed_dims=64, num_heads=4, num_points=4):
        """
        `MacDeformableCrossAttention` 모듈을 초기화합니다.
        임베딩 차원, 어텐션 헤드 수, 샘플링할 점의 개수 등을 설정합니다.

        Args:
            embed_dims (int, optional): 임베딩 차원. 기본값은 64.
            num_heads (int, optional): 어텐션 헤드의 개수. 기본값은 4.
            num_points (int, optional): 각 헤드마다 샘플링할 점의 개수. 기본값은 4.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points  # head 당 샘플링할 점의 개수
        self.head_dim = embed_dims // num_heads

        # 1. Sampling Offsets 예측 (Query -> Offset)
        # 각 Head마다, 각 Point마다 x, y 오프셋이 필요하므로 output은 heads * points * 2
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 2)

        # 2. Attention Weights 예측 (Query -> Weight)
        # 각 샘플링된 점에 대한 중요도 (Scalar)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_points)

        # 3. Value Projection (BEV Features -> Value)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        # 4. Output Projection
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, query, reference_points, value_map):
        """
        `MacDeformableCrossAttention` 모듈의 포워드 패스입니다.
        쿼리, 레퍼런스 포인트, BEV 특징 맵을 사용하여 변형 가능한 크로스-어텐션을 수행하고,
        집계된 특징을 반환합니다.

        Args:
            query (torch.Tensor): 쿼리 텐서 [B, Num_Query, C].
            reference_points (torch.Tensor): 정규화된 레퍼런스 포인트 텐서 [B, Num_Query, 2]. (0~1 범위)
            value_map (torch.Tensor): BEV 특징 맵 텐서 [B, C, H, W]. (평탄화되지 않은 맵)

        Returns:
            torch.Tensor: 집계된 특징 텐서 [B, Num_Query, C].
        """
        B, Nq, C = query.shape
        B, C_val, H, W = value_map.shape

        # 1. Value 준비 (투영 및 헤드 분리)
        # value_map: [B, C, H, W] -> value: [B*heads, head_dim, H, W]
        value = self.value_proj(value_map.permute(0, 2, 3, 1))  # [B, H, W, C]
        value = value.view(
            B, H, W, self.num_heads, self.head_dim
        )  # [B, H, W, heads, head_dim]
        value = value.permute(0, 3, 1, 2, 4).reshape(
            B * self.num_heads, H, W, self.head_dim
        )  # [B*heads, H, W, head_dim]
        value = value.permute(0, 3, 1, 2)  # [B*heads, head_dim, H, W]

        # 2. 오프셋 및 가중치 예측
        sampling_offsets = self.sampling_offsets(query).view(
            B, Nq, self.num_heads, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Nq, self.num_heads, self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)  # [B, Nq, heads, pts]

        # 3. 샘플링 위치 계산
        ref_points_ex = reference_points.view(B, Nq, 1, 1, 2)
        sampling_locations = ref_points_ex + sampling_offsets  # [B, Nq, heads, pts, 2]

        # 4. 샘플링 ("Deformable" 부분)
        # grid_sample을 위한 형태 변경: value는 [B*heads,...], grid는 일치해야 함
        # locations: [B, Nq, heads, pts, 2] -> [B, heads, Nq, pts, 2]
        sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4)
        # -> [B*heads, Nq*pts, 1, 2]
        sampling_grid = sampling_locations.reshape(
            B * self.num_heads, Nq * self.num_points, 1, 2
        )
        sampling_grid = sampling_grid * 2.0 - 1.0  # [-1, 1] 범위로 정규화

        # 각 헤드별 특징 샘플링
        # 출력: [B*heads, head_dim, Nq*pts, 1]
        sampled_feats = F.grid_sample(
            value,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # 5. 집계
        # [B*heads, head_dim, Nq*pts, 1] -> [B, heads, head_dim, Nq, pts]
        sampled_feats = sampled_feats.view(
            B, self.num_heads, self.head_dim, Nq, self.num_points
        )
        # [B, heads, head_dim, Nq, pts] -> [B, Nq, heads, pts, head_dim]
        sampled_feats = sampled_feats.permute(0, 3, 1, 4, 2)

        # 어텐션 가중치 적용
        # [B, Nq, h, p, d] * [B, Nq, h, p, 1] -> p에 대해 합산 -> [B, Nq, h, d]
        agg_feats = (sampled_feats * attention_weights.unsqueeze(-1)).sum(dim=3)

        # 헤드 연결 및 투영
        # [B, Nq, C]
        agg_feats = agg_feats.flatten(2)
        output = self.output_proj(agg_feats)
        return output


class DecoupledMapDecoderLayer(nn.Module):
    """
    `MapTR` 디코더의 단일 레이어를 구현합니다.
    인스턴스 내(intra-instance) 및 인스턴스 간(inter-instance) Self-Attention과
    Deformable Cross-Attention을 포함합니다.
    """

    def __init__(self, embed_dims=64, nhead=4, dropout=0.1, dim_feedforward=128):
        """
        `DecoupledMapDecoderLayer` 모듈을 초기화합니다.
        임베딩 차원, 헤드 수, 드롭아웃 비율, 피드포워드 네트워크의 차원 등을 설정합니다.

        Args:
            embed_dims (int, optional): 임베딩 차원. 기본값은 64.
            nhead (int, optional): 어텐션 헤드의 개수. 기본값은 4.
            dropout (float, optional): 드롭아웃 비율. 기본값은 0.1.
            dim_feedforward (int, optional): 피드포워드 네트워크의 차원. 기본값은 128.
        """
        super().__init__()
        self.embed_dims = embed_dims

        # --- Self-Attention (기존 유지) ---
        self.intra_ins_sa = nn.MultiheadAttention(
            embed_dims, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dims)

        self.inter_ins_sa = nn.MultiheadAttention(
            embed_dims, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        # --- Cross-Attention (변경됨: MacDeformable) ---
        # 기존: self.cross_attn = nn.MultiheadAttention(...)
        # 변경: Reference Point 기반 Deformable Attention
        self.cross_attn = MacDeformableCrossAttention(
            embed_dims, num_heads=nhead, num_points=4
        )
        self.norm3 = nn.LayerNorm(embed_dims)

        # Reference Point Generator (Query -> 2D Point)
        # MapTRv2에서는 쿼리 자체가 위치 정보를 예측함
        self.reference_points_head = nn.Linear(embed_dims, 2)

        # FFN
        self.linear1 = nn.Linear(embed_dims, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dims)
        self.norm4 = nn.LayerNorm(embed_dims)
        self.activation = nn.ReLU()

    def forward(self, query, value_map, query_pos=None):
        """
        `DecoupledMapDecoderLayer` 모듈의 포워드 패스입니다.
        쿼리, BEV 특징 맵, 쿼리 위치 임베딩을 입력으로 받아 디코더 레이어 연산을 수행하고
        갱신된 쿼리 특징을 반환합니다.

        Args:
            query (torch.Tensor): 쿼리 텐서 [B, Num_Ins, Num_Pts, C].
            value_map (torch.Tensor): BEV 특징 맵 텐서 [B, C, H, W].
            query_pos (torch.Tensor, optional): 쿼리 위치 임베딩 텐서 [B, Num_Ins, Num_Pts, C]. 기본값은 None.

        Returns:
            torch.Tensor: 갱신된 쿼리 특징 텐서 [B, Num_Ins, Num_Pts, C].
        """
        B, N, Nv, C = query.shape

        # Q = Query + Pos
        q = query + query_pos if query_pos is not None else query

        # --- 1. Intra-Instance SA ---
        q_intra = q.view(B * N, Nv, C)
        tgt, _ = self.intra_ins_sa(q_intra, q_intra, q_intra)
        query = self.norm1(query.view(B * N, Nv, C) + tgt).view(B, N, Nv, C)

        # --- 2. Inter-Instance SA ---
        q = query + query_pos if query_pos is not None else query
        q_inter = q.permute(0, 2, 1, 3).reshape(B * Nv, N, C)
        tgt, _ = self.inter_ins_sa(q_inter, q_inter, q_inter)
        tgt = tgt.view(B, Nv, N, C).permute(0, 2, 1, 3)
        query = self.norm2(query + tgt)

        # --- 3. Deformable Cross-Attention ---
        # Query Flatten: [B, Total_Queries, C]
        q_flat = query.flatten(1, 2)
        q_pos_flat = query_pos.flatten(1, 2) if query_pos is not None else None
        q_in = q_flat + q_pos_flat if q_pos_flat is not None else q_flat

        # (A) Reference Points 생성 (Sigmoid로 0~1 정규화)
        reference_points = self.reference_points_head(q_in).sigmoid()

        # (B) Deformable Attention 수행
        # value_map을 Flatten 하지 않고 2D 이미지 형태 그대로 넘깁니다.
        tgt = self.cross_attn(
            query=q_in, reference_points=reference_points, value_map=value_map
        )

        q_flat = self.norm3(q_flat + tgt)

        # --- 4. FFN ---
        tgt = self.linear2(self.dropout(self.activation(self.linear1(q_flat))))
        q_flat = self.norm4(q_flat + tgt)

        return q_flat.view(B, N, Nv, C)


class MapDecoder(nn.Module):
    """
    `MapTR` 모델의 디코더를 구현하는 클래스입니다.
    여러 개의 `DecoupledMapDecoderLayer`를 통해 BEV 특징 맵에서 지도 요소를 반복적으로 추출하고,
    최종적으로 클래스 로짓과 포인트 좌표를 예측합니다.
    """

    def __init__(
        self,
        bev_h=200,
        bev_w=100,
        embed_dims=64,
        num_query=50,
        num_points=20,
        num_classes=3,
        num_layers=6,
    ):
        """
        `MapDecoder` 모듈을 초기화합니다.
        BEV 맵의 높이/너비, 임베딩 차원, 쿼리 개수, 포인트 개수, 클래스 개수, 레이어 개수 등을 설정합니다.

        Args:
            bev_h (int, optional): BEV 맵의 높이. 기본값은 200.
            bev_w (int, optional): BEV 맵의 너비. 기본값은 100.
            embed_dims (int, optional): 임베딩 차원. 기본값은 64.
            num_query (int, optional): 쿼리의 개수. 기본값은 50.
            num_points (int, optional): 각 쿼리당 포인트의 개수. 기본값은 20.
            num_classes (int, optional): 예측할 클래스의 개수. 기본값은 3.
            num_layers (int, optional): 디코더 레이어의 개수. 기본값은 6.
        """
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_points = num_points

        self.instance_embedding = nn.Embedding(num_query, embed_dims)
        self.point_embedding = nn.Embedding(num_points, embed_dims)

        self.decoder_layers = nn.ModuleList(
            [
                DecoupledMapDecoderLayer(embed_dims=embed_dims, nhead=4)
                for _ in range(num_layers)
            ]
        )

        # Deformable Attention은 Positional Embedding을 Reference Point로 대체하므로
        # BEV Positional Embedding은 이제 필수 요소는 아니지만, 초기화용으로 남겨둘 수 있습니다.
        # 여기서는 제거하거나 사용하지 않아도 무방합니다.

        self.cls_head = nn.Linear(embed_dims, num_classes)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, 2),
            nn.Sigmoid(),
        )

    def get_hierarchical_queries(self, batch_size):
        """
        계층적 쿼리(인스턴스 쿼리 + 포인트 쿼리)를 생성합니다.

        Args:
            batch_size (int): 배치 크기.

        Returns:
            torch.Tensor: 생성된 계층적 쿼리 텐서.
        """
        inst_query = self.instance_embedding.weight.unsqueeze(1)
        point_query = self.point_embedding.weight.unsqueeze(0)
        queries = inst_query + point_query
        queries = queries.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return queries

    def forward(self, bev_feat):
        """
        `MapDecoder` 모듈의 포워드 패스입니다.
        BEV 특징 맵을 입력으로 받아 디코더 레이어들을 거치며 지도 요소를 예측하고,
        클래스 로짓과 포인트 좌표를 반환합니다.

        Args:
            bev_feat (torch.Tensor): BEV 특징 맵 텐서 [B, C, H, W].

        Returns:
            tuple:
                - cls_logits (torch.Tensor): 클래스 로짓 텐서.
                - point_coords (torch.Tensor): 예측된 포인트 좌표 텐서.
        """
        B, C, H, W = bev_feat.shape

        # 기존: bev_flat 생성 -> 삭제 (Deformable은 2D feature map을 직접 사용)

        queries = self.get_hierarchical_queries(B)

        output = queries

        # 반복적 디코딩
        for layer in self.decoder_layers:
            # value=bev_feat (2D map) 전달
            # query_pos는 쿼리 자체에서 생성되거나 별도 임베딩 사용 가능
            # 여기서는 쿼리를 pos처럼 사용
            output = layer(query=output, value_map=bev_feat, query_pos=queries)

        # 예측
        instance_feat = output.mean(dim=2)
        cls_logits = self.cls_head(instance_feat)
        point_coords = self.reg_head(output)

        return cls_logits, point_coords
