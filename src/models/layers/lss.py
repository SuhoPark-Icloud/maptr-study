"""
`LSSTransform` 클래스를 정의하며, 이는 MapTR 모델의 Lift, Splat, Shoot (LSS) 모듈을 구현합니다.
다중 시점 이미지 특징과 깊이 분포를 차량의 BEV (Bird's Eye View) 특징으로 변환하는 역할을 담당합니다.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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


class LSSTransform(nn.Module):
    def __init__(self, grid_conf=None, input_size=(450, 800), downsample=16):
        """
        `MapTR` 모델을 위한 LSS(Lift, Splat, Shoot) 변환 모듈을 초기화합니다.
        BEV 그리드 설정, 그리드 파라미터 계산, Frustum 설정 등을 담당합니다.

        Args:
            grid_conf (dict, optional): BEV 그리드 설정을 포함하는 딕셔너리. 기본값은 None.
            input_size (tuple, optional): 입력 이미지의 (높이, 너비). 기본값은 (450, 800).
            downsample (int, optional): 특징 맵의 다운샘플링 비율. 기본값은 16.
        """
        super().__init__()

        # 1. BEV Grid Setting
        if grid_conf is None:
            self.grid_conf = {
                "xbound": [-15.0, 15.0, 0.15],
                "ybound": [-30.0, 30.0, 0.15],
                "zbound": [-2.0, 2.0, 4.0],
                "dbound": [1.0, 60.0, 1.0],
            }
        else:
            self.grid_conf = grid_conf

        # 2. Grid Parameters
        dx, bx, nx = self.gen_dx_bx(
            self.grid_conf["xbound"], self.grid_conf["ybound"], self.grid_conf["zbound"]
        )
        self.register_buffer("dx", dx)
        self.register_buffer("bx", bx)
        self.register_buffer("nx", nx)

        # 3. Frustum Setup
        self.input_size = input_size
        self.downsample = downsample
        self.fH, self.fW = input_size[0] // downsample, input_size[1] // downsample

        # Calculate D (Depth bins)
        d_bound = self.grid_conf["dbound"]
        self.D = int((d_bound[1] - d_bound[0]) / d_bound[2])

        self.register_buffer("frustum", self.create_frustum())

    def gen_dx_bx(self, xbound, ybound, zbound):
        """
        BEV 그리드의 해상도(`dx`), 시작점(`bx`), 셀 개수(`nx`)를 계산합니다.

        Args:
            xbound (list): X축 경계 [min, max, interval].
            ybound (list): Y축 경계 [min, max, interval].
            zbound (list): Z축 경계 [min, max, interval].

        Returns:
            tuple:
                - dx (torch.Tensor): 각 축의 복셀 해상도 텐서.
                - bx (torch.Tensor): 각 축의 첫 번째 복셀 중심 좌표 텐서.
                - nx (torch.LongTensor): 각 축의 복셀 개수 텐서.
        """
        dx = torch.tensor([xbound[2], ybound[2], zbound[2]])
        # bx is the center of the first voxel (min + interval / 2)
        bx = torch.tensor(
            [
                xbound[0] + xbound[2] / 2.0,
                ybound[0] + ybound[2] / 2.0,
                zbound[0] + zbound[2] / 2.0,
            ]
        )
        nx = torch.LongTensor(
            [
                (xbound[1] - xbound[0]) / xbound[2],
                (ybound[1] - ybound[0]) / ybound[2],
                (zbound[1] - zbound[0]) / zbound[2],
            ]
        )
        return dx, bx, nx

    def create_frustum(self):
        """
        카메라의 시야(frustum)를 3D 포인트 클라우드 형태로 생성합니다.
        이는 각 픽셀과 깊이 빈에 해당하는 3D 공간 상의 점들을 정의합니다.

        Returns:
            torch.Tensor: Frustum을 구성하는 3D 포인트 클라우드 텐서.
        """
        d_bound = self.grid_conf["dbound"]
        ds = torch.arange(d_bound[0], d_bound[1], d_bound[2])

        xs = torch.linspace(0, self.input_size[1] - 1, self.fW)
        ys = torch.linspace(0, self.input_size[0] - 1, self.fH)

        # Meshgrid
        ds = ds.view(-1, 1, 1).expand(-1, self.fH, self.fW)
        xs = xs.view(1, 1, -1).expand(self.D, self.fH, -1)
        ys = ys.view(1, -1, 1).expand(self.D, -1, self.fW)

        return torch.stack((xs, ys, ds), -1)

    def get_geometry(self, sensor2ego_mats, intrinsics):
        """
        2D 이미지 좌표와 깊이 정보, 카메라 내부/외부 파라미터를 사용하여
        각 프러스텀 포인트의 3D 기준 차량(ego) 좌표를 계산합니다.

        Args:
            sensor2ego_mats (torch.Tensor): 센서에서 기준 차량(ego) 좌표계로의 변환 행렬 텐서.
            intrinsics (torch.Tensor): 카메라 내부 파라미터(고유 행렬) 텐서.

        Returns:
            torch.Tensor: 각 프러스텀 포인트의 3D 기준 차량 좌표 텐서.
        """
        B, N = sensor2ego_mats.shape[:2]

        points = self.frustum.view(1, 1, self.D, self.fH, self.fW, 3).expand(
            B, N, -1, -1, -1, -1
        )

        intrinsics_ = intrinsics.view(B * N, 3, 3)
        intrinsics_inv_ = torch.inverse(intrinsics_)
        intrinsics_inv = intrinsics_inv_.view(B, N, 1, 1, 1, 3, 3)

        pts_uv1 = torch.cat(
            [points[..., :2], torch.ones_like(points[..., :1])], dim=-1
        ).unsqueeze(-1)
        pts_cam = torch.matmul(intrinsics_inv, pts_uv1).squeeze(-1)
        pts_cam = pts_cam * points[..., 2:3]

        pts_cam_hom = torch.cat(
            [pts_cam, torch.ones_like(pts_cam[..., :1])], dim=-1
        ).unsqueeze(-1)
        sensor2ego = sensor2ego_mats.view(B, N, 1, 1, 1, 4, 4)

        pts_ego = torch.matmul(sensor2ego, pts_cam_hom).squeeze(-1)[..., :3]

        return pts_ego

    def voxel_pooling(self, geom_feats, geom_coords):
        """
        3D 기준 차량 좌표와 해당 특징들을 BEV 복셀 그리드로 풀링(pooling)합니다.
        그리드 외부의 포인트를 필터링하고, 각 복셀에 해당하는 특징들을 합산합니다.

        Args:
            geom_feats (torch.Tensor): 기하학적 특징 텐서.
            geom_coords (torch.Tensor): 기하학적 좌표 텐서.

        Returns:
            torch.Tensor: 풀링된 BEV 특징 텐서.
        """
        # Filter points outside grid
        kept = (
            (geom_coords[:, 0] >= self.bx[0] - self.dx[0] / 2.0)
            & (
                geom_coords[:, 0]
                < self.bx[0] + self.dx[0] / 2.0 + self.dx[0] * (self.nx[0] - 1)
            )
            & (geom_coords[:, 1] >= self.bx[1] - self.dx[1] / 2.0)
            & (
                geom_coords[:, 1]
                < self.bx[1] + self.dx[1] / 2.0 + self.dx[1] * (self.nx[1] - 1)
            )
            & (geom_coords[:, 2] >= self.bx[2] - self.dx[2] / 2.0)
            & (
                geom_coords[:, 2]
                < self.bx[2] + self.dx[2] / 2.0 + self.dx[2] * (self.nx[2] - 1)
            )
        )

        geom_feats = geom_feats[kept]
        geom_coords = geom_coords[kept]

        if geom_coords.shape[0] == 0:
            return torch.zeros(
                (1, geom_feats.shape[1], self.nx[2], self.nx[1], self.nx[0]),
                device=geom_feats.device,
            )

        # [FIX] Correct Origin Calculation for Indexing
        # bx is center of first voxel. So min_bound = bx - dx/2.
        lower_bound = self.bx - self.dx / 2.0
        coords_ind = ((geom_coords - lower_bound) / self.dx).long()

        # [Safety] Clamp indices to be within valid range
        coords_ind[..., 0] = coords_ind[..., 0].clamp(0, self.nx[0] - 1)
        coords_ind[..., 1] = coords_ind[..., 1].clamp(0, self.nx[1] - 1)
        coords_ind[..., 2] = coords_ind[..., 2].clamp(0, self.nx[2] - 1)

        # Flatten Grid Indices
        ranks = (
            coords_ind[:, 0]
            + coords_ind[:, 1] * self.nx[0]
            + coords_ind[:, 2] * (self.nx[0] * self.nx[1])
        )

        sort_idx = ranks.argsort()
        ranks, geom_feats = ranks[sort_idx], geom_feats[sort_idx]

        bev_feat = torch.zeros(
            (self.nx[2] * self.nx[1] * self.nx[0], geom_feats.shape[1]),
            device=geom_feats.device,
        )

        # [MPS Optimization] Use index_add_
        bev_feat.index_add_(0, ranks, geom_feats)

        bev_feat = bev_feat.permute(1, 0).contiguous()
        bev_feat = bev_feat.view(-1, self.nx[2], self.nx[1], self.nx[0])

        return bev_feat

    def forward(self, img_feats, depth_probs, sensor2ego, intrinsics):
        """
        `LSSTransform` 모듈의 포워드 패스입니다.
        이미지 특징, 깊이 확률, 센서-기준 차량 변환 행렬, 내부 파라미터를 사용하여
        BEV 특징 맵을 생성합니다.

        Args:
            img_feats (torch.Tensor): 이미지 특징 텐서.
            depth_probs (torch.Tensor): 깊이 확률 텐서.
            sensor2ego (torch.Tensor): 센서에서 기준 차량(ego) 좌표계로의 변환 행렬 텐서.
            intrinsics (torch.Tensor): 카메라 내부 파라미터(고유 행렬) 텐서.

        Returns:
            torch.Tensor: 최종 BEV 특징 텐서.
        """
        B, N, C, H, W = img_feats.shape

        img_feats = img_feats.permute(0, 1, 3, 4, 2).unsqueeze(2)
        depth_probs = depth_probs.unsqueeze(-1)

        geom_feats = img_feats * depth_probs
        geom_coords = self.get_geometry(sensor2ego, intrinsics)

        # [FIX] Use reshape instead of view
        geom_feats = geom_feats.reshape(B, -1, C)
        geom_coords = geom_coords.reshape(B, -1, 3)

        final_bevs = []
        for b in range(B):
            bev = self.voxel_pooling(geom_feats[b], geom_coords[b])
            final_bevs.append(bev)

        return torch.stack(final_bevs)


# --- Testing Block ---
if __name__ == "__main__":
    print("🧪 Testing LSS Module (Pure PyTorch)...")

    lss = LSSTransform()

    B, N, C = 1, 6, 64

    img_feats = torch.randn(B, N, C, lss.fH, lss.fW)
    depth_probs = torch.randn(B, N, lss.D, lss.fH, lss.fW).softmax(dim=2)

    # [FIX] Use clone()
    sensor2ego = torch.eye(4).view(1, 1, 4, 4).expand(B, N, -1, -1).clone()
    intrinsics = torch.eye(3).view(1, 1, 3, 3).expand(B, N, -1, -1).clone()

    intrinsics[..., 0, 0] = 500
    intrinsics[..., 1, 1] = 500
    intrinsics[..., 0, 2] = 400
    intrinsics[..., 1, 2] = 225

    try:
        bev_map = lss(img_feats, depth_probs, sensor2ego, intrinsics)
        print("✅ LSS Forward Success!")
        print(f"   Input Feats: {img_feats.shape}")
        print(f"   Output BEV:  {bev_map.shape} (Expected: [B, C, Z, Y, X])")

        nx = lss.nx.numpy()
        print(f"   Grid Size:   {nx} (X={nx[0]}, Y={nx[1]}, Z={nx[2]})")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"❌ Error during LSS Forward: {e}")
