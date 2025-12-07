"""
`MapTR` 모델 학습 및 평가에 사용될 `MapTRDataset` 클래스를 정의합니다.
nuScenes 데이터셋을 기반으로 하며, 다중 시점 이미지와 벡터화된 지도(vector map) GT(Ground Truth)를
로드하고 전처리하는 역할을 담당합니다.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

# Matplotlib Patch for Mac
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

from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap


class MapTRDataset(Dataset):
    """
    `MapTR` 모델을 위한 PyTorch `Dataset` 클래스입니다.
    nuScenes 데이터셋의 샘플을 처리하여 모델 입력(이미지, 카메라 파라미터)과
    정답(벡터 맵)을 생성합니다.
    """

    def __init__(self, nusc, is_train=True):
        """
        `MapTRDataset`을 초기화합니다.
        nuScenes 객체, 학습/검증 모드 설정, 데이터 증강 및 변환 파이프라인 정의,
        지도 API 캐싱 등을 수행합니다.

        Args:
            nusc (NuScenes): nuScenes 데이터베이스 API 객체.
            is_train (bool, optional): 학습 모드 여부. 기본값은 True.
        """
        self.nusc = nusc
        self.is_train = is_train
        self.samples = self.nusc.sample

        # Ego Frame 기준 Patch Size
        # X (전후, Length): 60.0m (-30 ~ 30)
        # Y (좌우, Width): 30.0m (-15 ~ 15)
        self.patch_size = [60.0, 30.0]
        self.num_points = 20
        self.map_classes = ["divider", "ped_crossing", "boundary"]

        # Image Config
        self.img_scale = 0.5
        self.final_dim = (448, 800)  # (H, W)

        self.transform = T.Compose(
            [
                T.Resize(self.final_dim),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Map API Caching
        self.maps = {}
        for scene in nusc.scene:
            log = nusc.get("log", scene["log_token"])
            location = log["location"]
            if location not in self.maps:
                self.maps[location] = NuScenesMap(
                    dataroot=nusc.dataroot, map_name=location
                )

    def __len__(self):
        """
        데이터셋의 총 샘플 개수를 반환합니다.

        Returns:
            int: 데이터셋의 샘플 개수.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        주어진 인덱스(`idx`)에 해당하는 데이터 샘플을 로드하고 전처리하여 반환합니다.
        기준 차량의 포즈, 다중 시점 이미지, 카메라 파라미터(내부/외부),
        벡터 맵 GT를 포함하는 딕셔너리를 생성합니다.

        Args:
            idx (int): 가져올 샘플의 인덱스.

        Returns:
            dict: 모델 입력 및 정답 데이터를 포함하는 딕셔너리.
                  - 'imgs': 이미지 텐서
                  - 'intrinsics': 카메라 내부 파라미터 텐서
                  - 'sensor2egos': 센서-기준 차량 변환 행렬 텐서
                  - 'vectors': 벡터화된 지도 GT
                  - 'meta': 메타 정보 (토큰, 위치 등)
        """
        sample = self.samples[idx]
        scene = self.nusc.get("scene", sample["scene_token"])
        log = self.nusc.get("log", scene["log_token"])
        location = log["location"]
        nusc_map = self.maps[location]

        # --- 1. Ego Pose ---
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.nusc.get("sample_data", lidar_token)
        ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

        ego_trans = np.array(ego_pose["translation"])
        ego_rot = Quaternion(ego_pose["rotation"])

        # --- 2. Load Images & Camera Params ---
        imgs = []
        intrinsics = []
        sensor2egos = []

        cameras = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]

        for cam in cameras:
            cam_token = sample["data"][cam]
            cam_data = self.nusc.get("sample_data", cam_token)

            # A. Load Image
            img_path = os.path.join(self.nusc.dataroot, cam_data["filename"])
            img = Image.open(img_path)
            img = self.transform(img)
            imgs.append(img)

            # B. Get Calibration (Intrinsic)
            cs_record = self.nusc.get(
                "calibrated_sensor", cam_data["calibrated_sensor_token"]
            )
            intrinsic = np.array(cs_record["camera_intrinsic"])

            if self.img_scale != 1.0:
                resize_mat = np.eye(3)
                resize_mat[0, 0] = self.img_scale
                resize_mat[1, 1] = self.img_scale
                intrinsic = resize_mat @ intrinsic

            intrinsics.append(torch.from_numpy(intrinsic).float())

            # C. Get Extrinsic (Sensor -> Ego)
            sens_rot = Quaternion(cs_record["rotation"]).rotation_matrix
            sens_trans = np.array(cs_record["translation"])

            view_mat = np.eye(4)
            view_mat[:3, :3] = sens_rot
            view_mat[:3, 3] = sens_trans
            sensor2egos.append(torch.from_numpy(view_mat).float())

        imgs = torch.stack(imgs)
        intrinsics = torch.stack(intrinsics)
        sensor2egos = torch.stack(sensor2egos)

        # --- 3. Vector Map Extraction (GT) ---
        patch_box = (ego_trans[0], ego_trans[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(ego_rot) / np.pi * 180

        vectors = []
        for map_cat in self.map_classes:
            if map_cat == "divider":
                layers = ["road_divider", "lane_divider"]
            elif map_cat == "ped_crossing":
                layers = ["ped_crossing"]
            else:  # boundary
                layers = ["road_segment", "lane"]

            raw_geoms = nusc_map.get_map_geom(patch_box, patch_angle, layers)
            flat_geoms = self._flatten_geoms(raw_geoms)

            for geom in flat_geoms:
                if geom.is_empty:
                    continue
                points = None
                if geom.geom_type == "Polygon":
                    points = np.array(geom.exterior.coords)
                elif geom.geom_type == "LineString":
                    points = np.array(geom.coords)

                if points is None or len(points) < 2:
                    continue

                # Ego 좌표계: X(전후), Y(좌우)
                # X Range: -30 ~ 30 (Size 60)
                # Y Range: -15 ~ 15 (Size 30)

                normalized_points = np.copy(points)

                # Normalize X: (x + 30) / 60 -> 0 ~ 1
                normalized_points[:, 0] = (normalized_points[:, 0] + 30.0) / 60.0

                # Normalize Y: (y + 15) / 30 -> 0 ~ 1
                normalized_points[:, 1] = (normalized_points[:, 1] + 15.0) / 30.0

                resampled_pts = self.resample_polyline(
                    normalized_points, self.num_points
                )
                vectors.append((map_cat, torch.from_numpy(resampled_pts).float()))

        return {
            "imgs": imgs,
            "intrinsics": intrinsics,
            "sensor2egos": sensor2egos,
            "vectors": vectors,
            "meta": {"token": sample["token"], "location": location},
        }

    def _flatten_geoms(self, item):
        """
        중첩된 리스트 형태의 지도 기하학적 데이터(`item`)를
        평탄한(flat) 리스트로 변환하는 재귀 함수입니다.

        Args:
            item: nuScenes map API에서 반환된 기하학적 데이터.

        Returns:
            list: 평탄화된 shapely 기하학적 객체 리스트.
        """
        geoms = []
        if isinstance(item, (list, tuple)):
            for sub in item:
                if not isinstance(sub, str):
                    geoms.extend(self._flatten_geoms(sub))
        elif hasattr(item, "geom_type"):
            geoms.append(item)
        return geoms

    def resample_polyline(self, points, num_points):
        """
        폴리라인을 구성하는 점들의 배열(`points`)을
        일정한 간격의 `num_points` 개 점으로 리샘플링합니다.

        Args:
            points (numpy.ndarray): 리샘플링할 원본 포인트 배열 (N, 2).
            num_points (int): 리샘플링 결과로 나올 포인트의 개수.

        Returns:
            numpy.ndarray: 리샘플링된 포인트 배열 (`num_points`, 2).
        """
        if len(points) < 2:
            return np.repeat(points[0:1], num_points, axis=0)
        dists = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cum_dists = np.cumsum(dists)
        cum_dists = np.insert(cum_dists, 0, 0.0)
        total_dist = cum_dists[-1]
        if total_dist == 0:
            return np.repeat(points[0:1], num_points, axis=0)
        step_dists = np.linspace(0, total_dist, num_points)
        new_xs = np.interp(step_dists, cum_dists, points[:, 0])
        new_ys = np.interp(step_dists, cum_dists, points[:, 1])
        return np.stack([new_xs, new_ys], axis=1)
