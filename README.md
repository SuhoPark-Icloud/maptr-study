# MapTR-MPS: Educational MapTRv2 Implementation (Pure PyTorch)

이 저장소는 **MapTRv2 (Map Transformer)** 논문의 핵심 아이디어를 **Apple Silicon (M1/M2/M3) Mac** 환경의 **MPS (Metal Performance Shaders)** 가속을 활용하여 학습할 수 있도록 구현한 프로젝트입니다.

복잡한 `mmdetection3d`나 CUDA 컴파일 의존성을 모두 제거하고, **Pure PyTorch**로만 구현하여 코드의 가독성을 높이고 설치 과정을 단순화했습니다.

> ⚠️ **Disclaimer (주의사항)**
> 1. **교육 및 연구 목적:** 이 코드는 MapTR의 동작 원리를 이해하기 위해 작성된 교육용 코드입니다.
> 2. **AI Assisted:** 이 코드는 생성형 AI의 도움을 받아 작성되었으며, 인간의 검토를 거쳤으나 잠재적인 버그가 존재할 수 있습니다.
> 3. **학습 검증 미완료:** 개발자의 하드웨어 성능 한계(MacBook) 및 시간상의 제약으로 인해 Full Epoch 학습을 통한 최종 성능(mAP) 검증은 완료되지 않았습니다. (단, `debug_overfit.py`를 통한 수렴성 검증은 완료됨)

## 🌟 Key Features (주요 특징)

*   **Mac(MPS) Native Support:** CUDA 전용인 `MSDeformAttn`을 사용하지 않고, PyTorch의 `F.grid_sample`을 활용한 **MacDeformableCrossAttention**을 구현하여 맥북에서도 학습이 가능합니다.
*   **Pure PyTorch:** `mmcv`, `mmdet3d` 등 설치가 까다로운 라이브러리 의존성을 제거했습니다.
*   **MapTRv2 Core Implemented:**
    *   계층적 쿼리 (인스턴스 + 포인트)
    *   분리된 셀프 어텐션
    *   보조적인 One-to-Many 매칭 & 밀집 감독 (깊이/분할)
    *   순열 동등 매칭 (이동 & 뒤집기)

## 🛠️ Getting Started (시작하기)

이 프로젝트는 `uv`와 Python 3.11을 기반으로 합니다.

### 1. 환경 설정 (Environment Setup)

먼저, `uv`를 사용하여 Python 3.11 가상 환경을 생성하고 활성화합니다.

```bash
# Python 3.11 가상 환경 생성
uv venv

# 가상 환경 활성화 (macOS/Linux)
source .venv/bin/activate
```

### 2. 의존성 설치 (Install Dependencies)

`uv`를 사용하여 `pyproject.toml`에 명시된 프로젝트 의존성을 설치합니다.

```bash
uv pip install -e .
```

### 3. 데이터 준비 (Data Preparation)
NuScenes 데이터셋을 준비해야 합니다.

*   **필수:** Full Dataset (`v1.0-trainval`) 또는 Mini (`v1.0-mini`)
*   **필수:** [Map Expansion](https://www.nuscenes.org/download) (벡터 지도를 만드는 데 필요)

데이터셋은 아래와 같은 구조로 `data/nuscenes` 폴더에 위치해야 합니다.

```
maptr-study/
└── data/
    └── nuscenes/
        ├── maps/           # Map expansion 파일
        │   ├── basemap/
        │   ├── expansion/
        │   └── prediction/
        ├── samples/        # 카메라 이미지, 라이다 데이터 등
        ├── sweeps/
        └── v1.0-mini/      # 또는 v1.0-trainval
```
## 🚀 Usage (사용법)

### 1. 파이프라인 검증 (Visualization)
데이터 로더와 모델이 정상적으로 연결되어 점들이 찍히는지 확인합니다.
```bash
python tools/verify_full_pipeline.py
```

### 2. 과적합 디버깅 (Overfitting Debug)
작은 데이터(1개 샘플)에 대해 모델이 수렴하는지 테스트합니다. 모델 로직이 정상이라면 Loss가 줄어들고 예측된 점들이 GT(초록색)와 일치(빨간색)해야 합니다.
```bash
python tools/debug_overfit.py
```

### 3. 학습 (Train)
본격적인 학습을 시작합니다.
```bash
python tools/train.py
```
*   `logs/` 폴더에 TensorBoard 로그가 저장됩니다.
*   `checkpoints/` 폴더에 모델 가중치가 저장됩니다.

TensorBoard를 사용하여 학습 과정을 시각화할 수 있습니다.
```bash
tensorboard --logdir logs/
```
위 명령어를 실행하면 터미널에 `TensorBoard 2.x at http://localhost:6006/ (Press CTRL+C to quit)`와 같은 메시지가 표시됩니다. 해당 URL을 웹 브라우저에 입력하여 TensorBoard 대시보드를 확인할 수 있습니다.

### 4. 추론 (Inference)
저장된 체크포인트를 불러와 추론 결과를 시각화합니다.
```bash
python tools/inference.py
```

## 🤝 Contribution (기여하기)
이 프로젝트는 아직 완벽하지 않습니다. 다음과 같은 기여는 언제나 환영합니다!
*   Mac/MPS 호환성 유지: CUDA 커널을 강제하지 않는 Pure PyTorch 구현 개선.
*   버그 수정: 코드 내 잠재적 오류 수정.
*   성능 검증: 고성능 GPU 환경에서 Full Training을 돌려보고 성능 리포트 공유.

Note: PR을 보내실 때는 코드가 Mac 환경에서도 돌아갈 수 있도록 가능한 순수 PyTorch API를 유지해 주세요.

## 📄 License & Acknowledgements

### License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

### Acknowledgements
This project is an educational re-implementation based on the official **MapTR** repository. We deeply respect the original authors and their contribution to the autonomous driving community.

*   **Original Paper:** [MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction](https://arxiv.org/abs/2308.05736) [1]
*   **Original Code:** [hustvl/MapTR](https://github.com/hustvl/MapTR) [2]

If you use this code or the original MapTR ideas, please cite the original papers:

```bibtex
@inproceedings{liao2022maptr,
  title={MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction},
  author={Liao, Bencheng and Chen, Shaoyu and Wang, Xinggang and Cheng, Tianheng and Zhang, Qian and Liu, Wenyu and Huang, Chang},
  booktitle={ICLR},
  year={2023}
}

@article{liao2023maptrv2,
  title={MapTRv2: An End-to-End Framework for Online Vectorized HD Map Construction},
  author={Liao, Bencheng and Chen, Shaoyu and Zhang, Yunchi and Jiang, Bo and Zhang, Qian and Liu, Wenyu and Huang, Chang and Wang, Xinggang},
  journal={arXiv preprint arXiv:2308.05736},
  year={2023}
}

This implementation also references concepts from:
• Lift, Splat, Shoot (LSS): [Philion and Fidler, ECCV 2020]
• Deformable DETR: [Zhu et al., ICLR 2021]
