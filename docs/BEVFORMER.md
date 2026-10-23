# BEVFormer-tiny: 카메라 6대 기반 3D 검출 실험

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

이 통합 코드는 공식 BEVFormer-tiny 설정을 상속하고, 단일 GPU 학습을 실행합니다. 모델 자체를 재구현하지 않습니다. RTX 3080 Laptop 16GB에서는 기본 `fp32`, GPU당 배치 1, 로더 작업자 2부터 시작합니다. 실제 메모리·속도는 환경에서 측정해야 합니다. 이 모델의 출력은 3D 객체 검출이며, 경로 계획·조향 제어·occupancy 학습 기능을 제공하는 것은 아닙니다.

기준 소스는 [공식 저장소의 커밋 `66b65f3a1f58caf0507cb2a971b9c0e7f842376c`](https://github.com/fundamentalvision/BEVFormer/commit/66b65f3a1f58caf0507cb2a971b9c0e7f842376c)입니다. 기본값은 이 커밋의 수정되지 않은 checkout을 요구합니다. 다른 버전이나 수정본을 의도적으로 쓰는 경우에만 `--allow-unpinned`를 지정하고 그 차이를 별도로 검증하세요. 로컬 Git 상태는 실행 계획에 기록됩니다.

## 별도 환경

현재 프로젝트의 일반 PyTorch 학습 환경과 **분리된 Ubuntu/WSL2 Python 3.8 환경**에서 실행하세요. 설정 생성과 이 폴더의 테스트는 Windows의 Python 3.8 이상에서도 됩니다. 학습은 Linux의 CUDA/NCCL 환경을 전제로 하며 Windows 네이티브 실행은 차단합니다.

공식 설치 문서가 지정한 핵심 조합은 다음과 같습니다. 이는 최신 버전 권장이 아닌, 해당 연구 코드의 호환 기준입니다.

| 구성 | 기준 버전 |
|---|---|
| Python | 3.8 |
| PyTorch / torchvision | 1.9.1+cu111 / 0.10.1+cu111 |
| CUDA 빌드 | 11.1 |
| mmcv-full | 1.4.0 |
| mmdet / mmsegmentation | 2.14.0 / 0.14.1 |
| mmdetection3d | 소스 태그 v0.17.1 |

전체 보조 라이브러리, 컴파일 도구와 설치 순서는 [해당 커밋의 공식 설치 문서](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/docs/install.md)를 따르세요. CUDA 확장 빌드에는 호환되는 CUDA toolkit/nvcc와 컴파일러가 필요할 수 있습니다. 시스템의 최신 CUDA나 최신 MMDetection으로 임의 교체하면 호환된다고 볼 수 없습니다. 이 프로젝트는 환경을 자동 설치하거나 전역 Python을 변경하지 않습니다.

예를 들어 conda로 환경과 기준 소스를 별도로 마련할 수 있습니다. 다음 명령은 사용자가 직접 실행하는 설치 단계입니다.

```bash
conda create -n bevformer-legacy python=3.8
conda activate bevformer-legacy
git clone https://github.com/fundamentalvision/BEVFormer.git "$HOME/BEVFormer"
git -C "$HOME/BEVFormer" checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
```

공식 문서의 나머지 설치를 마친 후 CUDA 및 mmcv 확장이 로드되는지 확인합니다.

```bash
python -c "import torch; from mmcv.ops import MultiScaleDeformableAttention; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

학습 래퍼는 Python과 핵심 패키지 버전을 확인하고, CUDA가 없거나 버전이 다르면 학습 전에 종료합니다. 이 검사만으로 모든 확장·드라이버의 정상 작동이 보장되지는 않습니다. 초기 소량 실행으로 실제 연산도 확인해야 합니다.

## 데이터 준비

nuScenes 사용 약관에 동의하고 [공식 다운로드 페이지](https://www.nuscenes.org/download)에서 mini 또는 trainval 및 CAN bus 확장을 준비합니다. tiny는 RGB 카메라 6대와 보정·자차 상태·3D 정답을 이용합니다. 차량에 LiDAR를 장착할 필요는 없지만 **공식 전처리기는 nuScenes의 LiDAR 원본 경로도 확인**하므로, 영상만 복사한 데이터로 원본 전처리가 된다고 가정하면 안 됩니다.

예시 배치:

```text
workspace/
  camera-rc-training/       # 이 프로젝트
  data/
    can_bus/               # CAN bus 압축 해제 결과
    nuscenes-mini/
      samples/CAM_FRONT/ ... 나머지 5개 카메라
      samples/LIDAR_TOP/
      sweeps/
      maps/
      v1.0-mini/
```

Windows와 WSL의 경로는 서로 다릅니다. 실제 마운트 위치를 확인하고 `PROJECT_ROOT`와 `DATA_ROOT`를 자신의 WSL 경로로 설정하세요. **WSL 안에서 생성한 경로를 계속 사용**해야 하며 Windows 경로가 들어 있는 설정·PKL을 WSL에 그대로 가져가면 안 됩니다. 작은 파일이 많은 경우 WSL Linux 파일시스템으로 데이터를 배치하면 I/O 차이가 있을 수 있습니다.

전처리 도우미는 공식 `create_nuscenes_infos` 함수를 호출해 temporal PKL 두 개를 만듭니다. trainval만 준비하는 경우 stock CLI의 추가 test 변환을 요청하지 않습니다. 모델에 필요 없는 2D COCO 주석도 만들지 않습니다. 기본 실행은 경로와 명령 확인뿐이고, `--execute`를 붙여야 변환합니다.

```bash
cd "$PROJECT_ROOT"
conda activate bevformer-legacy
python integrations/bevformer/prepare_infos.py \
  --upstream "$HOME/BEVFormer" \
  --data-root "$DATA_ROOT/nuscenes-mini" \
  --canbus-root "$DATA_ROOT" \
  --version v1.0-mini --prepare-only
```

경로가 맞으면 `--prepare-only`를 `--execute`로 바꿉니다. 결과는 기본적으로 해당 data root의 `nuscenes_infos_temporal_train.pkl`, `nuscenes_infos_temporal_val.pkl`입니다. full 데이터에서는 data root를 별도 폴더로 바꾸고 `--version v1.0-trainval`을 사용합니다. mini/full을 같은 폴더에 출력해 덮어쓰지 마세요.

**일반 MMDetection3D의 `nuscenes_infos_train.pkl`은 이 모델의 temporal 주석과 다릅니다.** PKL 안에는 변환 당시의 카메라 경로가 저장됩니다. 설정의 `data_root`만 바꾸어도 내부 경로가 자동 교체되지는 않습니다. 데이터를 이동했다면 새 위치에서 주석을 다시 생성하세요. 자세한 형식은 [공식 데이터 준비 문서](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/docs/prepare_dataset.md)와 [공식 converter](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/tools/data_converter/nuscenes_converter.py)에 있습니다.

## 설정 생성과 학습

아래는 **mini 데이터로 연결을 확인하는 예시**입니다. mini 성능을 실제 주행 성능이나 최종 모델 성능으로 해석하지 마세요. `--random-init`은 다운로드 없이 파이프라인을 확인하기 위한 선택입니다.

```bash
python integrations/bevformer/run.py \
  --upstream "$HOME/BEVFormer" \
  --data-root "$DATA_ROOT/nuscenes-mini" \
  --dataset-version v1.0-mini \
  --output-dir "$PROJECT_ROOT/runs/bevformer-mini" \
  --lr 0.000025 --epochs 1 --workers 2 \
  --random-init --prepare-only
```

이 명령은 `bevformer_tiny_local.py`와 `launch_plan.json`을 작성합니다. 경로의 공백·한글·따옴표는 Python 문자열로 안전하게 보존합니다. 원본 설정은 수정하지 않습니다. 실행 계획에는 데이터 경로, 소스 커밋, 초기화 방법, GPU 수, 학습률과 전체 명령이 기록됩니다. 생성 파일의 내용이 달라지는 경우 새 출력 폴더를 사용하거나 `--overwrite`로 명시적으로 교체합니다. 체크포인트는 이 옵션으로 삭제되지 않습니다.

실제 학습은 같은 명령의 `--prepare-only`를 `--execute`로 바꾸면 됩니다. 래퍼는 shell을 사용하지 않고 인자 목록을 전달하며, upstream 폴더에서 다음 형태로 실행합니다.

```text
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 
  <upstream>/tools/train.py <생성된 설정> --launcher pytorch --work-dir <출력>/work --seed 42
```

전역 배치는 1입니다. `--lr`는 반드시 직접 지정하도록 했습니다. 예시 `0.000025`는 탐색 출발점일 뿐, 다중 GPU 논문 결과를 재현하는 최적값이나 검증된 노트북 설정이 아닙니다. 자동 GPU 비례 학습률 조절은 사용하지 않습니다. 동일 epoch 수도 학습 배치와 optimizer step 수가 달라지므로 논문 설정과 동일한 실험이라고 볼 수 없습니다.

mini 연결을 확인했다면 실제 학습에는 미리 받은 신뢰 가능한 가중치를 지정하세요. 자동으로 모델 가중치를 다운로드하지 않습니다.

| 초기화 옵션 | 의미 |
|---|---|
| `--backbone-weights /path/resnet50.pth` | torchvision ResNet50과 호환되는 로컬 backbone 가중치로 초기화 |
| `--load-from /path/bevformer_tiny.pth` | 로컬 BEVFormer 전체 모델을 로드하고 optimizer를 새로 시작 |
| `--resume-from /path/latest.pth` | 중단한 학습의 optimizer·epoch 등 상태를 이어서 실행 |
| `--random-init` | 사전학습 가중치 없이 시작; 연결 시험용 권장 |

네 옵션 중 하나만 선택합니다. 전체 모델 체크포인트는 모델 variant와 일치해야 합니다. 재개 시에는 데이터 분할·모델·정밀도를 유지하고 `--epochs`에는 추가 epoch 수가 아닌 **최종 총 epoch 수**를 넣습니다. PKL과 PyTorch 체크포인트는 로드 과정에서 코드를 실행할 수 있으므로 직접 생성했거나 신뢰하는 출처의 파일만 사용하세요.

예를 들어 full 학습은 이전 명령에서 `--dataset-version v1.0-trainval`, full data root, 별도 output dir, `--epochs 24`, `--backbone-weights ...`를 사용합니다. 작은 학습률 후보를 비교하고 검증 NDS/mAP를 확인한 다음 장시간 실행하세요.

## 정밀도·로그·한계

- 기본 `--precision fp32`는 공식 standard tiny 설정입니다. 옵션 `--precision fp16`은 **공식 FP16 설정과 `tools/fp16/train.py`를 함께 선택**합니다. 단순히 standard 코드에 FP16 설정만 추가하지 않습니다. 기본 temporal queue 3과 800×450급 영상 설정은 upstream에서 상속합니다.
- `--workers 2`가 기본이며 RAM/로더 문제가 있으면 0으로 시험합니다. `--gpu 0`은 사용할 GPU 하나를 선택합니다.
- epoch마다 검증·체크포인트 저장이 기본이고 최근 체크포인트 3개를 보관합니다. `--eval-interval`, `--checkpoint-interval`, `--log-interval`로 변경할 수 있습니다. 로그와 체크포인트는 `<output-dir>/work/`에 저장됩니다. 디스크 여유와 온도·전력 제한을 관찰하세요.
- 래퍼는 입력 파일·카메라 디렉터리의 존재를 확인하지만 모든 이미지의 무결성, PKL 내용, train/val 장면 중복을 검사하지는 않습니다. 공식 scene 분할을 유지하세요. 같은 주행 장면으로 학습하고 검증하지 마세요.
- BEVFormer 코드의 Apache-2.0과 nuScenes 데이터 이용 조건은 별개입니다. 데이터·가중치의 사용·재배포 조건을 각각 확인하세요. 공식 코드·데이터·가중치는 이 프로젝트에 동봉하지 않았습니다.
- **이 환경에서는 학습이나 nuScenes 전처리를 실행하지 않았습니다.** 아래 테스트는 설정 생성·경로·명령·실패 처리만 검증합니다. CUDA 확장 빌드, 실제 GPU 학습, 성능·소요시간 검증은 별도입니다.

```bash
python -m unittest discover -s integrations/bevformer -p "test_*.py" -v
```

참고: [공식 Tiny 설정](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/projects/configs/bevformer/bevformer_tiny.py), [공식 FP16 실행 안내](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/docs/getting_started.md), [코드 라이선스](https://github.com/fundamentalvision/BEVFormer/blob/66b65f3a1f58caf0507cb2a971b9c0e7f842376c/LICENSE).
