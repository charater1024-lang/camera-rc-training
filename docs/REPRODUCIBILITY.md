# 재현 절차와 실행 범위

모든 명령은 저장소 루트에서 실행합니다. 로컬 자료·실험 결과·가중치는 공개 저장소에 포함되지 않습니다. 아래 절차는 코드와 계약을 재현하는 방법이며, 공개 파일만으로 과거 비공개 스냅샷의 수치를 정확히 재생할 수 있다는 뜻은 아닙니다. 자신의 적법한 자료로 새 실험을 만들면 다른 해시와 결과를 기록해야 합니다.

## 1. CPU 개발·테스트 설치

기록된 주 환경은 Windows / Python 3.13 / PyTorch 2.9.1+cu126 / TorchVision 0.24.1+cu126 / Ultralytics 8.4.182입니다. `requirements-tested.txt`는 그 환경의 관측 목록이고 모든 OS에서 검증한 lockfile이 아닙니다. 패키지 메타데이터의 최소 Python 지원과 아래 고정 테스트 프로필은 구분합니다.

Linux shell 예 (macOS용 PyTorch wheel 조합은 이 프로필에서 검증하지 않았습니다):

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]' -c constraints/cpu.txt
python -m pip check
python -m camera_rc --help
python -c "from pathlib import Path; Path('runs').mkdir(exist_ok=True)"
python -m pytest tests integrations/bevformer/test_run.py -q -p no:cacheprovider --basetemp=runs/cpu-check-001
```

Windows PowerShell 예:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[test]' -c constraints/cpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m camera_rc --help
$env:CUDA_VISIBLE_DEVICES = ''
.\.venv\Scripts\python.exe -c "from pathlib import Path; Path('runs').mkdir(exist_ok=True)"
.\.venv\Scripts\python.exe -m pytest tests integrations/bevformer/test_run.py -q -p no:cacheprovider --basetemp=runs/cpu-check-001
```

재실행할 때 `--basetemp`는 새 이름을 사용합니다. pytest는 지정한 임시 디렉터리를 관리하므로 기존 실험 폴더를 넣지 않습니다. CPU 테스트는 작은 합성 이미지·마스크와 mock을 사용하며 일부 작은 모델의 CPU 학습도 검사합니다. 데이터셋/가중치 다운로드, 실제 CUDA 학습, 차량 연결, 실제 속도 벤치마크는 실행하지 않습니다. 의존성 설치 자체는 패키지 서버에 접근합니다.

`.github/workflows/cpu-tests.yml`은 Ubuntu/Python 3.13의 CPU 전용 프로필입니다. GPU wheel이나 데이터/weights를 설치하지 않고 `.[test]`만 설치합니다. optional semantic 통합을 설치하지 않았을 때 관련 테스트는 해당 테스트의 skip 조건을 따릅니다. CI 파일이 있다는 사실을 원격 CI 통과로 표현하지 않습니다. 실제 통과 여부는 해당 commit의 Actions 결과로 확인합니다.

`test` extra는 수집에 필요한 pytest, requests, PyYAML, psutil, OpenCV, pyserial을 포함합니다. 다운로드 도구만 필요하면 `.[download]`, serial/runtime은 `.[runtime]`, 그림 출력은 `.[preview]`를 선택합니다. Torch는 기본 의존성입니다. OpenCV의 GUI/headless 패키지를 동시에 설치하지 않습니다.

## 2. 선택적 CUDA semantic 환경

이는 학습을 명시적으로 진행할 때만 필요한 별도 환경입니다. CPU 설치에 추가하지 않고 새 venv를 권장합니다.

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[semantic,runtime]' -c constraints/semantic-cu126.txt
python -m pip check
python -m camera_rc doctor
python -c "import ultralytics; from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer; print(ultralytics.__version__)"
```

`ultralytics==8.4.182`를 고정합니다. 이 프로젝트는 native semantic API를 사용하므로 이름이 YOLO라는 이유만으로 임의의 detection/instance-segmentation 버전으로 교체할 수 없습니다. CUDA 실패 시 본 semantic 학습기는 CPU로 조용히 전환하지 않습니다. 시스템 GPU 드라이버 설치나 다른 프로세스 중지는 이 절차에 포함되지 않습니다.

`setup.ps1 -CpuOnly`와 `setup_ubuntu.sh`는 편의 도구이며, 정확한 프로필이 필요하면 위의 wheel index와 constraints를 명시한 명령을 사용합니다. BEVFormer 연동은 레거시 MMDetection 계열의 **별도 환경**이며 이 venv에 섞지 않습니다. [BEVFormer 연구 문서](BEVFORMER.md)를 참고하십시오.

## 3. 가중치 취득과 검증

현재 비교 모델의 학습된 checkpoint는 배포하지 않습니다. 공식 초기 가중치와 AMP 검사용 보조 detector도 직접 취득해야 합니다. 후자는 정확도 비교 모델이 아니라 Ultralytics의 AMP 점검을 위한 입력입니다.

| 파일 | 공식 출처 | 기대 SHA256 |
|---|---|---|
| `yolo26x-sem.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) | `eae8ff0d95bf96863c59c40bade61037870c7a39f2c53a8c49fd1b5022ed331d` |
| `yolo26n.pt` | [Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/tag/v8.4.0) | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |

공식 release에서 취득한 파일을 `models/pretrained/`에 둡니다. 보조 파일은 정확히 `models/pretrained/yolo26n.pt`여야 합니다. 해시가 다르면 무시하거나 상수를 바꾸지 말고 출처/버전을 먼저 조사합니다.

```powershell
Get-FileHash -LiteralPath 'models/pretrained/yolo26x-sem.pt' -Algorithm SHA256
Get-FileHash -LiteralPath 'models/pretrained/yolo26n.pt' -Algorithm SHA256
```

```bash
sha256sum models/pretrained/yolo26x-sem.pt models/pretrained/yolo26n.pt
```

공식 초기 모델 대신 자신이 완료한 2클래스 모델을 이어 사용할 때는 `--weights`와 **그 모델 자신의** `--init-training-report`를 함께 사용합니다. 보고서·모델·부모 계보·데이터 출처를 검증하고 새 optimizer로 시작합니다. 이것은 epoch 중단 재개가 아닙니다. 다른 모델의 보고서나 과거 실행 파일을 고쳐 경로만 맞추지 않습니다. 신뢰하지 않는 `.pt`는 로드하지 않습니다.

## 4. 데이터 취득·고정·전처리

1. [DATASETS](DATASETS.md)의 원 제공자 조건을 확인하고 자료를 취득합니다. ZIP 선택 목록 또는 다운로드 완료 증거를 기록합니다. 파일 크기가 잠시 같다는 이유만으로 전체 완료를 선언하지 않습니다.
2. `verify_aihub_archives.py --root <download-root> --output <new-intake-dir>`로 SHA256·CRC를 검사합니다. 출력은 새 디렉터리여야 합니다.
3. `extract_aihub_verified.py --help`에 따라 검증 보고서와 명시적 선택/완료 증거를 연결합니다. 원본과 실패 폴더를 보존하고 별도 폴더에 해제합니다. 선택 스냅샷과 전체 다운로드 완료는 별개의 상태입니다.
4. `audit_aihub_annotations.py --source-spec <local-spec.json> --output <new-audit-dir>`로 원본 형식과 클래스/오류를 확인합니다. 12번 QA 자료를 polygon에 섞지 않습니다.
5. `prepare_aihub_road.py --source-spec <local-spec.json> --output <new-road-dir> --quarantine-ledger <ledger.json>`로 명시적 매핑을 적용합니다. 격리는 annotation SHA와 사유를 보존하며 조용히 누락시키지 않습니다.
6. `build_aihub_cohort.py`로 기존 학습 출처의 replay 표본을 연결하고, `prepare_yolo_semantic.py --config <local-config.json> --output <new-export-dir> --workers 4`로 byte/RGB identity와 마스크 통계를 고정합니다. diagnostic 제한 옵션으로 만든 자료를 전체 준비 완료로 표시하지 않습니다.
7. `prepare_evaluation_contract.py --spec <local-evaluation-spec.json> --output <new-contract-dir>`로 보고용 영상/라벨/threshold/분모를 고정합니다. 모델의 학습 출처와 평가 대상을 분리하되 겹침 검사를 유지합니다.

`<...>`는 자신의 실제 로컬 입력으로 치환하는 자리표시자이며 그대로 실행할 명령은 아닙니다. 각 script의 `--help`와 공개 테스트의 작은 fixture가 입력 schema 예제입니다. 이 저장소는 타인의 원본 데이터 경로를 포함한 local spec을 배포하지 않습니다. 실행 전에 manifest의 task/class/shape와 출처 조건을 검토해야 합니다.

`run_aihub_cpu_preparation.py`는 이름과 달리 CPU 준비 뒤 **GPU runner까지 연결하는 orchestration 도구**입니다. CPU 전처리만 할 때 이 스크립트를 실행하지 않습니다. 전처리 전용 명령은 다음 절의 `preprocess_aihub_revision.py`입니다.

## 5. 보정 전처리만 실행

이미 고정된 training export에 적용합니다. `configs/local/preprocess.json`에 다음 필수 값을 자신의 파일로 작성합니다. 모든 `inputs.*.path`에는 `Path(...).resolve()`로 얻은 절대경로를 넣습니다. 독립 검증기는 정책에 저장한 경로와 보고서의 정규화된 절대경로도 비교하므로 상대경로를 사용하지 않습니다. JSON의 해시는 실행 직전에 실제 파일의 SHA256으로 채워야 하며, 과거 실험 해시를 다른 파일에 재사용하지 않습니다.

```json
{
  "inputs": {
    "export_report": {"path": "/absolute/path/to/project/data/my_export/report.json", "sha256": "ACTUAL_SHA256"},
    "export_marker": {"path": "/absolute/path/to/project/data/my_export/.complete", "sha256": "ACTUAL_SHA256"},
    "train_manifest": {"path": "/absolute/path/to/project/data/my_export/train.jsonl", "sha256": "ACTUAL_SHA256"},
    "reporting_manifest": {"path": "/absolute/path/to/project/data/my_export/val.jsonl", "sha256": "ACTUAL_SHA256"},
    "old_adapter": {"path": "/absolute/path/to/project/scripts/prepare_aihub_road.py", "sha256": "ACTUAL_SHA256"},
    "quarantine": {"path": "/absolute/path/to/project/runs/my_intake/quarantine.json", "sha256": "ACTUAL_SHA256"},
    "official_contract": {"path": "/absolute/path/to/project/runs/my_contract/contract.json", "sha256": "ACTUAL_SHA256"}
  },
  "expected_training_pool_count": 37864,
  "seed": 42,
  "development_fraction_target": 0.1,
  "limitations": ["Training-derived development data may have been seen by parent models."]
}
```

`/absolute/path/to/project/`는 자신의 실제 프로젝트 절대경로로 바꿉니다. Windows에서는 `str(Path(path).resolve())`의 결과를 `json.dump`로 저장해 역슬래시가 자동으로 이스케이프되게 합니다. 정규화한 경로의 슬래시를 수동으로 바꾸면 정책/보고서 문자열 검사가 실패할 수 있으므로 그대로 저장합니다. `expected_training_pool_count`도 자신의 검증된 풀 수로 바꿉니다. 정책에는 추가로 허가된 범위, mapping 변경 사유, 원본 표본 검토, 중단 조건과 한계를 기록하는 것을 권장합니다. 과거 export를 생성한 rasterizer와 현재 `old_adapter`가 일치해야 원래 마스크가 재현됩니다.

```bash
python scripts/preprocess_aihub_revision.py --policy configs/local/preprocess.json --output data/my_preprocessed_v1 --workers 4
python scripts/verify_aihub_preprocessing.py --revision data/my_preprocessed_v1 --expected-contract-sha256 ACTUAL_FROZEN_CONTRACT_SHA256
```

두 번째 명령은 stdout JSON만 출력하는 독립 검증입니다. 기본 계약 해시는 기록된 과거 스냅샷에 고정되어 있으므로 자신의 새 계약은 `--expected-contract-sha256`로 명시합니다. 검사 실패를 통과시키기 위해 그때그때 바꾸는 값이 아니라, 실행 전에 따로 보존한 기대 해시여야 합니다.

전처리는 원본·기존 mask를 덮어쓰지 않고 수정 mask만 새로 저장합니다. `.complete`, `complete=true`, 보고서·manifest·mapping·changes·split_report 해시와 실제 mask diff를 함께 확인합니다. `.incomplete`는 이 스크립트의 시작 저널로 남을 수 있으며 성공 판정의 유일 기준이 아닙니다. 타 어댑터의 완료 표시 규칙과 혼동하지 않습니다.

## 6. 명시적 학습과 평가

아래는 API 사용 예이며 현재 작업을 자동 재시작하라는 지시가 아닙니다. 데이터와 계보가 실제 존재해야 합니다. 새 출력 폴더를 사용하고, 이미 완료된 학습은 반복하지 않습니다.

```bash
python scripts/train_yolo_semantic.py --dataset data/my_export --config configs/local/my_training.json --weights models/pretrained/yolo26x-sem.pt --output runs/my_training_v1 --epochs 3 --image-size 384 --batch 8 --nominal-batch 32 --workers 2 --lr 0.0001 --gpu-memory-gib 5 --device 0
```

후보 선택용 `val`에는 사전에 고정한 개발셋을 연결합니다. 공식 보고셋과 public 40장을 tuning/사례 선택에 쓰지 않습니다. 부모가 개발셋을 이미 학습했다면 해당 노출을 표시하는 별도 개발 평가 계약을 사용하며, 공식 평가기의 조상 데이터 누수 검사를 완화하지 않습니다.

```bash
python scripts/evaluate_frozen_road.py --evaluation-contract runs/my_contract/contract.json --model-family yolo26x-sem --config configs/local/my_training.json --checkpoint runs/my_training_v1/weights/best.pt --training-report runs/my_training_v1/training_report.json --input-size 384 640 --device cuda --batch-size 4 --output runs/my_evaluation_v1.json
```

checkpoint 실제 위치는 완료한 `training_report.json`에서 확인합니다. 위 `weights/best.pt`는 예시 경로입니다. TinyUNet은 자신의 config/checkpoint/training-report를 사용하고 `--model-family tinyunet`으로 평가합니다. 필요할 때 `--training-content-export`로 그 모델의 기존 출처를 보강하되 다른 모델의 보고서를 수정하지 않습니다.

각 모델에서 계약 SHA, 입력 크기, precision, threshold, 분모, source 이름과 count, checkpoint SHA, 완료 여부가 같거나 의도대로 대응하는지 검증한 뒤 비교합니다. `evaluation_complete=true`와 데이터 provenance 확인 전의 중간 로그를 최종 수치로 쓰지 않습니다. 학습 중 OOM/AMP skip/중단은 보존하며 자동 epoch 반복으로 실패를 숨기지 않습니다. 큐를 잠깐 예약했다면 자식 종료 후 원복해야 합니다.

## 7. 완료한 의미 분할 모델로 로컬 이미지 한 장 읽기

`VerifiedYoloSemantic`은 완료한 2클래스 YOLO26x-sem 보고서와 그 보고서가 지정한 가중치만 로드합니다. `.[semantic]` 설치와 직접 취득·학습한 로컬 모델이 필요합니다. 아래는 CPU FP32 추론 예이며 카메라, serial, 계획기, 제어기, 다운로드를 시작하지 않습니다. 이 예제는 도로 확률 배열을 반환하며 장애물 검출기나 거리 추정기를 자동 추가하지 않습니다.

```python
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"

import numpy as np
import torch
from PIL import Image
from camera_rc.driving.yolo_semantic import VerifiedYoloSemantic

torch.set_num_threads(2)
config = {
    "segmentation_input_size": [384, 640],  # height, width
    "drivable_class_ids": [0],
    "segmentation_training_report": str(Path("runs/my_training_v1/training_report.json").resolve()),
    "segmentation_training_report_sha256": "REPLACE_WITH_PREVIOUSLY_VERIFIED_REPORT_SHA256",
    "segmentation_checkpoint_role": "best",
    "allow_diagnostic": False,
    "yolo_settings_dir": str(Path("runs/offline_image_check/.ultralytics").resolve()),
}
checkpoint = Path("runs/my_training_v1/fit/weights/best.pt").resolve()
predictor = VerifiedYoloSemantic(config, checkpoint, device="cpu", half=False)
with Image.open("my_local_image.png") as image:
    rgb = np.array(image.convert("RGB"))
road_probability = predictor(rgb)  # original H x W, floating probabilities
road_mask = road_probability >= 0.65
print({"shape": list(road_mask.shape), "predicted_road_fraction": float(road_mask.mean())})
```

경로와 보고서 SHA는 실행 전에 독립적으로 확인한 자신의 값으로 바꿉니다. 출력의 도로 비율은 정답과 비교한 정확도가 아닙니다. 이 어댑터의 검증 범위는 보고서 완료 증거·클래스/아키텍처·선택한 체크포인트 바이트이며, 전체 학습 데이터와 조상 계보 재감사는 별도 평가 도구가 담당합니다. 원본 해상도로 확률을 복원하는 이 예제와 공식 일반 검증의 192×320 GT 격자 지표는 구분합니다.

기존 `CameraPerception`에 연결할 때는 `segmentation_backend="yolo26x_semantic"`를 명시하고 위 설정 및 `segmentation_checkpoint`를 `perception` 설정에 넣습니다. 기본 native/TinyUNet 경로는 바뀌지 않습니다. YOLO semantic 백엔드의 실제 차량 제어는 코드에서 차단되며, 이 예제로 제어 승인 상태를 바꾸지 않습니다.

## 8. 공개 증빙과 검증 상태

[RESULTS](RESULTS.md)는 완료된 기존 비교의 정확도 수치와 원본 해시를 공개합니다. [DATASETS](DATASETS.md)는 새 전처리 결과를 구분합니다. 보정 후보는 완료된 보고서를 검증하기 전까지 pending입니다. 이번 속도 비교는 취소되었으므로 다른 실행 시간을 끼워 넣지 않습니다.

공개 소프트웨어 테스트는 데이터 파이프라인과 실패 경로의 작동 검사입니다. 기존 218개 준비 검사, 107개 오프라인 fixture, 135개 작은 batch 통합, 20개 초기 검토기, 43개 전처리 검사, 14개 독립 검증기 검사는 서로 다른 시점/범위의 기록이며 단순 합산한 총 테스트 수나 현재 릴리스의 결과가 아닙니다. 현재 릴리스의 정확한 pass/skip 수는 실행 로그와 해당 commit CI에서 확인합니다. 어떤 pass 수도 실제 인식 정확도나 무사고를 의미하지 않습니다.
