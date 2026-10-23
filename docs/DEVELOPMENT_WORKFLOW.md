# 현재 PC에서 학습하고 MSI에는 추론 환경 배포

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

현재 사용할 개발 장비는 **Windows / RTX 3080 Laptop 16GB / RAM 32GB**입니다. 나중에 사용할 탑재 장비는 **Ubuntu 24.04 LTS / MSI Prestige 15 A10SC / GTX 1650 Max-Q 4GB / RAM 16GB 가정 / 모니터 없음**입니다. 차량에는 MSI 한 대만 탑재합니다. 모델 학습 장비와 상시 연결하지 않아도 동작하는 로컬 추론을 준비합니다.

Ubuntu 장비가 현재 수중에 없으므로 이 프로젝트에서는 파일·설치 스크립트·서비스 예시만 준비했습니다. MSI에 접속하거나 OS·드라이버·패키지·서비스를 변경하지 않았습니다. 현재 Windows 가상 환경을 그대로 사용합니다.

## 사용할 환경과 모델

| 작업 | 현재 선택 | 실행 위치 |
|---|---|---|
| 학습·평가 | Python, PyTorch 2.9.1 CUDA 12.6, torchvision 0.24.1 | 현재 Windows `.venv` |
| 주행 표면 분할 | TinyUNet width16, 입력 192×320, 2클래스, 하나의 가중치 공유 | Windows 학습 → MSI 추론 |
| 선택적 객체 탐지 | Ultralytics YOLO11n, 공개 사전학습 가중치 미세조정 | Windows 학습 → MSI 추론 |
| 경로 계획 | 보정된 지면 투영, 여러 방향의 관측 결합, 차체 이동 영역 검사 | MSI CPU |
| 조향·속도 명령 | Pure Pursuit + 측정 속도 PI, 독립 제어 루프 | MSI CPU → USB MCU |
| 배포 환경 후보 | Python 3.12 + 동일 Torch/vision 버전, batch1 추론 | MSI `.venv-linux` |

TinyUNet은 사전학습 기반이 아닌 작은 연구용 기준 모델입니다. 모델의 정확도는 실제 학습·독립 평가로 판단해야 합니다. 현재 선택은 처리 비용과 전체 흐름을 검증하기 위한 기준이며 다른 모델보다 정확도가 높다는 의미가 아닙니다. 정밀도는 MSI FP32를 비교 기준으로 두고 FP16의 실제 지연·예측 품질을 비교합니다. 현재 네 방향 진단은 별도 표기에 따라 FP16을 사용합니다.

BEVFormer 같은 큰 공동 다중 카메라 모델은 4GB 운용 기본 구성에 포함하지 않습니다. 네 방향 입력을 하나의 큰 배치로 묶거나 모델을 네 개 띄우지 않고, **모델 하나로 방향별 이미지를 순차 처리**합니다. 모델 가중치 메모리는 공유하지만 네 번의 추론 비용과 카메라·CPU·USB 비용은 여전히 발생합니다. 현재 내보내기는 PyTorch 체크포인트이며 TensorRT 엔진을 생성하지 않습니다.

## 지금 할 작업

프로젝트 위치: `.`.

```powershell
Set-Location -LiteralPath '.'
.\.venv\Scripts\python.exe scripts/development.py status --prepared-data data/public_v2
.\.venv\Scripts\python.exe scripts/development.py verify --device cuda --with-tests
```

`status`는 경로·디스크 여유와 준비 상태, 체크포인트의 존재를 보고합니다. 현재 BDD에는 `--prepared-data data/public_v2`를 명시합니다. 파일 존재만으로 준비 완료를 판단하지 않고 `data/public_v2/report.json`과 `.incomplete` 여부를 함께 확인합니다. `verify`는 환경, 선택한 회귀 테스트, 2단계 합성 학습, 네 방향 벤치마크, 50묶음의 무화면 재생을 차례로 실행합니다. 새 `runs/development/<실행 ID>/summary.json`과 단계별 로그에 결과가 남고, 실패한 단계 이후는 실행하지 않습니다. CUDA 실패 시 CPU로 자동 전환하지 않습니다. CPU로 점검하려면 `--device cpu`를 명시해야 하며 설정된 지연 목표 검사는 그대로 적용됩니다.

검증은 녹화 파일을 모사한 입력만 사용하므로 실제 카메라·시리얼을 열지 않습니다. 합성 학습의 가중치를 재생 모델로 사용하지 않으며, 별도의 일정 출력 진단 모델로 운용 연결을 검사합니다. 이 묶음에는 객체 탐지가 포함되지 않습니다. 선택적 탐지 의존성 검사는 `scripts/check_runtime_environment.py --with-detection`으로 따로 실행할 수 있습니다.

BDD100K 2018년 이미지·주행 가능 영역 라벨의 다운로드, ZIP 해시 검증, 선택 추출과 `data/public_v2`의 이진 마스크 변환·전체 이미지·분할 검사를 완료했습니다. `images/100k`의 train 70,000장 모두를 CUDA에서 1 epoch 학습하고 val 10,000장 전체를 검증했습니다. 320×192 입력의 2클래스 mIoU는 80.03%, 도로 IoU는 67.28%이며 공식 3클래스 점수와 다릅니다. 실제 충돌 회피나 MSI 운용 검증은 포함하지 않습니다. [BDD100K 기준 모델 학습](BDD100K_BASELINE.md)에 결과와 재개 절차가 있습니다.

이전 중단 실행의 `data/public_v1`은 미완료 상태로 보존하며 이번 학습 입력으로 사용하지 않았습니다. [출처 기록](BDD100K_SOURCES.md)에 무가입 미러, 원 이용 조건과 구형 ID 검증을 남겼습니다.

다른 자료를 추가할 때는 [공개 데이터 학습 준비](PUBLIC_TRAINING_SETUP.md)로 새 BDD 단독 또는 BDD+Cityscapes 설정을 생성합니다. 준비 도구는 원본 정답과 분할을 검사하고 별도 마스크를 만들며, 그 자체로 다운로드·학습을 시작하지 않습니다. 더 많은 소스와 자체 자료를 연결할 때는 [단계별 학습](CURRICULUM.md), [자체 네 방향 데이터](MULTIVIEW_TRAINING.md)를 따릅니다.

권장 학습 순서는 BDD 기반 표면 분할 → Cityscapes와 혼합 → 자체 네 방향 영상과 공개 데이터 일부 재학습입니다. 데이터셋 개수 자체를 목표로 삼지 않고, 표면 라벨 의미·촬영 환경·검증 성능에 맞는 출처를 추가합니다. 모든 데이터셋이 동일 정답을 제공하지 않으므로 객체 탐지·깊이·경로 정답을 분할 학습에 그대로 섞지 않습니다.

```powershell
# 완료된 데이터 준비 결과 확인
Get-Content -LiteralPath 'data/public_v2/report.json'
# 완료된 첫 1 epoch 학습 및 전체 검증 결과 확인
Get-Content -LiteralPath 'runs/bdd100k_2018_baseline_v1/summary.json'
```

`configs/curriculum_multiview.json`은 자체 네 방향 정답까지 갖춘 뒤 사용하는 확장 예시입니다. 공개 데이터만 확보한 첫 학습에 이 예시의 모든 소스를 요구하지 않습니다.

일반 공개 데이터 준비 도구의 기본 학습값은 batch 2, gradient accumulation 4, AMP, worker 0입니다. **이번 BDD 기준 모델의 별도 설정은 batch 16, accumulation 1, worker 2**이며, 입력 192×320과 모델 width 16은 유지합니다. 학습 배치를 변경해도 모델 자체가 커지지는 않습니다. 완료된 첫 1 epoch 설정은 `configs/local/bdd2018_baseline_v1.json`, 준비된 총 20 epoch 재개 설정은 `configs/local/bdd2018_continue20_v1.json`입니다. 20 epoch 재개는 아직 실행하지 않았습니다. 기존 결과 폴더에 최초 시작 명령을 다시 실행하지 않고 [재개 절차](BDD100K_BASELINE.md)를 사용합니다. 실제 결과는 summary.json (로컬 비공개 기록: `runs/bdd100k_2018_baseline_v1/summary.json`)에서 확인하며, 합성 GPU 실행 기록은 [현재 검증 범위](REPRODUCIBILITY.md)과 구분합니다.

데이터 없이 할 수 있는 학습·운용 검증은 명시적으로 진단용입니다. 출력은 매번 새 폴더를 사용합니다.

```powershell
.\.venv\Scripts\python.exe scripts/smoke_multiview_training.py --output runs/my_multiview_training_check --device cuda
.\.venv\Scripts\python.exe -m camera_rc drive-fixture --multiview --device cuda --output runs/my_multiview_fixture
.\.venv\Scripts\python.exe -m camera_rc drive-benchmark --config runs/my_multiview_fixture/runtime.json --output runs/my_multiview_benchmark.json --frames 100 --warmup 10
.\.venv\Scripts\python.exe scripts/run_headless.py --config runs/my_multiview_fixture/runtime.json --output-root runs/headless_checks --dry-run --max-frames 100
```

## 나중에 MSI를 확보하면

1. [Ubuntu 준비 안내](HEADLESS_UBUNTU.md)에 따라 소스와 운용 모델을 복사하고 `.venv-linux`를 새로 만듭니다. Windows `.venv`와 원본 학습 데이터는 배포하지 않습니다.
2. 카메라 한 대로 해상도·지연·CUDA·로그를 확인하고, [네 방향 카메라 안내](MULTICAMERA.md)에 따라 방향을 늘립니다.
3. 실제 카메라 보정, 각 방향 시각 차이, USB 대역폭, 발열, 메모리와 기한을 측정합니다. 현재 RTX 3080의 결과를 GTX 1650 성능으로 환산하지 않습니다.
4. 검증한 모델·보정·MCU가 준비되기 전까지 dry-run으로 유지합니다. systemd 예시도 dry-run만 제공하며 현재 설치·활성화하지 않았습니다.
