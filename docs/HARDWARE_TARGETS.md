# Jetson Orin Nano / MSI Prestige 15 A10SC 운용

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

현재 선택한 기본 대상은 **이미 Ubuntu 24.04 LTS가 설치된 MSI, 모니터 없음**입니다. 아래 Jetson 비교는 이전 검토 자료로 유지합니다. 지금은 현재 Windows PC에서 학습·개발하고 MSI에는 나중에 배포합니다. 우선 [개발 흐름](DEVELOPMENT_WORKFLOW.md)과 [Ubuntu 준비 안내](HEADLESS_UBUNTU.md)를 사용하세요.

2026-10-19 기준 설계와 공식 자료 확인입니다. 아래 설정은 **초기 측정용 후보**이며, 실제 두 장치의 FPS·발열·카메라·시리얼 구동을 검증한 결과가 아닙니다. Windows RTX 3080에서 확인한 소프트웨어 검증 범위는 [현재 검증 범위](REPRODUCIBILITY.md)에 있습니다.

차량에는 **MSI 노트북 또는 Jetson 중 한 대만** 탑재합니다. 선택한 한 대가 인식·경로 계획·제어 명령 계산을 모두 수행합니다. 두 설정은 대체 가능한 선택지이며 동시 사용하거나 서로 연산을 나누는 구성이 아닙니다. 대규모 학습은 차량 밖의 기존 RTX 3080 Laptop 16GB / RAM 32GB에서 계속하고, 선택한 운용 장치에는 내보낸 모델과 보정 파일만 배치합니다. 운용 중 학습 PC에 접속할 필요는 없습니다.

```text
USB 카메라 → [MSI Ubuntu 또는 Orin Nano 8GB 중 한 대]
          → USB 시리얼 MCU → 조향 장치·모터
                       ← 측정 속도·비상 정지·arm 상태
```

이미 가진 장비로 시작한다면 MSI가 추가 보드 구매 없이 연결·디버깅하기 편합니다. 차량 탑재 크기·전력·무게를 우선하면 Jetson Orin Nano **8GB / Super 개발자 키트**를 후보로 권합니다. 두 장치를 모두 구매하거나 순차적으로 사용해야 한다는 뜻은 아닙니다. MCU는 AI 연산용 컴퓨터의 대안이 아니라 구동기 제어와 watchdog을 맡는 별도 제어기입니다.

## 하드웨어 차이

| 항목 | MSI Prestige 15 A10SC | Jetson Orin Nano Super 개발자 키트 |
|---|---|---|
| CPU / 아키텍처 | 10세대 Intel Core U 계열 / x86_64, 세부 SKU 확인 | 6코어 Arm Cortex-A78AE / aarch64 |
| GPU | GTX 1650 Max-Q, 전용 GDDR5 4GB | Ampere GPU, 1024 CUDA / 32 Tensor cores |
| 메모리 | 시스템 RAM 16GB 가정, VRAM과 별개 | CPU·GPU·OS가 **총 8GB LPDDR5 공유** |
| RAM 확장 | MSI 공통 사양은 2슬롯·최대 64GB, 현재 장착량 확인 | 노트북 RAM처럼 추가 장착하는 구성 아님 |
| 초기 용도 | 연결·디버깅·녹화·경량 추론 | 차체 탑재와 경량 추론 |

하드웨어 수치는 [MSI 공식 사양](https://www.msi.com/Business-Productivity/Prestige-15-A10X-r/Specification), [NVIDIA 공식 사양](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)을 확인했습니다. Jetson의 CPU와 GPU 메모리가 같은 DRAM을 사용한다는 설명은 [CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/)에 있습니다. A10SC의 RAM·SSD 용량은 SKU마다 다릅니다. MSI 공식 표는 Windows 제품 사양이며 이 노트북의 Ubuntu 인증이나 주변기기 호환성을 보장하는 자료는 아닙니다.

Orin Nano 4GB 모델은 이번 기본 대상으로 잡지 않습니다. 공유 4GB는 앞서 가정한 전용 VRAM 4GB + RAM 16GB보다 전체 메모리 여유가 훨씬 작습니다. Jetson의 최대 INT8 TOPS와 노트북 GPU의 FP32 성능 수치로 이 파이프라인의 FPS 순위를 계산할 수 없습니다.

## 추가한 설정

| 설정 | MSI Ubuntu | Orin Nano 8GB |
|---|---|---|
| 파일 | `configs/runtime_msi_prestige15_ubuntu.json` | `configs/runtime_jetson_orin_nano_8gb.json` |
| CUDA | 필수, CPU로 자동 대체하지 않음 | 필수, CPU로 자동 대체하지 않음 |
| 입력 | USB UVC 카메라 1대, 640×360 | 동일 |
| 모델 후보 | width16 영역 분할, 입력 192×320, batch 1 | 동일 |
| 초기 정밀도 | FP32 (`half=false`) | FP16 autocast (`half=true`) |
| 인식 / 제어 목표 | 10Hz / 50Hz | 10Hz / 50Hz |
| PyTorch allocator 상한 | 3GiB | 2.5GiB |
| 프로세스 RSS 상한 | 12GiB | 4GiB |
| 시스템 가용 RAM 하한 | 2GiB | 1.5GiB |
| 물리 출력 | dry-run | dry-run |

MSI의 FP32는 비교 측정을 위한 출발값입니다. 같은 영상·모델로 `half=true`와 처리 시간 및 예측값을 비교해 결정하세요. 정밀도를 낮춘다고 모든 GPU에서 빨라지는 것으로 가정하지 않습니다. 모델 입력 크기는 런타임 JSON이 아니라 **체크포인트의 학습 설정**에서 읽습니다. 위 모델 후보로 학습·내보낸 가중치를 사용해야 합니다.

Jetson 설정의 `ram_budget_gib=8`은 공유 메모리 목표입니다. `gpu_budget_gib=4`는 CUDA에 배정하려는 작업 예산일 뿐, 별도 4GB VRAM이 있다는 뜻이 아닙니다. RSS와 CUDA 메모리를 더해 총 사용량으로 해석하지 않습니다. 시스템 가용 RAM과 CUDA/PyTorch 지표를 함께 기록합니다. allocator 상한은 CUDA context나 다른 라이브러리의 모든 할당을 제한하지 않습니다.

두 설정 모두 처리 기한 150ms, 경로 만료 200ms를 유지합니다. 낮은 인식 주기는 계산 부하를 줄이기 위한 초기 목표이며 주행 성능 보장은 아닙니다. 정지 오류를 없애려고 기한만 늘리지 말고, 모델·입력·속도·관측 거리와 제동 조건을 함께 검토하세요. 기본 차체 치수와 최대 속도 0.4m/s도 실제 차체 측정값으로 교체해야 합니다.

객체 탐지는 초기 설정에서 꺼져 있습니다. 분할 실행을 확인한 뒤 로컬 `perception.detection_checkpoint`를 지정하여 320 입력의 경량 탐지를 추가하고 다시 측정하세요. 분할만으로 주행 경로에 장애물이 없다고 보장하지 않습니다. 단안 깊이·BEVFormer·다중 카메라 모델을 동시에 구동하는 구성은 아닙니다. 현재 런타임은 PyTorch이며 TensorRT 엔진 추론은 구현하지 않았습니다.

## MSI의 Ubuntu 설치 후 준비

새 설치 기준 후보는 **Ubuntu 24.04 LTS 64비트 / Python 3.12**입니다. 이 코드의 Linux 설치 절차는 대상 노트북에서 아직 실행하지 않았습니다. 프로젝트 폴더를 Ubuntu 사용자 홈의 작업 폴더로 복사하고, Windows `.venv`는 사용하지 않습니다. 아래 명령은 Ubuntu의 프로젝트 루트에서 실행합니다.

Ubuntu의 추가 드라이버 도구 또는 [Canonical 공식 안내](https://ubuntu.com/desktop/docs/en/latest/how-to/graphics/install-nvidia-drivers/)에 따라 해당 GPU용 NVIDIA 드라이버를 설치하고 재부팅합니다. `nvidia-smi`에서 GTX 1650과 드라이버를 확인합니다. Secure Boot를 쓰면 설치 안내에 따른 모듈 서명 키 등록이 필요할 수 있습니다.

```bash
sudo apt update
sudo apt install python3-venv libgl1 libglib2.0-0
python3 -m venv .venv-linux
source .venv-linux/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[runtime]'
python -m pip check
mkdir -p runs
python scripts/check_runtime_environment.py --output runs/environment_msi.json
```

Torch/torchvision 조합은 기존 Windows 테스트와 버전을 맞추기 위한 후보이며 최신 버전을 뜻하지 않습니다. 공식 [PyTorch 이전 버전 설치표](https://pytorch.org/get-started/previous-versions/)에서 Linux용 CUDA 12.6 조합을 확인했습니다. 드라이버 호환 여부와 실제 CUDA 실행은 점검 스크립트로 확인합니다. 별도 CUDA Toolkit 전체 설치부터 시작할 필요는 없습니다. 객체 탐지 사용 시:

```bash
python -m pip install -e '.[detection]'
python scripts/check_runtime_environment.py --with-detection --output runs/environment_msi_detection.json
```

초기 테스트는 전원 어댑터를 연결하고 절전·자동 절전 진입을 확인한 상태에서 진행합니다. 이후 실제 배터리 운용 상태에서도 같은 연속 실행을 측정해야 합니다.

## Jetson의 소프트웨어 준비

Jetson은 NVIDIA **JetPack / Jetson Linux** 설치 경로를 사용합니다. 일반 PC용 Ubuntu 이미지나 위 x86 CUDA wheel 설치 명령을 그대로 적용하지 않습니다. 조사 시점의 [Orin Nano 공식 빠른 시작](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html)은 JetPack **7.2.1 / L4T r39.2.1**을 안내합니다. [JetPack 제품 페이지](https://developer.nvidia.com/embedded/jetpack)의 7 계열 기반은 Ubuntu 24.04입니다. 이전 6.2.x 경로는 Ubuntu 22.04 기반이며, 두 계열의 설치법·CUDA·Python 조합을 섞지 않습니다.

JetPack 7.2 이상은 USB에 Jetson ISO를 기록한 뒤 보드의 저장장치에 설치하는 경로입니다. 오래된 QSPI 펌웨어는 먼저 6.x 업데이트 경로가 필요할 수 있으므로 공식 안내를 따라야 합니다. NVMe SSD를 사용하면 모델·로그·컨테이너 저장 공간을 확보하기 편합니다. 이 작업에서는 OS나 펌웨어를 설치하거나 변경하지 않았습니다.

PyTorch는 **해당 JetPack/L4T 및 Orin 지원이 명시된 NVIDIA aarch64 CUDA wheel 또는 컨테이너**를 선택합니다. [NVIDIA 호환성 표](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)를 확인하세요. 이 표의 6.2 행에는 25.06 / Torch 2.8.0a0 조합이 있지만, 조사 시 표에 나온 7.1 조합을 Orin의 7.2.1에서 검증된 것으로 추정하면 안 됩니다. 프로젝트에 JetPack 7.2.1의 특정 wheel/container 조합을 검증·고정한 상태는 아닙니다.

프로젝트 선언 범위는 Python≥3.10, Torch≥2.7,<3, torchvision≥0.22,<1, NumPy≥1.26,<3, Pillow≥10,<13입니다. 오래된 JetPack용 Torch 2.5 wheel을 가져오면 이 범위를 충족하지 않습니다. torchvision도 같은 Torch/CUDA ABI와 맞아야 합니다. 공급자가 제공한 조합에서 기본 모델 실행에 필요한 Pillow·NumPy·OpenCV·pyserial·psutil을 준비한 뒤, **이미 설치한 Jetson Torch를 pip가 교체하지 않도록** 프로젝트 자체만 설치합니다.

```bash
# NVIDIA가 제공하는 호환 Python 환경 안에서 실행
python -m pip install --no-deps -e .
python -m pip check
mkdir -p runs
python scripts/check_runtime_environment.py --output runs/environment_jetson.json
```

`--no-deps`는 누락 의존성을 해결하는 명령이 아닙니다. `pip check`와 점검 스크립트가 통과해야 다음 단계로 진행합니다. 스크립트의 버전 검사는 NVIDIA prerelease 표기를 release 숫자로 비교하므로, 통과하더라도 패키지 메타데이터와 JetPack 호환성 검토가 별도로 필요합니다. 컨테이너를 사용하면 GPU와 선택한 카메라·시리얼 장치를 컨테이너에 전달하는 설정도 필요합니다. 이 문서는 그러한 장치 전달을 검증한 컨테이너 실행 명령을 제공하는 단계는 아닙니다.

객체 탐지를 추가할 때도 Torch/vision을 유지하는 의존성 구성을 사용하고 `--with-detection` 검사를 실행합니다. NumPy→Torch→CUDA 연산과 CUDA NMS까지 확인합니다. Jetson에서 NVML 또는 프로세스 GPU 사용량이 제공되지 않으면 null로 남습니다. `tegrastats`를 함께 기록해 메모리·온도·클럭·전력 모드 변화를 관찰하세요. CSI 카메라의 Argus/GStreamer 통합은 현재 구현 범위 밖이며, 우선 USB UVC 카메라를 사용합니다.

## 모델 배치와 장치 측정

1. 학습 PC에서 [단계별 학습](CURRICULUM.md) 후 `export-runtime`으로 `runs/deployment/road_surface.pt`를 만듭니다. 배포에는 모든 원본 데이터셋이 필요하지 않습니다.
2. 고정 장착된 카메라·렌즈·차체를 측정하고 [보정 안내](RUNTIME.md)에 따라 `configs/calibration_measured.json`을 만듭니다. 예시 보정을 실차 값으로 사용하지 않습니다.
3. 운영체제에서 원하는 UVC 카메라의 인덱스를 확인해 `source.index`를 지정합니다. 요청한 640×360을 실제 지원하는지 확인합니다. 노트북 내장 카메라가 index 0일 수 있으므로 차량에 고정한 카메라를 선택합니다.
4. 장치별 설정으로 벤치마크와 카메라 dry-run을 실행합니다. 다음은 MSI 예시이며 Jetson은 config 파일명만 교체합니다.

```bash
python -m camera_rc drive-benchmark --config configs/runtime_msi_prestige15_ubuntu.json --frames 300 --warmup 20 --output runs/msi_benchmark.json
python -m camera_rc drive --config configs/runtime_msi_prestige15_ubuntu.json --duration-s 600 --output runs/msi_camera_dry
```

벤치마크는 한 프레임 반복 추론이며 모터를 연결하지 않습니다. 연속 dry-run에서 프레임 나이·평균/p95/p99/최대 지연·제어 주기·메모리 여유를 확인합니다. 10Hz 목표에는 평균 주기뿐 아니라 지연 꼬리와 기한 초과 여부가 중요합니다. 카메라 내부 지연은 호스트 수신 시각으로 측정할 수 없어 별도 검증이 필요합니다. 두 장치의 10Hz/50Hz 달성을 아직 보장하지 않습니다.

실제 MCU를 연결할 때 `ls -l /dev/serial/by-id/`로 고정된 장치 경로를 확인하여 `REPLACE_WITH_MCU`를 교체합니다. 계정에 `dialout` 권한이 없으면 다음 명령 후 로그아웃·재로그인합니다. 카메라 권한은 해당 장치 그룹을 확인해 필요한 경우 `video` 그룹도 부여합니다.

```bash
sudo usermod -aG dialout "$USER"
```

실구동 전 나머지 조건은 [시리얼 규약](SERIAL_PROTOCOL.md)과 [운용 안내](RUNTIME.md)를 따릅니다. CPU/Linux 제어 루프의 50Hz 설정은 실시간 보장이 아닙니다. 독립 MCU watchdog, 측정 속도 피드백, 물리 arm/비상 정지는 그대로 필요합니다.

## 증설 판단

MSI의 RAM 16GB는 현재 경량 추론의 시작 용량으로 유지합니다. 실제 가용 RAM이 부족하거나 녹화·개발 도구를 함께 실행해 압박이 생기면 32GB를 검토합니다. GPU 계산 지연이나 VRAM 부족은 시스템 RAM 증설만으로 해결되지 않습니다. Jetson 8GB에서 모델을 줄여도 공유 메모리·연산 한계를 넘으면 16GB 이상 상위 보드를 검토할 시점입니다. 우선 모델 하나와 카메라 하나로 측정하고, 필요한 탐지·추가 카메라를 한 단계씩 늘리는 편이 판단에 도움이 됩니다.
