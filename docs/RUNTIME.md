# 전용 GPU 4GB / 시스템 RAM 16GB 운용

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

이 구성은 PC GPU와 USB 시리얼 MCU를 사용하는 전방 카메라 기반 저속 주행 연구용입니다. 실제 보드·카메라·차체가 아직 선정되지 않았으므로 실제 구동 및 제동 성능은 검증되지 않았습니다. 범용 도로 자율주행이나 실차 적용 검증을 의미하지 않습니다.

**Jetson Orin Nano 8GB / MSI Prestige 15 A10SC Ubuntu**의 별도 설정은 [장치별 안내](HARDWARE_TARGETS.md)를 사용하세요. 두 설정은 인식 목표 10Hz, 제어 목표 50Hz로 시작하며, 아래 범용 설정의 인식 목표 20Hz와 다릅니다. Jetson 설정에는 공유 메모리 감시가 포함됩니다.

현재 확정한 운용 대상은 **MSI Ubuntu 24.04, 모니터 없는 환경**입니다. 현재 Windows PC에서 학습·검증을 하고 MSI 환경은 나중에 적용할 준비만 합니다. [개발·배포 흐름](DEVELOPMENT_WORKFLOW.md)과 [네 방향 카메라 확장](MULTICAMERA.md)을 참고하세요. 아래는 기본 단일 전방 카메라 설명이며 다중 카메라 설정은 `runtime_msi_4cam_dryrun.json`입니다.

## 현재 연결한 흐름

```text
카메라 최신 프레임 → 렌즈 보정 → 경량 영역 분할 + 선택적 객체 검출
  → 보정된 지면 좌표 → 차량 전체 면적을 검사하는 곡률 후보 경로
  → Pure Pursuit 조향 + 측정 속도 PI 제어
  → USB 시리얼 명령 → 독립 watchdog을 가진 MCU → 차체별 구동기
```

영상 처리와 제어는 별도 루프입니다. 제어 기본 목표는 50Hz, 영상 처리 목표는 20Hz이며 달성 보장이 아닙니다. MCU는 측정 속도와 물리 arm/비상 정지 상태를 보내야 합니다. 속도 명령을 측정 속도로 되돌려 보내면 안 됩니다. 오래된 경로, 통신 손실, 처리 기한 초과는 정지 상태로 고정되며 원인을 해결한 뒤 프로그램을 재시작합니다. 장애물로 경로가 막히면 정지하고, 정상 관측으로 경로가 다시 열리면 재계획할 수 있습니다.

계획은 평평한 지면·고정 카메라를 가정합니다. 경사·턱·공중에 튀어나온 장애물·가려진 물체의 3D 충돌을 해결하지 않습니다. 전방 카메라가 회전 시 새롭게 차지할 측면 영역을 관찰하지 못하면 정지합니다. 미관측 영역을 도로로 간주하지 않습니다. 전역 지도·목적지 내비게이션·교차로 우선권 판단·이동 객체의 미래 궤적 예측은 현재 구현 범위 밖입니다. `route_hint`는 로컬 직진/좌/우 곡률 선호이며 목적지 탐색이 아닙니다.

## 4GB와 16GB가 의미하는 것

| 항목 | 기본 구성 |
|---|---|
| 학습 PC | 기존 RTX 3080 Laptop 16GB / RAM 32GB 사용 |
| 운용 목표 | 전용 VRAM 4GB + 별도 시스템 RAM 16GB |
| 모델 | width16, 192×320 영역 분할; 선택적 YOLO 320 입력 |
| 동시 실행 제외 | 단안 깊이·학습 경로·BEVFormer |
| 배치 | 한 프레임씩 추론 |
| 정밀도 | CUDA에서 FP16 autocast, CPU에서는 FP32 |
| GPU 여유 | PyTorch allocator 상한 3GiB, 전체 목표 4GiB |
| RAM 여유 | 프로세스 RSS 12GiB 상한, 전체 목표 16GiB |

학습 데이터를 늘리는 것과 운용 모델 메모리 크기는 별개입니다. 운용에는 모델 파일만 필요합니다. 여러 모델·카메라·시계열 입력을 추가할 때 메모리와 처리 지연을 다시 측정합니다. CUDA context, Ultralytics 등 다른 라이브러리의 메모리는 PyTorch 상한에 전부 포함되지 않습니다. 따라서 이 상한은 4GB 전체 사용량을 보장하지 않습니다. Jetson처럼 CPU/GPU가 메모리를 공유하는 보드의 4GB를 같은 사양으로 해석하면 안 됩니다.

우선 4GB/16GB 장치에서 측정합니다. 전체 GPU 메모리 초과면 모델·해상도를 줄이거나 6~8GB 이상을 검토하고, 메모리 여유가 있어도 지연을 넘으면 연산 성능이 더 높은 GPU가 필요합니다. RAM은 OS·카메라 버퍼·로그·전체 프로세스 사용량으로 판단합니다. 별도 대규모 학습을 운용 장치에서 동시에 수행하지 않습니다.

## 학습된 모델을 준비하기

`docs/CURRICULUM.md`의 road_surface_v1은 0=road, 1=non-road, 255=ignore입니다. 도로 분류 결과만으로 빈 공간이 보장되지는 않습니다. 차량 데이터에는 실제 장애물과 노면 경계를 포함해 평가해야 합니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc export-runtime --checkpoint '.\runs\curriculum_road_surface\stages\road_vehicle\best.pt' --output '.\runs\deployment\road_surface.pt'
```

기본 `configs/runtime_4gb.json`은 이 파일을 참조합니다. 모델의 전처리 해상도는 체크포인트에 저장된 값을 사용합니다. 진단용 모델은 실구동에서 거부합니다. 객체 검출도 사용하려면 `perception.detection_checkpoint`에 학습한 로컬 YOLO `.pt` 파일 경로를 설정하세요. 런타임은 모델을 자동 다운로드하지 않습니다. 두 모델의 라벨 의미와 `drivable_class_ids`, `obstacle_labels`를 일치시켜야 합니다.

## 카메라·지면·차체 보정

1. 실제 장착 높이·각도와 영상 해상도를 고정하고 렌즈 왜곡을 보정합니다.
2. 후륜 축 중심을 원점으로 전방 x, 좌측 y를 미터 단위로 측정합니다.
3. 보정된 영상의 픽셀과 지면 좌표를 대응시킵니다. 학습용 대응점 4개 이상과 별도 검증점 2개 이상이 필요합니다. 사용 범위 사각형이 측정점들의 볼록 껍질 안에 들어가야 합니다.
4. `configs/calibration_points.example.json`을 형식 참고용으로 사용합니다. 예시 숫자를 실제 카메라 값으로 사용하지 마세요. 실제 측정 입력은 `diagnostic_only=false`로 기록하고 그 출처를 보존합니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc calibrate-ground --points '.\data\my_camera_points.json' --output '.\configs\calibration_measured.json'
```

카메라가 이미 왜곡 보정된 영상을 보내면 `input_is_rectified=true`를 사용합니다. 원본 영상이면 camera_matrix와 distortion_coefficients를 제공하며, 대응점은 런타임과 동일한 보정 결과에서 측정해야 합니다. 보정 후 영상 바깥에서 생긴 픽셀은 미관측 영역으로 처리합니다. 카메라를 재장착하거나 해상도를 바꾸면 다시 보정합니다.

`planner`와 `controller`의 wheelbase_m, max_steering_rad, max_speed_mps는 같아야 합니다. 차량 너비·길이, 실제 조향 한계, 제동 감속도, 최대 스로틀을 측정해 설정합니다. 기본 최대 속도 0.4m/s는 예시 제한이며 안전을 보장하는 속도가 아닙니다. 제동 감속도는 실제 최악 조건 측정값보다 낙관적으로 설정하면 안 됩니다.

## 구동 전 출력 확인과 시리얼 연결

카메라 dry-run:

```powershell
.\.venv\Scripts\python.exe -m camera_rc drive --config '.\configs\runtime_4gb.json' --output '.\runs\camera_dry_01' --duration-s 30
```

카메라 dry-run에는 차량 측정 상태가 없으므로 조향·구동 명령은 STOP이며 인식/계획은 기록됩니다. 기록 영상을 사용하려면 source.kind=replay, source.path=JSONL을 설정합니다. 각 줄은 `{"image":"frame.png","speed_mps":0.2,"yaw_rate_rps":0.0}` 형식입니다. replay는 기록 상태를 사용해 제어 출력을 계산하지만 구동기에 전달할 수 없습니다.

실구동은 다음 조건을 갖춘 설정에서만 가능합니다.

- `transport.kind=serial`, 정확한 port, baudrate 230400.
- 실제 MCU에서 [규약](SERIAL_PROTOCOL.md)의 watchdog·물리 arm·비상 정지·측정 속도를 구현.
- 차체별 조향/모터/브레이크 매핑과 보정을 끝낸 뒤 calibration_confirmed=true.
- 실제 환경에서 모델과 경로 행동을 검증한 뒤 models_validated_for_control=true. 이 값은 인증이나 안전 보장이 아닌 운영자의 설정입니다.
- 진단 모델·진단 보정·미관측 영역을 임의로 비워두는 assumed_clearance 설정은 사용 불가.

```powershell
.\.venv\Scripts\python.exe -m camera_rc drive --config '.\configs\runtime_4gb_serial.json' --output '.\runs\vehicle_01' --actuate
```

`runtime_4gb_serial.json`은 보정 후 사용자가 기본 설정을 복사해 작성하는 파일이며 초기 프로젝트에는 생성하지 않습니다. 프로그램은 MCU를 자동 arm하지 않습니다. 종료 시 STOP을 보내며 통신 자체가 끊기면 MCU watchdog이 정지해야 합니다. 실제 카메라·USB 장치를 이번 작업에서 열거나 구동하지 않았습니다.

## 성능 측정과 로그

```powershell
.\.venv\Scripts\python.exe -m camera_rc drive-benchmark --config '.\configs\runtime_4gb.json' --frames 100 --output '.\runs\target_benchmark.json'
```

벤치마크에는 렌즈 보정, CPU/GPU 전송, 두 인식 모델(설정된 경우), 계획, 제어 계산이 포함됩니다. 같은 프레임 반복 실행이며 카메라 노출·USB 전달·서보 응답은 제외합니다. `drive`의 실제 실행 기록으로 프레임 나이와 연속 실행 시 지연을 추가 확인합니다. 카메라 타임스탬프는 호스트 수신 시각이므로 카메라 내부 노출/버퍼 지연은 별도 측정해야 합니다.

출력은 `perception.jsonl`, `control.jsonl`, `summary.json`, 실행 설정입니다. 지연·제어 주기의 평균·p95·p99·최대값, PyTorch peak allocated/reserved, 프로세스 RSS, 장치 전체 GPU 사용량, 지원되는 경우 프로세스 전체 GPU 메모리를 기록합니다. Windows에서 프로세스 GPU 메모리가 null이면 미측정이며 0이 아닙니다. 장치 전체 값에는 OS와 다른 프로세스가 포함됩니다. `success=true`는 소프트웨어 오류가 없다는 의미이고, dry-run 무구동 상태나 모델 주행 정확도를 증명하지 않습니다.

시스템 가용 RAM 최솟값도 기록합니다. 프로세스 RSS 상한이나 `runtime.system_available_min_gib` 하한을 위반하면 새 경로를 전달하기 전에 오류로 정지합니다. 초기 모델 준비 중 위반하면 시리얼 연결을 열지 않습니다. 표본 감시로 순간적인 메모리 폭증이나 OS 강제 종료까지 방지할 수는 없으므로 MCU watchdog은 별도로 필요합니다. `hardware.memory_architecture=shared`에서는 CUDA 메모리를 별도 VRAM으로 표시하지 않고, 전용 GPU 예산의 벤치마크 판정은 null로 둡니다. 메모리 구조는 설정값이며 하드웨어 자동 탐지가 아닙니다.

입력 신선도와 모델 처리 기한은 `runtime`에서 설정합니다. 기본 정지 반응 예산은 경로 유효기간+제어 주기+시리얼 쓰기 시간+MCU TTL을 포함하며, 카메라 내부 지연·실제 제동 지연은 측정해 추가해야 합니다. 지연 제한을 늘려 오류만 없애면 정지 거리도 늘어납니다. 처리 속도, 최대 속도, 관측 거리, 제동 조건을 함께 조정하세요.

## 재현 가능한 소프트웨어 진단

```powershell
.\.venv\Scripts\python.exe -m camera_rc drive-sim --output '.\runs\sim_01'
.\.venv\Scripts\python.exe '.\scripts\smoke_curriculum.py' --output '.\runs\curriculum_smoke_01' --device cuda
.\.venv\Scripts\python.exe -m pytest tests integrations/bevformer/test_run.py -q
```

시뮬레이션은 전진, 장애물 앞 정지, 인식 입력 중단 시 정지를 확인합니다. MCU 참조 모델 테스트는 지연/중복 명령, 통신 단절, 재활성화 조건을 확인합니다. 단순 시뮬레이션의 통과를 실제 차량의 충돌 회피 성능으로 해석하지 않습니다.
