# 네 방향 카메라의 공유 분할 모델 학습

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

학습·개발은 현재 Windows PC(RTX 3080 Laptop 16GB, RAM 32GB)에서 진행하고, 내보낸 경량 체크포인트를 MSI Ubuntu 24.04 LTS 장치에서 추론합니다. Ubuntu 장치에 원본 학습 데이터 전체를 옮길 필요는 없습니다. 아래 명령은 프로젝트 폴더에서 실행하는 **Windows PowerShell** 기준입니다.

## 모델이 배우는 것

`configs/curriculum_multiview.json`은 BDD 공개 데이터 → BDD+Cityscapes 혼합 → 장착한 전방·후방·좌측·우측 카메라 데이터와 공개 데이터 재학습의 세 단계입니다. 앞선 공개 데이터 준비 방법은 [CURRICULUM.md](CURRICULUM.md)에 있습니다. 실제 파일·정답·라이선스를 준비한 출처만 설정에 넣습니다. 데이터 다운로드나 학습이 자동으로 시작되지는 않습니다.

네 방향의 이미지는 **같은 경량 분할 모델을 학습하는 개별 표본**으로 사용합니다. 입력은 이미지 한 장이며 출력은 그 이미지의 도로 표면 분류입니다. 한 장치에서 모델 가중치를 공유해 각 방향을 처리할 수 있습니다. 이 구성은 네 영상을 동시에 입력받는 BEV 모델, 깊이·움직임 예측 모델, 충돌 회피를 직접 학습하는 모델이 아닙니다. 방향별 관측을 차량 좌표계에 맞추고 경로·차체 영역과 비교하는 처리는 별도의 런타임 역할입니다.

모든 단계는 `width=16`, 입력 `[높이192, 너비320]`, 2클래스 모델을 유지합니다. batch 2, gradient accumulation 4, worker 0, CUDA 자동 선택 및 AMP를 사용합니다. 이 값은 보수적인 시작값이며 Windows 학습 GPU의 여유에 따라 batch를 조정할 수 있습니다. 이미지 전체를 RAM에 캐시하지 않고 필요할 때 디코딩하며 manifest 메타데이터만 메모리에 둡니다. 학습 배치 크기를 키워도 배포 모델 구조가 커지지는 않습니다.

## 자체 데이터와 정답 만들기

장착 위치·렌즈·높이·노출 조건을 실제 운용 환경에 맞춰 전후좌우 영상을 수집합니다. 마스크는 사람이 작성하거나 자동 주석 후 사람이 검수한 **동일 해상도 정수 ID 이미지**여야 합니다.

| ID | 의미 |
| --- | --- |
| 0 | 도로/주행 표면 |
| 1 | 그 외 표면·객체 |
| 255 | 보이지 않거나 주석이 불확실해 학습에서 제외할 픽셀 |

색으로 칠한 RGB 마스크, 객체 탐지 박스, 단순 촬영 영상은 이 분할 모델의 정답을 대신하지 못합니다. 0은 도로 표면이라는 의미이며 장애물이 없다는 보증, 통행 허가, 안전한 속도를 뜻하지 않습니다. 누락한 정답을 모두 0으로 채우면 위험한 오학습이 되므로 불확실한 영역은 255로 둡니다. 도구는 형식과 값만 검사하며 사람이 실제로 검수했는지는 판단할 수 없습니다.

원본 JSONL은 각 줄에 다음 필드를 포함합니다. `image`, `mask` 상대 경로는 JSONL 파일의 폴더 기준입니다. 다음은 형식 설명을 위한 두 줄이며 실제 입력에는 **모든 방향의 train과 val 표본**이 필요합니다.

```jsonl
{"image":"images/drive_001/front/000001.jpg","mask":"masks/drive_001/front/000001.png","camera":"front","clip_id":"site_a_20261019_drive_001","split":"train"}
{"image":"images/drive_002/rear/000010.jpg","mask":"masks/drive_002/rear/000010.png","camera":"rear","clip_id":"site_a_20261019_drive_002","split":"val"}
```

`camera`는 `front`, `rear`, `left`, `right` 중 하나입니다. `clip_id`는 프레임 번호가 아니라 **원본 주행/장면 전체의 ID**입니다. 같은 주행의 네 카메라와 인접 시간 프레임에는 같은 값을 사용합니다. 긴 주행을 파일로 잘라 저장하더라도 서로 비슷한 인접 구간을 train/val로 나누지 말고 원본 주행 ID를 유지합니다. 촬영 날짜와 주행 번호를 포함해 다른 수집 회차와 구분하세요.

`split`은 사용자가 주행 단위로 정한 `train` 또는 `val`입니다. 무작위 프레임 분할은 제공하지 않습니다. 같은 장면이 다른 방향에서 찍혔다는 이유로 검증에 넣으면 성능이 과대평가될 수 있습니다. 최종 시험용 별도 주행은 이 JSONL에 넣지 않고 보존하세요. 카메라 간 정확한 동기화는 이 개별 이미지 학습의 요구사항은 아니지만, 실제 운용에서는 여러 방향의 관측 시각과 지연을 검사해야 합니다.

```powershell
Set-Location '.'
.\.venv\Scripts\python.exe scripts/prepare_multiview_training.py --manifest data/vehicle_labeled.jsonl --output data/road/multiview
```

스크립트는 `data/road/multiview/train_front.jsonl`, `val_front.jsonl` 등 8개 목록과 `report.json`을 만듭니다. 영상과 마스크는 복사하지 않습니다. 파일 존재·정렬·ID·학습 해상도에서 유효한 정답 픽셀·중복 이미지 경로·네 방향의 분할 완전성·클립 누출을 확인합니다. 데이터 검사에 실패하면 출력 폴더를 만들지 않고, 파일 쓰기에 실패해 `.incomplete`가 남았다면 해당 결과로 학습하지 마세요. 기존 출력은 덮어쓰지 않으므로 재생성할 때 새 출력 폴더와 설정 경로를 사용합니다.

모든 방향에 `dataset_id=vehicle_multiview`, `group_id=clip_id`, `origin_id=vehicle_multiview:clip_id`를 공통 기록하므로 기존 학습기의 출처 간 누출 검사와 연결됩니다. `--dataset-id`를 바꿀 때에도 같은 원본을 다른 이름으로 중복 등록하지 마세요. 이미지 내용의 유사도를 자동 비교하지 않으므로 이름을 바꾼 복사본·재인코딩본에도 원본 `clip_id`를 보존해야 합니다. 생성 목록에는 절대 경로를 쓰므로 데이터 위치를 옮겼다면 원본 JSONL의 경로를 수정하고 새 출력으로 다시 준비합니다.

## 검증하고 단계적으로 학습

공개 데이터와 자체 네 방향의 정답 파일을 준비한 후 실행합니다.

```powershell
# 파일 경로, 의미, 모든 단계의 학습/검증 누출을 점검합니다. 학습하지 않습니다.
.\.venv\Scripts\python.exe -m camera_rc curriculum --config configs/curriculum_multiview.json

# 마스크를 실제 디코딩해 검사합니다. 공개 데이터 stage에도 각각 적용하세요.
.\.venv\Scripts\python.exe -m camera_rc check-data --config configs/stages/multiview_vehicle.json

# 실제 학습은 명시적으로 실행합니다.
.\.venv\Scripts\python.exe -m camera_rc curriculum --config configs/curriculum_multiview.json --execute

# 같은 설정·입력의 중단된 실행을 이어갑니다.
.\.venv\Scripts\python.exe -m camera_rc curriculum --config configs/curriculum_multiview.json --execute --resume-run
```

마지막 단계의 기대 표본 비율은 앞·뒤·좌·우 **각 20%**, BDD·Cityscapes **각 10%**입니다. 영상 개수가 적은 방향도 같은 비중으로 복원 추출됩니다. 많은 반복 표본이 다양한 새 장면을 대체하지 않으므로 한 방향의 수집량이 매우 적으면 추가 수집이 필요합니다. stage 완료 시 각 방향과 기존 공개 검증 자료를 따로 평가합니다. 평균값뿐 아니라 가장 취약한 방향, 어두운 장면, 역광, 가까운 장애물, 교차 이동 상황의 결과를 확인하세요. 현재 검증 지표만으로 충돌률이나 실차 안전도를 추정할 수는 없습니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc export-runtime --checkpoint runs/curriculum_multiview/stages/multiview_vehicle/best.pt --output runs/deployment/road_surface.pt
```

내보낸 체크포인트를 Ubuntu MSI의 배포 설정에서 가리키도록 합니다. 합성·진단 자료로 만든 모델의 진단 표시는 내보낼 때도 보존됩니다. 자체 데이터의 실제 학습·독립 평가와 장치에서의 지연 측정을 마친 뒤 운용용 모델로 판단해야 합니다. 카메라 수를 늘리는 것만으로 충돌 회피가 완성되지 않으며, 특히 바닥 투영이 맞지 않는 돌출물·턱·움직이는 물체는 별도 인식과 검증이 필요합니다.

## 실제 데이터가 없을 때 코드만 점검

다음 실행은 명시적으로 합성 정답 8장을 만들고 전방 단계 → 네 방향 단계의 학습·가중치 전달·방향별 검증·완료 단계 재개를 확인합니다. 각 단계는 1 epoch이며 모든 설정과 체크포인트에 진단 표시가 남습니다. 성공해도 실제 데이터 학습이나 주행 성능 검증을 완료했다는 뜻은 아닙니다.

```powershell
.\.venv\Scripts\python.exe scripts/smoke_multiview_training.py --output runs/verification_multiview_training --device cuda
```

결과는 지정한 새 출력 폴더의 `verification.json`과 `training/report.json`에 저장됩니다. 재실행 시 기존 결과를 덮어쓰지 않도록 다른 출력 폴더를 사용합니다.

진단의 짧은 단계는 `training.amp_init_scale=1024`로 시작합니다. 일반 학습의 기본값은 65536이며, AMP 중간 계산이 넘치면 해당 optimizer 갱신을 건너뛰고 scale을 줄입니다. 실제 적용한 `optimizer_steps`와 건너뛴 `amp_skipped_steps`를 별도 기록합니다. 연속 8회 실패하거나 한 epoch 동안 적용한 갱신이 0회이면 완료로 처리하지 않고 중단합니다. 이때 입력·모델을 확인한 후 `amp_init_scale`을 낮추거나 `amp=false`로 새 출력 폴더에서 재시도합니다. 비유한 loss나 AMP를 사용하지 않을 때의 비유한 gradient는 계속 오류로 처리합니다.
