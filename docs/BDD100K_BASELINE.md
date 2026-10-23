# BDD100K 실제 데이터 기준 모델 학습

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

현재 Windows 학습 PC의 RTX 3080 Laptop 16GB와 RAM 32GB에서 **BDD100K 전체 학습 분할의 첫 1 epoch와 전체 검증을 완료했습니다.** MSI Ubuntu 노트북은 이후 추론·제어 대상으로 유지하며, 이번에는 사용하지 않았습니다.

## 첫 실행 결과

2026-10-21 확인한 학습 원본 summary.json (로컬 비공개 기록: `runs/bdd100k_2018_baseline_v1/summary.json`)과 데이터 준비 원본 report.json (로컬 비공개 기록: `data/public_v2/report.json`)을 기준으로 기록했습니다. 준비 보고서의 `training_started=false`는 준비 도구 자체가 학습을 시작하지 않았다는 뜻이며, 그 뒤 별도로 실행한 실제 학습 결과는 `summary.json`에서 확인합니다.

| 항목 | 실제 결과 |
| --- | ---: |
| 학습 / 검증 이미지 | 70,000장 / 10,000장 |
| 완료 epoch | 1 / 요청 1 |
| 학습 미니배치 / 실제 optimizer 갱신 | 4,375 / 4,375 |
| AMP 갱신 건너뜀 | 0 |
| 전체 검증 배치 / 유효 픽셀 | 625 / 614,400,000 |
| 학습 손실 / 검증 손실 | 0.189423 / 0.150932 |
| 2클래스 mIoU | 80.03% |
| 도로 IoU / 도로 외 영역 IoU | 67.28% / 92.78% |
| 픽셀 정확도 | 93.71% |
| epoch 학습·검증 소요 시간 | 529.01초, 약 8분 49초 |
| 최대 PyTorch 할당 VRAM | 1.401GiB |
| 완료 / 진단 실행 여부 | `training_complete=true` / `diagnostic=false` |

부분 배치 제한 없이 학습했고, `partial_validation=false`로 전체 검증을 완료했습니다. 시간은 다운로드·전처리를 제외한 해당 epoch 기록입니다. VRAM 값은 PyTorch 텐서 할당량이며 장치 전체 메모리나 MSI의 필요 사양을 측정한 값이 아닙니다.

이 점수는 **너비 320×높이 192의 이진 도로 분할** 결과입니다. 직접·대체 주행 영역을 합쳤으므로 BDD 공식 3클래스 점수와 다르며, 충돌 회피·실제 주행·MSI 성능 검증 결과가 아닙니다. 총 20 epoch까지 이어 학습하는 설정은 준비되어 있지만 아직 실행하지 않았습니다.

## 데이터와 설정

이번 입력은 Archive에서 받은 BDD100K 2018년 이미지 및 `drivable_maps` 라벨입니다. 원본 ZIP 다운로드·해시 대조·선택 추출과 `data/public_v2`의 전체 학습 준비를 완료했습니다. 출처·라이선스·해시·샘플 확인 기록은 [BDD100K_SOURCES.md](BDD100K_SOURCES.md)에 있습니다.

구형 원본 ID는 `0=배경`, `1=직접 주행`, `2=대체 주행`입니다. 프로젝트 학습 클래스는 `0=road_surface`, `1=not_road_surface`이므로 다음처럼 바꿉니다. 무시 대상 `255`는 유지합니다.

```json
{"0": 1, "1": 0, "2": 0, "255": 255}
```

데이터 준비 시 `--bdd-encoding legacy_2018`을 지정합니다. 신형 BDD ID 변환 규칙을 이번 원본에 적용하면 배경이 도로로 바뀌므로 두 형식을 혼용하지 않습니다. 원본 사진은 그대로 참조하고 학습용 이진 마스크를 별도 생성합니다.

| 항목 | 경로 또는 설정 |
| --- | --- |
| 프로젝트 | `.` |
| 원본 이미지 루트 | `datasets\bdd100k_2018\extracted\bdd100k\images\100k` |
| 원본 ID 마스크 루트 | `datasets\bdd100k_2018\extracted\bdd100k\drivable_maps\labels` |
| 준비 데이터 | `data/public_v2` |
| 첫 실행 설정 | `configs/local/bdd2018_baseline_v1.json` |
| 20 epoch까지 이어 학습하는 설정 | `configs/local/bdd2018_continue20_v1.json` |
| 결과 폴더 | `runs/bdd100k_2018_baseline_v1` |
| 모델 | TinyUNet, width 16, 2개 클래스 |
| 입력 크기 | 높이 192 × 너비 320 |
| batch / gradient accumulation | 16 / 1 |
| 데이터 로더 worker | 2 |
| 첫 실행 | 전체 학습 분할 1 epoch 후 전체 검증 완료 |
| 이어 학습 | 완료 epoch를 포함한 누적 20 epoch 설정 준비, 미실행 |

공식 train/val 분할을 유지합니다. 추출한 `images/100k`의 규모는 train 70,000장, val 10,000장이며 실제 준비 결과는 `data/public_v2/report.json`의 검사 결과와 샘플 수를 확인합니다. 같은 추출본에는 별도 `images/10k`의 train 7,000장·val 1,000장도 있지만, 이번 drivable 학습에는 포함하지 않습니다. test 이미지도 이번 학습·검증에 사용하지 않습니다.

## 준비 결과와 최초 실행 명령 기록

PowerShell에서 프로젝트 폴더로 이동합니다.

```powershell
Set-Location -LiteralPath '.'
Get-Content -LiteralPath 'data/public_v2/report.json'
```

이번 `report.json`에는 `complete=true`와 train/val 전 샘플 해독 완료가 기록되어 있습니다. 이후 새 데이터를 준비할 때도 보고서의 성공 여부와 검사를 확인하고, `.incomplete`가 남아 있으면 해당 준비 실행을 완료 상태로 취급하지 않습니다. 구체적인 데이터 준비 명령은 [PUBLIC_TRAINING_SETUP.md](PUBLIC_TRAINING_SETUP.md)를 참고하세요.

아래는 이미 완료한 최초 실행 명령의 기록입니다. **현재 같은 결과 폴더로 다시 실행하지 않습니다.** 추가 학습에는 다음 절의 `--resume` 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc train `
  --config configs/local/bdd2018_baseline_v1.json
```

이미 같은 설정의 학습이 실행 중이면 두 번째 프로세스를 시작하지 않습니다. 기존 결과가 있는 폴더에 처음부터 다시 실행하면 코드가 덮어쓰기를 거부합니다. 이어 학습할 때는 아래 `--resume` 명령을 사용하고, 별도 실험은 새 `training.output_dir`를 지정한 설정을 만듭니다.

첫 실행은 작은 진단용 일부 배치가 아니라 1 epoch의 기준 모델을 얻기 위한 구성입니다. `--max-steps`나 설정의 `max_train_steps`, `max_val_steps`로 제한한 실행은 전체 학습·검증과 구분합니다.

## 20 epoch까지 이어 학습

첫 epoch의 학습·검증을 마쳤고 `last.pt`를 저장했습니다. 총 20 epoch까지 이어 학습하려면 다음 명령을 사용합니다. 이 추가 실행은 아직 수행하지 않았습니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc train `
  --config configs/local/bdd2018_continue20_v1.json `
  --resume runs/bdd100k_2018_baseline_v1/last.pt
```

`epochs=20`은 **총 20 epoch**를 뜻합니다. 1 epoch를 완료했다면 19 epoch를 추가로 수행합니다. 모델 가중치뿐 아니라 optimizer, AMP scaler, 난수 상태와 진행 epoch를 복원합니다. 재개 설정의 `training.output_dir`는 체크포인트가 있는 기존 결과 폴더와 같아야 합니다.

현재 구현은 완료된 epoch 체크포인트에서 다음 epoch로 재개합니다. epoch 도중 중단되면 마지막 완료 epoch 이후의 작업은 다시 수행합니다. 첫 epoch를 끝내기 전에는 재개 가능한 `last.pt`가 없을 수 있습니다. 진단용으로 epoch 중간에 잘린 체크포인트는 재개할 수 없습니다.

총 20 epoch까지 이미 완료했다면 같은 설정으로 추가 학습하지 않습니다. 평가를 실행하거나, 더 학습할 필요가 있는지 검토한 뒤 누적 `epochs` 값을 늘린 새 설정으로 재개합니다.

## 결과 확인과 평가

학습 결과를 확인합니다.

```powershell
Get-Content -LiteralPath 'runs/bdd100k_2018_baseline_v1/summary.json'
Get-Content -LiteralPath 'runs/bdd100k_2018_baseline_v1/metrics.jsonl' -Tail 1
```

- `summary.json`: 해당 실행의 `requested_epochs`, `completed_epochs`, `training_complete`, `diagnostic`와 최종 검증 결과.
- `metrics.jsonl`: 완료 epoch별 학습 손실, 검증 지표, 실제 optimizer 갱신 횟수, AMP skip 수, 소요 시간과 학습 VRAM.
- `last.pt`: 마지막 완료 epoch에서 이어 학습하기 위한 체크포인트.
- `best.pt`: 검증 mIoU 기준으로 선택한 체크포인트.
- `config.json`, 재개 시 `resume_config.json`: 실제 사용한 학습 설정.

`summary.json`은 실행이 정상적으로 끝난 뒤 작성됩니다. 파일이 아직 없으면 완료 결과가 아닙니다. 20 epoch 재개 중에는 이전 1 epoch의 summary가 남아 있을 수 있으므로 `requested_epochs`, 최신 로그와 실행 상태를 함께 확인해야 합니다. `training_complete=true` 역시 그 설정의 epoch·검증을 마쳤다는 뜻이며, 자율주행 성능이 충분하다는 판정은 아닙니다.

선택된 모델을 검증 분할 전체에서 다시 평가하려면 다음 명령을 사용합니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc evaluate `
  --config configs/local/bdd2018_baseline_v1.json `
  --checkpoint runs/bdd100k_2018_baseline_v1/best.pt
```

실제 실행 결과의 `miou`, `class_iou`, `pixel_accuracy`를 확인합니다. `class_iou[0]`은 road, `class_iou[1]`은 non-road입니다. 두 클래스를 합친 이번 지표는 BDD의 직접·대체 주행 클래스를 나눠 평가하는 공식 벤치마크 수치와 같지 않습니다.

## 예측 이미지 미리보기

저장된 `best.pt`를 CPU FP32로 읽어 RGB·정답·예측 비교 이미지 (로컬 비공개 기록: `runs/bdd2018_preview_v1/preview.png`)를 생성했습니다. 미리보기 원본 report.json (로컬 비공개 기록: `runs/bdd2018_preview_v1/report.json`)의 `status=complete`를 확인했습니다. 선택한 검증 인덱스는 `[0, 2000, 4000, 5999, 7999, 9999]`입니다. 이 6장은 시각적 오류 확인용이며 전체 성능을 대표하는 표본은 아닙니다. 위 결과 표의 mIoU는 별도로 수행한 전체 10,000장 검증에서 나온 값입니다.

검토한 이미지에는 야간 상단 배경의 작은 도로 오탐(index 2000), 야간 도로 내부의 누락과 경계 오류(index 4000), 눈길의 일부 도로 누락(index 9999)이 남아 있었습니다. 해당 조건의 개선 여부를 후속 학습에서 확인해야 합니다.

아래는 완료한 미리보기 명령입니다. 다시 실행할 때는 기존 폴더를 덮어쓰지 않도록 `--output`을 새 이름으로 바꿉니다. CPU 실행이므로 CUDA 학습 성능을 측정하는 도구는 아닙니다.

```powershell
.\.venv\Scripts\python.exe scripts/preview_segmentation.py `
  --checkpoint runs/bdd100k_2018_baseline_v1/best.pt `
  --output runs/bdd2018_preview_v1 `
  --samples 6
```

기본 선택은 검증 manifest 순서에서 균등 간격으로 뽑은 6장입니다. 결과는 새 출력 폴더의 `preview.png`와 `report.json`에 저장됩니다. 다시 실행할 때는 새 출력 폴더를 지정합니다. 정답·예측의 도로 영역은 초록색, 무시 영역은 자홍색이며, 표시한 도로 IoU는 모델 입력 해상도에서 계산한 해당 샘플의 값입니다. 이 소량 미리보기는 전체 검증이나 train/val 누출 검사를 대신하지 않습니다.

미리보기에는 선택 의존성인 Matplotlib이 필요하며 현재 `.venv`에는 설치되어 있습니다. 새 환경에서는 다음 명령으로 설치합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[preview]'
```

## 이후 판단

1 epoch 결과는 이후 학습과 비교할 출발점입니다. 손실이 감소했는지, 검증 도로 IoU가 개선되는지, 예측에서 사람·차량·보도와 경계를 잘못 도로로 포함하지 않는지를 확인하고 이어 학습합니다. 전체 데이터 학습 자체만으로 목표 환경에 충분한 성능을 얻었다고 판단하지 않습니다.

이 모델은 전방 사진에서 도로 형태를 분할합니다. 충돌 회피, 물체와의 거리, 통행 허가, 다중 카메라의 시공간 일치나 조향·속도 제어를 이 학습만으로 검증하지 않습니다. 실제 장착 카메라 데이터로 평가·보정하고, MSI에서 지연·메모리·제어 연동을 별도로 측정해야 합니다. 학습 결과를 차량 운용 모델로 자동 배치하지 않습니다.
