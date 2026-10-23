# 추가 데이터 학습과 독립 평가 — 2026-10-21

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

작업 폴더는 `.`, 원본은 `datasets`입니다. 조사한 추가 후보 24종의 용도·용량·접근 조건·공식 출처는 [데이터셋 카탈로그](ADDITIONAL_DATASETS.md)에 있습니다. 이 문서의 학습은 경량 **도로 영역 분할**에 한정되며, 객체 검출·거리 추정·교통 규칙·예측·실제 제동을 모두 학습한 자율주행 시스템이 아닙니다.

## 실제 사용한 자료

| 자료 | 학습 | 검증/보류 | 이번 역할 |
|---|---:|---|---|
| BDD100K 2018 | 기존 기준 모델 70,000장; 추가 단계는 고정 10,000장 재사용 | 공식 val 10,000장 전부 | 기존 환경 성능 유지·비교 |
| CamVid SegNet 배포본 | 406장 | val 124장, 별도 보류 test 171장 | 추가 학습 |
| KITTI Road | 231장 | val 58장; 공식 test 290장은 정답 비공개 | 추가 학습 |
| RoadObstacle21 | 0장 | 공개 GT 30장만 평가 | 작은 장애물을 도로로 오인하는지 검사 |
| RoadAnomaly21 | 0장 | 공개 GT 10장만 평가 | 새로운 물체를 도로로 오인하는지 검사 |

CamVid는 같은 원본 영상이 겹치는 기존 프레임 분할 대신 원본 영상 전체를 분리했습니다. KITTI도 공식 mapping의 원본 drive를 분리했습니다. **두 수치는 공식 리더보드 분할 점수가 아닙니다.** CamVid의 보류 171장은 학습·체크포인트 선택·이번 성능 비교에 사용하지 않았습니다. 장애물 평가 자료도 학습에 넣지 않았습니다.

CamVid 파일은 고정 Git commit의 blob hash, RoadObstacle/RoadAnomaly ZIP은 게시자 MD5, 모든 ZIP은 추출 중 CRC를 확인했습니다. KITTI는 공개 S3의 고정 ETag를 Range 요청마다 확인하고 ZIP CRC 및 로컬 SHA-256을 기록했습니다. KITTI의 multipart ETag나 로컬 SHA-256을 게시자 공개 MD5로 표현하지 않습니다. 출처 기록은 각 원본 폴더의 `DOWNLOAD_COMPLETE.json`에 보존됩니다.

이번 추가 원본 폴더의 저장량은 합계 2,286,161,620 bytes(약2.29GB/2.13GiB)입니다. 압축본·추출본·KITTI 재개 조각과 중단된 부분 파일까지 포함하며 기존 BDD와 변환 마스크·학습 결과는 제외합니다. 원본 이용 조건도 보존했습니다. 연구용 공개 접근이 상용 이용·모델 재배포 허가를 자동으로 뜻하지 않습니다.

## 라벨과 학습 단계

공통 ID는 `road_surface_v1`: **0=도로, 1=그 외, 255=무시**입니다. BDD2018 원본은 0=배경, 1=직접 주행, 2=대체 주행이므로 1·2→0, 0→1로 변환했습니다. CamVid는 3→0, 11→255, 나머지→1입니다. KITTI는 R=유효 영역, B=도로이며 lane GT는 사용하지 않았습니다. 도로 주석 기준은 출처마다 다르므로 각 출처를 따로 평가합니다.

1. 기존 `runs/bdd100k_2018_baseline_v1/best.pt`에서 시작합니다.
2. `bdd_camvid_stage1_v1`: BDD 10,000 + CamVid 406장, 2 epochs.
3. `bdd_camvid_kitti_stage2_v1`: 이전 단계 best 가중치에서 BDD 10,000 + CamVid 406 + KITTI 231장, 2 epochs.

각 단계는 해당 이미지 전체를 epoch마다 한 번씩 사용합니다. optimizer는 새로 시작합니다. 진단용 step 제한은 사용하지 않았습니다. 두 단계 모두 TinyUNet width16, 192×320 입력, batch16, 학습률0.0002, CUDA AMP입니다. BDD 재사용 표본은 학습 목록에서 고정 SHA-256 순위로 선택했습니다. 검증·보류 목록과 이미지/장면/원본 ID가 겹치면 거부합니다.

이것은 BDD 중심의 짧은 추가 학습 실험입니다. 두 번째 단계 비중은 약94.0%/3.8%/2.2%이고, 완전한 수렴이나 전 환경 향상을 의미하지 않습니다. `best.pt`는 합산 검증 mIoU로 선택되어 BDD 비중이 큽니다. 따라서 아래 데이터별 점수와 장애물 오인율을 함께 판단해야 합니다. 데이터 추가 효과만 분리한 대조 실험도 아니므로, 개선 전체를 새 데이터의 효과로 단정할 수 없습니다.

## 결과

**판정: 연구용 보존, 실제 운용 적용 미승인.** 도로 IoU는 세 데이터셋 모두 올랐지만, 도로가 아닌 곳과 작은 장애물을 도로로 잘못 판단하는 비율도 올라갔습니다. 이 후보를 안전성이 개선된 모델로 채택하지 않았습니다. 기존 기준 모델 역시 실차 운용이 검증된 모델이 아닙니다.

| 검증 자료 | 이미지 | 도로 IoU: 기존 → 추가 학습 | 비도로 픽셀의 도로 오인율: 기존 → 추가 학습 |
|---|---:|---:|---:|
| BDD100K | 10,000 | 67.28% → **71.20%** | 2.75% → **3.32%** |
| CamVid (원본 영상 보류) | 124 | 58.25% → **60.58%** | 0.98% → **2.28%** |
| KITTI Road (원본 주행 보류) | 58 | 67.08% → **77.24%** | 1.54% → **2.16%** |

도로 IoU는 높을수록 좋습니다. 위 오인율은 `FP/(FP+TN)`이며 낮을수록 좋습니다. 일반 도로 검증은 192×320으로 변환된 정답과 argmax를 비교합니다. 예를 들어 BDD 도로 recall은 76.37%→82.78%로 올랐지만 precision은 84.97%→83.57%로 떨어졌습니다. 더 넓게 도로를 예측하는 변화가 평균 점수와 위험한 오인을 동시에 증가시킬 수 있습니다.

| 독립 공개 GT 평가 | 이미지 / 표시된 장애물 픽셀 | 장애물 픽셀의 도로 오인율: 기존 → 추가 학습 |
|---|---:|---:|
| RoadObstacle21 | 30장 / 129,782픽셀 | 10.13% → **18.55%** |
| RoadAnomaly21 | 10장 / 1,317,063픽셀 | 0.0060% → **0.0411%** |

두 번째 표는 원본 GT 크기를 그대로 두고 `p(road)>=0.65`를 적용한 픽셀 합산 비율입니다. 이 확률은 안전성이 보정된 확률이 아닙니다. 원본 GT=0은 도로 정답으로 가정하지 않았고, GT=1 장애물만 분모로 사용했습니다. **사고 확률·객체 검출률·공식 anomaly AP가 아닙니다.** 두 표는 해상도와 결정 규칙이 달라 합산하지 않습니다.

RoadObstacle21의 이미지별 오인율 평균도 18.53%→47.75%로 악화했습니다. 새 모델은 **30장 중 8장에서 표시된 장애물 픽셀 전부를 도로로 분류**했습니다. `validation_15`, `validation_16`, `validation_17` 등이 포함됩니다. 작은 물체가 전체 픽셀 평균에서 가려지는 문제도 확인됐습니다. 표본이 작고 인접 프레임도 있으므로 30장의 결과를 독립적인 30회 주행이나 전체 환경의 안전성으로 해석할 수 없습니다.

두 학습 단계는 모두 `training_complete=true`, `diagnostic=false`, 완전한 epoch/검증으로 종료됐습니다. optimizer 갱신은 1,302회와 1,330회, AMP 갱신 누락은 0회입니다. 학습+epoch 검증 시간은 각각 약372초와376초였습니다. PyTorch 최대 할당 VRAM은 각각 **1.401GiB**였으며 CUDA 드라이버·카메라·다른 프로세스 메모리는 포함하지 않습니다. 실제 MSI 4GB GPU에서의 전체 메모리·속도 검증은 아직 하지 않았습니다.

실제 D 프로젝트에서 UTF-8 환경으로 전체 테스트 **454개 통과**했습니다. 이는 소프트웨어 회귀 검사이며 하드웨어 제동 시험이 아닙니다.

- 전후 비교와 미승인 판정 JSON (로컬 비공개 기록: `runs/additional_data_evaluation_v1/comparison.json`)
- 기존 모델 전체 평가 (로컬 비공개 기록: `runs/additional_data_evaluation_v1/baseline.json`), 추가 학습 모델 전체 평가 (로컬 비공개 기록: `runs/additional_data_evaluation_v1/candidate.json`)
- 첫 단계 학습 기록 (로컬 비공개 기록: `runs/bdd_camvid_stage1_v1/summary.json`), 두 번째 단계 학습 기록 (로컬 비공개 기록: `runs/bdd_camvid_kitti_stage2_v1/summary.json`)
- 예측 이미지 6장 (로컬 비공개 기록: `runs/additional_data_preview_v1/preview.png`): 각 데이터셋에서 미리 정한 2장씩, 전체 성능을 대표하지 않습니다. 직접 확인한 예시에서도 건물·하늘의 도로 오인과 경계 오류가 남아 있습니다.

추가 학습 모델은 `runs/bdd_camvid_kitti_stage2_v1/best.pt`이며 SHA-256은 `6dd380745de6acee1372e2efb7e4fc33aa1e6f279bb9a1644b809b0235292f87`입니다. 초기 기준 모델과 두 단계의 가중치·설정·manifest 해시를 모두 보존했습니다. 일반 검증 manifest뿐 아니라 장애물 평가 이미지와 GT의 해시도 전후 동일함을 확인했습니다.

## 재현과 후속 단계

프로젝트 루트에서 `.venv\Scripts\python.exe -X utf8`를 사용합니다. 새 학습을 준비할 때는 새로운 `--name`을 지정해야 합니다. 기존 결과는 덮어쓰지 않습니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_replay_stage.py --project '.' --name my_next_stage --source-root data/camvid_road_v1 --source-root data/kitti_road_v2 --init-from runs/bdd_camvid_kitti_stage2_v1/best.pt --epochs 2
.\.venv\Scripts\python.exe -X utf8 -m camera_rc train --config configs/local/my_next_stage.json
```

실제 준비된 KITTI manifest는 `data/kitti_road_v2`입니다. `data/kitti_road_v1`은 잘못 지정한 ZIP 중첩 경로의 실패 기록이며 `.incomplete` 상태로 보존했습니다. 학습 설정은 v2만 참조합니다.

동일한 검증 설정으로 기존 모델과 새 모델을 각각 평가합니다. `--output`은 새로운 JSON 파일 경로입니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/evaluate_road_safety.py --config configs/local/bdd_camvid_kitti_stage2_v1.json --checkpoint runs/bdd_camvid_kitti_stage2_v1/best.pt --output runs/my_evaluation.json --obstacle-root 'datasets\road_obstacle21\dataset_ObstacleTrack\dataset_ObstacleTrack' --anomaly-root 'datasets\road_anomaly21\dataset_AnomalyTrack\dataset_AnomalyTrack' --device cuda
```

다음 단계에서는 단순히 같은 도로 데이터를 더 반복하기보다, 별도의 **학습용** 작은 장애물·인도·보행자 정답과 위험한 도로 오인에 더 큰 비용을 주는 학습 방법을 검토해야 합니다. 객체 검출·거리 추정도 별도 검증이 필요합니다. 추가 데이터 확장 후보는 카탈로그의 A2D2 전방 RGB+semantic 일부 또는 Virtual KITTI2 RGB+semantic입니다. 이번 공개 GT40장을 학습·임계값 조정에 사용하면 독립 평가 역할을 잃으므로, 이 결과를 보고 방법을 조정한 다음에는 새로운 최종 보류 자료도 필요합니다. 실제 장착 카메라의 낮은 시점·렌즈·야간·역광·가림 조건은 자체 수집 및 별도 주행 분할로 검증해야 합니다.

## 제어 코드와 운용 상태

실제 센서 표본이 멈춰도 새 통신 패킷만 계속 오면 오래된 속도값이 신선하게 보이던 문제를 수정했습니다. 독립 센서 watchdog, 표본 순번/취득 시각 검증, 고장 후 물리 OFF→ON 재무장이 추가됐습니다. [USB 규약 v2](SERIAL_PROTOCOL.md)는 v1과 호환되지 않으며 PC와 MCU가 함께 구현해야 합니다.

MCU 코드는 Python 참조 모델입니다. 실제 보드 펌웨어·비상 정지 회로·제동 거리·카메라 노출 동기·최악 처리 지연을 검증한 것은 아닙니다. 화면 수신 시각만으로 카메라 버퍼나 반복 프레임 고장을 모두 판별할 수도 없습니다. 현재 4카메라 운용 설정은 `dry_run`, `models_validated_for_control=false`, `capture_timing_validated=false`를 유지합니다. 이번 체크포인트는 자동으로 배포 경로에 적용하지 않았습니다.

**데이터 학습이나 픽셀 인식 점수만으로 사고 확률 0을 보장할 수 없습니다.** 현재 결과는 연구용 모델 비교이며 실제 차량 운용 승인으로 사용하면 안 됩니다.
