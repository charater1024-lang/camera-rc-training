# 데이터 계약과 실제 데이터 준비

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

계정 가입·개인 약관 동의가 필요한 데이터는 사용자가 처리합니다. 명시적으로 요청한 무가입 공개 자료는 검증용 다운로드 스크립트로 준비할 수 있습니다. 모든 데이터셋의 정답을 같은 것으로 취급하지 않습니다. 추가 후보·접근 조건은 [추가 데이터셋 조사](ADDITIONAL_DATASETS.md)를 참고하세요.

## 공통 JSONL

한 줄에 JSON 객체 하나입니다. `image`, 파일형 `target`은 JSONL 위치 기준 상대 경로 또는 절대 경로입니다. 한글·공백 경로를 지원합니다. 예시는 실제 파일로 바꿔야 합니다.

```json
{"image":"images/drive01/0001.jpg","target":"masks/drive01/0001.png","group_id":"drive01","dataset_id":"own_rc","task":"segmentation"}
```

- `group_id`: 원본 주행/장면 ID. 연속 프레임, 같은 시점의 다른 카메라, 파생 라벨은 같은 그룹을 사용합니다.
- `dataset_id`: 출처. 같은 원본의 파생 자료에는 같은 원본 dataset_id/group_id를 유지하세요.
- 정답 없는 픽셀/객체를 배경으로 바꾸지 마세요. 코드가 다른 데이터셋 클래스 체계를 자동 통합하지 않습니다.
- 공식 train/val 분할을 우선 사용합니다. `split` 명령은 자체 수집 자료를 그룹 단위로 분리할 때 사용합니다. 시험용 데이터는 별도로 보관하세요.
- `check-data`는 모든 샘플의 파일·형상·값을 읽으므로 큰 데이터에서는 오래 걸립니다.

## BDD100K 검출

공식 100K 이미지와 detection JSON을 사용합니다. 분할용 10K 이미지를 같은 이미지 묶음으로 간주하지 마세요.

```powershell
.\.venv\Scripts\python.exe -m camera_rc prepare-bdd-detection --train-labels 'datasets\bdd\labels\bdd100k_labels_images_train.json' --val-labels 'datasets\bdd\labels\bdd100k_labels_images_val.json' --train-images 'datasets\bdd\images\100k\train' --val-images 'datasets\bdd\images\100k\val' --output '.\data\bdd_yolo'
```

기본 클래스는 pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign입니다. 실제 순서는 생성된 dataset.yaml을 확인하세요. person/motor/bike 옛 명칭은 대응 클래스에 매핑합니다. 알 수 없는 box 클래스, 라벨 없는 이미지 파일, 손상된 상자 데이터는 명시적으로 실패합니다. labels=[]는 공식적으로 주석된 음성 샘플로 허용합니다. 원본 파일은 보존하고 이미지 하드링크 또는 복사를 생성합니다.

## BDD100K 주행 가능 영역

일반 `segmentation_bdd.json` 예제는 차선 선분 검출과 구별되는 신형 3클래스 픽셀 분할입니다. 신형 단일 채널 ID PNG는 `0=direct`, `1=alternative`, `2=background`, `255=ignore`입니다. **이번에 내려받은 2018 drivable_maps는 `0=background`, `1=direct`, `2=alternative`이므로 서로 바꾸어 사용할 수 없습니다.** 컬러 시각화 PNG는 입력할 수 없습니다.

현재 실제 학습의 `data/public_v2`와 `configs/local/bdd2018_*`, `bdd_camvid_*` 계열은 별도로 변환된 `road_surface_v1`입니다: `0=road`, `1=nonroad`, `255=ignore`. 2018 원본의 1·2→0, 0→1로 변환합니다. 이미 변환된 마스크를 다시 신형 BDD 매핑에 넣지 마세요. CamVid·KITTI Road도 이 공통 ID로 변환하되 원본 주석 기준의 차이는 데이터셋별 검증에서 확인합니다.

공통 `pair-manifest`는 이미지와 마스크의 상대 파일 stem이 같을 때만 동작합니다. `--group-pattern`의 캡처는 실제 원본 주행 ID여야 합니다. 소스별 전용 준비 명령은 README를 참고하세요.

## KITTI 깊이

RGB와 **projected ground-truth 16-bit depth PNG**가 필요합니다. raw RGB만으로 지도학습 깊이 정답이 생기지 않습니다. 공식 encoded depth/256 = metres, 0은 정답 없음입니다. `proj_depth/velodyne_raw` 희소 입력을 정답으로 선택하지 마세요. image_02와 image_03 프레임은 같은 drive 그룹입니다.

```json
{"image":"raw/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000005.png","target":"depth/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png","group_id":"2011_09_26_drive_0001_sync","dataset_id":"kitti_depth","task":"depth"}
```

깊이는 nearest 방식으로 정답을 축소하고 유효한 픽셀에서만 손실·RMSE·AbsRel을 계산합니다. 원본 이미지와 정답 크기가 일치해야 합니다. 이 간단한 baseline은 KITTI 리더보드용 crop/평가 절차를 재현하지 않습니다. 프로젝트 내부 비교용 지표입니다.

## 자체 RC 미래 경로

CSV 열:

```csv
image,group_id,timestamp_s,x_m,y_m,yaw_rad,speed_mps,yaw_rate_rps
images/run01/0001.jpg,run01,0.0,0.0,0.0,0.0,0.5,0.0
images/run01/0002.jpg,run01,0.1,0.05,0.0,0.0,0.5,0.0
```

위 두 행은 형식 예시이며 2.5초 미래 정답을 만들려면 더 긴 기록이 필요합니다. 각 이미지 시각의 보정된 평면 pose가 필요합니다. PWM/스로틀 기록만으로 미터 단위 경로 정답을 만들 수 없습니다. 휠·IMU/VIO 등의 위치 추정을 사용한다면 스케일·드리프트·시각 오차도 평가하세요. 동기화된 노출 시각을 기준으로 로그를 맞추세요.

- 좌표: world x/y는 고정 기준, yaw는 +x축 기준 반시계 방향 라디안.
- 출력 경로: **현재 차량 기준 x=전방, y=왼쪽**, 단위 m.
- 입력: 현재 RGB와 현재 `[speed_mps,yaw_rate_rps]`만 사용. 미래 pose/속도는 정답 생성용입니다.
- 정답: 0.5,1.0,1.5,2.0,2.5초 뒤 `[forward_m,left_m,speed_mps]`.
- 현재 구현은 전진/정지용이며 음수 속도·후진 주행은 거부합니다.
- 큰 로그 간격과 시퀀스 끝의 미래가 없는 샘플은 제외합니다. 서로 다른 주행 사이를 보간하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc prepare-rc --csv '.\data\rc\poses.csv' --output '.\data\rc\all.jsonl' --max-gap 0.2
.\.venv\Scripts\python.exe -m camera_rc split --manifest '.\data\rc\all.jsonl' --output '.\data\rc\split' --val-fraction 0.2
```

독립된 주행 그룹이 최소 2개 필요합니다. 신뢰할 평가를 위해 서로 다른 코스·날짜·조명을 별도 검증/시험으로 확보하세요. 모델 출력은 오프라인 연구용이며 이 프로젝트에는 모터·서보 제어 코드가 없습니다.

## 출처

- BDD100K 형식: https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst
- KITTI depth: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
- 데이터별 약관은 코드 라이선스와 별도입니다. Ultralytics 검출 모듈은 AGPL-3.0/상업 라이선스 조건을 확인하세요: https://www.ultralytics.com/license
