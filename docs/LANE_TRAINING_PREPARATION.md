# 차선 학습 자료 준비

`scripts/prepare_lane_training.py`는 사람이 명시적으로 주석한 **자차 주행 차선 영역**을 별도 마스크·manifest·설정 파일로 준비합니다. 도로면 분할 결과에서 차선 정답을 추정하지 않습니다. 준비를 실행해도 학습·다운로드·정확도 평가·모델 채택·자동 모델 교체는 시작하지 않습니다. 시뮬레이터의 가정 차선 기하와 학습된 차선 인식 모델도 구분합니다.

## 입력과 라벨 의미

UTF-8 JSONL 한 줄은 한 이미지입니다. 이미지 경로는 주석 JSONL의 폴더 기준 상대 경로 또는 절대 경로이며, `group_id`는 한 주행·장면의 연속 프레임을 함께 묶는 식별자입니다. 동일 주행을 train/val 양쪽에 나누지 않습니다. 준비 한 번에는 같은 `dataset_id`만 사용합니다.

```json
{"image":"images/train-001.png","group_id":"drive-001","split":"train","dataset_id":"my_ego_lane","label_space_id":"ego_lane_v1","diagnostic_only":true,"valid_polygon":[[20,100],[300,100],[319,179],[0,179]],"ego_lane_polygon":[[140,110],[180,110],[240,179],[80,179]]}
{"image":"images/val-001.png","group_id":"drive-002","split":"val","dataset_id":"my_ego_lane","label_space_id":"ego_lane_v1","diagnostic_only":true,"valid_polygon":[[20,100],[300,100],[319,179],[0,179]],"ego_lane_polygon":[[140,110],[180,110],[240,179],[80,179]]}
```

좌표는 원본 이미지의 픽셀 좌표 `[x,y]`이며, 범위는 `0..width-1`, `0..height-1`입니다. 예시는 320×180 이미지용 형식 예입니다. 실제 주석과 서로 다른 이미지를 사용해야 합니다. `valid_polygon`은 자차 차선 여부가 확인된 영역이고, `ego_lane_polygon`은 그 안에 완전히 포함되는 자차 차선 영역입니다. 점 3개 이상, 유한 좌표, 0이 아닌 면적, 자기 교차 없는 단순 다각형이 필요합니다. 오목한 valid 영역 밖으로 ego 다각형의 변이 통과하는 경우도 거부합니다.

| ID | `ego_lane_v1` 의미 |
|---|---|
| 0 | valid 영역 안에서 주석상 자차 차선 바깥 |
| 1 | 명시적으로 주석한 자차 주행 차선 |
| 255 | valid 영역 바깥의 관측·주석 미확정 영역, 학습 ignore |

`0`은 도로가 아니라는 뜻이 아닙니다. 다른 차선도 포함될 수 있습니다. `road_surface_v1` 등 도로면 label space는 거부하며, 도로 마스크만 주고 polygon을 생략할 수 없습니다. 원본 마스크를 수정하거나 원본 이미지를 다시 인코딩하지 않습니다. `diagnostic_only: true`인 행이 하나라도 있으면 생성 설정 전체에 진단 전용 표시를 남깁니다. 합성 fixture·예시 자료는 반드시 진단 전용으로 표시합니다. 이 플래그는 학습 실행을 막는 권한 잠금이 아니라 후속 checkpoint에 이어지는 출처 표시입니다.

## 준비 명령

필요한 의존성은 Python·NumPy·Pillow입니다. 저장소 루트에서 아직 존재하지 않는 새 출력 폴더를 지정합니다.

```powershell
python scripts/prepare_lane_training.py --annotations data/ego_lane_annotations.jsonl --output data/ego_lane_prepared_new --image-size 180 320
```

두 split에 각각 1개 이상의 행이 필요합니다. 동일한 resolved 이미지 경로·파일 SHA256·완전히 디코딩한 RGB SHA256가 반복되면 split 내·split 간 모두 거부합니다. 메타데이터만 다른 동일 픽셀 이미지도 중복입니다. 그룹 이름이 달라도 픽셀 중복은 허용하지 않습니다. 이 검사는 유사 이미지·연속 프레임·다른 각도 장면의 근접 중복을 자동 탐지하지 않으므로 `group_id`를 원 주행 단위로 정확히 작성해야 합니다.

정상 출력은 다음과 같습니다.

- `masks/*.png`: 원본 이미지와 같은 크기의 8-bit ID 마스크
- `train.jsonl`, `val.jsonl`: 원본 이미지·새 마스크 경로, 그룹·출처·라벨 의미
- `train_config.json`: 기존 baseline trainer 형식의 2클래스·CPU 설정 초안
- `report.json`: 원본 주석·이미지·마스크 및 manifest·설정 해시, split별 원본·학습 크기 픽셀 수, 준비·미실행 상태
- `.complete`: `preparation_complete: true` 및 `report.json`의 SHA256

완료 판단은 `.complete`의 존재만으로 하지 않습니다. **`.incomplete`가 없고, `.complete`의 report SHA256가 실제 보고서와 일치하며, 보고서가 기록한 파일·원본·마스크 해시까지 일치해야 합니다.** 출력 시작 후 실패하면 `.incomplete`를 남기고 완료 marker를 발행하지 않습니다. 입력 형식·그룹 누수 같은 사전 검증 실패는 출력 폴더를 만들기 전에 거부합니다. 실패 결과를 수작업으로 완료 처리하거나 재사용하지 말고 입력 수정 후 새 출력 폴더로 다시 준비합니다. 실행 중 원본 파일을 다른 프로그램으로 수정하지 않습니다.

## 기존 학습기와의 관계 및 검증 범위

기존 `config.load_config`는 config 파일 기준으로 train/val manifest와 출력 경로를 해석합니다. 생성 설정의 `task=segmentation`, `model.num_classes=2`, `data.label_space_id=ego_lane_v1`, `image_size`, class weights 2개, CPU·AMP 꺼짐 옵션은 현재 trainer schema에 맞습니다. `ManifestDataset`은 label space 일치·원본 이미지와 마스크 크기·ID 0/1/255를 검사하고 마스크를 nearest 방식으로 조정하며, 기존 segmentation loss는 255를 ignore합니다. `diagnostic_only`는 trainer의 진단 checkpoint 표시로 전달됩니다. 생성 설정은 향후 사용자가 학습을 별도로 지시할 때 검토하는 초안이며, 준비 코드가 trainer를 호출하지 않습니다. 기존 도로면 checkpoint의 label space를 차선으로 바꿔 이어 학습하거나 런타임 모델로 채택하는 절차는 포함하지 않습니다.

현재 baseline trainer는 이 준비 도구의 `.complete`·report SHA 계약을 자동으로 검증하지 않습니다. 후속 학습을 승인하기 전에 준비 산출물의 완료·해시·그룹·주석 품질을 별도로 확인해야 합니다.

`tests/test_lane_training_preparation.py`는 작은 합성 CPU fixture로 라벨 매핑·원본 보존·완료 해시·새 출력 강제·중복/그룹 누수 거부·잘못된 polygon/오목 영역 containment·실패 marker·CLI의 학습 미실행을 검사합니다. 실제 trainer import·실행, 실제 데이터 적합성·차선 인식 정확도·실차 성능을 검증한 결과는 아닙니다.
