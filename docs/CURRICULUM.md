# 여러 데이터셋을 단계적으로 학습하기

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

이 기능은 준비된 로컬 데이터로 `공개 데이터 → 여러 출처 혼합 → 장착 카메라 환경 적응`을 실행합니다. 인식·깊이·경로 예측은 각각 별도 모델 체인입니다. 모든 공개 데이터셋을 자동으로 내려받거나 동일한 정답으로 취급하지 않습니다. 라이선스와 사용할 분할은 각 데이터셋에서 확인하세요.

`configs/curriculum_example.json`은 BDD100K와 Cityscapes, 자체 차량 데이터를 활용하는 **2클래스 도로 표면 분할** 예제입니다. 클래스 0은 도로 표면, 1은 그 외, 255는 무시입니다. 도로 표면은 장애물이 없거나 주행이 허가되었다는 뜻이 아닙니다. BDD의 주행 가능 영역과 Cityscapes의 도로 표면은 주석 기준도 다르므로 출처별 평가로 효과를 확인해야 합니다.

## 1. 원본 정답을 공통 의미로 변환

아래 예제는 **신형 BDD 인코딩 전용**입니다. BDD 변환기는 기존 `prepare-bdd-drivable`로 `data/bdd_drivable/train.jsonl`, `val.jsonl`을 만듭니다. 신형 원본 0=direct, 1=alternative를 도로 표면 0으로 합치고, 원본 2=background를 1로 옮깁니다. 근거: [BDD 공식 형식](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst).

**현재 내려받아 학습한 BDD2018 `drivable_maps`에는 아래 신형 매핑을 적용하지 마세요.** 2018 원본은 0=background, 1=direct, 2=alternative입니다. 현재 `data/public_v2/road/bdd_train` 및 `bdd_val`은 이미 1·2→road 0, 0→nonroad 1로 변환되어 있습니다. 이 manifest를 그대로 연결합니다. 추가 CamVid/KITTI는 `scripts/prepare_additional_road.py`, BDD 재사용 표본을 포함한 새 단계는 `scripts/prepare_replay_stage.py`로 준비합니다. 두 스크립트는 기존 결과를 덮어쓰지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc remap-masks --manifest data/bdd_drivable/train.jsonl --mapping configs/label_maps/bdd_to_road.json --output data/road/bdd_train
.\.venv\Scripts\python.exe -m camera_rc remap-masks --manifest data/bdd_drivable/val.jsonl --mapping configs/label_maps/bdd_to_road.json --output data/road/bdd_val
```

Cityscapes는 `leftImg8bit/{train,val}/도시/`와 `gtFine/{train,val}/도시/`를 준비합니다. 다음 경로는 내려받은 실제 경로로 바꾸세요.

```powershell
.\.venv\Scripts\python.exe -m camera_rc prepare-cityscapes --images datasets/cityscapes/leftImg8bit --labels datasets/cityscapes/gtFine --output data/cityscapes
.\.venv\Scripts\python.exe -m camera_rc remap-masks --manifest data/cityscapes/train.jsonl --mapping configs/label_maps/cityscapes_to_road.json --output data/road/cityscapes_train
.\.venv\Scripts\python.exe -m camera_rc remap-masks --manifest data/cityscapes/val.jsonl --mapping configs/label_maps/cityscapes_to_road.json --output data/road/cityscapes_val
```

Cityscapes 매핑은 **`*_gtFine_labelIds.png` 원본 ID** 전용입니다. `trainIds`나 컬러 PNG에는 적용하지 마세요. 도로 ID 7→0, 인도 ID 8→1이며 기존 무시 클래스는 255로 유지합니다. 다른 평가 클래스는 1로 합칩니다. [Cityscapes 공식 ID 정의](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py).

변환 결과는 `<output>/manifest.jsonl`, `masks/`, `report.json`입니다. 영상은 복사하지 않습니다. 원본 영상 경로, `dataset_id`, `group_id`, `origin_id`를 보존하고 `target_original`을 기록합니다. 선언하지 않은 ID는 모두 255이며 보고서에 픽셀 수가 나옵니다. 해상도 정렬이 다르거나 RGB 마스크이면 실패합니다. 실패 후 `.incomplete`가 남은 출력은 사용하지 말고 원인을 수정한 뒤 새 출력 폴더로 실행하세요.

자체 데이터에도 같은 의미의 마스크가 필요합니다. `data/road/vehicle_train/manifest.jsonl`, `vehicle_val/manifest.jsonl`에 경로를 연결하거나 예제 설정을 수정하세요. 영상을 수집했다고 정답 마스크가 자동으로 생기지는 않습니다. 자체 데이터가 아직 없으면 해당 단계 전체를 커리큘럼에서 제외합니다.

## 2. 계획 검증 후 실행

```powershell
# 기본 동작은 경로·분할·연결 관계 검사입니다. 학습/다운로드/출력 생성 없음.
.\.venv\Scripts\python.exe -m camera_rc curriculum --config configs/curriculum_example.json

# 정답을 실제 디코딩해서 검사할 때
.\.venv\Scripts\python.exe -m camera_rc check-data --config configs/stages/road_public.json

# 검증된 모든 단계를 순서대로 학습
.\.venv\Scripts\python.exe -m camera_rc curriculum --config configs/curriculum_example.json --execute

# 설정과 입력 목록이 같은 기존 실행을 이어가기
.\.venv\Scripts\python.exe -m camera_rc curriculum --config configs/curriculum_example.json --execute --resume-run
```

계획은 모든 단계에서 경로·정답 파일 존재와 의미 메타데이터, 전체 학습/검증 분리를 검사합니다. 이미지를 모두 디코딩하지는 않으므로 `check-data`로 실제 마스크 값과 정렬도 확인하세요. 실행 중 사용되는 설정·목록을 변경하지 마세요. 결과는 `runs/curriculum_road_surface/stages/단계명/` 아래에 저장됩니다. 단계 설정의 `training.output_dir`은 커리큘럼의 출력 폴더로 대체됩니다.

`plan.json`은 설정·manifest의 SHA-256 및 데이터 출처, `state.json`과 `report.json`은 단계 결과·가중치 해시·검증 출처별 성능을 기록합니다. 큰 이미지/마스크 파일 자체를 모두 해시하지는 않으므로 데이터 버전 폴더는 변경하지 않는 방식으로 보관하세요. 기존 폴더는 기본적으로 덮어쓰지 않습니다.

완료된 단계는 `--resume-run`에서 체크포인트 해시를 확인하고 건너뜁니다. 네이티브 학습이 예외로 중단된 경우 완료된 epoch의 `last.pt`로 이어갈 수 있습니다. 미완료 epoch 진단 실행이나 중단된 YOLO 단계는 보존하며 자동으로 완료 처리하지 않습니다. 설정을 바꾸거나 학습 단계를 추가했다면 새 `output_dir`로 실행합니다.

## 3. 혼합 비율과 단계 연결

```json
{
  "data": {
    "label_space_id": "road_surface_v1",
    "train_sources": [
      {"name": "vehicle", "manifest": "vehicle_train.jsonl", "weight": 3},
      {"name": "public_replay", "manifest": "public_train.jsonl", "weight": 1}
    ],
    "val_manifests": [
      {"name": "vehicle", "manifest": "vehicle_val.jsonl"},
      {"name": "public", "manifest": "public_val.jsonl"}
    ]
  }
}
```

출처 가중치 3:1은 영상 개수와 무관하게 **기대 샘플 비율 75%:25%**를 뜻합니다. 복원 추출이므로 한 epoch에 모든 이미지가 한 번씩 사용되는 방식은 아닙니다. epoch 당 표본 수는 기본 전체 학습 행 수이며 `training.samples_per_epoch`로 바꿀 수 있습니다. 기존 `train_manifest`, `val_manifest`도 지원하지만 복수 출처 설정과 동시에 넣지 마세요.

이미지는 필요할 때 읽고 전체 영상을 RAM에 캐시하지 않습니다. manifest 메타데이터는 메모리에 두므로 수백만~수천만 행은 단계별 데이터 범위를 나누어 관리하세요. 예제는 RAM 16GB를 고려해 worker 0, batch 2, 누적 4로 시작합니다. 이 값은 학습 시작점이며 VRAM 4GB 실시간 추론 성능을 보장하는 측정치는 아닙니다.

단계의 `init_from: "previous"`는 바로 전 단계의 `best.pt`에서 **가중치만** 읽습니다. optimizer, 학습률, epoch, RNG는 새로 시작합니다. `init_from: "stage:road_public"`처럼 앞선 단계를 명시할 수도 있습니다. 명시하지 않으면 독립된 새 모델입니다. 단계마다 task, 모델 구조, 입력 전처리, `label_space_id`, 깊이 단위 또는 미래 시간 간격이 같아야 연결됩니다. 예전 체크포인트에 의미 ID가 없으면 자동 추정해서 연결하지 않습니다.

단독 실행도 지원합니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc train --config configs/stages/road_vehicle.json --init-from runs/curriculum_road_surface/stages/road_mixed/best.pt
```

동일 실행을 이어가는 `--resume`과 새 단계의 `--init-from`은 동시에 사용할 수 없습니다. 합성/제한 스텝 진단 provenance는 체크포인트에 기록되고 연결 학습에도 보존됩니다. 진단 모델의 실행 성공은 실제 주행 성능을 뜻하지 않습니다.

## 4. 누수 방지와 망각 확인

첫 단계를 시작하기 전에 **어느 단계의 학습 데이터라도 어느 단계의 검증 데이터와 겹치면 중단**합니다. 같은 영상, `source_image`, `(dataset_id, group_id)`, `origin_id`를 검사합니다. 반복 학습을 위한 동일 학습 출처의 재사용은 허용합니다.

파생 데이터셋은 `dataset_id`가 달라도 원본이 같을 수 있습니다. 모든 버전에 `origin_id: "원본데이터셋:원본주행시퀀스"`를 동일하게 기입하세요. `origin_id`는 프레임 번호가 아닌 **원본 장면/주행 시퀀스 ID**입니다. 이 정보를 제공하지 않고 파일 이름도 바꾸면 코드가 중복을 추론하지 못합니다. 공식 test 세트는 이 설정에 넣지 않고 최종 독립 평가용으로 보존합니다.

각 단계의 최적 체크포인트는 같은 구조·의미를 가진 이전 단계들의 검증 출처에도 다시 평가합니다. `report.json`의 `holdout_metrics`에서 자체 환경 개선과 기존 공개 데이터 성능 하락을 함께 확인하세요. 필요하면 앞 단계 데이터를 replay로 섞습니다. 검증 자료로 설정을 고른 뒤 최종 성능을 보고할 때는 별도 시험 자료를 사용하세요.

## 5. 탐지·깊이·경로 모델 확장

깊이·경로 모델도 동일한 네이티브 체인을 사용합니다. 깊이는 공통 미터 단위/PNG 배율, 경로는 동일 ego 좌표계·미래 시간 간격·속도 단위로 먼저 변환해야 합니다. 한 분할 모델의 가중치를 탐지나 경로 모델에 연결할 수는 없습니다.

탐지 단계 예:

```json
{
  "name": "det_vehicle",
  "config": "stages/detection_vehicle.json",
  "label_space_id": "bdd_detection_10_v1",
  "init_from": "stage:det_public",
  "train_manifests": ["../data/detection_vehicle/train.jsonl"],
  "val_manifests": ["../data/detection_vehicle/val.jsonl"]
}
```

탐지는 기존 YOLO 설정을 사용합니다. 모델명과 `names`의 순서가 같아야 앞선 감사된 탐지 단계의 가중치를 이어갑니다. 위의 manifest 경로는 커리큘럼 JSON 기준입니다. 생략하면 각 dataset.yaml 옆의 `train.jsonl`, `val.jsonl`을 사용하므로 BDD 변환 결과를 바로 연결할 수 있습니다. manifest는 실제 YOLO 이미지 분할과 정확히 일치하고 올바른 `.txt` 정답을 가리켜야 합니다. 검증된 배경 이미지는 빈 label 파일을 명시해야 합니다. 외부 YOLO 사전학습은 첫 단계 설정의 `model`에서 지정하며 명시적 실행 시 Ultralytics가 내려받을 수 있습니다.

YOLO의 조기 종료나 시간 제한으로 요청 epoch보다 적게 수행되면 커리큘럼은 그 단계를 incomplete로 보고 뒤 단계를 자동 진행하지 않습니다. 객체 탐지 역시 이전 단계의 검증 세트를 다시 평가합니다. 데이터셋 이름만 추가해서 주석 변환 없이 학습하는 기능은 아닙니다.
