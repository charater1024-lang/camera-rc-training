# 공개 데이터 학습 준비

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

현재 학습기는 Windows의 RTX 3080 Laptop 16GB / RAM 32GB이고, Ubuntu MSI는 추후 추론·제어용입니다. 데이터가 아직 없으면 다운로드를 직접 완료한 다음 아래 준비 명령을 실행합니다. 이 도구는 다운로드·학습·차량 구동을 시작하지 않습니다.

## 받을 자료와 폴더 구조

- BDD100K: **100K 이미지의 train/val**과 그에 대응하는 **drivable raw ID PNG**가 필요합니다. 10K semantic segmentation 이미지나 RGB 시각화 마스크는 이 입력에 사용하지 않습니다.
- 선택 사항인 Cityscapes: 기본 `leftImg8bit`와 `gtFine`의 train/val이 필요합니다. `*_gtFine_labelIds.png`를 사용하며 `labelTrainIds`, 컬러 시각화, test, train_extra는 사용하지 않습니다.
- 각 제공기관에서 이용 조건과 접근 절차를 확인하고 내려받으세요. 준비 프로그램은 파일 구조와 ID를 검사하지만 배포본의 진위·이용허락까지 인증하지 않습니다.

**원본 인코딩을 반드시 명시합니다.** `bdd100k_drivable_maps.zip`의 `drivable_maps/labels`와 확인한 HF 미러는 `0=배경, 1=direct, 2=alternative`이므로 `--bdd-encoding legacy_2018`을 사용합니다. `current_2021`은 `0=direct, 1=alternative, 2=배경`인 신형 원본에만 사용합니다. 파일명·값 범위만으로 구별할 수 없습니다. [출처와 실제 마스크 검증](BDD100K_SOURCES.md)에 근거를 보관했습니다.

```text
data/raw/bdd/images/100k/
  train/<clip>.jpg
  val/<different-clip>.jpg
data/raw/bdd/drivable_maps/
  train/<clip>_drivable_id.png
  val/<different-clip>_drivable_id.png
data/raw/cityscapes/leftImg8bit/
  train/<city>/<city>_<sequence>_<frame>_leftImg8bit.png
  val/<city>/<city>_<sequence>_<frame>_leftImg8bit.png
data/raw/cityscapes/gtFine/
  train/<city>/<city>_<sequence>_<frame>_gtFine_labelIds.png
  val/<city>/<city>_<sequence>_<frame>_gtFine_labelIds.png
```

위 BDD 마스크 루트 이름은 예시입니다. 실제 압축 해제본에서 **train과 val을 직접 포함하는 경로**를 지정하세요. 마스크 이름은 `<clip>.png` 형식도 지원합니다. 데이터셋 공식 분할을 그대로 사용하며 같은 클립·시퀀스를 train과 val에 나누지 않습니다.

## BDD만으로 시작

프로젝트 루트 `.`의 PowerShell에서 실행합니다. 경로는 실제 다운로드 위치로 변경하세요.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_public_training.py `
  --bdd-images ".\data\raw\bdd\images\100k" `
  --bdd-masks ".\data\raw\bdd\drivable_maps" `
  --bdd-encoding current_2021 `
  --output data/public_v1
```

위 예시의 인코딩은 신형 마스크 기준입니다. 이번에 받은 2018년 Archive 파일에는 `current_2021` 대신 `legacy_2018`을 지정합니다. 전처리는 `--workers 4`를 추가하면 최대 4개 스레드로 수행하며, 준비 중 이미지와 마스크는 한정된 개수만 메모리에 올립니다.

BDD 한 단계 curriculum이 생성됩니다. 자체 촬영 데이터나 Cityscapes가 없어도 준비와 학습을 시작할 수 있습니다.

BDD 단독 단계는 매 epoch에 각 학습 이미지를 한 번씩 방문합니다. 복원 추출은 여러 데이터셋의 비중을 맞추는 혼합 단계에만 사용합니다.

Cityscapes도 보유했다면 **같은 준비 명령에** 다음 두 인수를 추가합니다. 이미 `public_v1`을 만들었다면 `--output data/public_v2`처럼 새 경로를 지정합니다.

```text
--cityscapes-images ".\data\raw\cityscapes\leftImg8bit"
--cityscapes-labels ".\data\raw\cityscapes\gtFine"
```

이때 BDD 사전학습 → BDD+Cityscapes 혼합학습의 두 단계가 생성되며, 두 번째 단계는 첫 번째 단계의 최적 가중치를 전달받습니다. 두 자료의 샘플 수와 무관하게 혼합 단계의 기대 추출 비중은 각각 50%이고, 두 검증셋의 성능을 따로 평가합니다.

## 결과와 실제 학습

`data/public_v1` 아래에 다음 파일이 생성됩니다.

- `report.json`: 원본 경로, 변환 규칙 해시, 샘플 수, 모든 샘플 해독 및 분할 검사 결과
- `raw/`: 원본 이미지·라벨을 참조하는 manifest
- `road/`: 변환된 마스크와 학습용 manifest
- `label_maps/`: 사용한 ID 변환 규칙 사본
- `stages/road_public.json`, 선택적으로 `stages/road_mixed.json`
- `curriculum.json`: 실행 가능한 단계 구성

원본 이미지·라벨은 복사하거나 수정하지 않습니다. 새 이진 마스크만 추가되므로 변환용 여유 공간이 필요합니다. 검사는 모든 이미지를 한 장씩 읽기 때문에 전체 데이터에서는 시간이 걸립니다. 원본 이미지 경로는 절대 경로로 유지되며, **원본 자료를 이동하면 다시 준비해야 합니다.**

출력 폴더가 이미 있으면 덮어쓰지 않습니다. 시작 후 실패하면 `.incomplete`와 실패한 `report.json`을 남깁니다. 이 경우 입력 문제를 해결하고 새 출력 폴더로 다시 실행하세요. `.incomplete`를 임의로 지워 준비 완료로 간주하지 마세요.

준비 완료 후 다음 명령은 학습 없이 전체 단계와 분할을 다시 확인합니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc curriculum --config data/public_v1/curriculum.json
```

검사 결과를 확인한 뒤 실제 학습은 별도 명령으로 시작합니다.

```powershell
.\.venv\Scripts\python.exe -m camera_rc curriculum --config data/public_v1/curriculum.json --execute
```

생성 기본값은 TinyUNet width 16, 192×320, 두 클래스, batch 2, gradient accumulation 4, worker 0, CUDA AMP입니다. BDD 20 epoch, 혼합 15 epoch이며, 준비 명령의 `--epochs`와 `--mixed-epochs`로 지정할 수 있습니다. 기본 장치는 CUDA이므로 GPU가 없을 때 느린 CPU 학습으로 조용히 전환하지 않습니다. 소규모 진단용으로만 `--device cpu`를 선택할 수 있습니다.

학습 결과는 준비 폴더의 `training/stages/<stage>/`에 저장됩니다. 중단된 실행의 재개는 기존 curriculum 명령의 `--resume-run` 규칙을 따릅니다. 성능을 확인하지 않은 모델을 `runs/deployment`에 자동 배치하지 않습니다.

BDD의 direct·alternative 영역과 Cityscapes의 road를 0으로 통합합니다. 다른 평가 클래스는 1, 원래 무시 대상은 255입니다. **도로 영역 분류는 장애물 부재·통행 허가·충돌 회피를 의미하지 않습니다.** 이 단계는 인식 모델의 공개 데이터 학습이며, 카메라 장착 환경의 자체 데이터 보정과 실제 장치 지연·제어 검증은 별도로 필요합니다.
