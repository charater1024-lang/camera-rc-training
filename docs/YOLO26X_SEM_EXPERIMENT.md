# YOLO26x 도로 분할 비교 실험

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

이 실험은 기존 TinyUNet을 YOLO26x-sem과 비교한다. 일반 YOLO26x 객체 검출기가 아니라 픽셀마다 도로 확률을 출력하는 **semantic segmentation** 모델이다. 자동차·보행자 종류별 검출, 거리 추정, 여러 카메라의 공간 통합, 경로 계획·제어 성능은 이 실험에서 검증하지 않는다.

공식 Cityscapes 사전학습 가중치는 [Ultralytics v8.4.0 릴리스](https://github.com/ultralytics/assets/releases/tag/v8.4.0)의 `yolo26x-sem.pt`이다. SHA256은 `eae8ff0d95bf96863c59c40bade61037870c7a39f2c53a8c49fd1b5022ed331d`이다. [공식 semantic 학습 문서](https://docs.ultralytics.com/tasks/semantic/)와 설치된 패키지의 API를 함께 확인했다.

## 데이터와 학습

- 학습: BDD100K 70,000장 + Virtual KITTI 2 28,920장 = **98,920장**.
- 검증: BDD 10,000장 + VK2 8,940장 + CamVid 124장 + KITTI 58장 = **19,122장**.
- 클래스: `0=road_surface`, `1=not_road_surface`, `255=ignore`.
- 공개 RoadObstacle21 30장과 RoadAnomaly21 10장은 학습에서 제외한다.
- 원본을 바꾸지 않고 PNG 마스크와 이미지 목록을 구성한다. 원본 RGB·마스크·manifest의 해시와 장면 식별자를 기록하고 train/val 중복을 검사한다.
- 전체 1 epoch가 첫 비교 단계다. 384 입력, CUDA AMP, AdamW, 고정 CE 클래스 가중치 `[1,2]`를 사용한다. 실제 배치는 별도 용량 측정 후 선택한다.
- TinyUNet과 새 모델의 사전학습, 손실, 증강, 최적화 방식은 다르다. 따라서 결과는 두 완성 모델의 비교이며, 차이를 모델 크기만의 효과로 해석하지 않는다.

`prepare_yolo_semantic.py`의 제한 옵션은 진단용이며, 본 학습 runner는 제한된 export를 거부한다. `.complete`의 해시와 전체 데이터 개수를 확인해야 본 학습이 시작된다. `training_report.json`의 `training_complete=true`와 체크포인트 해시가 있어야 평가가 진행된다.

## 정확도 비교

공통 조건은 RGB PIL bilinear 192×320, GT nearest 192×320이다. 두 모델 모두 이 해상도에서 native confusion matrix와 도로 IoU를 계산한다. 도로를 놓치는 경우와 비도로를 도로로 판단하는 경우를 함께 기록한다.

YOLO의 낮은 해상도 logits를 입력 해상도로 bilinear 복원한 다음 softmax의 0번 채널을 도로 확률로 사용한다. 공개 장애물 평가에서는 이 확률을 **원본 GT 해상도**로 복원한다. 작은 장애물이 GT 축소 과정에서 사라지지 않도록 한다.

장애물 오인율은 `GT=1인 장애물 픽셀 중 도로 확률이 0.65 이상인 픽셀 비율`이다. 일반 도로 검증의 `GT=비도로인 픽셀 중 argmax가 도로인 비율`과는 분모와 판정 규칙이 다르다. 둘을 섞지 않는다. 각 이미지별 결과도 저장한다.

384×640 입력 평가는 별도 조건으로 기록한다. 이때 일반 검증 GT는 공통 192×320, 공개 장애물 GT는 원본 크기다. 기존 192×320 결과와 비교할 때 해상도 증가 효과를 포함한다고 표시한다.

이 공개 40장은 이미 여러 후보를 확인한 개발용 회귀 평가 자료다. 처음 보는 비공개 시험이나 사고 확률 측정이 아니다. VK2의 여러 시점·날씨 변형도 독립된 실제 주행 횟수로 세지 않는다.

## 실행 성능

학습 완료 체크포인트로 RTX 3080 Laptop GPU에서 가중치·입력을 FP16으로 변환해 측정한다. 같은 실제 이미지로 TinyUNet도 측정하며, 두 모델은 순차 로딩한다. 정확도 평가는 FP32 가중치와 AMP를 사용하므로 수치 연산 경로가 완전히 같지는 않다. 배포 엔진을 확정한 뒤에는 그 엔진의 출력으로 정확도를 재검증해야 한다.

- 입력: 192×320 / 384×640.
- 한 주기: 이미지 1장 / 공유 모델 하나로 이미지 8장 순차 처리 / 8장을 묶은 배치 처리.
- 모델 연산: GPU에 이미 있는 입력부터 도로 확률 계산까지.
- 호스트 포함: 메모리에 준비한 PIL 이미지의 resize, CPU→GPU 전송, 추론, 도로 확률의 GPU→CPU 복사까지.
- warmup 10회 제외, 50주기의 p50/p95/p99·평균·최대 지연과 메모리를 기록한다.

3GiB allocator 제한은 4GB 기기의 다른 메모리 사용을 고려한 후보 조건이다. OOM이면 7GiB 조건도 검사한다. 이 제한은 GPU 처리 능력, 메모리 대역폭, CUDA context 및 다른 프로세스 점유를 재현하지 않는다. RTX 3080 측정값을 GTX 1650 또는 Jetson 성능으로 표시하지 않는다.

8장 순차·배치 처리는 8개의 동기화된 실제 카메라나 BEV 통합 실험이 아니다. 카메라 노출·USB 전송·프레임 대기·시간 동기화·계획·제어·제동 시간은 포함하지 않는다. 80km/h 운용 가능 여부는 이 결과만으로 판정하지 않는다.

## 산출물

실행 폴더의 `status.json`에서 진행 상태를 확인한다. `training/training_report.json`은 실제 optimizer update와 AMP skip, GPU 메모리, 체크포인트 해시를 기록한다. `evaluation/comparison.json`은 기존 모델과의 공통 조건 비교이며, `benchmarks/*.json`은 실행 성능이다. 실패·회귀·OOM도 삭제하거나 성공으로 바꾸지 않는다.

`run_yolo_semantic_experiment.py`는 다운로드 대기 상태인 기존 학습 큐의 프로세스 ID와 자식 작업을 확인한 뒤 잠시 suspend하고, 자식 작업 종료가 확인된 성공·오류 종료 시 finally에서 resume한다. 자식 종료가 확인되지 않으면 `recovery_required`와 해당 PID를 남겨 GPU 작업이 겹치지 않도록 한다. 다운로드 자체는 계속된다. 기존 큐가 이미 학습 중이면 끼어들지 않고 종료한다. 해당 큐가 없는 다른 환경에서는 개별 prepare → train → evaluate → benchmark 스크립트를 사용한다.
