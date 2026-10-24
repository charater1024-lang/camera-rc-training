# 자율주행 구조 개선 진행 현황

최종 갱신: 2026-10-24

## 현재 상태

검토한 8개 영역 모두에 코드 개선을 반영했다. 이 중 모델 라벨·검출 상태 계약은 코드 반영과 회귀 검사를 완료했고, 나머지 7개는 기반 구현 후 실제 입력 연결 또는 실측 검증이 남아 있다. 전체 자율주행 기능의 완료나 실제 도로 운행 승인을 뜻하지 않는다. 작업량이 서로 달라 단순한 완료율 백분율을 주행 성능처럼 표시하지 않는다.

시뮬레이터 상단의 **개선 진행 현황** 또는 `/#improvement-panel`에서 항목별 반영 내용과 남은 조건을 확인할 수 있다. `/api/improvements`, `/api/state`, `/api/report`에도 동일한 상태가 포함된다. 이 상태는 마지막 검증 시점의 기록이며 자동 실차 검증이나 백그라운드 학습을 실행하지 않는다.

| 항목 | 반영한 개선 | 남은 작업 |
|---|---|---|
| 라벨·검출 상태 | native 모델의 label_space_id, 클래스 의미, 주행 클래스 대조. 검출기 미구동과 빈 검출 구분. 실제 CPU 체크포인트 로딩 시험 | 실제 차량 영상의 정확도 측정 |
| 안전 검사 경로 | 실제 ControlLoop에 시뮬레이터와 같은 ShadowSafetySupervisor 사용. 관측·정책 누락 시 정지. 카메라 프레임 사이에도 나이·제동·이동 객체 재검사. 적색 신호 대기와 고장 latch 구분 | 실제 관측 생산자 및 카메라 replay 장면 연결 |
| 거리·추적·위치 | 미터 단위 세계 좌표 입력을 요구하는 진단용 추적기. ego 이동 변환, 모호한 연관·소실·만료 상태와 오차 증가 | 거리 센서·위치 추정의 실제 입력 및 추적 정확도 검증 |
| 산지·속도·물리 | 평면 가정과 촬영 시각 확인 요구. 기본 평면 허용 범위는 경사 절댓값 0.02, 피치 변화 0.01 rad. 궤적 속도·가속도·곡률·횡가속도·좌표와 속도 일관성 검사 | 해당 범위의 실측 타당성, 3D 센서 기하 및 차량 동역학 |
| 주행 행동 | 방향·연결·폐쇄 차로를 지키는 차로 그래프 경로 탐색. 차체 원형 외곽을 포함한 경계 검사. 신호·우선권·경로 불명확 시 정지 | 실제 지도·차로·신호 생산자, 교차로 행동·다중 객체 예측 계획 |
| 정책 모델 | 차량 한계를 제공하면 오픈 루프 평가에 물리 제약 위반 비율 포함 | 정책 실행 어댑터·폐루프 평가. 학습 및 모델 채택 미실행 |
| 실측·MCU | 수신 시각과 노출 시각 필드 구분. 검증되지 않은 노출 시각 거부. 검증 선언값만으로 실제 출력을 켤 수 없도록 차단 | 실제 촬영 시각, 조향·제동, 독립 MCU 펌웨어 및 하드웨어 결합 시험 |
| 검증 증거 | 모델·보정·차량 설정·코드와 ODD·대상을 고정한 SHA256 증거 감사 | 실제 독립 시험 데이터와 측정 보고서 |

## 세 대상의 진행 경로

`configs/platforms/simulator.json`, `rc_car.json`, `road_vehicle.json`은 대상별 준비 명세다. 실행 가능한 실차 설정으로 간주하면 안 된다. 미측정 차량 치수·제동·조향·센서 항목은 null/빈 목록으로 남겼다.

1. **시뮬레이터:** 합성 차선·교통·지역·속도·안전 고장 회귀 검사를 계속 사용한다. 정답 거리·추적과 단순 2D 차량 운동을 실차 성능으로 바꾸어 해석하지 않는다.
2. **RC카:** 차량 치수·센서·제어 보드를 특정한 뒤 실제 영상 기록, 거리·위치 추정, 제동·조향·독립 정지를 검증한다. RC 저속 설정과 해당 차량의 검증 증거를 묶는다.
3. **실제 차량:** RC 치수나 속도를 재사용하지 않는다. 별도의 센서·차량 인터페이스·동역학·시험 보고서와 운행 조건을 정의하고 검증한다. RC/시뮬레이터 결과를 실차 증거로 승인하지 않는다.

실측을 진행하려면 각 차량의 모델/크기/축거, 카메라 개수·모델, 거리·위치 센서, 제어 보드와 조향·제동 인터페이스, 기록 데이터가 필요하다. 아직 없는 정보를 추정값으로 채우지 않았다.

## 실행 동작 변경과 호환성

- 기본 안전 모드는 `required`다. 기존 카메라·분할 경로에는 검증된 metric occupancy, tracking, localization, road context 생산자가 없으므로 정지를 유지한다. 분할 출력만으로 해당 관측이 있다고 가정하지 않는다.
- 오프라인 기존 제어기 연구는 dry-run에서만 `safety.mode=diagnostic_local_only`로 명시할 수 있다. 이 모드의 상태는 DIAGNOSTIC_ONLY이며 안전 승인으로 표시하지 않는다.
- 새 SafetyObservation은 장면 증거, 장면 기준 차량 상태, 차로·교통 정보, 거리 출처와 기하·시각 정보를 함께 요구한다. 코드의 measured/recorded 표시만으로 센서 정확도가 증명되지는 않는다.
- 실제 serial 실행은 현재 명시적으로 차단된다. 기존 models_validated_for_control/calibration_confirmed 값을 true로 바꾸어도 검증 미완료 실차 출력을 시작하지 않는다.
- native road 모델은 `expected_label_space_id=road_surface_v1`, 주행 클래스 `[0]`, 2클래스가 필요하다. 차선 모델은 `ego_lane_v1`, `[1]`이다. 라벨 공간이 없는 과거 범용 체크포인트를 임의 추정해 로드하지 않는다. 기존 가중치의 메타데이터만 바꿔 검증된 모델로 취급하면 안 된다.
- policy 평가 설정의 `evaluation.vehicle_limits`에 명시적 차량 한계를 제공하면 물리 제약 검사를 함께 수행한다. 값이 없으면 physical_constraints_checked=False다. 아직 모델 학습이나 자동 교체는 수행하지 않았다.

## 핵심 코드

- driving/label_contract.py: 모델 의미 계약
- driving/safety_boundary.py, shadow_safety.py: 제어 경계와 공통 감독기
- driving/behavior.py, route_graph.py: 차로·규칙 검사와 지도 기반 경로 탐색 기준선
- driving/metric_tracking.py: 이미 측정된 metric 입력의 진단용 추적
- driving/trajectory_checks.py: 궤적 물리 일관성 검사
- driving/readiness.py: 운행 조건·대상·파일 정체성에 결합된 측정 증거 감사

추적기는 거리 센서나 검출기를 대신하지 않고, 차로 그래프는 실제 지도를 자동 생성하지 않는다. 새 기반 모듈 중 실제 카메라 생산자가 없는 기능을 활성화한 것처럼 표시하지 않는다.

## 검증 결과

Python 481개 및 UI 18개 통과. 보고서는 `reports/autonomy_improvement_validation.json`에 저장했다. CPU의 합성 체크포인트 로딩·추론, 시계열 모델 기하/시간/수치 계약, 안전 결합, 추적, 차로, 규칙, 지역·속도·성능 표시와 HTTP/UI 회귀 검사를 수행했다.

모델 학습 작업, 가중치 다운로드·자동 채택, 실제 카메라 촬영, serial 출력, 실제 차량 운행은 실행하지 않았다. 테스트 전용 CPU 라이브러리는 ignored runs/simulator-dev-tools에 격리했다. 이 환경은 대상 GPU/차량의 지연 검증 환경이 아니다.

## 실측 증거 감사 사용

프로젝트 루트에서 PYTHONPATH에 src를 추가하고 다음 명령을 사용한다.

```text
python scripts/audit_autonomy_readiness.py configs/autonomy_validation.example.json --output runs/NEW-readiness-audit.json
```

출력 파일이 이미 있으면 덮어쓰지 않는다. 예제는 미측정 상태라 audit_passed=False, 종료 코드 2가 정상이다. 실제 증거가 모두 일치하면 종료 코드 0이지만 actuation_authorized는 여전히 false다. 해시는 파일의 일치 여부를 확인하며 측정의 진실성이나 차량 안전성을 인증하지 않는다.

보고서마다 check, passed, measurement_source=measured, platform, odd, artifact_hashes, independent_test_set을 요구한다. manifest는 각 보고서의 경로와 SHA256을 고정한다. simulator/rc_car/road_vehicle 대상이 달라지거나 차량·보정·모델·ODD가 달라지면 같은 보고서를 통과 증거로 재사용할 수 없다.

## GitHub 공개 전 추가 확인 · 2026-10-24

전체 저장소와 BEVFormer wrapper 검사에서 Python 2,028개 통과·2개 생략, UI 18개 통과를 확인했다. 선택적 Ultralytics와 Windows 심볼릭 링크 제약을 생략 사유로 기록했다. 기존 진단 도구를 명시적 diagnostic_local_only 모드와 도로 2클래스 계약으로 이행했고, export 시 label_space_id·labels를 보존하도록 보완했다. 이는 위 481개 범위 검사의 후속 전체 소프트웨어 검사이며 실제 차량 검증은 아니다. 상세 환경·소스 해시·재검사 경과는 reports/github_publication_validation_2026-10-24.json에 보존한다.
