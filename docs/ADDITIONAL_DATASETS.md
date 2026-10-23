# 추가 공개 데이터셋 조사와 적용 순서

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

확인일: 2026-10-21. 대상은 RTX 3080 Mobile 16GB에서 학습하고, Ubuntu 24.04의 MSI 노트북 GPU 4GB에서 카메라 인식을 실행하는 연구 구성입니다. 아래는 **24개 추가 후보의 조사 결과**이며, 모든 후보의 다운로드 완료·학습 완료·코드 지원을 뜻하지 않습니다. 이번 작업에서는 CamVid·KITTI Road를 추가 학습하고 RoadObstacle21·RoadAnomaly21 공개 GT를 독립 평가했습니다. 실제 수치와 운용 미승인 판정은 [추가 학습 결과](ADDITIONAL_TRAINING.md)에 있습니다. BDD100K는 이미 사용하는 기준 데이터이므로 추가 후보 수에서 제외했습니다.

우선순위는 데이터 개수보다 목표 환경의 부족한 사례를 보완하는 정도입니다. 현재 전방 도로 인식에는 CamVid/KITTI Road/ACDC를 먼저 연결하고, 추가로 Virtual KITTI 2 또는 A2D2의 일부를 검토할 만합니다. 작은 미지 장애물은 RoadObstacle21 공개 validation을 학습과 분리해 검사합니다. 이후 실제 장착 높이·렌즈·노면에서 수집한 별도 주행으로 검증해야 합니다.

## 표를 읽는 방법

- **공개 링크**: 공식 배포자가 직접 안내한 공개 경로입니다. 무료 데이터도 이용 조건은 존재합니다.
- **가입/신청**: 계정·약관 동의 또는 접근 심사가 필요합니다. 이를 우회하는 미러는 제안하지 않습니다.
- **접근 미확인**: 공식 자료는 확인했지만 현재 배포 링크나 최종 다운로드 약관을 이 조사에서 확인하지 못했습니다.
- 크기는 공식 페이지 표기 또는 실제 파일 메타데이터입니다. **전체 센서, RGB만, 압축 아카이브, 압축 해제 후 크기를 서로 바꾸어 읽으면 안 됩니다.** 미확인 항목에는 임의의 GB 추정치를 넣지 않았습니다.
- 라이선스 열은 원문 확인을 위한 요약입니다. 데이터 라이선스와 도구 코드 라이선스는 별개이며, 서로 다른 데이터셋의 재배포 조건이 자동 통합되지 않습니다.

## 도로 분할·야간·악천후

| 후보 / 공식 출처 | 내용과 규모 | 저장량 / 접근 / 이용 조건 | 이 프로젝트에 적용할 때 |
|---|---|---|---|
| 1. [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) | 주행 영상의 의미 분할. 이번에 확보한 SegNet 배포는 **701 RGB/mask 쌍**, 11 의미 클래스+void | [고정 SegNet commit](https://github.com/alexgkendall/SegNet-Tutorial/tree/bb68b64e72cb004b3ed927fee0f5624dd8097498/CamVid)의 **1,402개 파일 합 186,792,534 bytes**(약186.8 MB). ZIP 용량이 아닌 선택 원본 파일 합. 공개 GitHub 경로; 원 데이터 조건과 출처 보존 | ID mask의 `3→0 road`, `11→255 ignore`, 나머지 `→1 non-road`. 원본 영상별로 **406 train / 124 val / 171 reserve**로 새로 분리. 표준 CamVid split 점수와 직접 비교하지 않음 |
| 2. [KITTI Road](https://www.cvlibs.net/datasets/kitti/eval_road.php) | 도로/차선 영역. 전방 도시 주행 | `data_road.zip` **470,992,343 bytes**(약471 MB, 실제 S3 목록 확인). [공식 AWS registry](https://registry.opendata.aws/kitti/)가 `avg-kitti` S3의 **AWS 계정 없는 접근**을 명시. [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/) | 도로 인식 추가 학습 후보. RGB GT 인코딩·평가 유효 영역을 해석해야 함. road와 lane 정답을 혼동하지 않음 |
| 3. [ACDC](https://acdc.vision.ee.ethz.ch/) | 안개·비·눈·야간 및 대응 정상 조건, Cityscapes 계열 분할. 악천후 일반화 후보 | 공식 사이트는 JavaScript 기반. **이번 확인에서는 현재 다운로드 접근·가입 필요 여부·최종 약관·패키지 크기를 확인하지 못함**. 버전별 이미지/라벨 구성을 원 배포에서 재확인 후 도입 | 대응 정상/악천후 이미지와 같은 주행은 동일 그룹으로 취급. invalid/uncertain 영역을 background로 바꾸지 않음. 현재 확보·학습된 데이터로 세지 않음 |
| 4. [A2D2](https://a2d2-dataset.github.io/) | 의미 분할 41,277프레임, 그중 12,497프레임에 3D box; 6카메라·5LiDAR | 전체 용량 미확인. [공식 AWS registry](https://registry.opendata.aws/aev-a2d2/)가 계정 없는 원본 tar와 개별 추출 파일 S3 배포를 모두 안내. **CC BY-ND 4.0** | 전방 RGB+semantic 일부만 선택 가능. LiDAR를 함께 학습할 필요 없음. 원본 컬러 라벨→공통 ID 변환 필요. 변환 라벨의 외부 재배포는 ND 조건 확인 |
| 5. [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) | 합성 주행 5장소×10변형, 좌우 카메라; 날씨·조명·시점 변화 | 공개 링크. RGB tar **7.01 GB** + class segmentation tar **969 MB**. depth 별도 **7.58 GB**. **CC BY-NC-SA 3.0** | RGB+semantic 약8 GB의 다운로드로 실험 가능. `colors.txt`의 RGB 팔레트 사용. 장면의 날씨/시점 변형을 같은 분할에 넣어 누출 방지. 실사 보완용 |
| 6. [WildDash 2](https://wilddash.cc/) | 공개 4,256프레임; 어려운 조명·날씨·화질·시점, 확장 분할 클래스 | 크기 미확인. 가입·승인 필요. 이미지 CC BY-NC 4.0 및 [추가 도로 안전 연구 조건](https://www.wilddash.cc/license/wilddash); 메타데이터 재배포 제한 | Cityscapes ID와 추가 클래스를 구분. 776 benchmark 프레임은 학습용으로 사용하지 않음. 접근 준비 후 강건성 평가/학습 |
| 7. [IDD](https://idd.insaan.iiit.ac.in/) | 182 주행 시퀀스의 10,000이미지, 34클래스. 인도의 비정형 도로 | 크기 미확인. 다운로드 로그인 필요; 다운로드 시 표시되는 데이터 약관 확인 | 도로 경계가 모호한 환경 보완. 34클래스 및 계층을 해석하고, 주행 가능 노면과 단순 평면을 구별 |
| 8. [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) | 다양한 국가·카메라·도로의 조밀한 의미/인스턴스 분할 | 현재 공식 페이지 로그인 필요. 버전별 클래스와 크기, 데이터 약관을 다운로드 페이지에서 확인 | 일반화 후보. 버전별 label taxonomy 확인 필요. Mapillary 일반 이미지·Object Dataset·Traffic Sign Dataset의 약관을 Vistas 약관으로 대신하지 않음 |
| 9. [Dark Zurich / MGCDA](https://github.com/sakaridis/MGCDA) | 주간·황혼·야간 8,779이미지. 대부분 비라벨 적응 데이터; 야간 val 50 / test 151 | 공식 저자 저장소에서 ETH 배포 링크 제공. 크기·최종 데이터 약관 미확인; test 정답 비공개 | 야간 적응 연구 후보. 비라벨 영상을 지도학습 road GT처럼 사용하면 안 됨. 공개 검증 라벨은 독립 평가에 보존 |
| 10. [SYNTHIA](https://synthia-dataset.net/) | 합성 도시의 semantic·depth·다중 시점. [원 논문](https://synthia-dataset.net/wp-content/uploads/2016/06/gros_cvpr16-1.pdf)의 Rand와 Cityscapes 대응판은 서로 다른 구성 | 공식 배포 접근·현재 크기·다운로드 조건 미확인 | 합성 사전학습 후보. variant별 class ID·카메라·날씨와 split 확인 후 연결. 당장 대량 확보하는 우선순위는 낮음 |

Virtual KITTI 2는 semantic만 필요하면 optical flow와 scene flow를 받을 필요가 없습니다. A2D2 공식 AWS 배포는 `aev-autonomous-driving-dataset`에 원본 tar, `audi-autonomous-driving-dataset`에 개별 파일을 제공합니다. 이는 배포자가 명시한 공개 경로이며 RGB와 라벨을 함께 선택하는 데 유용합니다.

## 미지 장애물·도로 위 작은 물체

| 후보 / 공식 출처 | 내용과 규모 | 저장량 / 접근 / 이용 조건 | 이 프로젝트에 적용할 때 |
|---|---|---|---|
| 11. [RoadObstacle21](https://segmentmeifyoucan.com/datasets) | 도로 ROI의 작은 장애물. 공식 test 327 + 공개 라벨 validation 30 | [공식 Zenodo](https://zenodo.org/records/5281633) ZIP **207,582,233 bytes**(약207.6 MB). 무가입 공개; **CC BY 4.0** | **독립 평가 우선**. 공개 GT 30개로 obstacle→road 오인율 측정. test 이미지가 있어도 test GT가 공개되었다는 뜻은 아님 |
| 12. [RoadAnomaly21](https://segmentmeifyoucan.com/datasets) | 동물·미지 차량 등 장면 어디든 나타나는 이상 물체. test 100 + 공개 라벨 validation 10 | [공식 Zenodo](https://zenodo.org/records/5270237) ZIP **52,625,353 bytes**(약52.6 MB). 무가입 공개. **이미지별 라이선스 혼합** | **독립 평가 후보**. 정상 픽셀은 road GT가 아님. 공개 val10의 attribution 아래 참조. 전체 데이터에 하나의 CC 라이선스를 붙이지 않음 |
| 13. [Lost And Found](https://www.6d-vision.com/lostandfounddataset) | 실제 도로의 작은 장애물. 공식 benchmark loader에서 train 1,036 / test 1,203 프레임 | 현재 원본 접근 미확인. [SegmentMeIfYouCan](https://segmentmeifyoucan.com/datasets)이 안내하는 DHBW 미러는 조사 시 404. 크기·최종 이용 조건 미확인 | 원본 확보 후 train만 장애물 학습 검토. 라벨 0(ignore), 1(road), 2 이상(anomaly)의 규칙을 해당 release와 대조. 평가 분할 보존 |
| 14. [Fishyscapes](https://fishyscapes.com/) | Lost And Found 실제 영상 및 합성 이상 물체 기반의 분할 불확실성 평가 | 원본 이미지별 권리·접근 조건을 별도 적용. 현재 배포 패키지 크기·전체 다운로드 경로 미확인 | OOD 평가 후보. anomaly mask는 완전한 도로 분할 GT가 아님. Cityscapes 등 원본 데이터와 중복·권리 관계 확인 |

### RoadObstacle21/RoadAnomaly21 공개 GT 확인 결과

공식 배포 ZIP의 중앙 디렉터리와 [공식 loader](https://github.com/SegmentMeIfYouCan/road-anomaly-benchmark/blob/master/road_anomaly_benchmark/datasets/tracks.py)를 대조했습니다. 이것은 원본 파일 구조 확인이며 모델의 평가 결과가 아닙니다.

| 항목 | RoadObstacle21 | RoadAnomaly21 |
|---|---|---|
| ZIP 파일 | `dataset_ObstacleTrack.zip` | `dataset_AnomalyTrack.zip` |
| 원본 MD5 | `895fb36d18765482cc291f69e63d6da6` | `231bf79ed58924bcd33d9cbe22e61076` |
| 공개 integer GT | 30개. `validation_1`~`18`, `validation_20`, `validation_30`~`40` | 10개. `validation0000`~`validation0009` |
| ZIP 내 실제 이미지 수 | **442개 WebP**. 공식 benchmark는 validation 30 / test 327이며, 아카이브에는 이외 추가 조건 이미지도 포함 | **110개 WebP**. validation 10 / test 100 |
| 라벨 의미 | `0=road`, `1=obstacle`, `255=ignore` | `0=normal`, `1=anomaly`, `255=ignore` |
| 공개 test GT | 없음 | 없음 |
| 비교 가능한 내부 지표 | GT obstacle 픽셀 중 road로 예측된 비율; 유효 road 유지율도 함께 | GT anomaly 픽셀 중 road로 예측된 비율. 정상 영역 전체의 road 정확도는 계산 불가 |

라벨은 컬러 시각화가 아닌 integer GT를 읽습니다. 공개 validation을 이 프로젝트의 독립 holdout으로 고정한다면 학습·early stopping·threshold 탐색에서 제외합니다. 반복 결과를 보고 threshold나 모델을 고르면 그 시점부터 개발용 검증 자료가 되므로 최종 성능은 새 holdout으로 다시 확인해야 합니다. 공식 test 성능이나 실제 충돌 확률로 해석하지 않습니다.

작은 장애물은 192×320 등의 낮은 입력 해상도에서 소실될 수 있습니다. 동일한 전처리·해상도·threshold로 기준 모델과 새 모델을 비교하고, 예측을 원본 GT 크기로 맞추어 평가한 방식 및 ignore 처리를 기록합니다. 전체 픽셀 평균에 묻히지 않도록 이미지별 오인율과 장애물 크기별 결과도 확인하는 것이 좋습니다.

RoadAnomaly의 Zenodo metadata `other-at`는 단일 라이선스 이름이 아닙니다. [공식 ZIP](https://zenodo.org/records/5270237/files/dataset_AnomalyTrack.zip?download=1) 내부 `dataset_AnomalyTrack/LICENSE.txt`에 개별 이미지 출처와 조건이 있습니다. 공개 validation 10개의 목록은 다음과 같습니다. 원본 LICENSE를 이미지와 함께 보관하고, 전체 test 묶음에도 동일 조건을 가정하지 마세요.

| validation 이미지 | 원 저자 / LICENSE에 기재된 조건 | 원본 출처 |
|---|---|---|
| `validation0000` | Don DeBold / CC BY 2.0 | [Flickr](https://www.flickr.com/photos/ddebold/4591182738) |
| `validation0001` | Entropy1963 / public domain | [Wikimedia](https://commons.wikimedia.org/wiki/File:Giraffe_crossing_the_road_in_Kruger_National_Park.jpg) |
| `validation0002` | Richard Rhee / CC BY-SA 2.0 | [Flickr](https://www.flickr.com/photos/29916323@N07/10702374353/) |
| `validation0003` | Radosław Drożdżewski / CC BY-SA 4.0 | [Wikimedia](https://commons.wikimedia.org/wiki/File:Mannheim_City_Airport_-_Bell_206_-_D-HONY_-_2016-06-26_21-29-29.jpg) |
| `validation0004` | Mario Hains / CC BY-SA 3.0 | [Wikimedia](https://commons.wikimedia.org/wiki/File:Trailer_ride_for_horses_-_panoramio.jpg) |
| `validation0005` | robert wade / CC BY-SA 2.0 | [Wikimedia](https://commons.wikimedia.org/wiki/File:Telephone_Box,_Whalley_Road,_Sabden,_Lancashire_-_geograph.org.uk_-_2030892.jpg) |
| `validation0006` | Ian D. Keating / CC BY 2.0 | [Wikimedia](https://commons.wikimedia.org/wiki/File:Three-toed_sloth_crossing_road_in_Costa_Rica.jpg) |
| `validation0007` | Дмитрий Лобакин, www.volganet.ru / CC BY-SA 3.0 | [Wikimedia](https://commons.wikimedia.org/wiki/File:Tank_T-34_on_Mira_street_(Volgograd).JPG) |
| `validation0008` | Heribert Bechen / CC BY-SA 2.0 | [Flickr](https://www.flickr.com/photos/hb1248/36843015490) |
| `validation0009` | Ildar Sagdejev / CC BY-SA 1.0, 2.0, 2.5, 3.0, 4.0 여러 버전이 기재됨 | [Wikimedia](https://commons.wikimedia.org/wiki/File:2003-08-16_Wild_turkeys_crossing_the_road_in_Wyoming.jpg) |

## 객체 탐지·다중 카메라

| 후보 / 공식 출처 | 내용과 규모 | 저장량 / 접근 / 이용 조건 | 이 프로젝트에 적용할 때 |
|---|---|---|---|
| 15. [COCO 2017](https://cocodataset.org/#download) | 일반 객체 80클래스. train 약118k / val 5k 이미지 | [공식 다운로드 표](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm): train **18 GB**, val **1 GB**, train/val annotation **241 MB**. 공개 링크. 이미지별 권리와 annotation 약관 구별 | 사람·자전거·차량 등 검출기 사전학습. COCO JSON→선택 detector 형식 변환. 라벨 없는 배경을 road 정답으로 쓰지 않음 |
| 16. [EuroCity Persons](https://eurocity-dataset.tudelft.nl/) | [원 논문](https://arxiv.org/abs/1805.07193): 47,300장 이상, 사람/탑승자 238,200개 이상 | 크기 미확인. 현행 사이트는 기관 이메일·소속 증빙·수동 심사. 기존 계정도 재가입 필요. 데이터별 승인 약관 확인 | 보행자·자전거 검출 보완. Detection/2.5D/Dense Pose가 별도. 현재 detection benchmark는 종료됐지만 데이터 접근 신청은 가능 |
| 17. [nuImages](https://www.nuscenes.org/nuimages) | 93k annotated images; 과거·미래 포함 전체 약1.2M. 2D box·instance·surface masks | 계정·약관 동의, 비상업 사용. 크기 미확인. [공식 devkit](https://github.com/nutonomy/nuscenes-devkit)상 **metadata+samples 필수, sweeps 선택** | 도로 surface와 객체를 활용하는 유력 후속 후보. JSON/mask 변환 필요. 비라벨 과거·미래 이미지를 정답 있는 93k와 혼동하지 않음 |
| 18. [WoodScape](https://github.com/valeoai/WoodScape) | 전·후·좌·우 어안. 공개 8.2k 이미지+8.2k previous; 1.8k heldout. semantic/instance/boxes/motion/calibration | 공식 저장소가 Google Drive 배포 안내. 크기 미확인. **데이터 proprietary 약관, 코드 MIT** | 광각·측후방 시점에 적합. semantic ID GT 사용; RGB 시각화와 구별. 원래 10 semantic classes 및 void를 매핑. 어안 보정·카메라 모델 필요 |
| 19. [nuScenes](https://www.nuscenes.org/nuscenes) | 1,000개 20초 scene; 6카메라+LiDAR+Radar, 3D box·pose·calibration | 계정·약관 동의, 비상업 사용. 이번 조사에서 현재 전체/카메라 전용 패키지 크기 미확인 | camera-only 3D 연구가 가능하지만 2D 도로 mask와 다른 작업. LiDAR panoptic/segmentation을 이미지 GT로 그대로 사용할 수 없음 |
| 20. [Argoverse 2 Sensor](https://argoverse.github.io/user-guide/datasets/sensor.html) | 1,000 logs; ring 7 + stereo 2카메라, 3D box·지도·pose | [공식 안내](https://www.argoverse.org/av2.html): 약 **1 TB**, 전체 센서 구성. 공개 다운로드. **CC BY-NC-SA 4.0** | 후속 다중 시점/3D 검출 연구. RGB만의 용량이 1 TB라는 뜻은 아님. camera/annotation/calibration 단위 선택 검토; 도로 픽셀 분할의 직접 대체재 아님 |
| 21. [Waymo Open Dataset](https://waymo.com/open/) | Perception: 여러 카메라와 2D/3D box, 일부 camera segmentation. End-to-End는 별도 구성 | [공식 다운로드](https://waymo.com/open/download/): Google 계정·약관 동의. 작업·버전·modality별 크기가 다름. 비상업 등 데이터 약관 적용 | Perception modular tables에서 필요한 camera/label 선택 검토. End-to-End trajectory를 RC 제어로 그대로 이전할 수 없음. 전체 BEV 모델의 4GB 실시간 적합성은 별도 측정 |

## 깊이·시간축·시뮬레이션

| 후보 / 공식 출처 | 내용과 규모 | 저장량 / 접근 / 이용 조건 | 이 프로젝트에 적용할 때 |
|---|---|---|---|
| 22. [KITTI Depth](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php) | 93k 이상 depth maps와 대응 raw 이미지. monocular depth / completion은 별도 평가 | depth GT 다운로드 **14 GB**, sparse LiDAR **5 GB**, selected val/test **2 GB**. **RGB는 raw에서 별도**. [CC BY-NC-SA 3.0](https://www.cvlibs.net/datasets/kitti/) | 단안 깊이 학습 가능. 학습 정답이 LiDAR에서 유래해도 차량 추론에 LiDAR가 필수인 것은 아님. 기존 depth adapter의 유효 픽셀·단위·매칭 확인 |
| 23. [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/) | 동일 경로 100회 이상 반복, 1년간 계절·날씨·조명. localization/시간축 연구 | 등록 필요, **CC BY-NC-SA 4.0**. 전체/선택 stream 저장량 미확인 | 한 주행·한 카메라부터 선택. 기본 데이터에 모든 픽셀의 road GT가 있는 것은 아님. 대규모 비라벨 적응이나 VIO 연구를 분할 모델 학습과 구별 |
| 24. [CARLA](https://carla.org/) | 다운로드 완료형 학습셋이 아닌 시뮬레이터. [RGB·depth·semantic 센서](https://carla.readthedocs.io/en/latest/ref_sensors/)로 자체 생성 | 생성량=시간×FPS×카메라 수×인코딩. 고정 데이터 용량 없음. 코드와 에셋의 라이선스 구별 | RC에 가까운 카메라 높이·화각·동기 다중 시점·날씨를 설계 가능. 16GB GPU 학습 PC에서 별도 생성 작업. 시뮬레이터 동시 실행을 4GB 운용 조건에 포함하지 않음 |

## 현재 학습 코드로 옮기는 기준

현재 공통 학습 라벨은 **`road_surface_v1`: `0=road`, `1=non-road`, `255=ignore`**로 확정했습니다. [현재 공개 데이터 준비 문서](PUBLIC_TRAINING_SETUP.md)와 출처별 label map을 기준으로 변환합니다. BDD 원본의 direct/alternative는 공통 road로 합쳐지므로 이 공통 출력에서 두 영역을 다시 구분할 수는 없습니다.

| 입력 자료 | 원본 인코딩 | 공통 `road_surface_v1` 변환 |
|---|---|---|
| 이번에 사용하는 BDD 2018 | `0=background`, `1=direct`, `2=alternative` | `1 또는 2→0 road`, `0→1 non-road`, `255→255 ignore` |
| BDD 신형 raw ID 자료 | `0=direct`, `1=alternative`, `2=background` | `0 또는 1→0 road`, `2→1 non-road`, `255→255 ignore`. **2018 자료에 이 변환을 적용하면 안 됨** |
| 이번에 확보한 CamVid SegNet ID mask | `3=road`, `11=void`, 나머지 의미 클래스 | `3→0 road`, `11→255 ignore`, 나머지 `→1 non-road` |

CamVid는 commit `bb68b64e72cb004b3ed927fee0f5624dd8097498`을 고정합니다. 기존 배포 split을 합친 701쌍에서 원본 영상이 겹치지 않게 406/124/171쌍으로 재분할했으며, reserve는 학습과 모델 선택에서 제외합니다. 이 분할의 결과는 프로젝트 내부 도메인 평가이며 표준 CamVid benchmark 결과가 아닙니다.

1. **공통 작업 유지:** 추가 데이터도 위 `road_surface_v1` 의미에 맞춰 변환합니다. 같은 숫자 ID라도 배포 버전이 다르면 의미가 달라질 수 있으므로 출처·인코딩·label map을 함께 기록합니다. road는 장애물 부재나 통행 허가의 보장이 아닙니다.
2. **출처별 변환:** palette, class ID, void, 유효 ROI, annotation 누락을 점검합니다. box 밖 픽셀·비라벨 영상·invalid 영역을 도로 또는 장애물의 확정 정답으로 만들지 않습니다.
3. **그룹 분할:** 같은 주행·원본 장면·다중 카메라·합성 날씨 변형은 같은 `group_id`를 공유합니다. 공식 train/val/test를 우선하고 중복 원본도 검사합니다.
4. **단계 학습:** BDD 기준 → 추가 실사 1종 → 악천후 또는 합성 1종 → 목표 카메라 자체 데이터 순으로 비교합니다. 이전 BDD 일부를 함께 섞어 이전 환경 성능 저하를 확인합니다.
5. **데이터별 성능:** 단일 평균 대신 BDD, 추가 도메인, 야간, 미지 장애물, 자체 코스를 각각 보고합니다. 좋지 않은 데이터 추가는 되돌릴 수 있게 체크포인트와 manifest를 보관합니다.

## 16GB 학습 → 4GB 실행에서의 의미

학습 이미지 수가 늘어도 배포 모델의 파라미터 수는 자동으로 증가하지 않습니다. 작은 공유 모델 하나를 여러 데이터로 학습하는 접근은 4GB 실행 목표와 양립할 수 있습니다. 대신 입력 해상도, 모델 구조, temporal queue, 동시에 처리하는 카메라 수가 메모리와 지연을 결정합니다.

- 학습에서는 디스크에서 batch 단위로 읽고 mixed precision을 적용합니다. batch·해상도는 실제 메모리 측정으로 조정하며, 모든 원본을 RAM 32GB에 올리지 않습니다.
- 실행 PC는 RAM 16GB를 기준으로 설계할 수 있지만, 카메라 decode buffer·녹화·추론 session을 함께 측정해야 합니다. 데이터 전체를 운용 PC에 복사할 필요는 없습니다.
- 다중 카메라는 가중치를 공유하고 순차/작은 batch로 처리하는 경량 2D 방식부터 검증합니다. 모든 화면이 충분히 최신인지와 최악 지연을 측정해야 하며, 평균 FPS만으로 안전성을 판단할 수 없습니다.
- nuScenes/Waymo/Argoverse 데이터 사용이 대형 BEV 모델의 필수 사용을 뜻하지 않습니다. 4GB GPU에서 해당 대형 구조가 실시간 실행된다고 보장할 근거는 아직 없습니다.
- 도로/객체 인식의 향상은 충돌 회피 전체의 증명이 아닙니다. 거리·속도·캘리브레이션·시간 동기·가림·제동 거리·계획 및 제어 실패는 별도로 평가해야 합니다.

## 저장 공간을 잡는 방법

확정된 선택 파일의 크기는 CamVid RGB+mask 원본 합 약187 MB, KITTI Road ZIP 약471 MB, RoadObstacle ZIP 약208 MB, RoadAnomaly ZIP 약53 MB입니다. RoadAnomaly는 앞의 이미지별 조건을 적용합니다. Virtual KITTI 2 RGB+semantic을 추가하면 다운로드 약8 GB가 더 필요합니다. 이는 전체 프로젝트 용량이 아니라 선택 파일들의 크기이며, CamVid의 파일 합과 나머지 ZIP의 압축 해제 후 크기는 서로 다릅니다.

압축 해제본, 변환 마스크, cache, 체크포인트를 함께 보관한다면 다운로드 합보다 여유가 필요합니다. 특히 PNG/JPEG처럼 이미 압축된 파일의 tar와 raw sensor의 압축률은 다르므로 일률적으로 2배라고 단정하지 않습니다. 실제 선택 manifest의 총 bytes와 압축 해제 후 bytes를 기록한 뒤 저장장치를 확장하는 편이 정확합니다.

이 문서는 추가 후보를 폭넓게 비교하기 위한 것이며, 모든 후보의 일괄 다운로드나 학습을 권장하지 않습니다. 동일한 독립 평가에서 개선되는 데이터부터 유지하는 것이 저장 공간과 학습 시간 모두에 유리합니다.
