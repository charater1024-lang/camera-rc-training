# BDD100K 출처와 레이블 검증

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

출처 확인일: 2026-10-20, 결과 갱신일: 2026-10-21. 이 문서는 다운로드 출처, 이용 조건, ZIP 무결성과 소량 샘플의 레이블 검증 기록입니다. 이미지·라벨 ZIP 두 개의 다운로드, 해시 대조와 선택 추출을 완료했습니다. 이후 전체 데이터 준비와 첫 1 epoch 학습·검증도 완료했으며, 학습 결과는 프로젝트의 `docs/BDD100K_BASELINE.md`와 `runs/bdd100k_2018_baseline_v1/summary.json`에 별도로 기록했습니다.

## 이용 조건과 가입

[BDD100K 공식 다운로드 문서](https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst)는 ETH 배포 경로를 안내하며, 다운로드하면 BDD100K 데이터 이용 조건에 동의한다고 설명합니다. 확인 당시 해당 ETH 경로에 연결할 수 없어 공개 미러를 조사했습니다.

[공식 데이터 라이선스](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst)는 교육·연구·비영리 목적의 사용·복사·변경·배포를 무료로 허용하고, 별도 서명 계약을 요구하지 않습니다. 복사·변경·배포할 때 원 저작권 고지와 이용 조건을 유지해야 합니다. 개인 실험 목적의 준비이며, 상업적 이용 권한을 확보했다는 의미는 아닙니다. 상업적 활용은 원 라이선스의 별도 조건을 확인해야 합니다.

데이터 라이선스와 툴킷 코드의 BSD 3-Clause 라이선스는 서로 다릅니다. 다운로드 폴더와 파생 데이터에는 원 데이터 라이선스 및 출처 기록을 함께 보존합니다.

아래 미러의 디렉터리 조회, 샘플 확인 및 Archive ZIP 두 개의 다운로드는 로그인·토큰·가입 없이 이루어졌습니다. 미러 접근에 가입이 필요하지 않더라도 원 데이터 이용 조건은 적용됩니다.

## 미러와 압축 파일 식별

[Internet Archive의 BDD100K 목록](https://archive.org/download/bdd100k/)에서 다음 2018년 배포 파일을 확인했습니다.

| 파일 | 바이트 크기 | Archive 메타데이터 MD5 |
| --- | ---: | --- |
| `bdd100k_images.zip` | 6,941,520,264 | `b538e3731a132e28dae37f18c442c51e` |
| `bdd100k_drivable_maps.zip` | 693,038,872 | `1bd70019468a81572c70d374751eb9e2` |

- [이미지 ZIP](https://archive.org/download/bdd100k/bdd100k_images.zip)
- [주행 가능 영역 ZIP](https://archive.org/download/bdd100k/bdd100k_drivable_maps.zip)
- [Archive 파일별 메타데이터](https://archive.org/download/bdd100k/bdd100k_files.xml)

위 MD5는 Archive가 게시한 해당 파일의 값이며, 전체 다운로드 파일에서 계산한 MD5와 일치했습니다. 추가로 계산한 전체 ZIP의 SHA256은 다음과 같습니다.

| 파일 | 다운로드 완료본 SHA256 |
| --- | --- |
| `bdd100k_images.zip` | `6f0e90b6860aeb68f02d63285475adcf325fe7d5e212d8149593f340234dab83` |
| `bdd100k_drivable_maps.zip` | `6f28c5b56fab54dc2f9855e96449d9fc4285023d585a3f5ca26f98baa522df29` |

이 검사는 내려받은 파일이 미러의 게시 체크섬과 일치함을 확인한 것입니다. 공식 배포본의 별도 체크섬 인증이나 모든 이미지·라벨의 의미 검증을 대신하지 않습니다. 아래의 샘플 검증은 전체 다운로드에 앞서 수행한 별도 기록입니다.

다운로드 폴더는 `datasets\bdd100k_2018\archives`, 추출 폴더는 `datasets\bdd100k_2018\extracted`입니다. 이번 추출에서 이미지 88,000장과 ID 마스크 80,000장이 나왔고, 추출 파일 크기 합계는 5,836,205,914바이트였습니다. 이미지 88,000장에는 `images/100k`의 train 70,000장·val 10,000장 외에 별도 `images/10k`의 train 7,000장·val 1,000장도 포함됩니다.

이번 drivable 학습 입력은 **`bdd100k/images/100k` 아래 80,000장만** 사용합니다. `images/10k`는 별도 태스크용으로 제공되는 분할이며 일부 이미지가 겹칠 수 있으므로 추가 학습 샘플로 섞지 않습니다. 마스크 루트는 `bdd100k/drivable_maps/labels`입니다. 준비 도구에는 이 두 경로를 명시적으로 전달합니다.

현재 공식 다운로드 문서의 다른 배포본은 100K 이미지 MD5를 `5a0359c86a0b8713adab1eee9a3041cb`, drivable 라벨 MD5를 `0abc320461200b1d7916f82fdcd64a96`으로 안내합니다. **이 값들은 위 2018년 Archive ZIP의 예상 체크섬이 아닙니다.** 파일명·폴더 구조·배포 시기가 다른 패키지 사이에 체크섬을 섞어 적용하지 않습니다.

보조 비교 출처는 [Hugging Face의 dronefreak 미러](https://huggingface.co/datasets/dronefreak/BDD100K-Drivable-Area-Segmentation/tree/92027ff251195fd7a97c298267b2a470829fc883)입니다. 확인에 사용한 revision은 다음과 같습니다.

```text
92027ff251195fd7a97c298267b2a470829fc883
```

이 미러는 공식 배포자가 아닌 제3자 미러입니다. 샘플 파일의 일치는 전체 미러가 공식 ZIP과 동일하다는 인증이 아닙니다.

## 구형과 신형 마스크의 차이

구형 정의는 추정이 아니라 공식 저장소의 과거 코드로 확인했습니다.

- [2018-05-23 공식 README, 34행](https://github.com/bdd100k/bdd100k/blob/3e762ca7337eb05ad80d54654dc93d943b75814f/README.md#L34): 구형 `drivable_maps` ID 정의.
- [같은 커밋의 `draw_drivable`, 373행부터](https://github.com/bdd100k/bdd100k/blob/3e762ca7337eb05ad80d54654dc93d943b75814f/bdd_data/show_labels.py#L373): 직접 주행 영역을 R 채널 1, 대체 영역을 2로 기록.
- [같은 커밋의 `convert_drivable_rgb`, 147행부터](https://github.com/bdd100k/bdd100k/blob/3e762ca7337eb05ad80d54654dc93d943b75814f/bdd_data/show_labels.py#L147): R 채널을 그대로 `_drivable_id.png`에 저장.
- [현재 공식 형식 문서](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst): 신형 drivable ID 정의.

| 의미 | 2018년 `drivable_maps` ID | 신형 `labels/drivable/masks` ID | 이 프로젝트의 이진 학습 ID |
| --- | ---: | ---: | ---: |
| 배경 / 도로 외 영역 | 0 | 2 | 1 |
| 직접 주행 가능 영역 | 1 | 0 | 0 |
| 대체 주행 가능 영역 | 2 | 1 | 0 |

**이 프로젝트의 이진 클래스는 `0=road`, `1=not_road`입니다.** 구형 원본에는 다음 변환을 사용합니다.

```json
{"0": 1, "1": 0, "2": 0}
```

설정 파일의 정식 클래스명은 `road_surface`, `not_road_surface`이며, 별도의 ignore 값 `255`는 그대로 보존합니다.

변환 설정은 [bdd_2018_to_road.json](../configs/label_maps/bdd_2018_to_road.json)에 있으며, 공개 데이터 준비 시 `--bdd-encoding legacy_2018`을 지정합니다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_public_training.py `
  --bdd-images "<이미지의 train·val을 포함하는 폴더>" `
  --bdd-masks "<drivable_maps\labels 폴더>" `
  --bdd-encoding legacy_2018 `
  --output data/bdd_2018_prepared
```

출력 폴더는 새 경로를 사용합니다. 실제 경로와 전체 준비 절차는 [공개 데이터 학습 준비](PUBLIC_TRAINING_SETUP.md)를 참고하세요.

이진 변환은 직접·대체 주행 영역을 합칩니다. 대체 영역을 현재 차량이 즉시 주행해도 된다는 의미로 해석해서는 안 됩니다. 차선 선택, 통행 방향, 다른 차량과 장애물에 대한 판단은 별도로 필요합니다.

파일명 접미사나 값의 범위 `{0, 1, 2}`만으로 구형·신형을 판별할 수 없습니다. 두 형식 모두 같은 값 범위를 사용하므로 출처와 명시적 인코딩 설정을 유지해야 합니다.

## HF 카드와 실제 파일의 불일치

확인한 HF 데이터 카드는 `0=직접 주행, 1=대체 주행, 2=배경`으로 설명했지만, 고정 revision에서 받은 실제 샘플은 **구형 `0=배경, 1=직접 주행, 2=대체 주행`**입니다.

HF에서 train ID 마스크 2장, val ID 마스크 2장과 각각의 대응 색상 마스크를 메모리에 읽었습니다. 각 샘플의 ID별 픽셀 수가 검정·빨강·파랑 색상 영역의 픽셀 수와 대응했습니다.

| 샘플 | ID 0 / 검정 픽셀 | ID 1 / 빨강 픽셀 | ID 2 / 파랑 픽셀 |
| --- | ---: | ---: | ---: |
| train `0000f77c-6257be58` | 726,634 | 194,966 | 0 |
| train `0000f77c-62c2a288` | 753,786 | 83,172 | 84,642 |
| val `b1c66a42-6f7d68ca` | 720,695 | 111,021 | 89,884 |
| val `b1c81faa-3df17267` | 697,750 | 137,656 | 86,194 |

샘플은 1280×720 단일 채널 ID PNG와 3채널 색상 PNG였습니다. 이 배포본의 색상은 검정 `(0,0,0)`, 빨강 `(255,0,0)`, 파랑 `(0,0,255)`입니다. 과거 코드의 다른 시각화 색상값과 혼동하지 않으며, 학습에는 색상 PNG 대신 ID PNG를 사용합니다.

## 샘플 SHA256과 교차 확인

Archive ZIP은 HTTPS Range 요청에 `206 Partial Content`를 반환했습니다. 처음 1MiB에서 색상 PNG 샘플을 읽고, 끝부분 128KiB의 ZIP 중앙 디렉터리로 ID PNG 한 장의 위치를 찾아 추가로 읽었습니다. 메모리에서 PNG를 해석하고 SHA256을 계산했습니다. 이 검증 단계에서 전체 ZIP을 내려받거나 샘플 파일을 디스크에 저장하지 않았습니다.

Archive와 HF에서 **같은 SHA256**을 확인한 파일:

| 파일 | SHA256 |
| --- | --- |
| `0000f77c-6257be58_drivable_color.png` | `8b9dd7ecfcb1a5d12c343913949c6158087c6276951b8bc5baf294901d810212` |
| `0000f77c-62c2a288_drivable_color.png` | `e52e50afb4ba0047bd55c4477c865a14aaedeb03290ea2e74103d82600f8ad51` |
| `c8266770-0cc2a4c7_drivable_id.png` | `12c4593bb66c1fd2e70581a360512485ea4815baa71b6dbf7a97a3c7d1eb2eaa` |

마지막 ID PNG의 Archive 경로는 `bdd100k/drivable_maps/labels/val/c8266770-0cc2a4c7_drivable_id.png`이며, ZIP 로컬 헤더 오프셋은 `667133730`입니다. HF 대응 경로는 `data/drivable_maps/labels/val/shard_001/c8266770-0cc2a4c7_drivable_id.png`입니다. 픽셀 수는 ID 0이 686,798개, ID 1이 81,995개, ID 2가 152,807개였습니다.

HF에서 추가로 읽은 ID PNG의 SHA256:

| 파일 | SHA256 |
| --- | --- |
| `0000f77c-6257be58_drivable_id.png` | `d7192f1cf4f8dee2f5aa4604a890887e0c849611b39dd4c671b5789df2795461` |
| `0000f77c-62c2a288_drivable_id.png` | `7b22aed19743b0569b02080d32f2028e67d8faaf6dbebba98e8b9f242a4e3d33` |
| `b1c66a42-6f7d68ca_drivable_id.png` | `235744433a86e2265b87420dc22e4b6c415027414df7c561636ee4a06545cd20` |
| `b1c81faa-3df17267_drivable_id.png` | `78d060eca6c9f3904214399632905e6647e197d3835762b2d50b8e9a78d3d95c` |

이 검증으로 두 미러의 확인된 샘플이 같은 구형 형식임을 확인했습니다. 이후 Archive ZIP 두 개의 전체 다운로드와 체크섬 대조를 완료했습니다. 학습 준비에서는 전체 입력의 이미지·마스크 대응, 크기, 허용 ID와 공식 train/val 분할을 추가로 검사합니다. 압축 해제·학습 준비·학습의 진행 상태와 결과는 각각의 실행 로그에서 확인합니다.
