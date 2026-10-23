# Ubuntu MSI의 화면 없는 운용 준비

> **과거 실험·연구/인터페이스 참고 문서.** 아래 수치와 진행 표현은 문서 작성 당시의 범위입니다. 현재 완료 결과와 취소/미검증 항목은 [RESULTS](RESULTS.md), 재현 절차는 [REPRODUCIBILITY](REPRODUCIBILITY.md), 운용 한계는 [LIMITATIONS](LIMITATIONS.md)를 우선 확인하십시오. 예제 명령은 설치 시 실행되지 않으며 학습·다운로드·서비스·제어의 새 실행을 뜻하지 않습니다. 로컬 경로는 이 공개본에서 프로젝트 상대 예제로 바꿨습니다.

대상은 **Ubuntu 24.04 LTS / Python 3.12 / x86_64 / MSI Prestige 15 A10SC / 전용 GPU 4GB / RAM 16GB**입니다. 현재 이 노트북이 수중에 없으므로 아래 내용은 나중에 적용할 준비물입니다. Ubuntu 설치, 패키지 설치, SSH 접속, 드라이버 변경, 서비스 설치·활성화는 이번 작업에서 수행하지 않았습니다. Windows RTX 3080 Laptop의 기존 `.venv`는 학습·개발에 계속 사용합니다.

현재 운용 코드는 GUI 창 없이 카메라 영상, GPU 추론, 계획·제어, JSON 로그를 처리합니다. 모니터나 데스크톱 로그인 자체는 요구하지 않습니다. 실제 노트북의 무화면 재부팅, NVIDIA 드라이버, 카메라·MCU, 발열과 지연은 장비를 확보한 뒤 확인해야 합니다. 이 준비물은 차량의 주행 성능 검증이 아닙니다.

## 1. 장비를 확보한 뒤 기본 상태 확인

일반 사용자 계정에서 실행합니다. GPU 확인 실패를 드라이버 자동 교체로 해결하지 않습니다.

```bash
cat /etc/os-release
uname -m
python3.12 --version
nvidia-smi
```

`nvidia-smi`가 작동하지 않으면 Ubuntu에서 기존 NVIDIA 드라이버 설치 상태와 Secure Boot 설정을 먼저 확인합니다. 기존 구성이 정상이라면 불필요하게 드라이버나 CUDA Toolkit을 교체할 필요가 없습니다. 설치 스크립트는 NVIDIA 드라이버와 시스템 CUDA를 설치하지 않습니다. PyTorch CUDA 연산까지 성공했는지는 별도 점검 스크립트가 확인합니다.

Ubuntu에 필요한 도구가 없을 때만 관리자가 아래 작업을 수동으로 수행합니다. 이 명령은 여기에서 실행하지 않았습니다.

```bash
sudo apt update
sudo apt install python3.12-venv openssh-server
```

SSH가 이미 구성되어 있으면 기존 설정을 사용합니다. 처음 준비하는 경우 서비스 상태와 로컬 네트워크 주소를 확인한 뒤, 다른 PC에서 실제 접속과 재부팅 후 재접속을 시험합니다. 방화벽·라우터 설정은 현재 네트워크 정책에 맞게 처리하고 인터넷 포트 개방을 기본 절차로 삼지 않습니다.

```bash
systemctl status ssh
hostname -I
# 필요할 때만 로컬 관리자 실행:
sudo systemctl enable --now ssh
# 다른 PC에서 실제 사용자명/IP를 넣어 실행:
ssh ubuntu_user@192.168.0.100
```

부팅 시 디스크 암호 입력이나 초기 SSH 설정이 필요하면 임시 화면 또는 다른 콘솔 경로가 필요할 수 있습니다. 덮개 닫힘·유휴 상태에서 suspend가 발생하는지도 확인합니다. **우선 덮개를 열고 냉각이 확보된 상태로 운용**하고, 설정을 바꿀 때는 해당 사용자와 전원 상태에 필요한 항목만 변경합니다. 이 프로젝트는 suspend 대상 전체를 mask하거나 시스템 절전 설정을 일괄 변경하지 않습니다.

## 2. 프로젝트와 운용 모델 복사

아래 예시는 `$HOME/자율 주행/camera-rc-training`에 프로젝트를 복사했다고 가정합니다. Windows의 `.venv`는 Linux에 복사해 재사용할 수 없습니다. 소스, `configs`, `scripts`, `deploy`, 필요한 운용 체크포인트를 복사하고 대규모 학습 데이터와 Windows 가상 환경은 제외합니다. 복사 방법은 실제 네트워크와 저장 장치를 확보한 뒤 선택합니다.

```bash
cd "$HOME/자율 주행/camera-rc-training"
bash scripts/setup_ubuntu.sh
# 객체 탐지까지 사용한다면 위 명령 대신:
# bash scripts/setup_ubuntu.sh --with-detection
```

스크립트는 Ubuntu 24.04·x86_64·Python 3.12와 기존 NVIDIA 드라이버 상태를 확인한 뒤, **별도 `.venv-linux`**에 Torch 2.9.1 / torchvision 0.24.1 / cu126과 프로젝트 runtime 의존성을 설치합니다. 일반 사용자로 실행하며 `sudo`는 사용하지 않습니다. 지원 대상으로 정하지 않은 OS나 Jetson에서는 중단합니다. 다운로드 및 실제 Ubuntu 설치는 아직 검증하지 않았습니다.

프로젝트의 일반 `opencv-python` 의존성을 유지합니다. GUI 함수를 호출하지 않으므로 화면은 필요하지 않지만, 최소 설치 Ubuntu에서는 `import cv2`에 필요한 `libGL`·GLib 공유 라이브러리가 빠져 있을 수 있습니다. 해당 오류가 확인될 때 Ubuntu 패키지 `libgl1`, [`libglib2.0-0t64`](https://packages.ubuntu.com/noble/libglib2.0-0t64)를 검토합니다. `opencv-python`과 `opencv-python-headless`를 함께 설치하지 마세요. 이 준비물은 의존성 조합을 임의로 헤드리스 패키지로 바꾸지 않습니다.

## 3. 소프트웨어 점검과 장치 접근

```bash
.venv-linux/bin/python scripts/check_runtime_environment.py --require-cuda
# 객체 탐지를 설치한 경우:
.venv-linux/bin/python scripts/check_runtime_environment.py --require-cuda --with-detection
```

이 점검은 버전·NumPy/Torch/OpenCV 호환성·CUDA 연산·선택적 CUDA NMS를 확인합니다. 카메라, 시리얼 포트, 물리 구동기를 열지 않습니다. `--allow-cpu`는 CPU 진단에만 사용하며 CUDA 운용 검증으로 해석하지 않습니다.

USB 장치 경로를 확인하고 시리얼 설정에는 가능한 한 안정적인 `/dev/serial/by-id/...`를 사용합니다. MCU 규약은 [SERIAL_PROTOCOL.md](SERIAL_PROTOCOL.md)를 따릅니다.

```bash
id
ls -l /dev/video*
ls -l /dev/serial/by-id/
```

기존 권한으로 접근할 수 있으면 추가 그룹 변경은 필요 없습니다. 장치 그룹이 `video`, `dialout`이고 해당 권한이 부족한 경우에만 관리자가 그룹을 추가한 뒤 완전히 로그아웃·재로그인합니다. 서비스 사용자도 같은 계정을 사용합니다.

```bash
# 필요할 때만; root로 운용하거나 chmod 666으로 장치를 공개하지 않습니다.
sudo usermod -aG video,dialout "$USER"
```

현재 기본 카메라 연결은 USB UVC를 전제로 합니다. `/dev/video0` 순서와 실제 카메라 해상도가 맞는지 확인하고, [RUNTIME.md](RUNTIME.md)에 따라 실제 장착 위치에서 카메라·지면·차체 보정을 완료합니다. `configs/runtime_msi_prestige15_ubuntu.json`의 모델 파일과 보정 파일은 실제 준비한 파일로 연결해야 합니다. 예시 값이나 진단 모델로 실구동 검증을 대신하지 않습니다.

## 4. 수동으로 화면 없는 dry-run

```bash
.venv-linux/bin/python scripts/run_headless.py \
  --config configs/runtime_msi_prestige15_ubuntu.json \
  --output-root runs/headless --dry-run --duration-s 30
```

매번 UTC 시각과 UUID로 새로운 출력 폴더를 생성하여 기존 로그를 덮어쓰지 않습니다. 해당 폴더의 `summary.json`, `perception.jsonl`, `control.jsonl`, `launcher.json`을 확인합니다. 시작 실패처럼 런타임 요약을 만들기 전 종료되면 표준 출력의 오류도 확인합니다. 카메라 dry-run에서는 측정 차량 상태가 없으므로 물리 구동 출력을 보내지 않습니다.

Ctrl+C 또는 SIGTERM은 기존 `run_drive`의 종료 경로로 전달되어 STOP을 보내고 연결을 닫습니다. 반복 종료 신호가 STOP 처리 중간을 끊지 않도록 처리합니다. 전원 손실이나 SIGKILL은 Python 종료 처리를 실행할 수 없으므로 MCU의 독립 watchdog은 계속 필요합니다.

실구동은 현재 [RUNTIME.md](RUNTIME.md)의 모든 보정·모델 검증·물리 arm 조건을 갖춘 별도 serial 설정과, 사람이 직접 실행하는 `--actuate`가 모두 필요합니다. dry-run 설정을 자동으로 실구동으로 바꾸거나 MCU를 소프트웨어에서 자동 arm하지 않습니다. 아래 서비스 예시는 항상 dry-run입니다.

## 5. 선택 사항: 사용자 systemd 서비스로 dry-run 유지

서비스는 **아직 설치·활성화하지 않았습니다.** 수동 dry-run이 성공한 뒤에만 사용합니다. `deploy/systemd/camera-rc-dry-run.service`의 프로젝트 경로를 실제 위치에 맞추세요. 경로에 공백이 있어도 각 인수가 따옴표로 묶여 있으며 `sh -c`를 사용하지 않습니다. `%h`는 서비스 사용자의 홈 디렉터리입니다. 직접 경로에 `%`가 들어가는 경우 systemd 지정자와 혼동되지 않도록 `%%`로 이스케이프해야 합니다.

```bash
mkdir -p "$HOME/.config/systemd/user"
cp deploy/systemd/camera-rc-dry-run.service "$HOME/.config/systemd/user/"
systemd-analyze --user verify "$HOME/.config/systemd/user/camera-rc-dry-run.service"
systemctl --user daemon-reload
systemctl --user start camera-rc-dry-run.service
journalctl --user -u camera-rc-dry-run.service -f
```

정지는 다음 명령으로 수행합니다. 서비스는 SIGTERM으로 종료를 요청하고 최대 15초를 기다립니다. 그 뒤 강제 종료가 필요하면 MCU watchdog이 정지를 담당해야 합니다.

```bash
systemctl --user stop camera-rc-dry-run.service
```

`Restart=no`이므로 오류·통신 단절·종료 후 자동 재시작하지 않습니다. `Type=exec`는 실행 파일을 시작하지 못한 경우 서비스 시작 실패를 알릴 수 있도록 사용합니다. 이 설정이 모델 로딩이나 운용 준비 완료를 보장하지는 않습니다. [systemd 서비스 설정 문서](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)

`--dry-run`이 명시되어 serial 설정을 잘못 연결하면 시작이 거부됩니다. 실제 주행용 자동 시작·자동 재시작 서비스는 제공하지 않습니다.

로그인 시 자동 dry-run 시작이 필요할 때만 `systemctl --user enable camera-rc-dry-run.service`를 수동으로 선택합니다. 로그아웃 후나 부팅 시에도 사용자 서비스를 유지하려면 해당 계정에 대한 lingering 구성이 별도로 필요할 수 있습니다. 기본 작업에는 포함하지 않았으며 `loginctl show-user "$USER" -p Linger`로 현재 상태부터 확인합니다.

## 검증 범위

Windows 개발 환경에서 실행 인수·구동 차단·고유 로그 경로·종료 신호 처리에 대한 자동 테스트를 수행합니다. 실제 Ubuntu의 bash 설치 실행, systemd 유닛 검증·활성화, SSH 연결, 덮개 동작, GTX 1650 처리 속도, USB 카메라와 MCU 제어는 장비 확보 후 수행할 항목입니다. MSI가 준비되기 전에도 Windows에서 학습·평가·모델 내보내기와 기록 영상 검증을 계속 진행할 수 있습니다.
