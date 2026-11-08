# Focus Guard

> "왜 집중 안 해?" 묻고, 변명까지 듣는 AI 공부 감독관

가천대학교 Physical AI **Vision-LLM 융합 시청각 멀티모달 시스템** 최종 프로젝트

카메라로 공부하는 모습을 보고, 집중이 깨지면 Gemma가 장면을 보고 음성으로 이유를 묻습니다.
사용자가 음성으로 변명하면 Gemma가 그 이유가 타당한지 판단하여 다시 음성으로 대답합니다.

## 파이프라인

```text
Camera ─▶ YOLO11n INT8 (휴대폰 · 책 · 노트북)
       └▶ MediaPipe Pose (손목 · 코 · 귀 · 어깨)
            └▶ 규칙 기반 상태 판단 (6가지 상태 + 지속 시간)
                 └▶ 이벤트 (이탈 3초 지속)
                      └▶ 상태 JSON → 자연어 Context + 프레임 이미지 ─▶ Gemma 4 E2B
                           └▶ Piper TTS "왜 집중 안 해?"
                                └▶ Whisper STT (사용자 변명) ─▶ Gemma 판단 ─▶ Piper TTS
```

## 파일 구성

| 파일 | 설명 |
|---|---|
| `focus_guard.py` | 실행 코드 |
| `Final_Project_Focus_Guard.ipynb` | 제출용 노트북 (설명 + 코드) |
| `Focus_Guard_발표.pdf` | 발표자료 |

## 상태 판단 규칙

| 상태 | 의미 | 규칙 |
|---|---|---|
| `FOCUS` | 집중 | 아래 조건 없음 (책을 들고 고개를 숙이면 독서로 보고 집중) |
| `PHONE` | 휴대폰 사용 | 손목·검지가 휴대폰 bbox(+25%) 안에 있거나, 휴대폰이 사람 영역과 겹침 |
| `LOOKING_AWAY` | 딴 곳 보기 | \|코x − 귀중심x\| ÷ 귀 간격 > 0.35, 또는 한쪽 귀만 보임 |
| `HEAD_DOWN` | 졸음 의심 | (어깨중심y − 코y) ÷ 어깨 너비 < 0.15, 공부 물건을 들고 있지 않음 |
| `BAD_POSTURE` | 자세 흐트러짐 | 어깨선 기울기 > 12°, 또는 양 귀를 잇는 선 기울기 > 20° (턱 괴기·삐딱하게 앉기) |
| `ABSENT` | 자리 비움 | 사람도, 자세 랜드마크도 탐지되지 않음 |

- 판단 우선순위: 자리 비움 → 휴대폰 → 딴 곳 보기 → 졸음 → 자세 흐트러짐 → 집중
- 이탈 상태가 3초(자리 비움은 6초) 지속되면 경고, 경고 간 쿨다운 15초

## 실행 방법

### Jetson

```bash
cd ~/vision-llm
git clone https://github.com/pygmettalion/focus-guard.git
cp focus-guard/focus_guard.py .
source .venv/bin/activate
python3 focus_guard.py
```

`focus_guard.py` 상단 설정 수정:

```python
USE_JETSON_CSI = True
FLIP_CODE = 0
MIC_DEVICE = "plughw:?,0"   # arecord -l 결과에 맞게
```

업데이트:

```bash
cd ~/vision-llm/focus-guard
git pull
cp focus_guard.py ..
```

### 노트북 (테스트용)

```bash
pip install ultralytics mediapipe opencv-python
mkdir -p src/models/MediaPipe
curl -L -o src/models/MediaPipe/pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
python focus_guard.py
```

Gemma, Whisper, Piper가 없으면 자동으로 대체됩니다.

| 구성요소 | 없을 때 |
|---|---|
| TensorRT 엔진 | `yolo11n.pt` 사용 |
| Gemma mmproj | 텍스트 전용 입력 |
| Gemma | 규칙 기반 문장 |
| Whisper | 터미널 키보드 입력 |
| Piper | 콘솔 출력 |

## 조작 키

| 키 | 기능 |
|---|---|
| `q` | 종료 |
| `v` | 음성으로 질문하기 (STT → Gemma → TTS) |
| `r` | 지금까지의 공부 리포트를 음성으로 듣기 |
| `e` | 테스트용 강제 경고 이벤트 |

## 사용 기술

| 수업 섹션 | 프로젝트에서 사용 |
|---|---|
| 2 · Computer Vision | CSI 카메라, MediaPipe Pose 랜드마크로 자세 판단 |
| 4 · Object Detection | YOLO11n TensorRT INT8로 휴대폰 · 책 · 노트북 탐지 |
| 5 · LLM & Gemma | 꾸짖기 · 변명 판단 · 질문 답변 · 리포트 프롬프트 |
| 6-A · Vision-to-Text | 상태 JSON 저장 → 자연어 Context로 Gemma에 전달 |
| 6-A · 파이프라인 분리 | 카메라 루프와 LLM · 음성 워커 스레드 분리 |
| 6-B · 멀티모달 | 이탈 순간의 카메라 프레임을 Gemma에 함께 전달 |
| 6-C · STT / TTS | whisper.cpp로 변명 인식, Piper로 음성 출력 |
