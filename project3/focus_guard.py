"""
Focus Guard : Vision-LLM 융합 시청각 멀티모달 공부 감독 시스템
=====================================================================
[파이프라인]
 Camera ─▶ YOLO (휴대폰/책/노트북 탐지)
        └▶ MediaPipe Pose (손목·고개 방향·자세)
             └▶ 규칙 기반 이벤트 판단 (휴대폰 사용 / 딴 곳 보기 / 졸음 / 자세 흐트러짐 / 자리 비움)
                  └▶ 상태 JSON + 현재 프레임 이미지 ─▶ Gemma (멀티모달)
                       └▶ Piper TTS "왜 집중 안 해?"  ─▶ 사용자 음성(Whisper STT) ─▶ Gemma 판단 ─▶ TTS

[키]
 q : 종료
 v : 음성으로 질문하기 (STT → Gemma → TTS)
 r : 지금까지 공부 리포트 음성으로 듣기
 e : (발표용) 강제로 경고 이벤트 발생

[실행] 아무 폴더에서나:  python3 focus_guard.py
  - 모델이 있는 수업 폴더(~/vision-llm 등)를 자동으로 찾아서 그 폴더 기준으로 실행
  - llama_cpp가 없는 파이썬으로 실행하면 수업 가상환경(.venv)으로 자동 재실행
  - 카메라(USB/CSI), 마이크(arecord -l)를 자동 감지
  - Gemma(필수): 모델 파일이 없으면 Hugging Face에서 자동 다운로드, 못 불러오면 종료
  - 포즈 모델 없으면 자동 다운로드, TensorRT 엔진 없으면 yolo11n.pt
  - whisper.cpp 없으면 키보드 입력, Piper 없으면 콘솔 출력
"""

import os
import re
import sys
import json
import time
import wave
import base64
import queue
import shutil
import platform
import threading
import subprocess

# ============================================================
# 1. 설정
# ============================================================
IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"

# ---- 카메라 (기본: 자동 감지) ----
CAMERA_MODE = "auto"        # "auto" | "usb" | "csi" | "index"
CAM_INDEX = 0
FLIP_CODE = None            # 상하반전 필요하면 0, 좌우반전(거울)은 1, 없으면 None
USB_DEVICE = "/dev/video0"
CAM_WIDTH, CAM_HEIGHT = 1280, 720   # 화면이 확대돼 보이면 16:9 해상도를 사용 (안 되면 자동으로 640x480)

USB_MJPG_PIPELINE = (
    f"v4l2src device={USB_DEVICE} ! "
    f"image/jpeg,width={CAM_WIDTH},height={CAM_HEIGHT},framerate=30/1 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# Jetson USB 카메라 (수업 tt.py와 동일한 GStreamer 파이프라인)
USB_PIPELINE = (
    f"v4l2src device={USB_DEVICE} ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

CSI_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# ---- 모델 경로 (수업 폴더 구조 기준, 아래 find_base_dir()가 수업 폴더를 자동으로 찾음) ----
YOLO_CANDIDATES = [
    "src/models/YOLO/yolo11n_int8.engine",
    "src/models/YOLO/yolo11n_fp16.engine",
    "src/models/YOLO/yolo11n_fp32.engine",
    "src/models/YOLO/yolo11n.pt",
]
POSE_MODEL_PATH = "src/models/MediaPipe/pose_landmarker_full.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 120

# ---- STT (whisper.cpp) ----
MIC_DEVICE = "auto"         # "auto"면 arecord -l 결과에서 USB 마이크를 자동 선택 (직접 지정: "plughw:3,0")
RECORD_SECONDS = 5
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"

# ---- TTS (Piper) ----
PIPER_PYTHON = ".piper_venv/Scripts/python.exe" if IS_WINDOWS else ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
TTS_FILE = "src/audio/response.wav"
SPEAKER_DEVICE = "auto"     # "auto"면 기본 장치 → USB 스피커 순서로 자동 시도 (직접 지정: "plughw:3,0")

# ---- 출력 ----
JSON_PATH = "src/json/focus_state.json"
TEMP_JSON_PATH = "src/json/focus_state_tmp.json"
SNAPSHOT_PATH = "src/output/focus_event.jpg"

# ---- 이벤트 규칙 ----
DISTRACT_SEC = 3.0          # 집중 안 하는 상태가 이만큼 지속되면 경고
ABSENT_SEC = 6.0            # 자리 비움은 조금 더 여유
COOLDOWN_SEC = 15.0         # 경고 후 다음 경고까지 최소 간격
LISTEN_AFTER_WARNING = True # 경고 후 사용자 변명 듣기

YAW_RATIO_TH = 0.35         # 코가 귀 중심에서 이만큼 벗어나면 옆을 보는 중
EAR_VIS_DIFF_TH = 0.55      # 한쪽 귀만 보이면 고개 돌린 것
HEAD_DOWN_RATIO_TH = 0.15   # (어깨중심y - 코y) / 어깨너비 가 이보다 작으면 고개 숙임
HAND_BOX_MARGIN = 0.25      # 물체 bbox를 이 비율만큼 키워서 손목이 들어오면 "손에 듦"
SHOULDER_TILT_TH = 12.0     # 어깨선이 이 각도(도) 이상 기울면 자세 흐트러짐
HEAD_TILT_TH = 20.0         # 양 귀를 잇는 선이 이 각도(도) 이상 기울면 고개를 옆으로 꺾음 (턱 괴기 등)

# COCO 클래스: 0 person, 63 laptop, 64 mouse, 66 keyboard, 67 cell phone, 73 book
YOLO_CLASSES = [0, 63, 64, 66, 67, 73]
DISTRACT_OBJECTS = {"cell phone"}
STUDY_OBJECTS = {"book", "laptop", "keyboard", "mouse"}

STATE_FOCUS = "FOCUS"
STATE_PHONE = "PHONE"
STATE_AWAY = "LOOKING_AWAY"
STATE_DROWSY = "HEAD_DOWN"
STATE_POSTURE = "BAD_POSTURE"
STATE_ABSENT = "ABSENT"

STATE_KO = {
    STATE_FOCUS: "집중 중",
    STATE_PHONE: "휴대폰 사용",
    STATE_AWAY: "고개를 돌려 다른 곳을 보는 중",
    STATE_DROWSY: "공부 물건 없이 고개를 숙이고 있음 (졸음 의심)",
    STATE_POSTURE: "어깨나 고개가 한쪽으로 기울어 자세가 흐트러짐",
    STATE_ABSENT: "자리에 사람이 없음",
}
STATE_COLOR = {
    STATE_FOCUS: (80, 200, 80),
    STATE_PHONE: (0, 0, 255),
    STATE_AWAY: (0, 165, 255),
    STATE_DROWSY: (255, 0, 200),
    STATE_POSTURE: (0, 220, 220),
    STATE_ABSENT: (160, 160, 160),
}


# ============================================================
# 1-1. 자동 환경 설정 (수업 폴더 / 마이크 / 포즈 모델)
# ============================================================
HOME = os.path.expanduser("~")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _base_score(d):
    """이 폴더에 수업 모델이 얼마나 있는지 점수화"""
    checks = [
        (GEMMA_MODEL_PATH, 4), (PIPER_MODEL, 2), (WHISPER_MODEL, 2),
        (POSE_MODEL_PATH, 1), (YOLO_CANDIDATES[0], 1), (YOLO_CANDIDATES[-1], 1),
    ]
    return sum(w for p, w in checks if os.path.exists(os.path.join(d, p)))


def find_base_dir():
    """모델이 들어있는 수업 폴더(src/models가 있는 곳)를 찾음"""
    candidates = [os.getcwd(), SCRIPT_DIR, os.path.dirname(SCRIPT_DIR),
                  os.path.join(HOME, "vision-llm")]
    # 홈 아래 3단계까지 src/models 폴더가 있는 곳도 후보로 추가
    for root, dirs, _ in os.walk(HOME):
        depth = root[len(HOME):].count(os.sep)
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("node_modules", "snap")]
        if depth >= 3:
            dirs[:] = []
        if os.path.isdir(os.path.join(root, "src", "models")):
            candidates.append(root)

    best, best_score = os.getcwd(), -1
    for d in candidates:
        if os.path.isdir(d):
            score = _base_score(d)
            if score > best_score:
                best, best_score = d, score
    return best


def detect_mic():
    """arecord -l 결과에서 USB 마이크(없으면 첫 번째 장치)를 plughw:N,M으로 반환"""
    if not (IS_LINUX and shutil.which("arecord")):
        return None
    out = subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout
    cards = re.findall(r"card (\d+): (\S+) \[([^\]]*)\], device (\d+):", out)
    if not cards:
        return None
    pick = next((c for c in cards if "usb" in (c[1] + c[2]).lower()), cards[0])
    return f"plughw:{pick[0]},{pick[3]}"


def ensure_pose_model():
    """포즈 모델이 없으면 자동 다운로드"""
    if os.path.exists(POSE_MODEL_PATH):
        return
    try:
        import urllib.request
        os.makedirs(os.path.dirname(POSE_MODEL_PATH), exist_ok=True)
        print("[Pose] 모델 다운로드 중...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("[Pose] 다운로드 완료")
    except Exception as e:
        print("[Pose] 다운로드 실패 (인터넷 확인):", e)


REQUIRED_MODULES = ["llama_cpp", "ultralytics", "cv2", "mediapipe"]


def _python_has_modules(python):
    code = "import " + ", ".join(REQUIRED_MODULES)
    return subprocess.run([python, "-c", code], capture_output=True).returncode == 0


def ensure_python_env():
    """지금 파이썬에 llama_cpp 등이 없으면, 수업 가상환경(.venv)의 파이썬으로 자동 재실행"""
    if os.environ.get("FOCUS_GUARD_REEXEC") == "1":
        return
    if _python_has_modules(sys.executable):
        return

    candidates = [
        os.path.join(BASE_DIR, ".venv", "bin", "python"),
        os.path.join(HOME, "vision-llm", ".venv", "bin", "python"),
        os.path.join(os.path.dirname(SCRIPT_DIR), ".venv", "bin", "python"),
    ]
    for python in candidates:
        if os.path.exists(python) and _python_has_modules(python):
            print(f"[Env] 수업 가상환경으로 다시 실행: {python}")
            os.environ["FOCUS_GUARD_REEXEC"] = "1"
            script = os.path.join(SCRIPT_DIR, os.path.basename(__file__))
            os.execv(python, [python, script] + sys.argv[1:])

    sys.exit(
        "[Env] llama_cpp / ultralytics / cv2 / mediapipe 가 모두 설치된 파이썬을 찾지 못했습니다.\n"
        "      수업 가상환경을 켜고 실행하세요:  source ~/vision-llm/.venv/bin/activate"
    )


def ensure_gemma_files():
    """Gemma GGUF / mmproj가 없으면 수업과 동일한 저장소에서 자동 다운로드"""
    missing = [p for p in (GEMMA_MODEL_PATH, MMPROJ_PATH) if not os.path.exists(p)]
    if not missing:
        return
    try:
        from huggingface_hub import hf_hub_download
        token = os.getenv("HF_TOKEN")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            token = token or os.getenv("HF_TOKEN")
        except ImportError:
            pass
        for path in missing:
            print(f"[Gemma] {os.path.basename(path)} 다운로드 중... (용량이 커서 몇 분 걸림)")
            hf_hub_download(
                repo_id="bartowski/google_gemma-4-E2B-it-GGUF",
                filename=os.path.basename(path),
                token=token,
                local_dir=os.path.dirname(path),
            )
        print("[Gemma] 다운로드 완료")
    except Exception as e:
        print("[Gemma] 다운로드 실패:", e)


BASE_DIR = find_base_dir()
os.chdir(BASE_DIR)          # 이후 모든 상대 경로(src/models/..., whisper.cpp/...)는 수업 폴더 기준
ensure_python_env()

import cv2                   # noqa: E402  (가상환경 확인 후 import)
import numpy as np           # noqa: E402
from ultralytics import YOLO # noqa: E402

if MIC_DEVICE == "auto":
    MIC_DEVICE = detect_mic() or "default"

for d in ["src/audio", "src/json", "src/output"]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 2. 음성 (STT / TTS)
# ============================================================
def _clean_for_speech(text):
    text = re.sub(r"[*#`>\[\]_~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_speaker_device = None   # 한 번 소리가 난 스피커 장치를 기억


def _playback_devices():
    """aplay -l 결과에서 재생 장치 목록 (USB 우선)"""
    out = subprocess.run(["aplay", "-l"], capture_output=True, text=True).stdout
    cards = re.findall(r"card (\d+): (\S+) \[([^\]]*)\], device (\d+):", out)
    cards.sort(key=lambda c: "usb" not in (c[1] + c[2]).lower())
    return [f"plughw:{c[0]},{c[3]}" for c in cards]


def play_wav(path):
    global _speaker_device
    try:
        if IS_LINUX and shutil.which("aplay"):
            # 기본 장치 → 실패하면 USB 스피커 → 나머지 장치 순서로 시도
            devices = [_speaker_device] if _speaker_device else (
                [SPEAKER_DEVICE] if SPEAKER_DEVICE != "auto" else [None] + _playback_devices())
            for dev in devices:
                cmd = ["aplay", "-q", path] if dev is None else ["aplay", "-q", "-D", dev, path]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0:
                    if _speaker_device is None:
                        _speaker_device = dev or "default"
                        if dev:
                            print(f"[TTS] 스피커: {dev}")
                    return
                print(f"[TTS] aplay 실패 ({dev or 'default'}): {r.stderr.strip()[:150]}")
        elif IS_WINDOWS:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        elif shutil.which("afplay"):
            subprocess.run(["afplay", path], check=True)
    except Exception as e:
        print("[TTS] 재생 실패:", e)


def text_to_speech(text):
    text = _clean_for_speech(text)
    print(f"\n[AI 🔊] {text}")
    if not text:
        return

    # 1순위: 수업에서 쓴 Piper
    if os.path.exists(PIPER_PYTHON) and os.path.exists(PIPER_MODEL):
        r = subprocess.run(
            [PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", TTS_FILE, "--", text],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and os.path.exists(TTS_FILE):
            play_wav(TTS_FILE)
            return
        print("[TTS] Piper 음성 생성 실패:", (r.stderr or r.stdout).strip()[-300:])
    else:
        print(f"[TTS] Piper 없음: {os.path.abspath(PIPER_PYTHON)} / {os.path.abspath(PIPER_MODEL)}")

    # 2순위: pyttsx3 (노트북 테스트용, 설치돼 있으면)
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass  # 콘솔 출력으로 대체


def record_audio(path, seconds):
    # 1순위: 수업 방식 (arecord)
    if IS_LINUX and shutil.which("arecord"):
        cmd = ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1",
               "-d", str(seconds), path]
        if shutil.which("pasuspender"):
            cmd = ["pasuspender", "--"] + cmd
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print("[STT] arecord 실패:", e)

    # 2순위: sounddevice (MediaPipe 설치 때 같이 설치함)
    try:
        import sounddevice as sd
        data = sd.rec(int(seconds * 16000), samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(data.tobytes())
        return True
    except Exception as e:
        print("[STT] 녹음 실패:", e)
        return False


def _whisper_exe():
    for p in [WHISPER_PATH, WHISPER_PATH + ".exe",
              "whisper.cpp/build/bin/whisper-cli", "whisper.cpp/build/bin/Release/whisper-cli.exe"]:
        if os.path.exists(p):
            return p
    return None


def speech_to_text(status_cb=None):
    exe = _whisper_exe()
    if exe and os.path.exists(WHISPER_MODEL):
        if status_cb:
            status_cb("LISTENING")
        print(f"\n[🎤] {RECORD_SECONDS}초 동안 말씀하세요...")
        if record_audio(AUDIO_FILE, RECORD_SECONDS):
            if status_cb:
                status_cb("THINKING")
            result = subprocess.run(
                [exe, "-m", WHISPER_MODEL, "-f", AUDIO_FILE, "-l", "ko", "--no-timestamps"],
                text=True, capture_output=True,
            )
            text = re.sub(r"\[.*?\]|\(.*?\)", "", result.stdout).strip()
            print(f"[STT] {text}")
            return text

    # 대체: 키보드 입력 (터미널에서 입력)
    if status_cb:
        status_cb("TYPE IN TERMINAL")
    try:
        return input("\n[STT 대체] 하고 싶은 말을 입력하세요: ").strip()
    except EOFError:
        return ""


# ============================================================
# 3. Gemma (멀티모달 LLM)
# ============================================================
class GemmaBrain:
    """Gemma는 필수: 모델을 못 불러오면 대체하지 않고 종료"""

    def __init__(self):
        self.llm = None
        self.vision = False

        from llama_cpp import Llama

        ensure_gemma_files()
        if not os.path.exists(GEMMA_MODEL_PATH):
            sys.exit(f"[Gemma] 모델 파일이 없습니다: {os.path.abspath(GEMMA_MODEL_PATH)}\n"
                     "        인터넷 연결 후 다시 실행하면 자동으로 다운로드합니다.")

        chat_handler = None
        if os.path.exists(MMPROJ_PATH):
            try:
                from llama_cpp.llama_chat_format import Gemma4ChatHandler
                chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)
                self.vision = True
            except Exception as e:
                print("[Gemma] ⚠ 이미지 핸들러 로드 실패 → 이미지 없이 텍스트만 전달:", e)
        else:
            print("[Gemma] ⚠ mmproj 파일이 없어 이미지 없이 텍스트만 전달합니다.")

        self.llm = Llama(
            model_path=GEMMA_MODEL_PATH,
            chat_handler=chat_handler,
            n_gpu_layers=-1,
            n_ctx=CONTEXT_WINDOW,
            n_batch=32,
            n_ubatch=32,
            verbose=False,
        )
        print(f"[Gemma] 로드 완료 (이미지 입력: {self.vision})")

    def ask(self, system_prompt, user_text, image_bgr=None, max_tokens=MAX_TOKENS):
        if self.llm is None:
            return None

        print("[Gemma] 생각 중...", flush=True)
        t0 = time.time()

        if self.vision and image_bgr is not None:
            h, w = image_bgr.shape[:2]
            scale = 512 / max(h, w)   # 엣지 기기 속도를 위해 이미지 축소
            if scale < 1:
                image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", image_bgr)
            image_data = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data}},
            ]
        else:
            content = user_text

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            print(f"[Gemma] 응답 완료 ({time.time() - t0:.1f}초)")
            return answer
        except Exception as e:
            print("[Gemma] 추론 실패:", e)
            return None


# ---- 프롬프트 (수업 형식: Instruction / Context / Constraint / Output Format) ----
SCOLD_PROMPT = """
Instruction:
당신은 사용자의 공부를 지켜보는 엄격하지만 따뜻한 AI 공부 감독관이다.
주어진 상황 정보와 카메라 이미지를 보고, 사용자가 집중하지 않는 이유를 지적하고 왜 집중하지 않느냐고 물어보시오.

Constraint:
상황 정보와 이미지에서 확인되지 않는 내용은 추측하지 마시오.
반드시 "왜"로 시작하는 질문을 하나 포함하시오.
이모티콘과 특수문자를 쓰지 마시오.

Output Format:
한국어 구어체 두 문장 이내.
"""

EXCUSE_PROMPT = """
Instruction:
당신은 AI 공부 감독관이다. 방금 집중하지 않는 사용자에게 이유를 물었고, 사용자가 대답했다.
상황 정보와 사용자의 대답을 보고 그 이유가 타당한지 판단하여 대답하시오.

Constraint:
타당한 이유(공부 관련 검색, 급한 연락, 몸이 아픔, 잠깐의 휴식 등)면 짧게 인정하고 다시 공부하라고 격려하시오.
핑계라면 단호하지만 유머 있게 반박하고 다시 집중하라고 하시오.
이모티콘과 특수문자를 쓰지 마시오.

Output Format:
한국어 구어체 두 문장 이내.
"""

QA_PROMPT = """
Instruction:
당신은 사용자 옆에서 공부를 도와주는 AI 공부 도우미이다.
주어진 상황 정보와 카메라 이미지를 참고하여 사용자의 질문에 답하시오.

Constraint:
상황 정보와 이미지에서 확인되지 않는 내용은 추측하지 마시오.
이모티콘과 특수문자를 쓰지 마시오.

Output Format:
한국어 구어체 세 문장 이내.
"""

REPORT_PROMPT = """
Instruction:
당신은 AI 공부 감독관이다. 주어진 공부 세션 통계를 바탕으로 사용자에게 오늘 공부를 평가해 주시오.

Constraint:
통계에 있는 숫자만 사용하고 새로운 숫자를 만들지 마시오.
가장 많이 집중을 깬 원인을 짚어주고, 다음에 할 한 가지 조언을 하시오.
이모티콘과 특수문자를 쓰지 마시오.

Output Format:
한국어 구어체 세 문장 이내.
"""

FALLBACK_SCOLD = {
    STATE_PHONE: "휴대폰 내려놓으세요. 왜 공부 안 하고 휴대폰 보고 있어요?",
    STATE_AWAY: "어디 보고 있어요? 왜 화면에 집중을 안 하는 거예요?",
    STATE_DROWSY: "고개가 계속 숙여져 있네요. 왜 졸고 있어요? 일어나세요!",
    STATE_POSTURE: "자세가 많이 흐트러졌어요. 왜 그렇게 삐딱하게 앉아 있어요?",
    STATE_ABSENT: "자리가 비어 있네요. 왜 공부하다 말고 사라졌어요?",
}


# ============================================================
# 4. Vision: YOLO + MediaPipe Pose → 상태 판단
# ============================================================
def load_yolo():
    for path in YOLO_CANDIDATES:
        if os.path.exists(path):
            print(f"[YOLO] {path} 사용")
            return YOLO(path)
    # 수업 모델이 없으면 ultralytics가 yolo11n.pt를 자동 다운로드
    os.makedirs("src/models/YOLO", exist_ok=True)
    print("[YOLO] 수업 모델 없음 → yolo11n.pt 자동 다운로드")
    model = YOLO("yolo11n.pt")
    return model


def load_pose():
    ensure_pose_model()
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"[Pose] ⚠ {POSE_MODEL_PATH} 없음 → 자세 판단 없이 YOLO만 사용")
        return None, None
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = mp_vision.PoseLandmarker.create_from_options(options)
    print(f"[Pose] {POSE_MODEL_PATH} 로드 완료")
    return detector, mp


def open_camera():
    """USB(GStreamer) → CSI → 일반 인덱스 순으로 열리는 카메라를 자동 선택"""
    attempts = []
    if IS_LINUX and CAMERA_MODE in ("auto", "usb") and os.path.exists(USB_DEVICE):
        # 1순위: 16:9 1280x720 (MJPG). 640x480(4:3)은 센서를 잘라 써서 화면이 확대돼 보임
        attempts.append((f"USB 카메라 {CAM_WIDTH}x{CAM_HEIGHT} (MJPG)", open_usb_wide))
        attempts.append(("USB 카메라 (GStreamer MJPG)", lambda: cv2.VideoCapture(USB_MJPG_PIPELINE, cv2.CAP_GSTREAMER)))
        attempts.append(("USB 카메라 640x480 (GStreamer)", lambda: cv2.VideoCapture(USB_PIPELINE, cv2.CAP_GSTREAMER)))
    if IS_LINUX and CAMERA_MODE in ("auto", "csi") and shutil.which("gst-inspect-1.0"):
        attempts.append(("CSI 카메라", lambda: cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)))
    if CAMERA_MODE in ("auto", "index"):
        backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
        attempts.append((f"카메라 {CAM_INDEX}번", lambda: cv2.VideoCapture(CAM_INDEX, backend)))

    for name, opener in attempts:
        try:
            cap = opener()
        except Exception:
            continue
        if cap is not None and cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"[Camera] {name} 사용 → 실제 해상도 {frame.shape[1]}x{frame.shape[0]}")
                return cap
        if cap is not None:
            cap.release()
    return None


def open_usb_wide():
    """V4L2로 USB 카메라를 MJPG 1280x720으로 열기 (화각 넓게)"""
    index = int(re.sub(r"\D", "", USB_DEVICE) or 0)
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def yolo_to_objects(result):
    """수업 6-A 방식: YOLO 결과 → Structured Output(dict list)"""
    objects = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        objects.append({
            "class": result.names[class_id],
            "confidence": round(float(box.conf[0].item()), 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    return objects


def point_in_box(pt, bbox, margin):
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (x1 - mx) <= pt[0] <= (x2 + mx) and (y1 - my) <= pt[1] <= (y2 + my)


def box_overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy > 0


def analyze_pose(landmarks, w, h):
    """MediaPipe Pose 33 landmark → 손 위치, 고개 방향, 고개 숙임"""
    info = {"hands": [], "yaw_ratio": 0.0, "looking_away": False, "head_down": False,
            "shoulder_tilt": 0.0, "head_tilt": 0.0, "bad_posture": False, "visible": True}

    def px(i):
        p = landmarks[i]
        return (p.x * w, p.y * h)

    def vis(i):
        # mediapipe 버전에 따라 visibility가 None으로 오는 경우가 있어 그때는 보이는 것으로 처리
        v = getattr(landmarks[i], "visibility", None)
        return 1.0 if v is None else float(v)

    # 손: 손목(15,16) + 검지(19,20)
    for i in (15, 16, 19, 20):
        if vis(i) > 0.3:
            info["hands"].append(px(i))

    nose = px(0)
    l_ear, r_ear = px(7), px(8)
    l_sh, r_sh = px(11), px(12)

    # 고개 좌우 회전
    ear_dist = abs(l_ear[0] - r_ear[0])
    if ear_dist > 1:
        ear_mid_x = (l_ear[0] + r_ear[0]) / 2
        info["yaw_ratio"] = (nose[0] - ear_mid_x) / ear_dist
    ear_vis_diff = abs(vis(7) - vis(8))
    info["looking_away"] = abs(info["yaw_ratio"]) > YAW_RATIO_TH or ear_vis_diff > EAR_VIS_DIFF_TH

    # 고개 숙임: 코가 어깨선에 가까워짐
    shoulder_w = abs(l_sh[0] - r_sh[0])
    if shoulder_w > 20 and vis(11) > 0.5 and vis(12) > 0.5:
        shoulder_mid_y = (l_sh[1] + r_sh[1]) / 2
        info["head_down"] = (shoulder_mid_y - nose[1]) / shoulder_w < HEAD_DOWN_RATIO_TH
        # 자세 흐트러짐 ①: 어깨선 기울기
        info["shoulder_tilt"] = float(np.degrees(np.arctan2(abs(l_sh[1] - r_sh[1]), shoulder_w)))

    # 자세 흐트러짐 ②: 양 귀를 잇는 선의 기울기 (고개를 옆으로 꺾음, 턱 괴기)
    if ear_dist > 1 and vis(7) > 0.5 and vis(8) > 0.5:
        info["head_tilt"] = float(np.degrees(np.arctan2(abs(l_ear[1] - r_ear[1]), ear_dist)))

    info["bad_posture"] = info["shoulder_tilt"] > SHOULDER_TILT_TH or info["head_tilt"] > HEAD_TILT_TH

    info["keypoints"] = {k: px(i) for k, i in
                         [("nose", 0), ("l_ear", 7), ("r_ear", 8), ("l_sh", 11),
                          ("r_sh", 12), ("l_wr", 15), ("r_wr", 16)] if vis(i) > 0.3}
    # 화면에 스켈레톤을 그리기 위한 전체 33개 점
    info["all_points"] = [(px(i), vis(i)) for i in range(len(landmarks))]
    return info


# 상반신 위주 스켈레톤 연결선 (MediaPipe Pose index)
POSE_CONNECTIONS = [
    (0, 2), (2, 7), (0, 5), (5, 8), (9, 10),                 # 얼굴
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),        # 어깨·팔
    (15, 19), (16, 20), (11, 23), (12, 24), (23, 24),        # 손·몸통
]


def decide_state(objects, pose_info):
    """규칙 기반 상태 판단 (우선순위: 자리비움 > 휴대폰 > 딴곳 > 졸음 > 자세 흐트러짐 > 집중)"""
    persons = [o for o in objects if o["class"] == "person"]
    person_present = bool(persons) or pose_info is not None

    held, nearby = [], []
    for o in objects:
        if o["class"] == "person":
            continue
        is_held = bool(pose_info) and any(
            point_in_box(hp, o["bbox"], HAND_BOX_MARGIN) for hp in pose_info["hands"])
        if is_held:
            held.append(o["class"])
        elif any(box_overlap(o["bbox"], p["bbox"]) for p in persons):
            nearby.append(o["class"])

    if not person_present:
        state = STATE_ABSENT
    elif DISTRACT_OBJECTS & set(held) or DISTRACT_OBJECTS & set(nearby):
        state = STATE_PHONE
    elif pose_info and pose_info["looking_away"]:
        state = STATE_AWAY
    elif pose_info and pose_info["head_down"] and not (STUDY_OBJECTS & set(held)):
        state = STATE_DROWSY
    elif pose_info and pose_info["bad_posture"]:
        state = STATE_POSTURE
    else:
        state = STATE_FOCUS

    return state, sorted(set(held)), sorted(set(nearby))


# ============================================================
# 5. 세션 통계 / 컨텍스트
# ============================================================
class Session:
    def __init__(self):
        self.start = time.time()
        self.focus_sec = 0.0
        self.distract_sec = 0.0
        self.event_counts = {s: 0 for s in FALLBACK_SCOLD}
        self.history = []          # (time, state, user_answer, ai_answer)

    def update(self, state, dt):
        if state == STATE_FOCUS:
            self.focus_sec += dt
        else:
            self.distract_sec += dt

    @property
    def focus_rate(self):
        total = self.focus_sec + self.distract_sec
        return 100.0 * self.focus_sec / total if total > 0 else 100.0


def build_context(state, held, nearby, duration, session):
    ctx = {
        "state": state,
        "state_ko": STATE_KO[state],
        "held_objects": held,
        "nearby_objects": nearby,
        "duration_sec": round(duration, 1),
        "focus_rate": round(session.focus_rate, 1),
        "distraction_count": sum(session.event_counts.values()),
        "elapsed_min": round((time.time() - session.start) / 60, 1),
    }
    # 수업 6-A-3 : JSON 저장 (임시파일 → os.replace)
    with open(TEMP_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=4)
    os.replace(TEMP_JSON_PATH, JSON_PATH)
    return ctx


def context_to_text(ctx):
    """수업 6-A-4 : JSON → Natural Language"""
    lines = [
        f"현재 상태: {ctx['state_ko']}",
        f"손에 든 물체: {', '.join(ctx['held_objects']) if ctx['held_objects'] else '없음'}",
        f"주변 물체: {', '.join(ctx['nearby_objects']) if ctx['nearby_objects'] else '없음'}",
        f"이 상태가 지속된 시간: {ctx['duration_sec']:.0f}초",
        f"공부 시작 후 경과 시간: {ctx['elapsed_min']}분",
        f"지금까지 집중이 끊긴 횟수: {ctx['distraction_count']}번",
        f"전체 집중률: {ctx['focus_rate']:.0f}%",
    ]
    return "\n".join(lines)


# ============================================================
# 6. LLM/음성 작업 스레드 (카메라 루프가 멈추지 않도록 분리)
# ============================================================
class Assistant(threading.Thread):
    def __init__(self, brain, session):
        super().__init__(daemon=True)
        self.brain = brain
        self.session = session
        self.jobs = queue.Queue()
        self.status = "IDLE"
        self.last_message = ""

    @property
    def busy(self):
        return self.status != "IDLE" or not self.jobs.empty()

    def set_status(self, s):
        self.status = s

    def submit(self, kind, ctx, frame):
        self.jobs.put((kind, ctx, frame))

    def say(self, text):
        self.last_message = text
        self.set_status("SPEAKING")
        text_to_speech(text)

    def run(self):
        while True:
            kind, ctx, frame = self.jobs.get()
            try:
                if kind == "event":
                    self.handle_event(ctx, frame)
                elif kind == "voice":
                    self.handle_voice(ctx, frame)
                elif kind == "report":
                    self.handle_report(ctx)
            except Exception as e:
                print("[Assistant] 오류:", e)
            self.set_status("IDLE")

    def handle_event(self, ctx, frame):
        self.set_status("THINKING")
        ctx_text = context_to_text(ctx)
        print("\n========== 집중 이탈 이벤트 ==========\n" + ctx_text)

        scold = self.brain.ask(SCOLD_PROMPT, f"Context:\n{ctx_text}", frame)
        scold = scold or FALLBACK_SCOLD.get(ctx["state"], "왜 집중을 안 하고 있어요?")
        self.say(scold)

        if not LISTEN_AFTER_WARNING:
            return
        answer = speech_to_text(self.set_status)
        if not answer:
            self.say("대답이 없네요. 다시 공부 시작합시다.")
            return

        self.set_status("THINKING")
        reply = self.brain.ask(
            EXCUSE_PROMPT,
            f"Context:\n{ctx_text}\n\n감독관이 한 말: {scold}\n사용자의 대답: {answer}",
            frame,
        ) or "알겠어요. 그럼 이제 다시 집중해 봅시다."
        self.session.history.append((time.strftime("%H:%M:%S"), ctx["state"], answer, reply))
        self.say(reply)

    def handle_voice(self, ctx, frame):
        question = speech_to_text(self.set_status)
        if not question:
            return
        self.set_status("THINKING")
        reply = self.brain.ask(
            QA_PROMPT, f"Context:\n{context_to_text(ctx)}\n\n사용자 질문: {question}", frame
        ) or "지금은 답을 만들 수 없어요. 일단 공부부터 합시다."
        self.say(reply)

    def handle_report(self, ctx):
        s = self.session
        counts = ", ".join(f"{STATE_KO[k]} {v}번" for k, v in s.event_counts.items())
        stats = (
            f"공부 시간: {(time.time() - s.start) / 60:.1f}분\n"
            f"집중한 시간: {s.focus_sec / 60:.1f}분\n"
            f"집중률: {s.focus_rate:.0f}%\n"
            f"경고 횟수: {counts}"
        )
        print("\n========== 공부 리포트 ==========\n" + stats)
        self.set_status("THINKING")
        reply = self.brain.ask(REPORT_PROMPT, f"Context:\n{stats}") or \
            f"오늘 집중률은 {s.focus_rate:.0f}퍼센트입니다. 다음에는 더 집중해 봅시다."
        self.say(reply)


# ============================================================
# 7. 화면 표시
# ============================================================
def draw_overlay(frame, objects, held, pose_info, state, duration, session, assistant, pose_status=""):
    h, w = frame.shape[:2]
    color = STATE_COLOR[state]

    # 자세 추정 스켈레톤
    if pose_info:
        pts = pose_info["all_points"]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts) and pts[a][1] > 0.4 and pts[b][1] > 0.4:
                cv2.line(frame, tuple(map(int, pts[a][0])), tuple(map(int, pts[b][0])), (255, 255, 255), 2)
        for (x, y), v in pts[:25]:
            if v > 0.4:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)

    for o in objects:
        x1, y1, x2, y2 = o["bbox"]
        c = (0, 0, 255) if o["class"] in DISTRACT_OBJECTS else \
            (255, 200, 0) if o["class"] in STUDY_OBJECTS else (200, 200, 200)
        thick = 3 if o["class"] in held else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thick)
        label = f"{o['class']} {o['confidence']:.2f}" + (" [HELD]" if o["class"] in held else "")
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

    if pose_info:
        for name, (x, y) in pose_info["keypoints"].items():
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
        kp = pose_info["keypoints"]
        if "l_sh" in kp and "r_sh" in kp:
            p1, p2 = tuple(map(int, kp["l_sh"])), tuple(map(int, kp["r_sh"]))
            cv2.line(frame, p1, p2, STATE_COLOR[STATE_POSTURE] if pose_info["bad_posture"] else (0, 255, 255), 2)
        # 임계값 튜닝용 수치 표시
        cv2.putText(frame,
                    f"yaw {pose_info['yaw_ratio']:+.2f}  shoulder {pose_info['shoulder_tilt']:.0f}deg  "
                    f"head {pose_info['head_tilt']:.0f}deg",
                    (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # 상단 패널
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.putText(frame, f"STATE: {state}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, f"Focus {session.focus_rate:.0f}%  |  Warnings {sum(session.event_counts.values())}",
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)

    # 이탈 타이머 게이지
    if state != STATE_FOCUS:
        limit = ABSENT_SEC if state == STATE_ABSENT else DISTRACT_SEC
        ratio = min(1.0, duration / limit)
        cv2.rectangle(frame, (w - 220, 15), (w - 20, 35), (80, 80, 80), 1)
        cv2.rectangle(frame, (w - 220, 15), (w - 220 + int(200 * ratio), 35), color, -1)

    # 집중률 바
    bar_w = int((w - 24) * session.focus_rate / 100)
    cv2.rectangle(frame, (12, h - 20), (12 + bar_w, h - 10), (80, 200, 80), -1)

    if assistant.status != "IDLE":
        cv2.putText(frame, f"AI: {assistant.status}", (w - 260, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, "q:quit  v:voice  r:report  e:test", (12, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 자세 추정 동작 상태 표시
    pose_color = (80, 220, 80) if pose_status == "POSE: OK" else (0, 0, 255)
    cv2.putText(frame, pose_status, (w - 220, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pose_color, 2)
    return frame


# ============================================================
# 8. 메인 루프
# ============================================================
def main():
    print(f"[Env] 수업 폴더: {BASE_DIR}")
    print(f"[Env] 파이썬: {sys.executable}")
    print(f"[Mic] {MIC_DEVICE}")
    print(f"[STT] whisper.cpp: {'OK' if _whisper_exe() and os.path.exists(WHISPER_MODEL) else '⚠ 없음 → 키보드 입력'}")
    print(f"[TTS] Piper: {'OK' if os.path.exists(PIPER_PYTHON) and os.path.exists(PIPER_MODEL) else '⚠ 없음 → 콘솔 출력'}")

    brain = GemmaBrain()

    # 시작할 때 음성 출력 테스트 (여기서 소리가 안 나면 스피커/Piper 문제)
    print("[TTS] 음성 출력 테스트")
    text_to_speech("포커스 가드를 시작합니다.")

    yolo = load_yolo()
    pose_detector, mp = load_pose()
    session = Session()
    assistant = Assistant(brain, session)
    assistant.start()

    cap = open_camera()
    if cap is None:
        print("카메라를 열 수 없습니다. 카메라 연결을 확인하세요.")
        sys.exit(1)

    print("q: 종료 | v: 음성 질문 | r: 공부 리포트 | e: 테스트 경고")

    state, state_since = STATE_FOCUS, time.time()
    last_alert = 0.0
    prev_t = time.time()
    pose_error_printed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if FLIP_CODE is not None:
            frame = cv2.flip(frame, FLIP_CODE)

        now = time.time()
        dt, prev_t = now - prev_t, now
        h, w = frame.shape[:2]

        # --- 1) YOLO 물체 탐지 ---
        result = yolo.predict(source=frame, conf=0.35, iou=0.5, classes=YOLO_CLASSES, verbose=False)[0]
        objects = yolo_to_objects(result)

        # --- 2) MediaPipe 자세 추정 ---
        pose_info = None
        if pose_detector is None:
            pose_status = "POSE: NO MODEL"
        else:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                if pose_result.pose_landmarks:
                    pose_info = analyze_pose(pose_result.pose_landmarks[0], w, h)
                    pose_status = "POSE: OK"
                else:
                    pose_status = "POSE: NO PERSON"
            except Exception as e:
                pose_status = "POSE: ERROR"
                if not pose_error_printed:
                    print("[Pose] 추론 오류:", e)
                    pose_error_printed = True

        # --- 3) 규칙 기반 상태 판단 ---
        new_state, held, nearby = decide_state(objects, pose_info)
        if new_state != state:
            state, state_since = new_state, now
        duration = now - state_since
        session.update(state, dt)

        # --- 4) 이벤트 감지 → Gemma + TTS ---
        limit = ABSENT_SEC if state == STATE_ABSENT else DISTRACT_SEC
        triggered = (state != STATE_FOCUS and duration >= limit
                     and now - last_alert > COOLDOWN_SEC and not assistant.busy)

        display = draw_overlay(frame.copy(), objects, held, pose_info, state, duration, session, assistant,
                               pose_status)
        cv2.imshow("Focus Guard (YOLO + Pose + Gemma)", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("e") and not assistant.busy:      # 발표용 강제 트리거
            triggered = True
            if state == STATE_FOCUS:
                state = STATE_PHONE

        if triggered:
            last_alert = now
            session.event_counts[state] += 1
            ctx = build_context(state, held, nearby, duration, session)
            cv2.imwrite(SNAPSHOT_PATH, display)
            assistant.submit("event", ctx, frame.copy())

        if key == ord("q"):
            break
        elif key == ord("v") and not assistant.busy:
            ctx = build_context(state, held, nearby, duration, session)
            assistant.submit("voice", ctx, frame.copy())
        elif key == ord("r") and not assistant.busy:
            ctx = build_context(state, held, nearby, duration, session)
            assistant.submit("report", ctx, None)

    cap.release()
    cv2.destroyAllWindows()
    if pose_detector is not None:
        pose_detector.close()

    print(f"\n최종 집중률: {session.focus_rate:.0f}%  |  경고: {session.event_counts}")
    for t, st, ans, rep in session.history:
        print(f"[{t}] {st} | 사용자: {ans} | AI: {rep}")


if __name__ == "__main__":
    main()
