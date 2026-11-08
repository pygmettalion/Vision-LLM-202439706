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

* Jetson / 노트북 둘 다 동작하도록, 없는 구성요소는 자동으로 대체(fallback)합니다.
  - TensorRT 엔진 없으면 yolo11n.pt
  - Gemma mmproj 없으면 텍스트 전용, Gemma 자체가 없으면 규칙 기반 문장
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

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# 1. 설정
# ============================================================
IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"

# ---- 카메라 ----
USE_JETSON_CSI = False      # Jetson CSI 카메라면 True, 노트북 웹캠/USB캠이면 False
CAM_INDEX = 0
FLIP_CODE = None            # 수업 코드처럼 상하반전 필요하면 0, 좌우반전(거울)은 1, 없으면 None

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

# ---- 모델 경로 (수업 폴더 구조 기준) ----
YOLO_ENGINE_PATH = "src/models/YOLO/yolo11n_int8.engine"
YOLO_PT_PATH = "src/models/YOLO/yolo11n.pt"
POSE_MODEL_PATH = "src/models/MediaPipe/pose_landmarker_full.task"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"

CONTEXT_WINDOW = 2048
MAX_TOKENS = 120

# ---- STT (whisper.cpp) ----
MIC_DEVICE = "plughw:3,0"   # arecord -l 로 확인한 번호로 수정
RECORD_SECONDS = 5
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"

# ---- TTS (Piper) ----
PIPER_PYTHON = ".piper_venv/Scripts/python.exe" if IS_WINDOWS else ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
TTS_FILE = "src/audio/response.wav"

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

for d in ["src/audio", "src/json", "src/output"]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# 2. 음성 (STT / TTS)
# ============================================================
def _clean_for_speech(text):
    text = re.sub(r"[*#`>\[\]_~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def play_wav(path):
    try:
        if IS_LINUX and shutil.which("aplay"):
            subprocess.run(["aplay", "-q", path], check=True)
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
        try:
            subprocess.run(
                [PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", TTS_FILE, "--", text],
                check=True, capture_output=True,
            )
            play_wav(TTS_FILE)
            return
        except Exception as e:
            print("[TTS] Piper 실패:", e)

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
    def __init__(self):
        self.llm = None
        self.vision = False
        try:
            from llama_cpp import Llama
        except ImportError:
            print("[Gemma] llama_cpp 없음 → 규칙 기반 문장으로 대체")
            return

        if not os.path.exists(GEMMA_MODEL_PATH):
            print(f"[Gemma] 모델 없음({GEMMA_MODEL_PATH}) → 규칙 기반 문장으로 대체")
            return

        chat_handler = None
        if os.path.exists(MMPROJ_PATH):
            try:
                from llama_cpp.llama_chat_format import Gemma4ChatHandler
                chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)
                self.vision = True
            except Exception as e:
                print("[Gemma] 이미지 핸들러 로드 실패 → 텍스트 전용:", e)

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

        if self.vision and image_bgr is not None:
            h, w = image_bgr.shape[:2]
            scale = 640 / max(h, w)
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
            return response["choices"][0]["message"]["content"].strip()
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
    if os.path.exists(YOLO_ENGINE_PATH):
        print("[YOLO] TensorRT INT8 엔진 사용")
        return YOLO(YOLO_ENGINE_PATH)
    path = YOLO_PT_PATH if os.path.exists(YOLO_PT_PATH) else "yolo11n.pt"
    print(f"[YOLO] {path} 사용")
    return YOLO(path)


def load_pose():
    if not os.path.exists(POSE_MODEL_PATH):
        print(f"[Pose] {POSE_MODEL_PATH} 없음 → 자세 판단 없이 YOLO만 사용")
        print("  wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_full/float16/latest/pose_landmarker_full.task -O " + POSE_MODEL_PATH)
        return None, None
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(options), mp


def open_camera():
    if USE_JETSON_CSI:
        return cv2.VideoCapture(CSI_PIPELINE, cv2.CAP_GSTREAMER)
    if IS_WINDOWS:
        return cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    return cv2.VideoCapture(CAM_INDEX)


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
        return getattr(landmarks[i], "visibility", 1.0) or 0.0

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
    return info


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
def draw_overlay(frame, objects, held, pose_info, state, duration, session, assistant):
    h, w = frame.shape[:2]
    color = STATE_COLOR[state]

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
    return frame


# ============================================================
# 8. 메인 루프
# ============================================================
def main():
    yolo = load_yolo()
    pose_detector, mp = load_pose()
    brain = GemmaBrain()
    session = Session()
    assistant = Assistant(brain, session)
    assistant.start()

    cap = open_camera()
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        sys.exit(1)

    print("q: 종료 | v: 음성 질문 | r: 공부 리포트 | e: 테스트 경고")

    state, state_since = STATE_FOCUS, time.time()
    last_alert = 0.0
    prev_t = time.time()

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
        if pose_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = pose_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if pose_result.pose_landmarks:
                pose_info = analyze_pose(pose_result.pose_landmarks[0], w, h)

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

        display = draw_overlay(frame.copy(), objects, held, pose_info, state, duration, session, assistant)
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
