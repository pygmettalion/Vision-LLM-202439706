from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
CAMERA_INDEX = 0


JAW_INDICES = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]


def point(landmarks, index, width, height):
    lm = landmarks[index]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def distance(p1, p2):
    return float(np.linalg.norm(p1 - p2))


def extract_features(landmarks, width, height):
    p = lambda index: point(landmarks, index, width, height)

    jaw_left, jaw_right = p(172), p(397)
    jaw_width = distance(jaw_left, jaw_right)
    if jaw_width < 5:
        return None

    jaw_center = (jaw_left + jaw_right) / 2
    contour = np.array([p(i) for i in JAW_INDICES], dtype=np.float32)

    return {
        "mouth_ratio": distance(p(61), p(291)) / jaw_width,
        "height_ratio": distance(p(2), p(152)) / jaw_width,
        "taper_ratio": distance(p(148), p(377)) / jaw_width,
        "contour": (contour - jaw_center) / jaw_width,
    }


def ratio_similarity(a, b):
    return max(0.0, 1.0 - abs(a - b) / max(abs(a), abs(b), 1e-6))


def calculate_destiny(a, b):
    mouth = ratio_similarity(a["mouth_ratio"], b["mouth_ratio"])
    height = ratio_similarity(a["height_ratio"], b["height_ratio"])
    taper = ratio_similarity(a["taper_ratio"], b["taper_ratio"])

    contour_distances = np.linalg.norm(a["contour"] - b["contour"], axis=1)
    contour_error = float(np.mean(contour_distances))
    contour = max(0.0, 1.0 - contour_error / 0.30)

    score = 100 * (0.25 * mouth + 0.20 * height + 0.20 * taper + 0.35 * contour)
    return float(np.clip(score, 0.0, 100.0))


def draw_lower_face(frame, landmarks, color, label):
    height, width = frame.shape[:2]
    jaw = np.array([
        point(landmarks, idx, width, height).astype(int)
        for idx in JAW_INDICES
    ], dtype=np.int32)

    cv2.polylines(frame, [jaw], False, color, 2, cv2.LINE_AA)
    for xy in jaw:
        cv2.circle(frame, tuple(xy), 3, color, -1, cv2.LINE_AA)

    for idx in [2, 61, 291, 152]:
        xy = point(landmarks, idx, width, height).astype(int)
        cv2.circle(frame, tuple(xy), 5, (255, 255, 255), -1, cv2.LINE_AA)

    chin = point(landmarks, 152, width, height).astype(int)
    label_x = min(max(int(chin[0]) - 35, 10), max(width - 90, 10))
    label_y = min(max(int(chin[1]) + 30, 100), height - 10)
    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=2,
    )


    cap = cv2.VideoCapture(CAMERA_INDEX)


    
    if not cap.isOpened():
        raise RuntimeError("Camera unavailable. Check or permissions.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_timestamp = -1
    smooth_score = None

    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Could not read camera frame")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp = max(int(time.monotonic() * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                result = landmarker.detect_for_video(mp_image, timestamp)

                # Sort by position on screen; these are not permanent identity labels.
                faces = sorted(result.face_landmarks,
                               key=lambda lm: (lm[172].x + lm[397].x) / 2)

                for index, landmarks in enumerate(faces):
                    color = (0, 240, 255) if index == 0 else (255, 180, 50)
                    draw_lower_face(frame, landmarks, color, f"PERSON {index + 1}")

                message = f"FACES DETECTED: {len(faces)}/2"
                if len(faces) == 2:
                    feature_a = extract_features(faces[0], width, height)
                    feature_b = extract_features(faces[1], width, height)
                    if feature_a is not None and feature_b is not None:
                        score = calculate_destiny(feature_a, feature_b)
                        smooth_score = score if smooth_score is None else 0.75 * smooth_score + 0.25 * score
                        message = f"JAW DESTINY: {smooth_score:.1f}%"
                    else:
                        smooth_score = None
                        message = "MOVE CLOSER TO CAMERA"
                else:
                    smooth_score = None
                    message = "SHOW TWO FACES TO COMPARE"

                cv2.rectangle(frame, (0, 0), (width, 90), (20, 20, 26), -1)
                cv2.putText(frame, "JAWLINE DESTINY THEORY", (20, 33),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, message, (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 240, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press Q or ESC to exit", (20, height - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Jawline Destiny Theory", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()