"""
[261025] 03_DL-and-GPU
GPU 가속 1 – 이미지 잔상 효과 (NumPy CPU vs. CuPy GPU)

원리: 원본 이미지를 오른쪽으로 0, 1, 2, ... 픽셀 이동한 복사본에
     점점 작아지는 가중치를 곱해 모두 더함

 - CPU (NumPy) : 출력 픽셀을 Python 반복문으로 하나씩 계산
 - GPU (CuPy)  : 이동 횟수만 반복, 각 이동 단계의 전체 픽셀 연산은 GPU에서 병렬 처리

실행 (Jetson, 레포 최상위 폴더 어디서든):
    python3 261025/gpu_motion_trail.py
결과 이미지: 261025/motion_trail_result.png
※ CPU 반복문이 느려서 Jetson에서 1~3분 정도 걸릴 수 있음
"""

import time
from pathlib import Path

import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "src" / "images" / "seagull.jpg"
RESULT_PATH = Path(__file__).resolve().parent / "motion_trail_result.png"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32


def motion_trail_np(image, weights):
    """NumPy: 출력 픽셀을 Python 반복문으로 하나씩 계산"""
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float32)

    for output_y in range(height):
        for output_x in range(width):
            accumulated = 0.0

            for shift in range(len(weights)):
                # x값 shift
                input_x = output_x - shift

                # 이미지 왼쪽 밖이면 0(검정)으로 취급
                if input_x < 0:
                    break

                # shift한 픽셀을 누적
                accumulated += image[output_y, input_x] * weights[shift]

            # 출력 배열에 누적 픽셀 저장
            output[output_y, output_x] = accumulated

    return output


def motion_trail_cp(image, weights):
    """CuPy: 이동 거리만 순차 반복, 이미지 전체 계산은 GPU 병렬 처리"""
    height, width = image.shape
    output = cp.zeros((height, width), dtype=cp.float32)

    for shift in range(len(weights)):
        if shift >= width:
            break

        # 입력의 [0, width-shift) 구간이 출력의 [shift, width) 구간에 더해짐
        input_region = image[:, 0:width - shift]
        output_region = output[:, shift:width]

        # GPU가 배열 전체를 병렬로 계산
        output_region += input_region * weights[shift]

    return output


def main():
    # ---------------- 이미지 불러오기 ----------------
    img_gray = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    img_gray = cv2.resize(img_gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    input_np = img_gray.astype(np.float32) / 255.0

    # ---------------- 가중치 ----------------
    weights_np = np.linspace(1.0, 0.05, TRAIL_LENGTH, dtype=np.float32)
    weights_np /= np.sum(weights_np)

    # ---------------- NumPy CPU ----------------
    print("NumPy CPU 처리 중... (1~3분 걸릴 수 있음)")
    cpu_start = time.perf_counter()
    output_np = motion_trail_np(input_np, weights_np)
    cpu_time = time.perf_counter() - cpu_start

    # ---------------- CPU → GPU ----------------
    upload_start = time.perf_counter()
    input_cp = cp.asarray(input_np)
    weights_cp = cp.asarray(weights_np)
    cp.cuda.Stream.null.synchronize()
    upload_time = time.perf_counter() - upload_start

    # ---------------- GPU 워밍업 ----------------
    _ = motion_trail_cp(cp.zeros((64, 64), dtype=cp.float32), cp.asarray(weights_np[:4]))
    cp.cuda.Stream.null.synchronize()

    # ---------------- CuPy GPU ----------------
    cp.cuda.Stream.null.synchronize()
    gpu_start = time.perf_counter()
    output_cp = motion_trail_cp(input_cp, weights_cp)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - gpu_start

    # ---------------- GPU → CPU ----------------
    download_start = time.perf_counter()
    output_cp_np = cp.asnumpy(output_cp)
    download_time = time.perf_counter() - download_start

    # ---------------- 결과 비교 ----------------
    gpu_total_time = upload_time + gpu_time + download_time

    print("========== NumPy CPU ==========")
    print(f"Processing time: {cpu_time:.4f} seconds")
    print()
    print("========== CuPy GPU ==========")
    print(f"CPU to GPU: {upload_time:.4f} seconds")
    print(f"GPU processing: {gpu_time:.4f} seconds")
    print(f"GPU to CPU: {download_time:.4f} seconds")
    print(f"Total GPU time: {gpu_total_time:.4f} seconds")
    print()
    print("========== Comparison ==========")
    print(f"Compute-only speedup: {cpu_time / gpu_time:.2f}x")
    print(f"Including transfers: {cpu_time / gpu_total_time:.2f}x")
    print("CPU/GPU 결과 일치:", np.allclose(output_np, output_cp_np, atol=1e-5))

    # ---------------- 이미지 출력 ----------------
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
    plt.subplot(1, 3, 2), plt.imshow(output_np, cmap="gray"), plt.title(f"Motion Trail (CPU) {cpu_time:.2f}s")
    plt.subplot(1, 3, 3), plt.imshow(output_cp_np, cmap="gray"), plt.title(f"Motion Trail (GPU) {gpu_time:.4f}s")
    for ax in plt.gcf().axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULT_PATH, dpi=100)
    print(f"결과 저장: {RESULT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
