"""
[261025] 03_DL-and-GPU
GPU 가속 2 – Convolution2D (NumPy CPU vs. CuPy GPU)

 - CPU (NumPy) : 출력 픽셀마다 커널 크기 영역을 잘라 가중합 (출력 위치를 순차 반복)
 - GPU (CuPy)  : 커널 원소(ky, kx)를 순차 반복, 모든 출력 위치의 대응 입력 픽셀을
                 한 번에 선택해 가중치를 곱하고 출력 배열 전체에 누적 (GPU 병렬)

세로/가로 edge 필터 + 편향(-0.2) + ReLU로 Feature Map을 만들고 CPU/GPU 결과를 비교

실행 (Jetson, 레포 최상위 폴더 어디서든):
    python3 261025/gpu_convolution2d.py
결과 이미지: 261025/convolution2d_result.png
"""

import time
from pathlib import Path

import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "src" / "images" / "seagull.jpg"
RESULT_PATH = Path(__file__).resolve().parent / "convolution2d_result.png"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# x축 방향 edge (세로 선) 검출 필터
dx_edge_filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=np.float32)

# y축 방향 edge (가로 선) 검출 필터
dy_edge_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float32)

# 너무 약한 특징 (0.2 이하)은 무시
dx_bias = -0.2
dy_bias = -0.2


def relu(x):
    return np.maximum(0, x)


def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    """NumPy CPU convolution (수업 코드와 동일)"""
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(image, ((padding, padding), (padding, padding)),
                          mode="constant", constant_values=0)

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = np.zeros((output_height, output_width), dtype=np.float32)

    for output_y in range(output_height):
        for output_x in range(output_width):
            start_y = output_y * stride
            start_x = output_x * stride

            image_region = padded_image[start_y:start_y + kernel_height,
                                        start_x:start_x + kernel_width]

            output[output_y, output_x] = np.sum(image_region * kernel) + bias

    return output


def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """CuPy GPU convolution: 커널 원소만 순차 반복, 출력 배열 전체는 GPU 병렬 연산"""
    kernel_height, kernel_width = kernel.shape

    padded_image = cp.pad(image, ((padding, padding), (padding, padding)),
                          mode="constant", constant_values=0)

    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    output = cp.zeros((output_height, output_width), dtype=cp.float32)

    for ky in range(kernel_height):
        for kx in range(kernel_width):
            # 모든 출력 위치에서 현재 커널 원소와 대응되는 입력 픽셀을 한 번에 선택
            image_region = padded_image[
                ky : ky + stride * (output_height - 1) + 1 : stride,
                kx : kx + stride * (output_width - 1) + 1 : stride,
            ]

            # 현재 커널 가중치를 곱해 출력 배열 전체에 누적
            output += image_region * kernel[ky, kx]

    output += bias

    return output


def main():
    img_gray = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    img_gray = cv2.resize(img_gray, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    input_np = img_gray.astype(np.float32) / 255.0

    # ---------------- NumPy CPU ----------------
    print("NumPy CPU convolution 처리 중...")
    cpu_start = time.perf_counter()
    output_dx_np = convolution2d(input_np, dx_edge_filter, bias=dx_bias, stride=1, padding=1)
    output_dy_np = convolution2d(input_np, dy_edge_filter, bias=dy_bias, stride=1, padding=1)
    cpu_time = time.perf_counter() - cpu_start

    relu_dx_np = relu(output_dx_np)
    relu_dy_np = relu(output_dy_np)

    # ---------------- 1) 변환: CPU → GPU ----------------
    upload_start = time.perf_counter()
    input_cp = cp.asarray(input_np)
    dx_kernel_cp = cp.asarray(dx_edge_filter)
    dy_kernel_cp = cp.asarray(dy_edge_filter)
    cp.cuda.Stream.null.synchronize()
    upload_time = time.perf_counter() - upload_start

    # ---------------- 2) 워밍업 ----------------
    _ = convolution2d_cp(cp.zeros((64, 64), dtype=cp.float32), dx_kernel_cp, bias=dx_bias, padding=1)
    cp.cuda.Stream.null.synchronize()

    # ---------------- 3) 처리: GPU convolution + ReLU ----------------
    gpu_start = time.perf_counter()
    output_dx_cp = convolution2d_cp(input_cp, dx_kernel_cp, bias=dx_bias, stride=1, padding=1)
    output_dy_cp = convolution2d_cp(input_cp, dy_kernel_cp, bias=dy_bias, stride=1, padding=1)
    relu_dx_cp = cp.maximum(0, output_dx_cp)
    relu_dy_cp = cp.maximum(0, output_dy_cp)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - gpu_start

    # ---------------- 4) 결과: GPU → CPU ----------------
    download_start = time.perf_counter()
    relu_dx_cp_np = cp.asnumpy(relu_dx_cp)
    relu_dy_cp_np = cp.asnumpy(relu_dy_cp)
    download_time = time.perf_counter() - download_start

    # ---------------- CPU/GPU 결과 비교 ----------------
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
    print()
    print("========== Result Check ==========")
    print("Vertical edge 결과 일치:", np.allclose(relu_dx_np, relu_dx_cp_np, atol=1e-5))
    print("Horizontal edge 결과 일치:", np.allclose(relu_dy_np, relu_dy_cp_np, atol=1e-5))

    # ---------------- 이미지 출력 ----------------
    plt.figure(figsize=(18, 8))
    plt.subplot(2, 3, 1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
    plt.subplot(2, 3, 3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
    plt.subplot(2, 3, 4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
    plt.subplot(2, 3, 5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
    plt.subplot(2, 3, 6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")
    for ax in plt.gcf().axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(RESULT_PATH, dpi=100)
    print(f"결과 저장: {RESULT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
