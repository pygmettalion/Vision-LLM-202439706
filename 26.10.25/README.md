# [261025] 03_DL-and-GPU / 04_DL-Object-Detection

| 파일 | 내용 |
|---|---|
| `gpu_motion_trail.py` | GPU 가속 1: 이미지 잔상 효과 (NumPy CPU vs. CuPy GPU) |
| `gpu_convolution2d.py` | GPU 가속 2: Convolution2D (NumPy CPU vs. CuPy GPU) |
| `mnist_gpu.py` | MNIST `main_gpu()` 구현 및 CPU/GPU 학습 시간 비교 |
| `cifar10_gpu.py` | CIFAR-10 GPU 학습 + 모델 저장 (`CIFAR10_CNN.pth`) |
| `iou_nms.py` | IoU, Confidence Thresholding, NMS 구현 및 OpenCV NMS와 비교 |

```bash
python3 261025/gpu_motion_trail.py
python3 261025/gpu_convolution2d.py
python3 261025/mnist_gpu.py
python3 261025/cifar10_gpu.py
python3 261025/iou_nms.py
```
