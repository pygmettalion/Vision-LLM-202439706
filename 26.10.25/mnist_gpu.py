"""
[261025] 03_DL-and-GPU
MNIST 데이터셋 CPU vs. GPU 학습 비교 – main_gpu() 구현

 - main_cpu() : 수업 코드 (CPU 학습)
 - main_gpu() : 같은 모델/하이퍼파라미터를 CUDA GPU에서 학습
     * device = cuda, pin_memory=True
     * GPU Warm-up으로 CUDA Context / cuDNN 초기화 시간 제외
     * torch.cuda.synchronize()로 GPU 작업이 끝난 시점에 시간 측정

실행 (Jetson, 레포 최상위 폴더 어디서든):
    python3 261025/mnist_gpu.py
"""

import time
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]

SEED = 42
DATA_ROOT = str(ROOT / "src" / "datasets")

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 3
LEARNING_RATE = 0.001
NUM_WORKERS = 0


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct_count += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    return total_loss / sample_count, correct_count / sample_count


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct_count += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, correct_count / sample_count


def build_loaders(pin_memory):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory)

    return train_loader, test_loader


def print_epoch(epoch, train_loss, train_accuracy, test_loss, test_accuracy, epoch_time):
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: {test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )


def main_cpu():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader, test_loader = build_loaders(pin_memory=False)

    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        print_epoch(epoch, train_loss, train_accuracy, test_loss, test_accuracy,
                    time.perf_counter() - epoch_start_time)

    total_training_time = time.perf_counter() - training_start_time
    print(f"Total CPU training time: {total_training_time:.2f}s")

    return total_training_time


def main_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # pin_memory=True: CPU → GPU 전송 속도 향상
    train_loader, test_loader = build_loaders(pin_memory=True)

    # 장치를 GPU로 지정하고 모델을 GPU에 할당
    device = torch.device("cuda")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # GPU Warm-up: 첫 연산의 초기화 시간을 학습 시간에서 제외
    warmup_input = torch.zeros(BATCH_SIZE, 1, 28, 28, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(warmup_input)
    torch.cuda.synchronize()

    # GPU 작업이 모두 끝난 시점부터 측정
    torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        torch.cuda.synchronize()
        print_epoch(epoch, train_loss, train_accuracy, test_loss, test_accuracy,
                    time.perf_counter() - epoch_start_time)

    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    return total_training_time


if __name__ == "__main__":
    print("=== CPU Training ===")
    cpu_training_time = main_cpu()

    print("\n=== GPU Training ===")
    gpu_training_time = main_gpu()

    speedup = cpu_training_time / gpu_training_time
    time_difference = cpu_training_time - gpu_training_time

    print("\n=== CPU vs. GPU Comparison ===")
    print(f"CPU training time: {cpu_training_time:.2f}s")
    print(f"GPU training time: {gpu_training_time:.2f}s")
    print(f"Time difference: {time_difference:.2f}s")
    print(f"GPU speedup: {speedup:.2f}x")
