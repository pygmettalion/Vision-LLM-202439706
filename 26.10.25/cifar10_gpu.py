"""
[261025] 03_DL-and-GPU
CIFAR-10 이미지 분류 GPU 가속 학습 + 모델 저장 (.pth)

 1. 데이터 증강(RandomCrop, RandomHorizontalFlip) + 정규화
 2. ConvBlock x3 → Global Average Pooling → Dropout → Linear 구조의 CNN
 3. CUDA GPU에서 5 Epoch 학습/평가, Epoch별·전체 학습 시간 출력
 4. Loss / Accuracy 그래프 저장
 5. 학습된 모델을 .pth로 저장한 뒤 다시 불러와 정확도 확인

실행 (Jetson, 레포 최상위 폴더 어디서든):
    python3 261025/cifar10_gpu.py
결과
    261025/CIFAR10_CNN.pth              학습된 모델
    src/models/CIFAR10/CIFAR10_CNN.pth  (수업 노트북 경로에도 같이 저장)
    261025/cifar10_curves.png           Loss / Accuracy 그래프
    261025/cifar10_predictions.png      테스트 이미지 예측 결과
※ 첫 실행 시 CIFAR-10 데이터셋(약 170MB) 다운로드 필요
"""

import time
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent

DATA_ROOT = str(ROOT / "src" / "datasets")
MODEL_PATH = OUT_DIR / "CIFAR10_CNN.pth"
COURSE_MODEL_PATH = ROOT / "src" / "models" / "CIFAR10" / "CIFAR10_CNN.pth"

SEED = 42
EPOCHS = 5
LEARNING_RATE = 0.001

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ============================================================
# 모델
# ============================================================
class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU → MaxPool"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class CIFAR10CNN(nn.Module):
    """ConvBlock(64) → ConvBlock(128) → ConvBlock(256) → GAP → Dropout → Linear(10)"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),      # [B, 64, 16, 16]
            ConvBlock(64, 128),    # [B, 128, 8, 8]
            ConvBlock(128, 256),   # [B, 256, 4, 4]
        )
        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))   # [B, 256, 1, 1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=256, out_features=10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


# ============================================================
# 학습 / 평가
# ============================================================
def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            correct_count += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return total_loss / sample_count, correct_count / sample_count


# ============================================================
# 메인
# ============================================================
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------------- 데이터 ----------------
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=test_transform, download=True)
    print("Train data:", len(train_dataset), "| Test data:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    # ---------------- 모델 / GPU ----------------
    device = torch.device("cuda")
    model = CIFAR10CNN().to(device)
    print("Device:", device)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ---------------- GPU Warm-up ----------------
    warmup_input = torch.zeros(64, 3, 32, 32, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            _ = model(warmup_input)
    torch.cuda.synchronize()

    # ---------------- 학습 ----------------
    train_loss_history, train_accuracy_history = [], []
    test_loss_history, test_accuracy_history = [], []

    torch.cuda.synchronize()
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    torch.cuda.synchronize()
    total_training_time = time.perf_counter() - training_start_time
    print(f"Total GPU training time: {total_training_time:.2f}s")

    # ---------------- 학습 그래프 ----------------
    epoch_axis = range(1, EPOCHS + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_axis, train_loss_history, marker="o", label="Train loss")
    plt.plot(epoch_axis, test_loss_history, marker="o", label="Test loss")
    plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("CNN Loss"), plt.grid(), plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch_axis, np.array(train_accuracy_history) * 100, marker="o", label="Train accuracy")
    plt.plot(epoch_axis, np.array(test_accuracy_history) * 100, marker="o", label="Test accuracy")
    plt.xlabel("Epoch"), plt.ylabel("Accuracy (%)"), plt.title("CNN Accuracy"), plt.grid(), plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "cifar10_curves.png", dpi=100)

    # ---------------- 모델 저장 (.pth) ----------------
    save_data = {
        "model_name": "CIFAR10CNN",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": test_accuracy_history[-1],
        "epochs": EPOCHS,
        "class_names": CIFAR10_CLASSES,
        "mean": CIFAR10_MEAN,
        "std": CIFAR10_STD,
    }

    COURSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, MODEL_PATH)
    torch.save(save_data, COURSE_MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")
    print(f"모델 저장: {COURSE_MODEL_PATH}")

    # ---------------- 저장한 모델 불러와서 확인 ----------------
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    loaded_model = CIFAR10CNN()
    loaded_model.load_state_dict(state_dict["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()

    _, loaded_accuracy = evaluate(loaded_model, test_loader, criterion, device)
    print("모델 이름:", state_dict["model_name"])
    print("Epoch 체크포인트:", state_dict["epochs"])
    print(f"저장된 테스트 정확도: {state_dict['test_accuracy'] * 100:.2f}%")
    print(f"불러온 모델 테스트 정확도: {loaded_accuracy * 100:.2f}%")

    # ---------------- 불러온 모델로 예측 시각화 ----------------
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)

    plt.figure(figsize=(15, 10))
    with torch.no_grad():
        for i in range(40):
            image, true_label = test_dataset[i]
            logits = loaded_model(image.unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)
            predicted_label = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_label].item()

            image_rgb = (image * std + mean).permute(1, 2, 0).clamp(0, 1)

            plt.subplot(4, 10, i + 1)
            plt.imshow(image_rgb)
            plt.title(
                f"True: {CIFAR10_CLASSES[true_label]}\n"
                f"Pred: {CIFAR10_CLASSES[predicted_label]}\n"
                f"Conf: {confidence * 100:.1f}%",
                color="red" if predicted_label != true_label else "black",
                fontsize=8,
            )
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "cifar10_predictions.png", dpi=100)
    print(f"결과 저장: {OUT_DIR / 'cifar10_curves.png'}, {OUT_DIR / 'cifar10_predictions.png'}")
    plt.show()


if __name__ == "__main__":
    main()
