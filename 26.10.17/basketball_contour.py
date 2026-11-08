"""
[261017] 02_Computer-Vision
모폴로지 연산을 적용하여 최종 농구공 Contour 선택 및 출력

과정
 1. LAB 색공간 변환 + L 채널 CLAHE로 조명 오염 보정
 2. LAB 범위로 Binary Mask 생성 (Color Segmentation)
 3. Morphological Operation
    - Opening : 배경의 작은 노이즈 제거
    - Closing : 공 표면의 줄무늬·반사광으로 생긴 구멍을 메워 하나의 영역으로 연결
 4. cv2.findContours로 외곽선 검출
 5. 면적이 가장 큰 Contour를 최종 농구공으로 선택
 6. 최소 외접원 / 무게중심(Centroid)으로 공의 중심과 반지름 표시

실행 (레포 최상위 폴더 어디서든):
    python3 261017/basketball_contour.py
결과 이미지: 261017/basketball_result.png
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]           # 레포 최상위 폴더
IMAGE_PATH = ROOT / "src" / "images" / "basketball_crop.jpg"
RESULT_PATH = Path(__file__).resolve().parent / "basketball_result.png"

# 농구공 LAB 범위 (수업 자료 기준)
LOWER_BASKETBALL_LAB = np.array([70, 130, 70])
UPPER_BASKETBALL_LAB = np.array([180, 230, 185])


def enhance_and_segment(image_bgr):
    """LAB + CLAHE로 대비를 강화한 뒤 Binary Mask 생성"""
    basketball_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(basketball_lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

    mask_enhanced = cv2.inRange(enhanced_lab, LOWER_BASKETBALL_LAB, UPPER_BASKETBALL_LAB)
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return enhanced_rgb, mask_enhanced


def apply_morphology(mask):
    """Opening x2 → Closing x3 으로 최종 Binary Mask 생성"""
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=2)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, close_kernel, iterations=3)

    return mask_opened, mask_closed


def select_ball_contour(final_mask):
    """외곽선 검출 후 면적이 가장 큰 contour를 농구공으로 선택"""
    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,        # 바깥쪽 테두리만 검출
        cv2.CHAIN_APPROX_SIMPLE,  # 꼭짓점 좌표만 저장
    )

    if len(contours) == 0:
        return contours, None

    largest_contour = max(contours, key=cv2.contourArea)
    return contours, largest_contour


def draw_result(enhanced_rgb, final_mask, largest_contour):
    """최종 농구공 영역, 외곽선, 최소 외접원, 무게중심을 그림"""
    result = enhanced_rgb.copy()

    # 가장 큰 contour만 채운 마스크를 반투명 빨간색으로 overlay
    ball_mask = np.zeros_like(final_mask)
    cv2.drawContours(ball_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    mask_color = np.zeros_like(result)
    mask_color[:, :, 0] = ball_mask
    result = cv2.addWeighted(result, 1.0, mask_color, 0.15, 0)

    # 빨간 외곽선
    cv2.drawContours(result, [largest_contour], -1, (255, 0, 0), 2)

    # 최소 외접원 (파란색)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_circle = (int(x), int(y))
    radius = int(radius)
    cv2.circle(result, center_circle, radius, (0, 0, 255), 2)
    cv2.circle(result, center_circle, 5, (0, 0, 255), -1)

    # 무게중심 (노란색) + 외접원 반지름으로 그린 원
    M = cv2.moments(largest_contour)
    if M["m00"] != 0:
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        centroid = center_circle
    cv2.circle(result, centroid, radius, (255, 255, 0), 2)
    cv2.circle(result, centroid, 5, (255, 255, 0), -1)

    return result, ball_mask, center_circle, centroid, radius


def main():
    basketball = cv2.imread(str(IMAGE_PATH))
    if basketball is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

    basketball_rgb = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

    enhanced_rgb, mask_enhanced = enhance_and_segment(basketball)
    mask_opened, final_ball_mask = apply_morphology(mask_enhanced)
    contours, largest_contour = select_ball_contour(final_ball_mask)

    if largest_contour is None:
        print("농구공 contour를 찾지 못했습니다.")
        return

    result, ball_mask, center_circle, centroid, radius = draw_result(
        enhanced_rgb, final_ball_mask, largest_contour
    )

    print(f"검출된 contour 수: {len(contours)}")
    print(f"선택된 농구공 contour 면적: {cv2.contourArea(largest_contour):.0f} px")
    print(f"최소 외접원 중심: {center_circle}, 반지름: {radius}")
    print(f"무게중심: {centroid}")

    plt.figure(figsize=(20, 9))

    titles_images = [
        ("Original", basketball_rgb, None),
        ("LAB (CLAHE) Enhanced", enhanced_rgb, None),
        ("Enhanced Binary Mask", mask_enhanced, "gray"),
        ("Opening x2", mask_opened, "gray"),
        ("Opening x2 + Closing x3", final_ball_mask, "gray"),
        ("Largest Contour (Ball)", ball_mask, "gray"),
    ]
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for (title, img, cmap), pos in zip(titles_images, positions):
        plt.subplot2grid((2, 4), pos), plt.imshow(img, cmap=cmap), plt.title(title)

    # 최종 결과는 오른쪽 열 전체를 사용
    plt.subplot2grid((2, 4), (0, 3), rowspan=2), plt.imshow(result)
    plt.title("Final Basketball Contour\nred: contour / blue: min circle\nyellow: centroid")

    for ax in plt.gcf().axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(RESULT_PATH, dpi=100)
    print(f"결과 저장: {RESULT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
