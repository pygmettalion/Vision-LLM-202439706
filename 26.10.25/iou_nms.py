"""
[261025] 04_DL-Object-Detection
IoU, NMS 구현

 1. convert_bbox_coord : (x, y, w, h) → (x1, y1, x2, y2)
 2. calculate_iou      : 두 BBox의 IoU = 교집합 / 합집합
 3. prune_low_conf     : Confidence Thresholding
 4. apply_nms          : Non-Maximum Suppression
 5. 사과 이미지 3장으로 IoU 계산, 랜덤 BBox에 NMS 적용 후 OpenCV NMS와 비교

실행 (레포 최상위 폴더 어디서든):
    python3 261025/iou_nms.py
결과 이미지: 261025/iou_result.png, 261025/nms_result.png
"""

from pathlib import Path
from random import randint, gauss, seed

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "src" / "images"
OUT_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. BBox 좌표 변환 / IoU
# ============================================================
def convert_bbox_coord(bbox):
    """(x, y, w, h) → (x1, y1, x2, y2)"""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def calculate_iou(bbox_a, bbox_b):
    """두 BBox (x1, y1, x2, y2)의 IoU 계산"""
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b

    # 교집합 박스 좌표
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)

    # 겹치지 않으면 너비/높이 = 0
    intersection_w = max(0, intersection_x2 - intersection_x1)
    intersection_h = max(0, intersection_y2 - intersection_y1)

    area_a = max(0, xa_2 - xa_1) * max(0, ya_2 - ya_1)
    area_b = max(0, xb_2 - xb_1) * max(0, yb_2 - yb_1)

    intersection_area = intersection_w * intersection_h
    union_area = area_a + area_b - intersection_area

    # ZeroDivisionError 방지
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# 2. Confidence Thresholding / NMS
# ============================================================
def prune_low_conf(boxes, scores, conf_threshold=0.5):
    """Confidence Score가 임계값보다 낮은 BBox 제거"""
    pruned_boxes = []
    pruned_scores = []

    for box, score in zip(boxes, scores):
        if score >= conf_threshold:
            pruned_boxes.append(box)
            pruned_scores.append(score)

    return pruned_boxes, pruned_scores


def apply_nms(boxes, scores, iou_threshold=0.5):
    """
    NMS
     1) Confidence가 가장 높은 BBox 선택
     2) 선택한 BBox와 나머지 BBox의 IoU 계산
     3) IoU가 임계값보다 큰 BBox 제거
     4) 남은 BBox로 1~3 반복
    """
    if len(boxes) == 0:
        return [], []

    order = list(np.argsort(scores)[::-1])   # Confidence 내림차순 인덱스
    keep_indices = []

    while len(order) > 0:
        best = order.pop(0)
        keep_indices.append(best)

        order = [
            index for index in order
            if calculate_iou(boxes[best], boxes[index]) <= iou_threshold
        ]

    nms_boxes = [boxes[index] for index in keep_indices]
    nms_scores = [scores[index] for index in keep_indices]

    return nms_boxes, nms_scores


def apply_nms_cv2(boxes, scores, conf_threshold=0.5, iou_threshold=0.5):
    """비교용: OpenCV 내장 NMS"""
    if len(boxes) == 0:
        return [], []

    boxes_xywh = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for x1, y1, x2, y2 in boxes]
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )
    keep_indices = np.array(keep_indices).reshape(-1).tolist()

    return [boxes[i] for i in keep_indices], [scores[i] for i in keep_indices]


# ============================================================
# 3. 시각화 도우미
# ============================================================
def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1] - txt_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def get_contour(img_bgr):
    """빨간 사과 / 초록 사과의 가장 큰 contour"""
    red_lower1, red_upper1 = np.array([0, 100, 50]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([170, 100, 50]), np.array([180, 255, 255])
    green_lower, green_upper = np.array([25, 10, 30]), np.array([100, 255, 255])

    blr_hsv = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (7, 7), 0), cv2.COLOR_BGR2HSV)

    red_mask = cv2.bitwise_or(
        cv2.inRange(blr_hsv, red_lower1, red_upper1),
        cv2.inRange(blr_hsv, red_lower2, red_upper2),
    )
    grn_mask = cv2.inRange(blr_hsv, green_lower, green_upper)

    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_grn, _ = cv2.findContours(grn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cntr_red = max(contours_red, key=cv2.contourArea) if contours_red else None
    cntr_grn = max(contours_grn, key=cv2.contourArea) if contours_grn else None

    return cntr_red, cntr_grn


def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
    img_h, img_w = img.shape[:2]
    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))
    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)
    return x1, y1, x1 + w, y1 + h


def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))


# ============================================================
# 4. 실행
# ============================================================
def test_functions():
    assert convert_bbox_coord((10, 20, 30, 40)) == (10, 20, 40, 60)
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0      # 완전히 동일
    assert calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0    # 겹치지 않음
    assert abs(calculate_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9
    assert calculate_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0          # 0으로 나누기 방지
    print("IoU 단위 테스트 통과")


def iou_demo():
    plt.figure(figsize=(18, 6))

    for k in range(1, 4):
        img_bgr = cv2.imread(str(IMAGE_DIR / f"two_apples{k}.png"))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        cntr_red, cntr_grn = get_contour(img_bgr)
        bbox_red = cv2.boundingRect(cntr_red)     # (x, y, w, h)
        bbox_grn = cv2.boundingRect(cntr_grn)

        red_xyxy = convert_bbox_coord(bbox_red)
        grn_xyxy = convert_bbox_coord(bbox_grn)
        iou = calculate_iou(red_xyxy, grn_xyxy)
        print(f"two_apples{k}.png IoU: {iou:.4f}")

        draw_bbox(img_rgb, red_xyxy[:2], red_xyxy[2:], (255, 0, 0), txt="Red Apple")
        draw_bbox(img_rgb, grn_xyxy[:2], grn_xyxy[2:], (0, 255, 0), txt="Green Apple")

        plt.subplot(1, 3, k), plt.imshow(img_rgb), plt.title(f"IoU: {iou:.4f}"), plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "iou_result.png", dpi=100)


def nms_demo(n_boxes=20, conf_threshold=0.5, iou_threshold=0.2):
    seed(42)
    hubble = skimage.data.hubble_deep_field()

    boxes = [random_bbox(hubble) for _ in range(n_boxes)]
    scores = [random_conf() for _ in range(n_boxes)]

    # 직접 구현: Confidence Thresholding → NMS
    pruned_boxes, pruned_scores = prune_low_conf(boxes, scores, conf_threshold)
    nms_boxes, nms_scores = apply_nms(pruned_boxes, pruned_scores, iou_threshold)

    # OpenCV 내장 NMS
    cv2_boxes, cv2_scores = apply_nms_cv2(boxes, scores, conf_threshold, iou_threshold)

    print(f"전체 BBox: {len(boxes)}")
    print(f"Confidence Thresholding 후: {len(pruned_boxes)}")
    print(f"NMS 후 (직접 구현): {len(nms_boxes)}")
    print(f"NMS 후 (OpenCV):   {len(cv2_boxes)}")
    print("직접 구현 결과 == OpenCV 결과:", sorted(nms_boxes) == sorted(cv2_boxes))

    panels = [("Before NMS", boxes, scores), ("Custom NMS", nms_boxes, nms_scores),
              ("OpenCV NMS", cv2_boxes, cv2_scores)]

    plt.figure(figsize=(21, 7))
    for k, (title, bxs, scs) in enumerate(panels):
        canvas = hubble.copy()
        for box, score in zip(bxs, scs):
            color = (randint(80, 255), randint(80, 255), randint(80, 255))
            draw_bbox(canvas, box[:2], box[2:], color, txt=f"{score:.2f}")
        plt.subplot(1, 3, k + 1), plt.imshow(canvas), plt.title(f"{title} ({len(bxs)} boxes)"), plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "nms_result.png", dpi=100)


def main():
    test_functions()
    print()
    iou_demo()
    print()
    nms_demo()
    print(f"\n결과 저장: {OUT_DIR / 'iou_result.png'}, {OUT_DIR / 'nms_result.png'}")
    plt.show()


if __name__ == "__main__":
    main()
