"""Shared drawing utilities for visualization and video generation."""

import os

import cv2
import numpy as np

# COCO 80 class names + face (class 80)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "face",
]

np.random.seed(42)
DET_COLORS = np.random.randint(50, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)
DET_COLORS[80] = [255, 255, 0]  # face = cyan

# Ultralytics pose palette (BGR for cv2)
POSE_PALETTE = np.array([
    [0, 128, 255], [51, 153, 255], [102, 178, 255], [0, 230, 230], [255, 153, 255],
    [255, 204, 153], [255, 102, 255], [255, 51, 255], [255, 178, 102], [255, 153, 51],
    [153, 153, 255], [102, 102, 255], [51, 51, 255], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 255],
], dtype=np.uint8)

# Ultralytics skeleton connections (1-indexed keypoint pairs)
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]

LIMB_COLOR = POSE_PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
KPT_COLOR = POSE_PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]


def draw_box(image, x1, y1, x2, y2, cls_id):
    """Draw a detection box with class label on the image."""
    color = tuple(int(c) for c in DET_COLORS[cls_id % len(COCO_CLASSES)])
    label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls_{cls_id}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
    cv2.putText(image, label, (x1 + 1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)


def draw_detections_from_label(image, label_path):
    """Draw detection boxes from a YOLO label file onto the image."""
    h, w = image.shape[:2]
    if not os.path.exists(label_path):
        return image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        draw_box(image, x1, y1, x2, y2, cls_id)

    return image


def _draw_skeleton(image, keypoints, h, w):
    """Draw skeleton limbs and keypoints for one person."""
    radius = max(round(min(h, w) / 160), 1)
    lw = max(round(min(h, w) / 160), 1)

    for i, (a, b) in enumerate(SKELETON):
        kp_a = keypoints[a - 1]
        kp_b = keypoints[b - 1]
        if kp_a[2] > 0 and kp_b[2] > 0:
            if kp_a[0] % w == 0 or kp_a[1] % h == 0:
                continue
            if kp_b[0] % w == 0 or kp_b[1] % h == 0:
                continue
            color = tuple(int(c) for c in LIMB_COLOR[i])
            cv2.line(
                image, (kp_a[0], kp_a[1]), (kp_b[0], kp_b[1]),
                color, thickness=max(lw // 2, 1), lineType=cv2.LINE_AA,
            )

    for i, (kx, ky, kv) in enumerate(keypoints):
        if kv > 0 and kx % w != 0 and ky % h != 0:
            color = tuple(int(c) for c in KPT_COLOR[i])
            cv2.circle(image, (kx, ky), radius, color, -1, lineType=cv2.LINE_AA)


def draw_poses_from_label(image, label_path):
    """Draw pose skeletons from a YOLO pose label file onto the image."""
    h, w = image.shape[:2]
    if not os.path.exists(label_path):
        return image

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5 + 17 * 3:
            continue

        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 1)

        keypoints = []
        for k in range(17):
            idx = 5 + k * 3
            kx = float(parts[idx]) * w
            ky = float(parts[idx + 1]) * h
            kv = int(parts[idx + 2])
            keypoints.append((int(kx), int(ky), kv))

        _draw_skeleton(image, keypoints, h, w)

    return image


def draw_pose_from_result(image, pose_result):
    """Draw pose skeletons from an ultralytics pose inference result."""
    h, w = image.shape[:2]
    if pose_result.keypoints is None:
        return

    for i in range(len(pose_result.boxes)):
        kps = pose_result.keypoints.data[i]
        keypoints = []
        for kp in kps:
            kx, ky, conf = int(kp[0].item()), int(kp[1].item()), kp[2].item()
            kv = 2 if conf >= 0.6 else (1 if conf >= 0.3 else 0)
            keypoints.append((kx, ky, kv))

        _draw_skeleton(image, keypoints, h, w)
