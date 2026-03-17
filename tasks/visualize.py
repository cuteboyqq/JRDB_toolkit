"""Stage 8: Visualize detection/pose/both on sampled images."""

import os
import glob
from pathlib import Path

import cv2
from tqdm import tqdm

from utils.drawing import draw_detections_from_label, draw_poses_from_label


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("VISUALIZE", {})
    root = Path(g["DATASET_ROOT"])

    every = g.get("SAMPLE_EVERY", 20)
    mode = tcfg.get("MODE", "both")  # "detection", "pose", or "both"
    skip = tcfg.get("SKIP", 3)

    img_dir = str(root / f"{g['YOLO_IMAGES_DIR']}_every{every}")

    # Default label dirs based on mode
    det_label_dir = str(root / tcfg.get(
        "DETECTION_LABEL_DIR",
        f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}_coco80_filtered_with_face",
    ))
    pose_label_dir = str(root / tcfg.get(
        "POSE_LABEL_DIR",
        f"{g['YOLO_POSE_LABELS_DIR']}_every{every}",
    ))

    out_dir = str(root / tcfg.get("OUTPUT_DIR", f"visualized_{mode}"))
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    selected = img_files[::skip]

    print(f"Total images: {len(img_files)}")
    print(f"Skip: {skip}, visualizing: {len(selected)} images")
    print(f"Mode: {mode}")
    if mode in ("detection", "both"):
        print(f"Detection labels: {det_label_dir}")
    if mode in ("pose", "both"):
        print(f"Pose labels: {pose_label_dir}")
    print(f"Output: {out_dir}")

    for img_path in tqdm(selected, desc="Visualizing", ncols=80):
        basename = os.path.splitext(os.path.basename(img_path))[0]

        image = cv2.imread(img_path)
        if image is None:
            continue

        if mode in ("detection", "both"):
            det_path = os.path.join(det_label_dir, basename + ".txt")
            image = draw_detections_from_label(image, det_path)

        if mode in ("pose", "both"):
            pose_path = os.path.join(pose_label_dir, basename + ".txt")
            image = draw_poses_from_label(image, pose_path)

        out_path = os.path.join(out_dir, basename + ".jpg")
        cv2.imwrite(out_path, image)

    print(f"Done! {len(selected)} images saved to {out_dir}")
