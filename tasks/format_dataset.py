"""Stage 8: Organize processed files into a training dataset directory structure.

Output layout (matching coco2017_face_pose format):
    {OUTPUT_DIR}/
    └── {SPLIT}/
        ├── images/                  ← sampled YOLO images
        ├── labels/
        │   ├── detection/           ← final detection labels (COCO 80 + face, filtered)
        │   └── pose/
        │       └── points/          ← inferred pose labels
        └── labels-GT/
            ├── detection/           ← ground truth person-only detection labels
            └── pose/
                └── points/          ← ground truth pose labels from JRDB
"""

import os
import shutil
import glob
from pathlib import Path

from tqdm import tqdm


def _copy_or_link(src, dst, symlink=False):
    """Copy or symlink a file."""
    if dst.exists():
        return
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("FORMAT_DATASET", {})
    root = Path(g["DATASET_ROOT"])

    every = g.get("SAMPLE_EVERY", 20)
    output_dir = root / tcfg.get("OUTPUT_DIR", "jrdb_train_dataset")
    split = tcfg.get("SPLIT", "train")
    use_symlink = tcfg.get("SYMLINK", True)

    # Source directories
    src_images = root / f"{g['YOLO_IMAGES_DIR']}_every{every}"
    src_det_labels = root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}_coco80_filtered_with_face"
    src_pose_labels = root / f"{g['YOLO_POSE_LABELS_DIR']}_every{every}"
    src_gt_det_labels = root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}"
    src_gt_pose_labels = root / g["YOLO_POSE_LABELS_DIR"]  # full GT (not overwritten by inference)

    # Target directories
    split_dir = output_dir / split
    dst_images = split_dir / "images"
    dst_det = split_dir / "labels" / "detection"
    dst_pose = split_dir / "labels" / "pose" / "points"
    dst_gt_det = split_dir / "labels-GT" / "detection"
    dst_gt_pose = split_dir / "labels-GT" / "pose" / "points"

    for d in [dst_images, dst_det, dst_pose, dst_gt_det, dst_gt_pose]:
        d.mkdir(parents=True, exist_ok=True)

    # Get list of sampled images as the reference set
    img_files = sorted(src_images.glob("*.jpg"))
    if not img_files:
        print(f"No images found in {src_images}")
        return

    print(f"Formatting {len(img_files)} files into {split_dir}")
    print(f"  Images:          {src_images}")
    print(f"  Detection:       {src_det_labels}")
    print(f"  Pose:            {src_pose_labels}")
    print(f"  GT Detection:    {src_gt_det_labels}")
    print(f"  GT Pose:         {src_gt_pose_labels}")
    print(f"  Symlink: {use_symlink}")

    stats = {"images": 0, "det": 0, "pose": 0, "gt_det": 0, "gt_pose": 0}

    for img_path in tqdm(img_files, desc="Formatting", ncols=80):
        base = img_path.stem

        # Image
        _copy_or_link(img_path, dst_images / img_path.name, use_symlink)
        stats["images"] += 1

        # Detection labels (final: COCO 80 + face, filtered)
        src = src_det_labels / f"{base}.txt"
        if src.exists():
            _copy_or_link(src, dst_det / f"{base}.txt", use_symlink)
            stats["det"] += 1

        # Pose labels (inferred)
        src = src_pose_labels / f"{base}.txt"
        if src.exists():
            _copy_or_link(src, dst_pose / f"{base}.txt", use_symlink)
            stats["pose"] += 1

        # GT detection labels (person-only from JRDB)
        src = src_gt_det_labels / f"{base}.txt"
        if src.exists():
            _copy_or_link(src, dst_gt_det / f"{base}.txt", use_symlink)
            stats["gt_det"] += 1

        # GT pose labels (from JRDB original, re-sampled from full set)
        src = src_gt_pose_labels / f"{base}.txt"
        if src.exists():
            _copy_or_link(src, dst_gt_pose / f"{base}.txt", use_symlink)
            stats["gt_pose"] += 1

    print(f"\n=== Format Complete ===")
    print(f"Output: {split_dir}")
    print(f"  images:          {stats['images']}")
    print(f"  labels/detection: {stats['det']}")
    print(f"  labels/pose:     {stats['pose']}")
    print(f"  labels-GT/detection: {stats['gt_det']}")
    print(f"  labels-GT/pose:  {stats['gt_pose']}")
