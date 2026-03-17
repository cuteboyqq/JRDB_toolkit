"""Stage 10: Analyze overlap statistics in detection labels."""

import os
import glob
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.drawing import COCO_CLASSES
from utils.yolo import parse_yolo_label, is_contained


def run(cfg):
    g = cfg["GLOBAL"]
    root = Path(g["DATASET_ROOT"])

    every = g.get("SAMPLE_EVERY", 20)
    label_dir = str(root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}_coco80")

    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

    total_files = 0
    problematic_files = 0
    total_contained_pairs = 0
    contained_class_pairs = {}
    worst_cases = []

    for lf in tqdm(label_files, desc="Analyzing", ncols=80):
        total_files += 1
        boxes, _ = parse_yolo_label(lf)
        if len(boxes) < 2:
            continue

        person_boxes = [b for b in boxes if b[0] == 0]
        file_contained = 0

        # Person-in-person containment
        for i, outer in enumerate(person_boxes):
            for j, inner in enumerate(person_boxes):
                if i == j:
                    continue
                if inner[5] >= outer[5]:
                    continue
                inner_xyxy = (inner[1], inner[2], inner[3], inner[4])
                outer_xyxy = (outer[1], outer[2], outer[3], outer[4])
                if is_contained(inner_xyxy, outer_xyxy, threshold=0.8):
                    file_contained += 1
                    key = ("person", "person")
                    contained_class_pairs[key] = contained_class_pairs.get(key, 0) + 1

        # Any box contained inside a large person box
        large_person_boxes = [b for b in person_boxes if b[5] > 0.05]
        for outer in large_person_boxes:
            outer_xyxy = (outer[1], outer[2], outer[3], outer[4])
            for inner in boxes:
                if inner is outer:
                    continue
                if inner[0] == 0:
                    continue
                if inner[5] >= outer[5]:
                    continue
                inner_xyxy = (inner[1], inner[2], inner[3], inner[4])
                if is_contained(inner_xyxy, outer_xyxy, threshold=0.8):
                    file_contained += 1
                    inner_cls = COCO_CLASSES[inner[0]] if inner[0] < len(COCO_CLASSES) else f"cls_{inner[0]}"
                    key = ("person", inner_cls)
                    contained_class_pairs[key] = contained_class_pairs.get(key, 0) + 1

        if file_contained > 0:
            problematic_files += 1
            total_contained_pairs += file_contained
            worst_cases.append((os.path.basename(lf), file_contained))

    print(f"=== Overlap Analysis ===")
    print(f"Total label files: {total_files}")
    if total_files > 0:
        print(f"Files with contained boxes: {problematic_files} ({100 * problematic_files / total_files:.1f}%)")
    print(f"Total contained pairs: {total_contained_pairs}")
    print()

    print(f"=== Contained class pairs (outer -> inner) ===")
    for (outer, inner), count in sorted(contained_class_pairs.items(), key=lambda x: -x[1])[:20]:
        print(f"  {outer} contains {inner}: {count}")

    print()
    print(f"=== Top 20 worst files (most contained boxes) ===")
    worst_cases.sort(key=lambda x: -x[1])
    for fname, count in worst_cases[:20]:
        print(f"  {fname}: {count} contained pairs")

    # Person box size distribution
    print()
    print(f"=== Person bounding box area distribution ===")
    all_person_areas = []
    for lf in tqdm(label_files, desc="Box stats", ncols=80):
        boxes, _ = parse_yolo_label(lf)
        for b in boxes:
            if b[0] == 0:
                all_person_areas.append(b[5])
    all_person_areas = np.array(all_person_areas)
    if len(all_person_areas) > 0:
        print(f"  Total person boxes: {len(all_person_areas)}")
        print(f"  Mean area (normalized): {all_person_areas.mean():.4f}")
        print(f"  Median area: {np.median(all_person_areas):.4f}")
        print(f"  >10% image area: {(all_person_areas > 0.10).sum()} ({100 * (all_person_areas > 0.10).mean():.1f}%)")
        print(f"  >20% image area: {(all_person_areas > 0.20).sum()} ({100 * (all_person_areas > 0.20).mean():.1f}%)")
        print(f"  >50% image area: {(all_person_areas > 0.50).sum()} ({100 * (all_person_areas > 0.50).mean():.1f}%)")
