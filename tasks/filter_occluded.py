"""Stage 5: Remove occluded person boxes (boxes contained inside larger person boxes)."""

import os
import glob
from pathlib import Path

from tqdm import tqdm

from utils.yolo import parse_yolo_label, is_contained


def _filter_labels(boxes, raw_lines, threshold=0.8):
    """Remove boxes that are mostly contained inside a larger person box."""
    person_boxes = [(i, b) for i, b in enumerate(boxes) if b[0] == 0]
    remove_indices = set()

    for idx, box in enumerate(boxes):
        for pidx, pbox in person_boxes:
            if idx == pidx:
                continue
            if box[5] >= pbox[5]:
                continue
            inner_xyxy = (box[1], box[2], box[3], box[4])
            outer_xyxy = (pbox[1], pbox[2], pbox[3], pbox[4])
            if is_contained(inner_xyxy, outer_xyxy, threshold=threshold):
                remove_indices.add(idx)
                break

    kept = [raw_lines[i] for i in range(len(raw_lines)) if i not in remove_indices]
    return kept, len(remove_indices)


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("FILTER_OCCLUDED", {})
    root = Path(g["DATASET_ROOT"])

    every = g.get("SAMPLE_EVERY", 20)
    threshold = tcfg.get("CONTAINMENT_THRESHOLD", 0.8)

    label_dir = str(root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}_coco80")
    out_dir = str(root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}_coco80_filtered")
    os.makedirs(out_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

    total_boxes = 0
    removed_boxes = 0
    files_modified = 0

    for lf in tqdm(label_files, desc="Filtering", ncols=80):
        boxes, raw_lines = parse_yolo_label(lf)
        total_boxes += len(raw_lines)

        if len(boxes) < 2:
            out_path = os.path.join(out_dir, os.path.basename(lf))
            with open(out_path, "w") as f:
                f.write("\n".join(raw_lines) + "\n" if raw_lines else "")
            continue

        kept_lines, num_removed = _filter_labels(boxes, raw_lines, threshold)
        removed_boxes += num_removed
        if num_removed > 0:
            files_modified += 1

        out_path = os.path.join(out_dir, os.path.basename(lf))
        with open(out_path, "w") as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")

    print(f"\n=== Filtering Complete ===")
    print(f"Total files: {len(label_files)}")
    print(f"Files modified: {files_modified}")
    print(f"Total boxes: {total_boxes}")
    if total_boxes > 0:
        print(f"Removed boxes: {removed_boxes} ({100 * removed_boxes / total_boxes:.1f}%)")
    print(f"Remaining boxes: {total_boxes - removed_boxes}")
    print(f"Output: {out_dir}")
