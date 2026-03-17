"""Stage 1: Convert JRDB JSON labels to YOLO format (detection + pose + face)."""

import json
from pathlib import Path

from tqdm import tqdm


def _path_to_yolo_name(file_name):
    """Convert 'image_2/scene_0/000003.jpg' to 'image_2_scene_0_000003'."""
    parts = Path(file_name).parts
    if len(parts) < 3:
        return Path(file_name).stem
    return "_".join(parts[:-1]) + "_" + Path(parts[-1]).stem


def _convert_box_jrdb_to_yolo(box, img_w, img_h):
    """JRDB box [x, y, w, h] in pixels -> normalized (cx, cy, w, h)."""
    x, y, w, h = box[0], box[1], box[2], box[3]
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def _coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO bbox [x, y, w, h] -> normalized (cx, cy, w, h)."""
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return (x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h


def _keypoints_to_bbox(keypoints, img_w, img_h):
    """Compute bbox from COCO keypoints (x,y,v)*17. Returns normalized (cx, cy, w, h)."""
    xs, ys = [], []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if v > 0:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return 0.5, 0.5, 0.01, 0.01
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return x_center / img_w, y_center / img_h, w / img_w, h / img_h


def _convert_detection(labels_dir, out_dir):
    """Convert JRDB labels_2d JSON to YOLO detection format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_w, img_h = 752.0, 480.0

    json_files = sorted(labels_dir.glob("*_image*.json"))
    if not json_files:
        print(f"  No *_image*.json files in {labels_dir}")
        return

    total_labels = 0
    for jpath in tqdm(json_files, desc="  Detection", ncols=80):
        with open(jpath, "r") as f:
            data = json.load(f)
        labels_dict = data.get("labels", {})
        stem = jpath.stem
        if "_image" not in stem:
            continue
        scene_part = stem.split("_image")[0]
        cam_part = "image_" + stem.split("_image")[1]

        for frame_name, anns in labels_dict.items():
            base = f"{cam_part}_{scene_part}_{Path(frame_name).stem}"
            lines = []
            for ann in (anns or []):
                box = ann.get("box")
                if not box or len(box) != 4:
                    continue
                xc, yc, nw, nh = _convert_box_jrdb_to_yolo(box, img_w, img_h)
                lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            out_path = out_dir / f"{base}.txt"
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            total_labels += len(lines)

    print(f"  Detection: wrote {total_labels} labels to {out_dir}")


def _convert_pose(labels_dir, out_dir):
    """Convert JRDB labels_2d_pose_coco JSON to YOLO pose format."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labels_dir.glob("*_image*.json"))
    if not json_files:
        print(f"  No *_image*.json files in {labels_dir}")
        return

    for jpath in tqdm(json_files, desc="  Pose", ncols=80):
        with open(jpath, "r") as f:
            data = json.load(f)
        anns_by_image = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)

        for im in data.get("images", []):
            img_id = im["id"]
            file_name = im["file_name"]
            img_w, img_h = float(im["width"]), float(im["height"])
            base = _path_to_yolo_name(file_name)
            lines = []
            for ann in anns_by_image.get(img_id, []):
                kpts = ann.get("keypoints", [])
                if len(kpts) < 51:
                    continue
                xc, yc, nw, nh = _keypoints_to_bbox(kpts, img_w, img_h)
                line_parts = [f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"]
                for i in range(0, 51, 3):
                    x = kpts[i] / img_w
                    y = kpts[i + 1] / img_h
                    v = int(kpts[i + 2])
                    line_parts.append(f"{x:.6f} {y:.6f} {v}")
                lines.append(" ".join(line_parts))
            out_path = out_dir / f"{base}.txt"
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"  Pose: wrote YOLO pose labels to {out_dir}")


def _convert_face(labels_dir, out_dir):
    """Convert JRDB labels_2d_head JSON to YOLO face detection format."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(labels_dir.glob("*_image*.json"))
    if not json_files:
        print(f"  No *_image*.json files in {labels_dir}")
        return

    for jpath in tqdm(json_files, desc="  Face", ncols=80):
        with open(jpath, "r") as f:
            data = json.load(f)
        anns_by_img = {}
        for ann in data.get("annotations", []):
            iid = ann["image_id"]
            if iid not in anns_by_img:
                anns_by_img[iid] = []
            anns_by_img[iid].append(ann)

        for im in data.get("images", []):
            img_id = im["id"]
            file_name = im["file_name"]
            img_w, img_h = float(im["width"]), float(im["height"])
            base = _path_to_yolo_name(file_name)
            lines = []
            for ann in anns_by_img.get(img_id, []):
                bbox = ann.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                xc, yc, nw, nh = _coco_bbox_to_yolo(bbox, img_w, img_h)
                lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            out_path = out_dir / f"{base}.txt"
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"  Face: wrote YOLO labels to {out_dir}")


def run(cfg):
    g = cfg["GLOBAL"]
    root = Path(g["DATASET_ROOT"])

    print("Converting detection labels...")
    _convert_detection(root / g["JRDB_LABELS_2D_DIR"], root / g["YOLO_DETECTION_LABELS_DIR"])

    print("Converting pose labels...")
    _convert_pose(root / g["JRDB_LABELS_POSE_DIR"], root / g["YOLO_POSE_LABELS_DIR"])

    print("Converting face labels...")
    _convert_face(root / g["JRDB_LABELS_HEAD_DIR"], root / g["YOLO_FACE_LABELS_DIR"])
