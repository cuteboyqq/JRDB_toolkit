"""Stage 9: Generate annotated MP4 videos with detection + face + pose overlays."""

import os
import glob
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from utils.drawing import draw_box, draw_pose_from_result
from utils.yolo import filter_person_boxes


def _draw_jrdb_person_boxes(image, label_path):
    """Draw filtered person boxes from JRDB detection labels."""
    h, w = image.shape[:2]
    if not os.path.exists(label_path):
        return

    all_boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id != 0:
                continue
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            all_boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))

    filtered = filter_person_boxes(all_boxes)

    for x1_n, y1_n, x2_n, y2_n in filtered:
        draw_box(image, int(x1_n * w), int(y1_n * h), int(x2_n * w), int(y2_n * h), 0)


def _draw_model_detections(image, det_result, face_result):
    """Draw COCO detections (skip person) + face detections."""
    if det_result.boxes is not None:
        for box in det_result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            draw_box(image, x1, y1, x2, y2, cls_id)

    if face_result.boxes is not None:
        for box in face_result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            draw_box(image, x1, y1, x2, y2, 80)


def _group_sequences(img_dir):
    """Group images by video sequence name."""
    sequences = defaultdict(list)
    for f in sorted(os.listdir(img_dir)):
        if not f.endswith(".jpg"):
            continue
        parts = f.rsplit("_", 1)
        seq_name = parts[0]
        sequences[seq_name].append(f)
    for seq in sequences:
        sequences[seq].sort()
    return sequences


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("GENERATE_VIDEOS", {})
    root = Path(g["DATASET_ROOT"])

    batch_size = tcfg.get("BATCH_SIZE", g.get("BATCH_SIZE", 16))
    conf = tcfg.get("CONF", g.get("CONF_THRESHOLD", 0.25))
    device = g.get("DEVICE", "0")
    fps = tcfg.get("FPS", 15)
    num_videos = tcfg.get("NUM_VIDEOS", 0)

    img_dir = str(root / g["YOLO_IMAGES_DIR"])
    person_label_dir = str(root / g["YOLO_DETECTION_LABELS_DIR"])
    out_dir = str(root / tcfg.get("OUTPUT_DIR", "videos_detection_pose"))
    os.makedirs(out_dir, exist_ok=True)

    print("Loading models...")
    det_model = YOLO(cfg["MODELS"]["DETECTION"])
    face_model = YOLO(cfg["MODELS"]["FACE"])
    pose_model = YOLO(cfg["MODELS"]["POSE"])

    sequences = _group_sequences(img_dir)
    sorted_seqs = sorted(sequences.items())

    if num_videos > 0:
        sorted_seqs = sorted_seqs[:num_videos]

    total_frames = sum(len(v) for _, v in sorted_seqs)
    print(f"Generating {len(sorted_seqs)} videos, total frames: {total_frames}")

    for seq_idx, (seq_name, frames) in enumerate(tqdm(sorted_seqs, desc="Videos", ncols=80)):
        video_path = os.path.join(out_dir, f"{seq_name}.mp4")
        tqdm.write(f"[{seq_idx + 1}/{len(sorted_seqs)}] {seq_name} ({len(frames)} frames)")

        sample = cv2.imread(os.path.join(img_dir, frames[0]))
        h, w = sample.shape[:2]

        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        for batch_start in tqdm(range(0, len(frames), batch_size), desc=f"  {seq_name}", leave=False, ncols=80):
            batch_frames = frames[batch_start:batch_start + batch_size]
            batch_paths = [os.path.join(img_dir, f) for f in batch_frames]

            det_results = det_model.predict(
                source=batch_paths, conf=conf, device=device, verbose=False)
            face_results = face_model.predict(
                source=batch_paths, conf=conf, device=device, verbose=False)
            pose_results = pose_model.predict(
                source=batch_paths, conf=conf, device=device, imgsz=736, verbose=False)

            for frame_name, det_r, face_r, pose_r in zip(
                    batch_frames, det_results, face_results, pose_results):
                image = cv2.imread(os.path.join(img_dir, frame_name))

                basename = os.path.splitext(frame_name)[0]
                label_path = os.path.join(person_label_dir, basename + ".txt")
                _draw_jrdb_person_boxes(image, label_path)

                _draw_model_detections(image, det_r, face_r)

                draw_pose_from_result(image, pose_r)

                writer.write(image)

        writer.release()
        tqdm.write(f"  Saved: {video_path}")

    print(f"\nDone! {len(sorted_seqs)} videos saved to {out_dir}")
