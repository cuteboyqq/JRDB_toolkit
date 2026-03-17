"""Stage 6: Run face inference and append face detections as class 80."""

import os
import glob
import shutil
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("INFERENCE_FACE", {})
    root = Path(g["DATASET_ROOT"])

    every = g.get("SAMPLE_EVERY", 20)
    face_class_id = tcfg.get("FACE_CLASS_ID", 80)
    batch_size = tcfg.get("BATCH_SIZE", g.get("BATCH_SIZE", 32))
    conf = tcfg.get("CONF", g.get("CONF_THRESHOLD", 0.25))
    device = g.get("DEVICE", "0")

    model_path = cfg["MODELS"]["FACE"]
    img_dir = str(root / f"{g['YOLO_IMAGES_DIR']}_every{every}")
    filtered_label_dir = str(root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}_coco80_filtered")
    out_label_dir = str(root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}_coco80_filtered_with_face")
    os.makedirs(out_label_dir, exist_ok=True)

    # Copy all filtered labels to the output directory first
    print("Copying filtered labels to output directory...")
    for f in glob.glob(os.path.join(filtered_label_dir, "*.txt")):
        shutil.copy2(f, out_label_dir)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"Total images: {len(img_files)}")
    print(f"Batch size: {batch_size}, Conf: {conf}, Device: {device}")

    total_faces = 0
    files_with_faces = 0

    for batch_start in tqdm(range(0, len(img_files), batch_size), desc="Face inference", ncols=80):
        batch_paths = img_files[batch_start:batch_start + batch_size]

        results = model.predict(
            source=batch_paths,
            conf=conf,
            device=device,
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(out_label_dir, basename + ".txt")

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            img_h, img_w = result.orig_shape
            face_lines = []

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                face_lines.append(f"{face_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if face_lines:
                total_faces += len(face_lines)
                files_with_faces += 1
                with open(label_path, "a") as f:
                    f.write("\n".join(face_lines) + "\n")

    print(f"\n=== Done ===")
    print(f"Total face detections: {total_faces}")
    print(f"Images with faces: {files_with_faces}")
    print(f"Output: {out_label_dir}")
