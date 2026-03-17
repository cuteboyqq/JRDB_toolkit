"""Stage 7: Run pose inference and save YOLO pose labels."""

import os
import glob
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("INFERENCE_POSE", {})
    root = Path(g["DATASET_ROOT"])

    every = g.get("SAMPLE_EVERY", 20)
    batch_size = tcfg.get("BATCH_SIZE", g.get("BATCH_SIZE", 16))
    conf = tcfg.get("CONF", g.get("CONF_THRESHOLD", 0.25))
    device = g.get("DEVICE", "0")
    imgsz = tcfg.get("IMGSZ", 720)

    model_path = cfg["MODELS"]["POSE"]
    img_dir = str(root / f"{g['YOLO_IMAGES_DIR']}_every{every}")
    out_label_dir = str(root / f"{g['YOLO_POSE_LABELS_DIR']}_every{every}")
    os.makedirs(out_label_dir, exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    print(f"Total images: {len(img_files)}")
    print(f"Batch size: {batch_size}, Conf: {conf}, Device: {device}, Imgsz: {imgsz}")

    total_poses = 0
    files_with_poses = 0
    PERSON_CLASS_ID = 0

    for batch_start in tqdm(range(0, len(img_files), batch_size), desc="Pose inference", ncols=80):
        batch_paths = img_files[batch_start:batch_start + batch_size]

        results = model.predict(
            source=batch_paths,
            conf=conf,
            device=device,
            imgsz=imgsz,
            verbose=False,
        )

        for img_path, result in zip(batch_paths, results):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(out_label_dir, basename + ".txt")

            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is None or len(boxes) == 0:
                open(label_path, "w").close()
                continue

            img_h, img_w = result.orig_shape
            lines = []

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i].xyxy[0].tolist()
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h

                parts = [f"{PERSON_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]

                if keypoints is not None and keypoints.data.shape[0] > i:
                    kps = keypoints.data[i]
                    for kp in kps:
                        kx = kp[0].item() / img_w
                        ky = kp[1].item() / img_h
                        conf_val = kp[2].item()
                        if conf_val < 0.3:
                            kv = 0
                        elif conf_val < 0.6:
                            kv = 1
                        else:
                            kv = 2
                        parts.append(f"{kx:.6f} {ky:.6f} {kv}")

                lines.append(" ".join(parts))

            if lines:
                total_poses += len(lines)
                files_with_poses += 1

            with open(label_path, "w") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")

    print(f"\n=== Done ===")
    print(f"Total pose detections: {total_poses}")
    print(f"Images with poses: {files_with_poses}")
    print(f"Output: {out_label_dir}")
