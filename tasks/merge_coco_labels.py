"""Stage 4: Run COCO 80-class inference and merge with person-only labels."""

import os
import sys
import glob

import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm


def _letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def _xywh2xyxy(x):
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def _non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    output = []
    for xi, x in enumerate(prediction):
        cls_scores = x[:, 4:]
        conf, cls_id = cls_scores.max(1, keepdim=True)
        mask = conf.squeeze(-1) > conf_thres
        x = x[mask]
        conf = conf[mask]
        cls_id = cls_id[mask]
        if x.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=x.device))
            continue
        boxes = _xywh2xyxy(x[:, :4])
        detections = torch.cat([boxes, conf, cls_id.float()], 1)
        offset = detections[:, 5:6] * 4096
        boxes_offset = detections[:, :4] + offset
        scores = detections[:, 4]
        keep = torchvision.ops.nms(boxes_offset, scores, iou_thres)
        output.append(detections[keep])
    return output


def _preprocess_batch(image_paths, img_size):
    batch_imgs = []
    orig_shapes = []
    ratios_pads = []
    for path in image_paths:
        img = cv2.imread(path)
        orig_shapes.append(img.shape[:2])
        img_lb, ratio, (dw, dh) = _letterbox(img, img_size)
        ratios_pads.append((ratio, dw, dh))
        img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1)
        img_lb = np.ascontiguousarray(img_lb, dtype=np.float32) / 255.0
        batch_imgs.append(img_lb)
    return torch.from_numpy(np.stack(batch_imgs)), orig_shapes, ratios_pads


def _scale_boxes_to_orig(boxes, ratio, dw, dh, orig_h, orig_w):
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
    return boxes


def _xyxy_to_yolo(boxes, orig_h, orig_w):
    lines = []
    for det in boxes:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cx = ((x1 + x2) / 2) / orig_w
        cy = ((y1 + y2) / 2) / orig_h
        w = (x2 - x1) / orig_w
        h = (y2 - y1) / orig_h
        lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("MERGE_COCO_LABELS", {})
    root = Path(g["DATASET_ROOT"])

    every = g.get("SAMPLE_EVERY", 20)
    image_dir = str(root / f"{g['YOLO_IMAGES_DIR']}_every{every}")
    person_label_dir = str(root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}")
    output_label_dir = str(root / f"{g['YOLO_DETECTION_LABELS_DIR']}_every{every}_coco80")
    os.makedirs(output_label_dir, exist_ok=True)

    model_path = cfg["MODELS"]["DETECTION"]
    conf_threshold = tcfg.get("CONF_THRESHOLD", g.get("CONF_THRESHOLD", 0.25))
    iou_threshold = tcfg.get("IOU_THRESHOLD", 0.45)
    img_size = tcfg.get("IMG_SIZE", 640)
    batch_size = tcfg.get("BATCH_SIZE", g.get("BATCH_SIZE", 16))
    device = f"cuda:{g.get('DEVICE', '0')}"

    # Add ultralytics source path for model unpickling
    ul_path = g.get("ULTRALYTICS_SOURCE_PATH")
    if ul_path and ul_path not in sys.path:
        sys.path.insert(0, ul_path)

    # Load model
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ckpt["model"].float().to(device)
    model.eval()
    model.half()

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    print(f"Found {len(image_paths)} images")

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Inferencing", ncols=80):
            batch_paths = image_paths[batch_start:batch_start + batch_size]
            batch_tensor, orig_shapes, ratios_pads = _preprocess_batch(batch_paths, img_size)
            batch_tensor = batch_tensor.to(device).half()

            preds = model(batch_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = preds.permute(0, 2, 1)

            detections = _non_max_suppression(preds, conf_threshold, iou_threshold)

            for idx, (img_path, dets) in enumerate(zip(batch_paths, detections)):
                stem = Path(img_path).stem
                orig_h, orig_w = orig_shapes[idx]
                ratio, dw, dh = ratios_pads[idx]

                # Read original person labels
                person_label_path = os.path.join(person_label_dir, f"{stem}.txt")
                person_lines = []
                if os.path.exists(person_label_path):
                    with open(person_label_path, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                person_lines.append(line)

                # Process COCO predictions (skip class 0 = person)
                coco_lines = []
                if len(dets) > 0:
                    dets = dets.clone()
                    dets = _scale_boxes_to_orig(dets, ratio, dw, dh, orig_h, orig_w)
                    non_person_mask = dets[:, 5] != 0
                    dets = dets[non_person_mask]
                    if len(dets) > 0:
                        coco_lines = _xyxy_to_yolo(dets, orig_h, orig_w)

                merged = person_lines + coco_lines
                output_path = os.path.join(output_label_dir, f"{stem}.txt")
                with open(output_path, "w") as f:
                    f.write("\n".join(merged))
                    if merged:
                        f.write("\n")

    print(f"Done! Merged labels saved to: {output_label_dir}")
    print(f"Total files: {len(os.listdir(output_label_dir))}")
