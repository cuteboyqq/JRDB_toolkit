"""Shared YOLO label parsing and box filtering utilities."""


def parse_yolo_label(label_path):
    """Parse a YOLO label file into boxes and raw lines.

    Returns:
        boxes: list of (cls_id, x1, y1, x2, y2, area)
        raw_lines: list of original line strings (stripped)
    """
    boxes = []
    raw_lines = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            area = w * h
            boxes.append((cls_id, x1, y1, x2, y2, area))
            raw_lines.append(line.strip())
    return boxes, raw_lines


def is_contained(inner_xyxy, outer_xyxy, threshold=0.8):
    """Check if inner box is mostly contained in outer box.

    Args:
        inner_xyxy: (x1, y1, x2, y2) tuple
        outer_xyxy: (x1, y1, x2, y2) tuple
        threshold: fraction of inner area that must overlap
    """
    ix1 = max(inner_xyxy[0], outer_xyxy[0])
    iy1 = max(inner_xyxy[1], outer_xyxy[1])
    ix2 = min(inner_xyxy[2], outer_xyxy[2])
    iy2 = min(inner_xyxy[3], outer_xyxy[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    inner_area = (inner_xyxy[2] - inner_xyxy[0]) * (inner_xyxy[3] - inner_xyxy[1])
    if inner_area == 0:
        return False
    return (inter_area / inner_area) >= threshold


def filter_person_boxes(boxes, threshold=0.8):
    """Remove person boxes that are mostly contained inside a larger person box.

    Args:
        boxes: list of (x1, y1, x2, y2) in normalized coords
        threshold: containment threshold

    Returns:
        Filtered list of (x1, y1, x2, y2).
    """
    if len(boxes) < 2:
        return boxes

    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    remove = set()

    for i, inner in enumerate(boxes):
        for j, outer in enumerate(boxes):
            if i == j:
                continue
            if areas[i] >= areas[j]:
                continue
            if is_contained(inner, outer, threshold=threshold):
                remove.add(i)
                break

    return [b for i, b in enumerate(boxes) if i not in remove]
