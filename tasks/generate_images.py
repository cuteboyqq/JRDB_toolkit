"""Stage 2: Flatten JRDB images to YOLO layout (one image per label)."""

import shutil
from pathlib import Path

from tqdm import tqdm


def _yolo_base_to_image_path(base, images_dir):
    """Convert YOLO label base name to source image path.

    e.g. 'image_2_cubberly-auditorium-2019-04-22_0_000003'
    -> images_dir / image_2 / cubberly-auditorium-2019-04-22_0 / 000003.jpg
    """
    idx = base.rfind("_")
    if idx < 0:
        return images_dir / f"{base}.jpg"
    frame = base[idx + 1:]
    rest = base[:idx]
    parts = rest.split("_", 2)
    if len(parts) < 3:
        return images_dir / f"{base}.jpg"
    image_n = "image_" + parts[1]
    scene = parts[2]
    return images_dir / image_n / scene / f"{frame}.jpg"


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("GENERATE_IMAGES", {})
    root = Path(g["DATASET_ROOT"])

    images_dir = root / g["JRDB_IMAGES_DIR"]
    label_dir = root / g["YOLO_DETECTION_LABELS_DIR"]
    out_dir = root / g["YOLO_IMAGES_DIR"]
    use_symlink = tcfg.get("SYMLINK", False)

    out_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(label_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt in {label_dir}. Run convert_labels first.")
        return

    done = 0
    skipped = 0
    for txt_path in tqdm(txt_files, desc="Generating images", ncols=80):
        base = txt_path.stem
        src = _yolo_base_to_image_path(base, images_dir)
        dst = out_dir / f"{base}.jpg"
        if not src.exists():
            skipped += 1
            continue
        if dst.exists() and not use_symlink:
            done += 1
            continue
        try:
            if use_symlink:
                if dst.exists():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            done += 1
        except OSError as e:
            print(f"Skip {base}: {e}")
            skipped += 1
    print(f"Done. YOLO images in {out_dir}: {done} written, {skipped} skipped (missing source).")
