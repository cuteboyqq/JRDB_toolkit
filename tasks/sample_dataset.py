"""Stage 3: Subsample every Nth frame from YOLO dataset."""

import shutil
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm


def _get_video_and_frame(stem):
    """From 'image_2_scene_0_000003' return (video_name, frame_num)."""
    idx = stem.rfind("_")
    if idx < 0:
        return stem, 0
    video_name = stem[:idx]
    frame_str = stem[idx + 1:]
    try:
        frame_num = int(frame_str)
    except ValueError:
        frame_num = 0
    return video_name, frame_num


def run(cfg):
    g = cfg["GLOBAL"]
    tcfg = cfg["TASKS"].get("SAMPLE_DATASET", {})
    root = Path(g["DATASET_ROOT"])

    every = tcfg.get("EVERY", g.get("SAMPLE_EVERY", 20))
    use_symlink = tcfg.get("SYMLINK", False)
    if every < 1:
        every = 1

    images_dir = root / g["YOLO_IMAGES_DIR"]
    suffix = f"_every{every}"
    out_images = root / f"{g['YOLO_IMAGES_DIR']}{suffix}"
    out_images.mkdir(parents=True, exist_ok=True)

    # Label dirs to sample
    label_names = [
        g["YOLO_DETECTION_LABELS_DIR"],
        g["YOLO_POSE_LABELS_DIR"],
        g["YOLO_FACE_LABELS_DIR"],
    ]
    out_label_pairs = []
    for name in label_names:
        src_d = root / name
        out_d = root / f"{name}{suffix}"
        out_d.mkdir(parents=True, exist_ok=True)
        out_label_pairs.append((src_d, out_d))

    # Group images by video sequence
    groups = defaultdict(list)
    for p in images_dir.iterdir():
        if p.suffix.lower() != ".jpg":
            continue
        video_name, frame_num = _get_video_and_frame(p.stem)
        groups[video_name].append((frame_num, p))

    if not groups:
        print(f"No .jpg images in {images_dir}")
        return

    # Keep every Nth frame per video
    to_keep = []
    for video_name, frames in groups.items():
        frames.sort(key=lambda x: x[0])
        for i, (frame_num, p) in enumerate(frames):
            if i % every == 0:
                to_keep.append(p)

    total = sum(len(f) for f in groups.values())
    print(f"Keeping {len(to_keep)} of ~{total} images (every {every}th frame)")

    for img_path in tqdm(to_keep, desc="Sampling", ncols=80):
        base = img_path.stem
        dst_img = out_images / img_path.name
        try:
            if use_symlink:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.symlink_to(img_path.resolve())
            else:
                shutil.copy2(img_path, dst_img)
        except OSError as e:
            print(f"Skip image {base}: {e}")
            continue

        for src_label_dir, out_label_dir in out_label_pairs:
            src_txt = src_label_dir / f"{base}.txt"
            if not src_txt.exists():
                continue
            dst_txt = out_label_dir / f"{base}.txt"
            try:
                if use_symlink:
                    if dst_txt.exists():
                        dst_txt.unlink()
                    dst_txt.symlink_to(src_txt.resolve())
                else:
                    shutil.copy2(src_txt, dst_txt)
            except OSError:
                pass

    print(f"Done. Sampled images: {out_images}")
    print(f"  Labels: {[str(d) for _, d in out_label_pairs]}")
