import os
import shutil
from pathlib import Path
from collections import defaultdict

import cv2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_gt(gt_path: Path):
    """
    Read MOT17 gt.txt
    Format:
    frame, id, x, y, w, h, conf, class, visibility
    """
    frame_dict = defaultdict(list)

    with gt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) < 9:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = int(float(parts[6]))
            cls = int(float(parts[7]))
            visibility = float(parts[8])

            frame_dict[frame_id].append({
                "track_id": track_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "conf": conf,
                "cls": cls,
                "visibility": visibility,
            })

    return frame_dict


def mot_to_yolo_bbox(x, y, w, h, img_w, img_h):
    """
    MOT: top-left x,y + width,height
    YOLO: normalized x_center y_center width height
    """
    x_center = x + w / 2.0
    y_center = y + h / 2.0

    x_center /= img_w
    y_center /= img_h
    w /= img_w
    h /= img_h

    return x_center, y_center, w, h


def is_valid_box(xc, yc, w, h):
    return (
        0.0 <= xc <= 1.0 and
        0.0 <= yc <= 1.0 and
        0.0 < w <= 1.0 and
        0.0 < h <= 1.0
    )


def convert_mot17_to_yolo(
    mot_train_dir: str,
    output_dir: str,
    val_sequences=None,
    min_visibility=0.0,
    copy_images=True,
):
    """
    Transform MOT17/train into YOLO detection format

    Parameters:
    - mot_train_dir: initial MOT17/train directory
    - output_dir: output YOLO directory
    - val_sequences: list of validation sequence, such as ["MOT17-02-FRCNN", "MOT17-10-FRCNN"]
    - min_visibility: minimum visability threshold
    - copy_images: True=copy image；False=create hard link
    """
    if val_sequences is None:
        val_sequences = []

    mot_train_dir = Path(mot_train_dir)
    output_dir = Path(output_dir)

    images_train_dir = output_dir / "images" / "train"
    images_val_dir = output_dir / "images" / "val"
    labels_train_dir = output_dir / "labels" / "train"
    labels_val_dir = output_dir / "labels" / "val"

    for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        ensure_dir(d)

    seq_dirs = sorted([p for p in mot_train_dir.iterdir() if p.is_dir()])

    if not seq_dirs:
        raise FileNotFoundError(f"Did not find any sequence directory under {mot_train_dir}.")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        img_dir = seq_dir / "img1"
        gt_path = seq_dir / "gt" / "gt.txt"

        if not img_dir.exists():
            print(f"[Skip] {seq_name}: no directory for img1")
            continue
        if not gt_path.exists():
            print(f"[Skip] {seq_name}: no gt.txt")
            continue

        split = "val" if seq_name in val_sequences else "train"
        out_img_dir = images_val_dir if split == "val" else images_train_dir
        out_label_dir = labels_val_dir if split == "val" else labels_train_dir

        print(f"[Processing] {seq_name} -> {split}")

        gt_by_frame = load_gt(gt_path)
        img_files = sorted(img_dir.glob("*.jpg"))

        if not img_files:
            print(f"[Skip] {seq_name}: no jpg in img1")
            continue

        # Read seqinfo.ini
        seqinfo_path = seq_dir / "seqinfo.ini"
        img_w, img_h = None, None

        if seqinfo_path.exists():
            with seqinfo_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("imWidth="):
                        img_w = int(line.split("=")[1])
                    elif line.startswith("imHeight="):
                        img_h = int(line.split("=")[1])

        if img_w is None or img_h is None:
            sample = cv2.imread(str(img_files[0]))
            if sample is None:
                raise RuntimeError(f"Cannot read image: {img_files[0]}")
            img_h, img_w = sample.shape[:2]

        for img_path in img_files:
            frame_id = int(img_path.stem)
            new_stem = f"{seq_name}_{img_path.stem}"
            out_img_path = out_img_dir / f"{new_stem}.jpg"
            out_label_path = out_label_dir / f"{new_stem}.txt"

            # Copy image
            if not out_img_path.exists():
                if copy_images:
                    shutil.copy2(img_path, out_img_path)
                else:
                    os.link(img_path, out_img_path)

            anns = gt_by_frame.get(frame_id, [])
            yolo_lines = []

            for ann in anns:
                # Keep valid annotation
                if ann["conf"] != 1:
                    continue
                if ann["cls"] != 1:   
                    continue
                if ann["visibility"] < min_visibility:
                    continue

                xc, yc, bw, bh = mot_to_yolo_bbox(
                    ann["x"], ann["y"], ann["w"], ann["h"], img_w, img_h
                )

                if not is_valid_box(xc, yc, bw, bh):
                    continue

                # Single-class verification
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            with out_label_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

    # Write dataset yaml
    yaml_path = output_dir / "mot17.yaml"
    yaml_text = f"""path: {output_dir.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: pedestrian
"""
    with yaml_path.open("w", encoding="utf-8") as f:
        f.write(yaml_text)

    print(f"\nTransformation complete. YOLO data saved to: {output_dir}")
    print(f"YAML file: {yaml_path}")


if __name__ == "__main__":
    mot_train_dir = r".\datasets\MOT17\train"
    output_dir = r".\datasets\MOT17_yolo"

    val_sequences = [
        "MOT17-02-FRCNN",
        "MOT17-05-FRCNN",
        "MOT17-09-FRCNN",
        "MOT17-10-FRCNN",
        "MOT17-13-FRCNN",
    ]

    convert_mot17_to_yolo(
        mot_train_dir=mot_train_dir,
        output_dir=output_dir,
        val_sequences=val_sequences,
        min_visibility=0.0,
        copy_images=True,
    )