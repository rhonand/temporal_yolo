from __future__ import annotations

import argparse
from pathlib import Path

import ultralytics
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO baseline on the converted MOT17 pedestrian dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets/MOT17_yolo/mot17.yaml"),
        help="Path to dataset yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="YOLO model checkpoint or config.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers. Windows usually works best with 0 first.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Training device, for example "0", "cpu", or "0,1".',
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Output project directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mot17_baseline",
        help="Run name under the project directory.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic augmentation in the last N epochs.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster training if memory allows.",
    )
    parser.add_argument(
        "--cos-lr",
        action="store_true",
        help="Use cosine learning rate schedule.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable mixed precision training.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the run directory.",
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        default=True,
        help="Train as a single-class detector.",
    )
    parser.add_argument(
        "--no-single-cls",
        dest="single_cls",
        action="store_false",
        help="Disable single-class mode.",
    )
    parser.add_argument(
        "--val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run validation during and after training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent
    data_path = args.data.resolve()
    ultralytics_path = Path(ultralytics.__file__).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")
    if repo_root not in ultralytics_path.parents:
        raise RuntimeError(
            "Imported ultralytics is not the repository copy. "
            f"Current import: {ultralytics_path}"
        )

    print(f"Using ultralytics from: {ultralytics_path}")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": str(args.project),
        "name": args.name,
        "patience": args.patience,
        "close_mosaic": args.close_mosaic,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "cache": args.cache,
        "cos_lr": args.cos_lr,
        "amp": args.amp,
        "resume": args.resume,
        "single_cls": args.single_cls,
        "val": args.val,
        "plots": True,
        "save": True,
        "exist_ok": False,
        "pretrained": True,
    }

    # Ultralytics expects omitted keys rather than None for some arguments.
    if args.device is None:
        train_kwargs.pop("device")

    print("Starting baseline training with:")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")

    results = model.train(**train_kwargs)

    print("\nTraining finished.")
    print(f"Best checkpoint: {results.save_dir / 'weights' / 'best.pt'}")
    print(f"Last checkpoint: {results.save_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
