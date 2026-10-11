"""Train a YOLO11 detector on the included PPE dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the dataset YAML")
    parser.add_argument("--model", default="yolo11n.pt", help="Pretrained model or checkpoint path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="For example: 0 for the first GPU, or cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="ppe")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_path}")
    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
        raise ValueError("epochs, imgsz, and batch must all be positive")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
        plots=True,
    )


if __name__ == "__main__":
    main()
