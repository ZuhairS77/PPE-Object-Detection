"""Run PPE detection on an image, folder, video, URL, or webcam."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train2" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image, folder, video, stream URL, or webcam index (for example 0)")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Trained .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="For example: 0 for the first GPU, or cpu")
    parser.add_argument("--show", action="store_true", help="Display detections while processing")
    parser.add_argument("--name", default="ppe-predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Trained weights were not found: {weights_path}\n"
            "Train the model first or pass an existing checkpoint with --weights."
        )
    if args.imgsz < 1 or not 0.0 <= args.conf <= 1.0:
        raise ValueError("imgsz must be positive and conf must be between 0 and 1")

    source = args.source
    is_camera_index = source.isdecimal()
    source_path = Path(source).expanduser()
    is_url = source.startswith(("http://", "https://", "rtsp://", "rtmp://"))
    if not is_camera_index and not is_url and not source_path.exists():
        raise FileNotFoundError(f"Detection source was not found: {source_path.resolve()}")

    model = YOLO(str(weights_path))
    model.predict(
        source=int(source) if is_camera_index else source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        show=args.show,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
    )


if __name__ == "__main__":
    main()
