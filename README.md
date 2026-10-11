# PPE Object Detection with YOLO11

A Python project for training a YOLO11 object detector and running it on images, folders, videos, network streams, or a webcam. The included Roboflow dataset has six classes: `Gloves`, `Hard_hat`, `Mask`, `Person`, `Safety_boots`, and `Vest`.

> **Important:** This model detects labeled objects. A PPE compliance decision (for example, deciding that a particular person is missing a helmet) also requires person-to-equipment association and application-specific rules; object detections alone do not establish compliance.

## Project contents

- `train.py` — train a YOLO11 model on the included dataset.
- `detect.py` — run a trained checkpoint on an image, directory, video, stream, or camera.
- `dataset/data.yaml` — dataset configuration and class names.
- `dataset/{train,valid,test}/images` and `labels` — image splits and YOLO-format labels.
- `runs/detect/train2/weights/best.pt` — checkpoint already present in this project; use it for inference, or train a fresh model.
- `code.ipynb` — notebook workflow.

## Setup

Use Python 3.10 or newer. From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch is installed as an Ultralytics dependency. For NVIDIA GPU training, install a PyTorch build compatible with the local CUDA driver by following the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). A GPU is optional; use `--device cpu` to request CPU execution.

## Train

The included pretrained Nano model is downloaded by Ultralytics on first use if needed. Train with the default settings (50 epochs, 640-pixel images):

```powershell
python train.py
```

You can adjust the run, for example:

```powershell
python train.py --epochs 100 --imgsz 640 --batch 8 --device 0 --name ppe-custom
```

Use `--device cpu` for CPU training. Lower `--batch` if you run out of memory. Training outputs and the best checkpoint are saved under `runs/detect/<run-name>/weights/best.pt`.

## Detect PPE

Use the checkpoint already included with this project:

```powershell
python detect.py "dataset/test/images"
```

Results are saved under `runs/detect/ppe-predict/`. To use a new training run's checkpoint, pass its path:

```powershell
python detect.py "path/to/image.jpg" --weights "runs/detect/ppe/weights/best.pt" --conf 0.25
```

Video and webcam examples:

```powershell
python detect.py "A building site labourer - 20 years ago Vs Today.mp4" --show
python detect.py 0 --show
```

For help with options, run `python train.py --help` or `python detect.py --help`.

## Notebook

Open `code.ipynb` with the project virtual environment selected as its kernel. The notebook should use the local dataset and checkpoint paths; no TensorFlow installation or hard-coded machine-specific paths are required. The script commands above are the recommended reproducible workflow.

## Dataset and labels

The dataset is from [Roboflow PPE Detection v14](https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb/dataset/14), licensed **CC BY 4.0**; retain attribution and follow that license when redistributing it. The dataset contains 1,101 training, 305 validation, and 213 test images. Labels use normalized YOLO coordinates and class IDs 0–5; `dataset/data.yaml` is the source of truth for the class order and split paths.

The dataset documentation supplied by its source describes a different set of label names than this repository's six-class `data.yaml`. This project follows the six classes actually configured in `data.yaml`; inspect the annotations and verify the intended class mapping before retraining or combining with other datasets.

## Troubleshooting

- **Dataset not found:** run training from the project root, and confirm that `dataset/train/images`, `dataset/valid/images`, and `dataset/test/images` exist. The YAML paths are relative to `dataset/`.
- **Checkpoint not found:** train first, or pass the correct `.pt` file with `--weights`.
- **CUDA/GPU error:** omit `--device` to use automatic selection, or pass `--device cpu`; install a compatible PyTorch CUDA build for GPU acceleration.
- **Poor detections:** evaluate on held-out test images, review label/class consistency, and tune confidence using `--conf`. Detection output is not by itself a safety-compliance verdict.

## References

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [Ultralytics training guide](https://docs.ultralytics.com/modes/train/)
- [Ultralytics prediction guide](https://docs.ultralytics.com/modes/predict/)
- [Dataset source and license](https://universe.roboflow.com/sdp-lfigk/ppe-detection-ozhfb/dataset/14)
