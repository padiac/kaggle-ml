# YOLO Playground

A dedicated workspace for experimenting with YOLO (You Only Look Once) object detection models.

## Structure

- **datasets/**: Stores datasets (e.g., COCO128).
- **scripts/**: Python scripts for training (`train.py`), prediction (`predict.py`), and verifying the environment (`sanity_check.py`).
- **configs/**: Configuration files (YAML).
- **experiments/**: Notes and logs for various experiments.
- **runs/**: Training outputs and artifacts (ignored by git).

## Getting Started

1.  Run `python scripts/sanity_check.py` to verify dependencies.
2.  Add your dataset to `data/`.
3.  Configure model in `configs/yolo_baseline.yaml`.
4.  Train: `python scripts/train.py`
