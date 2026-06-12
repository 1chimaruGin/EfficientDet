# EfficientDet

> Train and run EfficientDet object detection on your own custom dataset.

A PyTorch wrapper around [rwightman's efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch) and [timm](https://github.com/rwightman/pytorch-image-models) backbones, set up for single-class detection on a custom CSV-annotated dataset with a simple `train` / `predict` CLI.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Albumentations](https://img.shields.io/badge/Albumentations-cc0000)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

## Features

- EfficientDet D0–D7 backbones selectable via a single `--coef` flag.
- Custom dataset support driven by a CSV of bounding boxes (no COCO JSON required).
- Heavy data augmentation via Albumentations plus optional CutMix mosaics.
- Training with AdamW and a `ReduceLROnPlateau` scheduler, automatic best-checkpoint saving, and resume support.
- Inference with Weighted Boxes Fusion (WBF) and bounding-box visualization.

## Installation

```bash
git clone https://github.com/1chimaruGin/EfficientDet.git
cd EfficientDet
pip install -r requirements.txt
```

You will also need `torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib`, and `pycocotools` installed in your environment.

To train from a pretrained checkpoint, download the matching EfficientDet weights (e.g. `efficientdet_d4-5b370b7a.pth`) from the [efficientdet-pytorch releases](https://github.com/rwightman/efficientdet-pytorch).

## Dataset format

The custom loader reads a CSV alongside an image folder of the same name. For a dataset path `data/train`, provide:

- `data/train.csv` with the columns `image_id`, `bbox`, and `source`, where `bbox` is a string list `[x, y, w, h]` (top-left x/y, width, height).
- `data/train/<image_id>.jpg` for each referenced image.

The data is split into train/validation folds with stratified K-fold (fold 0 used as validation). All boxes are treated as a single class.

## Usage

**Train**

```bash
python main.py -m train -p data/train -coef 4 -w efficientdet_d4-5b370b7a.pth -b 4 -out outputs
```

Useful arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --mode` | `train` | `train` or `predict` |
| `-p, --path` | `data/train` | Dataset path (expects `<path>.csv` and `<path>/`) |
| `-coef, --coef` | `1` | EfficientDet variant (D0–D7) |
| `-w, --load-weights` | `efficientdet_d4-5b370b7a.pth` | Pretrained checkpoint to start from |
| `-b, --batch-size` | `4` | Batch size |
| `-nwk, --num-workers` | `2` | DataLoader workers |
| `-r, --resume` | off | Resume from `outputs/last-checkpoint.pth` |
| `-out, --output` | `outputs` | Output / checkpoint directory |

Checkpoints and a `log.txt` are written to the output directory; the best-validation-loss checkpoint is kept automatically.

**Predict**

```bash
python main.py -m predict -coef 4
```

This loads a trained checkpoint and visualizes predicted boxes on a test image.

## Project Structure

```
EfficientDet/
├── main.py            # CLI entry point (train / predict)
├── train.py           # Training loop
├── model.py           # Network / Inference model builders
├── config.py          # Training hyperparameters
├── data/              # Dataset, CSV preprocessing, loaders, transforms
├── effdet/            # Vendored efficientdet-pytorch + timm
└── utils/             # Inference + WBF visualization helpers
```

## License

Released under the [Apache License 2.0](LICENSE).

## Acknowledgements

EfficientDet and the backbones are from [rwightman](https://github.com/rwightman). Thanks to [Alex Shonenkov](https://www.kaggle.com/shonenkov) for the original [training kernel](https://www.kaggle.com/shonenkov/training-efficientdet).
