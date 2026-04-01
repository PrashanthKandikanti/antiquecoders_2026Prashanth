# Plant Validation and Disease Detection

This repo now uses a two-model computer-vision pipeline:

1. `validation` model: `plant` vs `non_plant`
2. `disease` model: `healthy`, `rust`, `blight`, `mildew`, `spot`

The training and preprocessing flow lives in `model/train.py`, and inference lives in `model/inference.py`. Root `train.py` and `predict.py` are CLI entrypoints.

## Dataset Structure

Processed validation data:

```text
training/data/processed/validation/
  train/
    plant/
    non_plant/
  val/
    plant/
    non_plant/
  test/
    plant/
    non_plant/
```

Processed disease data:

```text
training/data/processed/disease/
  train/
    healthy/
    rust/
    blight/
    mildew/
    spot/
  val/
    healthy/
    rust/
    blight/
    mildew/
    spot/
  test/
    healthy/
    rust/
    blight/
    mildew/
    spot/
```

Raw disease data should be labeled by class folder:

```text
training/data/raw/wheat_disease/
  healthy/
  rust/
  blight/
  mildew/
  spot/
```

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare and train the plant-vs-non-plant model:

```bash
python train.py --stage validation --prepare
```

Prepare and train the disease model:

```bash
python train.py --stage disease --prepare --disease-source "C:\path\to\wheat_dataset"
```

Train both models:

```bash
python train.py --stage all --prepare --disease-source "C:\path\to\wheat_dataset"
```

Use `--architecture mobilenet_v2` for faster demo inference or `--architecture resnet18` if you want a slightly heavier backbone.

## Inference

Run command-line inference:

```bash
python predict.py --image path/to/image.jpg --json
```

The JSON response includes:

- `status`
- `quality`
- `validation`
- `prediction`

The Django upload endpoint is available at `POST /plant-health/diagnose/` with a multipart `image` file.
