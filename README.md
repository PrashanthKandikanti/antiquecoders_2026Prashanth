# Wheat Disease Detection

This repo now uses one canonical ML structure:

- `model/` contains the training and hierarchical inference code.
- `plant_health/` contains the user upload pipeline for Django.
- root `train.py` and `inference.py` are thin entrypoints only.

## Hierarchical model flow

When a user uploads an image, the pipeline works like this:

1. Run image quality checks.
2. Stage 1 predicts `healthy` vs `diseased`.
3. If diseased, Stage 2 predicts the disease type:
   `rust`, `blight`, `mildew`, or `spot`.
4. Django formats the result into farmer-friendly advice.

## Dataset layout

Stage 1 training data:

```text
training/data/processed/stage1/
  train/
    healthy/
    diseased/
  val/
    healthy/
    diseased/
```

Stage 2 training data:

```text
training/data/processed/stage2/
  train/
    rust/
    blight/
    mildew/
    spot/
  val/
    rust/
    blight/
    mildew/
    spot/
```

Existing flat layout is also supported and will be used automatically if present:

```text
training/data/processed/
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

## Training

Install dependencies:

```bash
pip install -r requirements.txt
```

Train Stage 1:

```bash
python train.py --stage stage1
```

Train Stage 2:

```bash
python train.py --stage stage2
```

Checkpoints are written to `training/checkpoints/`.

## Inference

Run hierarchical inference from the command line:

```bash
python inference.py --image path/to/image.jpg --json
```

The Django upload endpoint is available at `POST /plant-health/diagnose/` with a multipart `image` file.
