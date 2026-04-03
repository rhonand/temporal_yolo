# YOLO + MOT17 Baseline

This project is an early-stage attempt to extend YOLO with temporal prediction.  
The current completed step is converting the MOT17 training set into standard YOLO detection format, followed by a baseline detector training pipeline.

## Current Status

- Convert `MOT17/train` to YOLO detection data
- Keep valid pedestrian annotations only
- Train a baseline detector on the converted dataset
- Temporal modeling is not implemented yet

## Project Structure

```text
yolo/
|-- transform_data.py
|-- train_baseline.py
|-- ultralytics/
|-- datasets/
|   |-- MOT17/
|   `-- MOT17_yolo/
|-- runs/
`-- yolo11n.pt
```

## Environment

This project targets Python 3.10.

The reason is practical: your system Python is `3.14`, while the current YOLO/PyTorch stack in this project is prepared around `3.10`.

## Recommended Workflow

Use a dedicated Python 3.10 virtual environment for this repository and always run project scripts from the repository root.

### 1. Create a Python 3.10 environment

If `py -3.10` is available on your machine:

```powershell
py -3.10 -m venv ouster-env
```

If `py -3.10` is not available, install Python 3.10 first, then create the environment with that interpreter.

### 2. Activate the environment

Preferred way:

```powershell
.\scripts\activate_env.ps1
```

Manual way:

```powershell
.\ouster-env\Scripts\activate
```

### 3. Install dependencies

Install PyTorch first.

For CUDA 12.1:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU only:

```powershell
python -m pip install torch torchvision torchaudio
```

Then install the remaining project packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Why keep a local `ultralytics/` folder

This repository contains a local `ultralytics/` source tree so you can modify YOLO internals such as loss functions, modules, or heads and track those changes in Git.

## Local Ultralytics Source

When you run project scripts such as `python train_baseline.py` from this repository, Python will import the local `ultralytics` package before the environment-installed one. This lets you modify files like:

- `ultralytics/utils/loss.py`
- `ultralytics/nn/modules/...`
- `ultralytics/models/...`

and track those framework changes directly in GitHub.

`train_baseline.py` also checks that the imported `ultralytics` package comes from this repository, so it will fail early if the wrong copy is being used.

## Dataset Preparation

This repository does not include the full MOT17 dataset or the converted YOLO images/labels.

You should place the original dataset like this:

```text
datasets/
`-- MOT17/
    `-- train/
        |-- MOT17-02-FRCNN/
        |-- MOT17-04-FRCNN/
        |-- ...
```

The conversion script expects:

- image files under `img1/`
- annotations under `gt/gt.txt`
- image size metadata in `seqinfo.ini`

## Convert MOT17 to YOLO Format

Run:

```powershell
python transform_data.py
```

This generates:

- `datasets/MOT17_yolo/images/train`
- `datasets/MOT17_yolo/images/val`
- `datasets/MOT17_yolo/labels/train`
- `datasets/MOT17_yolo/labels/val`
- `datasets/MOT17_yolo/mot17.yaml`

## Label Policy

The current conversion keeps only valid pedestrian boxes:

- `conf == 1`
- `cls == 1`

All kept boxes are written as YOLO class `0`, and the dataset is treated as:

```yaml
names:
  0: pedestrian
```

So this is a single-class pedestrian detection baseline, which matches standard MOT17 training usage for detection.

## Train Baseline

After activation, run the default baseline:

```powershell
python train_baseline.py
```

Or use the helper script:

```powershell
.\scripts\run_train.ps1
```

Example with explicit options:

```powershell
python train_baseline.py --device 0 --epochs 100 --batch 16 --imgsz 640
```

Helper script example:

```powershell
.\scripts\run_train.ps1 -Device 0 -Epochs 100 -Batch 16 -Imgsz 640 -Name mot17_baseline
```

Outputs are saved under:

```text
runs/detect/mot17_baseline/
```

## Files

- `transform_data.py`: converts MOT17 annotations and images into YOLO detection format
- `train_baseline.py`: baseline training entrypoint for Ultralytics YOLO

## GitHub Sync Recommendation

Recommended to upload:

- source code
- configs
- lightweight metadata files
- this README

Recommended not to upload:

- `datasets/MOT17/`
- converted YOLO images and labels
- `runs/`
- virtual environment directories
- model weight files such as `*.pt`

## Notes on the current environment

Your existing `ouster-env` was created from Python `3.10.11`, but its original base interpreter path may no longer exist. If `ouster-env\Scripts\python.exe` fails, the clean fix is to recreate the environment from a working Python 3.10 installation instead of patching the broken launcher.

## Next Step

The next natural step is to extend the dataset and model pipeline so temporal information is preserved, for example:

- keep `track_id`
- build frame windows or clip-based samples
- add temporal fusion on top of the detector backbone/head
