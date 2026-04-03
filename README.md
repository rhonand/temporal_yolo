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
|-- datasets/
|   |-- MOT17/
|   `-- MOT17_yolo/
|-- runs/
`-- yolo11n.pt
```

## Environment

This project currently runs in a local Python environment with Ultralytics YOLO.

Recommended package setup:

```powershell
python -m venv ouster-env
.\ouster-env\Scripts\activate
pip install ultralytics opencv-python
```

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

Run the default baseline:

```powershell
python train_baseline.py
```

Example with explicit options:

```powershell
python train_baseline.py --device 0 --epochs 100 --batch 16 --imgsz 640
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

## Next Step

The next natural step is to extend the dataset and model pipeline so temporal information is preserved, for example:

- keep `track_id`
- build frame windows or clip-based samples
- add temporal fusion on top of the detector backbone/head

