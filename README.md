# Emotion Recognition from Human Gait using MotionAGFormer and STEP

This project integrates MotionAGFormer and STEP for emotion recognition based on human gait analysis.

## Overview

The pipeline consists of two main components:

1. **Pose Estimation** (`Pose_Estimation/`): 
   - 2D pose estimation using HRNet
   - 3D pose estimation using MotionAGFormer
   - Temporal pose refinement with PersPose

2. **Emotion Recognition** (`Recognition/`):
   - STEP-based classifier for emotion recognition from gait patterns
   - Graph-based neural network approach (TGCN)

## Project Structure

```
├── integrated_pipeline.py      # Main pipeline script
├── requirements.txt            # Python dependencies
├── Pose_Estimation/            # Pose estimation models and utilities
├── Recognition/                # Emotion recognition models
├── demo/                       # Demo scripts and sample inputs
├── checkpoints/                # Pre-trained model checkpoints
└── output/                     # Generated outputs
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hoangkthieu/Integrated-Skeleton-based-Gait-Emotion-Recognition-from-MotionAGFormer-and-STEP.git

   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained model files:
   - See `checkpoints/` directory for model download links
   - See `Pose_Estimation/lib/yolov3/` directory for yolov3.weights download links

## Usage

Run the gait emotion recognition pipeline by providing an input video path, a STEP model
checkpoint path, and an output directory.

### Basic Command

```bash
python integrated_pipeline.py \
    --video /path/to/input_video.mp4 \
    --checkpoint /path/to/STEP_model.pth.tar \
    --output /path/to/output
```

### Examples

```bash
python integrated_pipeline.py \
    --video demo/input_video/VID_RGB_003.mp4 \
    --checkpoint checkpoints/Recognition/STEP/original_2D+3D/epoch172_acc99.48_model.pth.tar \
    --output demo/output

```

### Output Structure

Results are saved under `{output}/{video_name}/`:

```
{output}/{video_name}/
├── input_2D/
│   └── keypoints.csv                        # 2D HRNet keypoints per frame
├── output_3D/
│   └── keypoints_step_order.csv             # 3D MotionAGFormer keypoints (world-space)
├── keypoint_features/
│   ├── keypointFeatures.csv                 # Root-centered keypoints (for ST-GCN)
│   └── keypointFeatures_worldspace.csv      # World-space keypoints
└── affective_features/
    ├── affectiveFeatures.h5                 # Computed affective features
    └── affectiveFeatures.csv                # Human-readable affective features
```

### Emotion Classes

The model predicts one of four emotions:

| Class ID | Emotion |
|---|---|
| 0 | Angry |
| 1 | Neutral |
| 2 | Happy |
| 3 | Sad |


## Requirements

See `requirements.txt` for all dependencies.
