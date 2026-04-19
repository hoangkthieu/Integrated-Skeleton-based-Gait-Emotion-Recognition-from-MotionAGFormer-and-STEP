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
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download pre-trained checkpoints:
   - See `checkpoints/` directory for model download links

## Usage

Run the integrated pipeline:
```bash
python integrated_pipeline.py
```

## Requirements

See `requirements.txt` for all dependencies.

## License

[Add your license information here]

## Authors

[Add author information here]
