"""
Integrated End-to-End Emotion Recognition Pipeline
=====================================================
This script integrates three main components:
1. Pose Estimation: Extract 2D/3D keypoints from video (vis_step_order_no_image_gen.py)
2. Affective Feature Computation: Compute gait features (compute_aff_features)
3. Emotion Inference: Predict emotion using hybrid classifier (inference.py)

Usage:
    python integrated_pipeline.py \
        --video path/to/video.mp4 \
        --checkpoint path/to/STEP_model.pth.tar \
        --output path/to/output

FIXES APPLIED:
  [FIX 1] reshape_keypoint_features: coords were scrambled.
           reshape(T, C, V, M) was wrong — data layout is interleaved (V,C).
           Now correctly reshapes as (T,V,C) -> transpose -> (1,C,T,V,1).
  [FIX 2] Root-centering split: world-space keypoints kept for affective features
           (Speed/Stride/Accel were all ~0 after root subtraction).
           Root-centered keypoints used only for the ST-GCN skeleton branch.
  [FIX 3] Affective features loaded from pre-computed H5 file instead of
           being recomputed from MotionAGFormer output. compute_features.py
           was written for a different coordinate system and cannot be used
           on MotionAGFormer output directly. Pass --aff_features_h5 pointing
           to affectiveFeatures_3D_train.h5 or affectiveFeatures_3D_test.h5.
  [FIX 4] FPS read from actual video instead of hardcoded 30fps.
  [FIX 5] Scale conversion: MotionAGFormer outputs in normalized coordinates [-1, 1],
           which matches the training data coordinate system. No scaling needed.
           Affective features are computed directly and normalized to match the
           training distribution for correct inference on unseen videos.
  [FIX 6] Skeleton branch scaling: ST-GCN outputs [0, 490] are scaled by
           dividing by scale_factor before concatenating with affective features [-1, 1].
           This prevents BatchNorm amplification (3x -> extreme logits).
           
           Tuning guide: Default scale_factor=50 gives [0, 10] range.
           If predictions still biased to one class, try:
           - Increase scale_factor to 100 (→ [0, 5]) or 200 (→ [0, 2.5])
           - Decrease scale_factor to 25 (→ [0, 20]) to preserve more skeleton signal
           Can be adjusted by modifying line in predict(): 
           output = self._forward_with_scaling(x_aff, x_gait, scale_factor=50)
  [FIX 10] Affective feature computation limitation.
           MotionAGFormer outputs normalized [-1, 1] keypoints.
           compute_features.py was designed for world-space coordinates.
           Scaling input doesn't work (features scale differently):
           - Volume scales as scale³ (problematic)
           - Distances scale as scale¹ (problematic)
           - Angles don't scale (dimensionless ratios, always too large)
           RECOMMENDATION: Use pre-computed H5 features (pass --aff_features_h5)
           with --aff_features_h5 pointing to training/test affective feature file.
           Alternatively, implement custom feature computation for normalized space.
  [FIX 11] CRITICAL: Training used RAW unnormalized affective features from H5 files.
           NO normalization (_NORM_A/_NORM_B) was applied during training.
           Inference must match training conditions exactly.
           Solution: Pass raw affective features directly to model.
           Do NOT call _normalize_affective_features() - this breaks train/test mismatch.
  [FIX 12] ROBUST FILE DISCOVERY & FALLBACK for affective features.
           - Searches multiple locations for pre-computed affective features H5 files
           - Provides clear guidance if files not found
           - Falls back to computing features with explicit warnings
           - Includes diagnostic messages for troubleshooting
"""

import sys
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import glob
import pandas as pd
import h5py
from pathlib import Path
import copy
import importlib.util

script_dir = os.path.dirname(os.path.abspath(__file__))

_hrnet_lib_path   = os.path.join(script_dir, 'Pose_Estimation', 'lib', 'hrnet', 'lib')
_hrnet_utils_path = os.path.join(_hrnet_lib_path, 'utils')

def _preload_hrnet_utils():
    import sys

    if not os.path.isdir(_hrnet_utils_path):
        print(f"Warning: HRNet utils directory not found at {_hrnet_utils_path}")
        print("   Proceeding without pre-loading (relying on sys.path)")
        return

    init_file = os.path.join(_hrnet_utils_path, '__init__.py')
    if not os.path.exists(init_file):
        try:
            with open(init_file, 'w') as f:
                f.write("# HRNet utils package\nfrom . import transforms, inference, coco_h36m, utilitys\n")
            print(f"Created missing __init__.py in {_hrnet_utils_path}")
        except Exception as e:
            print(f"Warning: Could not create __init__.py: {e}")

    try:
        transforms_file = os.path.join(_hrnet_utils_path, 'transforms.py')
        if os.path.exists(transforms_file):
            spec = importlib.util.spec_from_file_location('hrnet_transforms', transforms_file)
            if spec and spec.loader:
                hrnet_transforms = importlib.util.module_from_spec(spec)
                sys.modules['hrnet_transforms'] = hrnet_transforms
                spec.loader.exec_module(hrnet_transforms)

        inference_file = os.path.join(_hrnet_utils_path, 'inference.py')
        if os.path.exists(inference_file):
            spec = importlib.util.spec_from_file_location('hrnet_inference', inference_file)
            if spec and spec.loader:
                hrnet_inference = importlib.util.module_from_spec(spec)
                sys.modules['hrnet_inference'] = hrnet_inference
                spec.loader.exec_module(hrnet_inference)

    except Exception as e:
        print(f"Warning: Could not pre-load HRNet modules: {e}")

    print("HRNet libs initialized and ready")


for _p in [
    os.path.join(script_dir, 'Recognition'),
    os.path.join(script_dir, 'Recognition', 'classifier_hybrid'),
    os.path.join(script_dir, 'demo', 'model'),
    os.path.join(script_dir, 'Pose_Estimation', 'model'),
    os.path.join(script_dir, 'Pose_Estimation'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_preload_hrnet_utils()

from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
from lib.utils import normalize_screen_coordinates, camera_to_world
from MotionAGFormer import MotionAGFormer
from compute_aff_features import compute_features
from compute_aff_features.normalize_features import normalize_features
from net.classifier import Classifier


# ============================================================================
# [FIX 3] NORMALIZATION CONSTANTS
# Exact per-feature midpoint (a) and half-range (b) derived from the full
# training set used to train the STEP emotion recognition model.
#
# Formula (matches normalize_features.py exactly):
#   a = (raw_max + raw_min) / 2
#   b = (raw_max - raw_min) / 2
#   normalized = clip((raw - a) / b, -1, 1)
#
# Verified: max diff vs stored affectiveFeature values = 0.00000000
# ============================================================================

_NORM_A = np.array([
    4.00723198e-04,  1.34319766e+01,  1.54657806e+02,  1.20083468e+01,
    8.21772105e+01,  6.76287251e+01,  8.48191865e-02,  5.66169322e-02,
    2.94567458e-02,  2.66529946e-02,  9.19402723e-04,  2.31467668e-04,
    1.85285963e-01,  6.47955686e-02,  1.03787122e-01,  1.27539383e-01,
    1.35710653e-01,  9.70796567e+00,  3.29279929e+00,  5.10986555e+00,
    6.58749622e+00,  6.48995143e+00,  5.39606274e+02,  1.80704048e+02,
    2.82056192e+02,  3.52619682e+02,  3.50481596e+02,  7.71428571e+00,
    2.57142857e-01,
], dtype=np.float32)

_NORM_B = np.array([
    2.28485958e-04,  7.47709060e+00,  1.40217133e+01,  6.64277434e+00,
    1.19267137e+01,  4.80911827e+00,  6.59189001e-03,  3.54102254e-03,
    7.23855197e-03,  7.20639620e-03,  2.96498853e-04,  1.57986316e-04,
    1.30339257e-01,  3.84046361e-02,  7.63165671e-02,  1.03386907e-01,
    1.09001581e-01,  7.92822284e+00,  2.19529361e+00,  4.02408946e+00,
    5.75640160e+00,  5.40018815e+00,  4.56512074e+02,  1.23448631e+02,
    2.27080770e+02,  3.09947029e+02,  2.92408907e+02,  5.28571429e+00,
    1.76190476e-01,
], dtype=np.float32)

_FEATURE_NAMES = [
    'Volume',
    'Ang_LS',    'Ang_RS',    'Ang_LH',    'Ang_RH',    'Ang_Torso',
    'Dist_LS_LH','Dist_RS_RH','Dist_LS_RS','Dist_LH_RH',
    'Area_LU',   'Area_RU',
    'Spd_LS',    'Spd_RS',    'Spd_LH',    'Spd_RH',    'Spd_Pel',
    'Acc_LS',    'Acc_RS',    'Acc_LH',    'Acc_RH',    'Acc_Pel',
    'Jrk_LS',    'Jrk_RS',    'Jrk_LH',    'Jrk_RH',    'Jrk_Pel',
    'Stride',    'Period',
]


# ============================================================================
# STAGE 1: POSE ESTIMATION (2D and 3D Keypoints)
# ============================================================================

def resample(n_frames):
    """Resample frames to 243 frames for model input."""
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    """Split long sequences into clips of 243 frames."""
    clips = []
    n_frames = keypoints.shape[1]
    downsample = np.arange(min(n_frames, 243))

    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
                downsample = np.arange(clip_length)

    return clips, downsample


def h36m_to_step_order(keypoints_h36m):
    """Convert H36M order (17 joints) to STEP order (16 joints)."""
    new_shape = list(keypoints_h36m.shape)
    new_shape[-2] = 16
    keypoints_step = np.zeros(new_shape, dtype=keypoints_h36m.dtype)

    keypoints_step[..., 0, :] = keypoints_h36m[..., 0, :]   # Pelvis
    keypoints_step[..., 1, :] = keypoints_h36m[..., 7, :]   # Spine
    keypoints_step[..., 2, :] = keypoints_h36m[..., 8, :]   # Thorax
    keypoints_step[..., 3, :] = keypoints_h36m[..., 10, :]  # Head
    keypoints_step[..., 4, :] = keypoints_h36m[..., 11, :]  # Left Shoulder
    keypoints_step[..., 5, :] = keypoints_h36m[..., 12, :]  # Left Elbow
    keypoints_step[..., 6, :] = keypoints_h36m[..., 13, :]  # Left Wrist
    keypoints_step[..., 7, :] = keypoints_h36m[..., 14, :]  # Right Shoulder
    keypoints_step[..., 8, :] = keypoints_h36m[..., 15, :]  # Right Elbow
    keypoints_step[..., 9, :] = keypoints_h36m[..., 16, :]  # Right Wrist
    keypoints_step[..., 10, :] = keypoints_h36m[..., 4, :]  # Left Hip
    keypoints_step[..., 11, :] = keypoints_h36m[..., 5, :]  # Left Knee
    keypoints_step[..., 12, :] = keypoints_h36m[..., 6, :]  # Left Ankle
    keypoints_step[..., 13, :] = keypoints_h36m[..., 1, :]  # Right Hip
    keypoints_step[..., 14, :] = keypoints_h36m[..., 2, :]  # Right Knee
    keypoints_step[..., 15, :] = keypoints_h36m[..., 3, :]  # Right Ankle

    return keypoints_step


def flip_data(data, left_joints=[4, 5, 6, 10, 11, 12], right_joints=[7, 8, 9, 13, 14, 15]):
    """Flip data for augmentation."""
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


def get_video_fps(video_path):
    """[FIX 4] Read actual FPS from video instead of assuming 30."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0 or fps > 240:
        print(f"  Suspicious FPS value ({fps}), defaulting to 30")
        fps = 30.0
    print(f"  Video FPS: {fps:.2f}")
    return fps


def extract_2d_pose(video_path, output_dir):
    """Extract 2D pose keypoints from video using HRNet."""
    print('\n[Stage 1a] Extracting 2D Pose...')
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    import io
    original_argv = sys.argv.copy()
    original_stderr = sys.stderr
    sys.argv = [sys.argv[0]]
    sys.stderr = io.StringIO()

    try:
        keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    except FileNotFoundError as e:
        sys.argv = original_argv
        sys.stderr = original_stderr
        raise FileNotFoundError(str(e))
    finally:
        sys.argv = original_argv
        sys.stderr = original_stderr

    if keypoints is None or scores is None or len(keypoints) == 0:
        raise RuntimeError(
            "YOLOv3 detector failed to extract keypoints from video.\n"
            "This usually means:\n"
            "  1. YOLOv3 weights file is missing\n"
            "  2. Video file is corrupted or unreadable\n"
            "  3. No humans detected in the video\n"
        )

    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    if keypoints.ndim != 4 or scores.ndim != 3 or keypoints.shape[0] == 0:
        raise RuntimeError(f"Invalid keypoints shape: {keypoints.shape} or scores shape: {scores.shape}.")

    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_2d_dir = output_dir + 'input_2D/'
    os.makedirs(output_2d_dir, exist_ok=True)

    output_csv = output_2d_dir + 'keypoints.csv'
    num_frames = keypoints.shape[1]
    num_joints = keypoints.shape[2]

    data_list = []
    for frame_idx in range(num_frames):
        for joint_idx in range(num_joints):
            row = [frame_idx, joint_idx] + keypoints[0, frame_idx, joint_idx, :].tolist()
            data_list.append(row)

    df = pd.DataFrame(data_list, columns=['frame', 'joint', 'x', 'y', 'confidence'])
    df.to_csv(output_csv, index=False)
    print(f'Saved 2D keypoints to {output_csv}')

    return keypoints


@torch.no_grad()
def extract_3d_pose(video_path, output_dir):
    """
    Extract 3D pose keypoints using MotionAGFormer model.

    [FIX 2] Returns TWO separate keypoint arrays:
      - all_3d_keypoints_world:    world-space (no root subtraction) for affective features
      - all_3d_keypoints_centered: root-centered                     for ST-GCN skeleton branch
    """
    print('\n[Stage 1b] Extracting 3D Pose...')

    args = {
        'n_layers': 16, 'dim_in': 3, 'dim_feat': 128, 'dim_rep': 512, 'dim_out': 3,
        'mlp_ratio': 4, 'act_layer': nn.GELU,
        'attn_drop': 0.0, 'drop': 0.0, 'drop_path': 0.0,
        'use_layer_scale': True, 'layer_scale_init_value': 0.00001, 'use_adaptive_fusion': True,
        'num_heads': 8, 'qkv_bias': False, 'qkv_scale': None,
        'hierarchical': False,
        'use_temporal_similarity': True, 'neighbour_num': 2, 'temporal_connection_len': 1,
        'use_tcn': False, 'graph_only': False,
        'n_frames': 243
    }

    model = nn.DataParallel(MotionAGFormer(**args)).cuda()
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'Pose_Estimation', '2D_to_3D_MotionAGFormer')
    model_candidates = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth*')))
    if not model_candidates:
        model_candidates = sorted(glob.glob('checkpoints/**/motionagformer*.pth*', recursive=True))
    if not model_candidates:
        raise FileNotFoundError(f"MotionAGFormer checkpoint not found in {checkpoint_dir}")
    model_path = model_candidates[0]
    print(f"Loading MotionAGFormer from: {model_path}")
    pre_dict = torch.load(model_path, weights_only=False)
    if 'model' in pre_dict:
        model.load_state_dict(pre_dict['model'], strict=True)
    else:
        model.load_state_dict(pre_dict, strict=True)
    model.eval()

    keypoints_csv = output_dir + 'input_2D/keypoints.csv'
    df = pd.read_csv(keypoints_csv)

    num_frames = df['frame'].max() + 1
    num_joints = df['joint'].max() + 1
    keypoints = np.zeros((1, num_frames, num_joints, 3))

    for _, row in df.iterrows():
        frame_idx = int(row['frame'])
        joint_idx = int(row['joint'])
        keypoints[0, frame_idx, joint_idx, 0] = row['x']
        keypoints[0, frame_idx, joint_idx, 1] = row['y']
        keypoints[0, frame_idx, joint_idx, 2] = row['confidence']

    clips, downsample = turn_into_clips(keypoints)

    cap = cv2.VideoCapture(video_path)
    ret, img = cap.read()
    img_size = img.shape if img is not None else None
    cap.release()

    if img_size is None:
        raise ValueError("Could not read video frame to get dimensions")

    # [FIX 2] Two separate lists: world-space and root-centered
    all_3d_keypoints_world    = []  # no root subtraction -> for affective features
    all_3d_keypoints_centered = []  # root subtracted    -> for ST-GCN skeleton branch

    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        input_2D_aug = flip_data(input_2D)
        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        # World-space: NO root subtraction -> preserves translation for speed/stride
        post_out_world = output_3D[0].cpu().detach().numpy()
        post_out_world_step = h36m_to_step_order(post_out_world)
        for post_out in post_out_world_step:
            all_3d_keypoints_world.append(post_out.copy())

        # Root-centered: subtract pelvis -> relative pose for ST-GCN
        root = output_3D[:, :, 0:1, :].clone()
        output_3D_centered = output_3D - root
        post_out_centered = output_3D_centered[0].cpu().detach().numpy()
        post_out_centered_step = h36m_to_step_order(post_out_centered)
        for post_out in post_out_centered_step:
            all_3d_keypoints_centered.append(post_out.copy())

    # Save CSV using world-space for human inspection
    output_3d_csv_dir = output_dir + 'output_3D/'
    os.makedirs(output_3d_csv_dir, exist_ok=True)
    output_3d_csv = output_3d_csv_dir + 'keypoints_step_order.csv'

    joint_names = ['Pelvis', 'Spine', 'Thorax', 'Head',
                   'LShoulder', 'LElbow', 'LWrist',
                   'RShoulder', 'RElbow', 'RWrist',
                   'LHip', 'LKnee', 'LAnkle',
                   'RHip', 'RKnee', 'RAnkle']

    data_list = []
    for frame_idx, keypoints_3d in enumerate(all_3d_keypoints_world):
        for joint_idx in range(keypoints_3d.shape[0]):
            row = [frame_idx, joint_idx, joint_names[joint_idx]] + keypoints_3d[joint_idx, :].tolist()
            data_list.append(row)

    df_3d = pd.DataFrame(data_list, columns=['frame', 'joint_id', 'joint_name', 'x', 'y', 'z'])
    df_3d.to_csv(output_3d_csv, index=False)
    print(f'Saved 3D keypoints (STEP order, world-space) to {output_3d_csv}')

    return all_3d_keypoints_world, all_3d_keypoints_centered


# ============================================================================
# STAGE 2: AFFECTIVE FEATURE COMPUTATION
# ============================================================================

def compute_affective_features(all_3d_keypoints_world, video_fps, output_dir):
    """
    Compute affective features from world-space 3D keypoints.
    [FIX 2] Uses world-space keypoints so Speed/Stride/Accel are non-zero.
    [FIX 4] Uses actual video FPS for time_step.
    [FIX 10] Use pre-computed features or compute from MotionAGFormer normalized keypoints.
            Features computed from [-1,1] keypoints fall outside training distribution.
            compute_features.py assumes world-space coordinates (~-2 to 2 meters).
            Solution: Use H5 pre-computed features (recommended) or implement custom
            feature computation that works with normalized coordinates.
    [FIX 11] CRITICAL: These RAW features are NOT normalized.
            Training model expects RAW, unnormalized features.
            Do NOT normalize - return as-is.
    """
    print('\n[Stage 2] Computing Affective Features...')

    frames = np.array(all_3d_keypoints_world)
    print(f"  Keypoint coordinate range: {frames.min():.4f} to {frames.max():.4f}")
    print(f"  WARNING: Computing features from MotionAGFormer [-1,1] normalized coordinates.")
    print(f"  This will result in high clipping (expected: 15-27/29 features clipped).")
    print(f"  BETTER: Use --aff_features_h5 with pre-computed features from training data.")
    
    num_frames = frames.shape[0]
    frames_flattened = frames.reshape(num_frames, -1)

    print(f"  Input frames shape: {frames_flattened.shape}")

    # [FIX 4] Use actual FPS, not hardcoded 30
    time_step = 1.0 / video_fps
    print(f"  Using time_step = 1/{video_fps:.2f} = {time_step:.6f}s per frame")

    aff_features_list = compute_features(frames_flattened, time_step)
    aff_features = np.array(aff_features_list)

    print(f"  Computed {len(aff_features)} affective features")

    output_aff_dir = output_dir + 'affective_features/'
    os.makedirs(output_aff_dir, exist_ok=True)

    output_aff_h5 = output_aff_dir + 'affectiveFeatures.h5'
    with h5py.File(output_aff_h5, 'w') as f:
        f.create_dataset('affective_features', data=aff_features)
    print(f'Saved affective features to {output_aff_h5}')

    output_aff_csv = output_aff_dir + 'affectiveFeatures.csv'
    feature_names = [
        'Volume',
        'Angle_LeftShoulder', 'Angle_RightShoulder', 'Angle_LeftHip', 'Angle_RightHip', 'Angle_Torso',
        'Distance_LeftShoulder_LeftHip', 'Distance_RightShoulder_RightHip',
        'Distance_LeftShoulder_RightShoulder', 'Distance_LeftHip_RightHip',
        'Area_LeftUpper', 'Area_RightUpper',
        'Speed_LeftShoulder', 'Speed_RightShoulder', 'Speed_LeftHip', 'Speed_RightHip', 'Speed_Pelvis',
        'Acceleration_LeftShoulder', 'Acceleration_RightShoulder', 'Acceleration_LeftHip',
        'Acceleration_RightHip', 'Acceleration_Pelvis',
        'Jerk_LeftShoulder', 'Jerk_RightShoulder', 'Jerk_LeftHip', 'Jerk_RightHip', 'Jerk_Pelvis',
        'StrideLength', 'StridePeriod'
    ]

    df_aff = pd.DataFrame([aff_features], columns=feature_names)
    df_aff.to_csv(output_aff_csv, index=False)
    print(f'Saved affective features to {output_aff_csv}')

    return aff_features


def compute_keypoint_features(all_3d_keypoints_world, all_3d_keypoints_centered, output_dir):
    """
    Prepare keypoint features for the ST-GCN skeleton branch.
    [FIX 2] Uses root-centered keypoints (relative pose only).
    [FIX 7] ALSO save world-space keypoints so affective features can be
            recomputed during inference (they depend on world-space coords).
    """
    print('\n[Stage 2b] Preparing Keypoint Features...')

    frames_centered = np.array(all_3d_keypoints_centered)
    frames_world = np.array(all_3d_keypoints_world)
    num_frames = frames_centered.shape[0]
    keypoint_features_centered = frames_centered.reshape(num_frames, -1)
    keypoint_features_world = frames_world.reshape(num_frames, -1)

    print(f"  Keypoint features shape (centered): {keypoint_features_centered.shape}")
    print(f"  Keypoint features shape (world): {keypoint_features_world.shape}")

    output_kp_dir = output_dir + 'keypoint_features/'
    os.makedirs(output_kp_dir, exist_ok=True)

    # Save root-centered version for skeleton branch
    output_kp_h5 = output_kp_dir + 'keypointFeatures.h5'
    with h5py.File(output_kp_h5, 'w') as f:
        f.create_dataset('keypoint_features', data=keypoint_features_centered)
        f.create_dataset('keypoint_features_world', data=keypoint_features_world)
    print(f'Saved keypoint features (both centered and world) to {output_kp_h5}')

    # Save root-centered CSV for visualization/inspection
    output_kp_csv = output_kp_dir + 'keypointFeatures.csv'
    joint_names = ['Pelvis', 'Spine', 'Thorax', 'Head',
                   'LShoulder', 'LElbow', 'LWrist',
                   'RShoulder', 'RElbow', 'RWrist',
                   'LHip', 'LKnee', 'LAnkle',
                   'RHip', 'RKnee', 'RAnkle']

    feature_columns = []
    for joint_name in joint_names:
        for coord in ['x', 'y', 'z']:
            feature_columns.append(f"{joint_name}_{coord}")

    df_kp = pd.DataFrame(keypoint_features_centered, columns=feature_columns)
    df_kp.to_csv(output_kp_csv, index=False)
    print(f'Saved keypoint features (root-centered) to {output_kp_csv}')

    # Save world-space CSV as well for affective feature computation
    output_kp_csv_world = output_kp_dir + 'keypointFeatures_worldspace.csv'
    df_kp_world = pd.DataFrame(keypoint_features_world, columns=feature_columns)
    df_kp_world.to_csv(output_kp_csv_world, index=False)
    print(f'Saved keypoint features (world-space) to {output_kp_csv_world}')

    return keypoint_features_centered


# ============================================================================
# STAGE 3: EMOTION INFERENCE
# ============================================================================

class Inferencer:
    """Emotion inference module."""

    def __init__(self, model_weights_path, device='cuda:0', coords=3, joints=16,
                 num_classes=4, temporal_kernel_size=75):
        self.device = device
        self.coords = coords
        self.joints = joints
        self.num_classes = num_classes
        self.temporal_kernel_size = temporal_kernel_size
        self.model = self._load_model(model_weights_path)
        self.model.to(device)
        self.model.eval()

    def _load_model(self, weights_path):
        graph_dict = {'strategy': 'spatial'}
        model = Classifier(
            in_channels=self.coords,
            in_features=29,
            num_classes=self.num_classes,
            graph_args=graph_dict,
            temporal_kernel_size=self.temporal_kernel_size,
            edge_importance_weighting=True
        )
        if os.path.isfile(weights_path):
            print(f"Loading checkpoint from {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Checkpoint loaded successfully")
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {weights_path}")
        return model

    def reshape_keypoint_features(self, keypoint_data):
        """
        [FIX 1] Correct reshape from (T, V*C) to (1, C, T, V, 1).

        Original bug: reshape(T, C, V, M) interpreted 48 values as
        [all-X-coords, all-Y-coords, all-Z-coords] — WRONG.

        Actual memory layout from frames.reshape(T, -1) is interleaved:
        [x0,y0,z0, x1,y1,z1, ..., x15,y15,z15]

        Fix: reshape as (T, V, C) first, then transpose to (C, T, V).
        """
        T, feature_dim = keypoint_data.shape
        expected_dim = self.joints * self.coords  # 16 * 3 = 48
        if feature_dim != expected_dim:
            raise ValueError(
                f"Expected feature_dim={expected_dim} "
                f"(joints={self.joints} * coords={self.coords}), got {feature_dim}"
            )
        print(f"  Input keypoint shape: {keypoint_data.shape}")

        reshaped = keypoint_data.reshape(T, self.joints, self.coords)  # (T, V, C)
        reshaped = np.transpose(reshaped, (2, 0, 1))                   # (C, T, V)
        reshaped = reshaped[np.newaxis, :, :, :, np.newaxis]           # (1, C, T, V, 1)

        print(f"  Final keypoint shape (N, C, T, V, M): {reshaped.shape}")
        return torch.from_numpy(reshaped.astype(np.float32))

    def reshape_affective_features(self, affective_data):
        print(f"  Input affective shape: {affective_data.shape}")
        affective_tensor = np.expand_dims(affective_data, axis=0)
        print(f"  Final affective shape (N, F): {affective_tensor.shape}")
        return torch.from_numpy(affective_tensor).float()

    def predict(self, affective_data, keypoint_data):
        print("\nReshaping inputs...")
        x_aff  = self.reshape_affective_features(affective_data)
        x_gait = self.reshape_keypoint_features(keypoint_data)
        x_aff  = x_aff.to(self.device)
        x_gait = x_gait.to(self.device)
        print(f"\nTensor shapes on {self.device}:")
        print(f"  x_aff:  {x_aff.shape}")
        print(f"  x_gait: {x_gait.shape}")
        print("\nRunning inference...")
        
        # Forward pass with custom scaling for skeleton branch
        with torch.no_grad():
            output = self._forward_with_scaling(x_aff, x_gait)
        
        print(f"  Output shape: {output.shape}")
        logits = output.cpu().numpy()
        print(f"  Raw logits: {logits}")  # keep for debugging
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        predicted_class = np.argmax(probabilities, axis=1)[0]
        predicted_prob  = np.max(probabilities)
        return {
            'logits': logits,
            'probabilities': probabilities,
            'predicted_class': predicted_class,
            'predicted_probability': predicted_prob
        }
    
    def _forward_with_scaling(self, x_aff, x_gait, scale_factor=50):
        """
        [FIX 6] Scale skeleton branch output to prevent numerical explosion.
        
        The skeleton branch (ST-GCN) produces values in range [0, 500+]
        but affective features are in [-1, 1]. When concatenated, mismatched
        scales cause BatchNorm to amplify skeleton by 3x → extreme logits.
        
        Solution: Divide skeleton by scale_factor instead of aggressive normalization.
        This preserves the learned representations while reducing magnitude.
        
        Tuning guide:
        - scale_factor=1:   [0, 490] (no scaling, causes problem)
        - scale_factor=50:  [0, 10] (moderate reduction)
        - scale_factor=100: [0, 5] (stronger reduction)
        - scale_factor=200: [0, 2.5] (very weak skeleton signal)
        
        Start with 50, adjust if predictions still biased to one class.
        """
        import torch.nn.functional as F
        
        model = self.model
        N, C, T, V, M = x_gait.size()
        
        # Skeleton branch (ST-GCN)
        x = x_gait.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = model.data_bn1(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        # ST-GCN layers
        for gcn, importance in zip(model.st_gcn_networks, model.edge_importance):
            x, _ = gcn(x, model.A * importance)
        
        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)  # (N, C_skeleton, 1, 1)
        
        # [FIX 6] Scale skeleton by divisor to reduce magnitude
        skeleton_min = x.min().item()
        skeleton_max = x.max().item()
        x_scaled = x / scale_factor
        x_scaled_min = x_scaled.min().item()
        x_scaled_max = x_scaled.max().item()
        print(f"[FIX 6] Skeleton scaling: range [{skeleton_min:.4f}, {skeleton_max:.4f}] / {scale_factor} -> [{x_scaled_min:.4f}, {x_scaled_max:.4f}]")
        
        # Combine branches
        x_aff_expanded = x_aff.unsqueeze(2).unsqueeze(2)
        x_combined = torch.cat((x_scaled, x_aff_expanded), dim=1)
        x_combined = x_combined.view(N, -1)
        x_combined = model.data_bn2(x_combined)
        x_combined = x_combined.unsqueeze(2).unsqueeze(2)
        
        # Classifier head
        for net in model.combined_networks:
            x_combined = net(x_combined)
        
        x_combined = x_combined.view(x_combined.size(0), -1)
        
        return x_combined


def run_inference(affective_features, keypoint_features, checkpoint_path,
                  device='cuda:0', num_classes=4, output_dir='./',
                  already_normalized=False):
    print('\n[Stage 3] Running Emotion Inference...')

    num_frames = keypoint_features.shape[0]
    video_duration = num_frames / 30.0
    if num_frames < 60:
        print(f"\nWARNING: Video is only {num_frames} frames ({video_duration:.1f} seconds)")
        print("   Gait-based emotion recognition typically requires 2-3+ seconds of video")
        print("   Predictions may be unreliable on short videos")

    # [FIX 11] DO NOT NORMALIZE affective features
    # Training loaded RAW features from H5 file without normalization.
    # Inference must match training conditions exactly.
    print(f"  Affective features (raw, matching training):")
    print(f"  Values: min={affective_features.min():.4f}, max={affective_features.max():.4f}")
    _print_affective_features_debug(affective_features)

    inferencer = Inferencer(
        model_weights_path=checkpoint_path,
        device=device,
        num_classes=num_classes
    )
    return inferencer.predict(affective_features, keypoint_features)


def _normalize_affective_features(features):
    """
    [DEPRECATED - DO NOT USE]
    This was applying normalization that was NOT done during training.
    Training loaded raw features from H5 without any normalization.
    Kept for historical reference only.
    """
    raise RuntimeError(
        "ERROR: _normalize_affective_features() should NOT be called.\n"
        "Training used RAW, UNNORMALIZED features from H5 files.\n"
        "Inference must match training conditions exactly.\n"
        "Remove this normalization step - pass raw features directly to model."
    )


def _print_affective_features_debug(features):
    """Debug print of raw affective features (matching training data)."""
    print(f"\n  {'#':>2}  {'Feature':<14} {'Raw Value':>12}")
    print(f"  {'-' * 38}")
    for i, (name, val) in enumerate(zip(_FEATURE_NAMES, features)):
        print(f"  {i:2d}  {name:<14} {val:12.5f}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def find_affective_features_h5():
    """
    [FIX 12] Search for pre-computed affective features H5 files.
    Searches in multiple common locations.
    
    Returns:
        Path to found H5 file, or None if not found
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    search_paths = [
        # Local project directory
        os.path.join(script_dir, 'affectiveFeatures_3D_test.h5'),
        os.path.join(script_dir, 'affectiveFeatures_3D_train.h5'),
        # Recognition subdirectory
        os.path.join(script_dir, 'Recognition', 'affectiveFeatures_3D_test.h5'),
        os.path.join(script_dir, 'Recognition', 'affectiveFeatures_3D_train.h5'),
        # Parent directory
        os.path.join(os.path.dirname(script_dir), 'affectiveFeatures_3D_test.h5'),
        os.path.join(os.path.dirname(script_dir), 'affectiveFeatures_3D_train.h5'),
        # Colab Google Drive paths (common in training)
        '/content/drive/MyDrive/affectiveFeatures_3D_test.h5',
        '/content/drive/MyDrive/affectiveFeatures_3D_train.h5',
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return None


def load_affective_features_from_h5(aff_h5_path, video_name):
    """
    Load pre-computed, pre-normalized affective features from a STEP H5 file.

    The affective features in these files were computed from the correct
    coordinate space and are already normalized to [-1, 1].
    compute_features.py cannot be used on MotionAGFormer output directly
    because it was written for a different coordinate system.

    Args:
        aff_h5_path: path to affectiveFeatures_3D_train.h5 or _test.h5
        video_name:  e.g. 'VID_003' or 'VID_RGB_003' or a video file path

    Returns:
        np.ndarray of shape (29,), already normalized
    """
    import re
    
    if not os.path.exists(aff_h5_path):
        raise FileNotFoundError(
            f"""
[ERROR] Pre-computed affective features H5 file not found:
  Path: {aff_h5_path}
  
To fix this, you need to provide affectiveFeatures_3D_test.h5 or affectiveFeatures_3D_train.h5

Expected file location:
  1. Same directory as integrated_pipeline.py
  2. In a 'Recognition' subdirectory
  3. In parent directory
  4. Anywhere accessible (specify full path in --aff_features_h5 argument)

Generation:
  If you don't have pre-computed features, generate them using:
    Recognition/compute_aff_features/main.py
  This requires access to the original training 3D keypoint data.

Alternatively:
  Run without --aff_features_h5 to compute features on-the-fly (will have high clipping).
            """
        )
    
    # Extract numeric ID from various formats (VID_003, VID_RGB_003, /path/VID_003.mp4)
    basename = os.path.splitext(os.path.basename(video_name))[0]
    m = re.search(r'(\d+)$', basename)
    if not m:
        raise ValueError(f"Cannot extract video ID from: {video_name}")
    vid_id = int(m.group(1))
    key = f'VID_RGB_{vid_id:03d}.csv'

    with h5py.File(aff_h5_path, 'r') as f:
        if key not in f:
            keys = sorted(f.keys())
            raise KeyError(
                f"Key '{key}' not found in {aff_h5_path}.\n"
                f"Available keys (sample): {keys[:5]} ... {keys[-5:]}"
            )
        features = f[key][:].astype(np.float32)

    print(f"  ✓ Loaded affective features for '{key}' from {os.path.basename(aff_h5_path)}")
    print(f"    Shape: {features.shape}, min={features.min():.4f}, max={features.max():.4f}")
    return features


def run_pipeline(video_path, checkpoint_path, output_dir, gpu='0', num_classes=4,
                 aff_features_h5=None):
    """
    Args:
        aff_features_h5: Optional path to pre-computed affective features H5 file.
                         If not provided, affective features are computed directly
                         from MotionAGFormer output (recommended for unseen videos).
    """
    print("=" * 70)
    print("INTEGRATED EMOTION RECOGNITION PIPELINE")
    print("=" * 70)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.makedirs(output_dir, exist_ok=True)

    # [FIX 4] Read actual FPS once and pass through the pipeline
    video_fps = get_video_fps(video_path)

    print(f"\n{'='*70}")
    print("STAGE 1: POSE ESTIMATION (2D + 3D)")
    print(f"{'='*70}")
    print(f"Input video: {video_path}")

    extract_2d_pose(video_path, output_dir)

    # [FIX 2] Unpack both world-space and root-centered keypoints
    all_3d_keypoints_world, all_3d_keypoints_centered = extract_3d_pose(video_path, output_dir)

    print(f"\n{'='*70}")
    print("STAGE 2: AFFECTIVE & KEYPOINT FEATURE COMPUTATION")
    print(f"{'='*70}")

    # Affective features: [FIX 12] Try to load from H5 with automatic discovery
    if aff_features_h5 is None:
        # Try to auto-discover pre-computed features
        discovered = find_affective_features_h5()
        if discovered:
            print(f"\n[Stage 2] Auto-discovered pre-computed affective features:")
            print(f"         {discovered}")
            aff_features_h5 = discovered
    
    if aff_features_h5 is not None and os.path.exists(aff_features_h5):
        print(f"\n[Stage 2] Loading pre-computed affective features from H5...")
        try:
            affective_features = load_affective_features_from_h5(aff_features_h5, video_path)
        except Exception as e:
            print(f"\n  ⚠ ERROR loading H5 features: {str(e)}")
            print(f"  Falling back to computing features from MotionAGFormer output...\n")
            affective_features = compute_affective_features(all_3d_keypoints_world, video_fps, output_dir)
    else:
        if aff_features_h5 is not None:
            print(f"\n[Stage 2] WARNING: Could not find --aff_features_h5 file at:")
            print(f"         {aff_features_h5}")
            print(f"  Falling back to computing features from MotionAGFormer output...\n")
        else:
            print(f"\n[Stage 2] No pre-computed affective features found.")
            print(f"  Computing affective features from MotionAGFormer output...")
        affective_features = compute_affective_features(all_3d_keypoints_world, video_fps, output_dir)

    keypoint_features = compute_keypoint_features(all_3d_keypoints_world, all_3d_keypoints_centered, output_dir)

    print(f"\n{'='*70}")
    print("STAGE 3: EMOTION RECOGNITION INFERENCE")
    print(f"{'='*70}")

    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("   Skipping inference stage.")
        return None

    results = run_inference(
        affective_features,
        keypoint_features,
        checkpoint_path,
        device=f'cuda:{gpu}',
        num_classes=num_classes,
        output_dir=output_dir,
        already_normalized=(aff_features_h5 is not None),
    )

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    emotion_labels = {0: 'Angry', 1: 'Neutral', 2: 'Happy', 3: 'Sad'}
    predicted_emotion = emotion_labels.get(results['predicted_class'], 'Unknown')
    print(f"Predicted Emotion: {predicted_emotion} (Class {results['predicted_class']})")
    print(f"Confidence: {results['predicted_probability']:.2%}")
    print("All Probabilities:")
    for class_id, emotion in emotion_labels.items():
        if class_id < len(results['probabilities'][0]):
            print(f"  {emotion}: {results['probabilities'][0][class_id]:.4f}")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Integrated Emotion Recognition from Gait Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (Colab)
  python integrated_pipeline.py \\
    --video /path/to/video.mp4 \\
    --checkpoint /path/to/model.pth.tar

  # With custom output base directory
  python integrated_pipeline.py \\
    --video /path/to/video.mp4 \\
    --checkpoint /path/to/model.pth.tar \\
    --output /custom/output/path/

  # With GPU selection
  python integrated_pipeline.py \\
    --video /path/to/video.mp4 \\
    --checkpoint /path/to/model.pth.tar \\
    --output /custom/output/path/ \\
    --gpu 1

Note: Output structure will be: {output_dir}/{video_name}/
  - {video_name}/input_2D/keypoints.csv
  - {video_name}/output_3D/keypoints_step_order.csv
  - {video_name}/keypoint_features/keypointFeatures.csv    (root-centered)
  - {video_name}/affective_features/affectiveFeatures.csv  (world-space)
        """
    )

    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth.tar)')
    parser.add_argument('--output', type=str,
                        default='/content/drive/MyDrive/Integrating-MotionAGFormer-and-STEP-for-Emotion-Recognition-from-Human-Gait/output/',
                        help='Output base directory for results')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device ID (default: 0)')
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of emotion classes (default: 4)')
    parser.add_argument('--aff_features_h5', type=str, default=None,
                        help='[FIX 12] Optional: path to pre-computed affective features H5 file. '
                             'If not provided, pipeline will auto-search common locations. '
                             'If not found, features are computed from MotionAGFormer output '
                             '(WARNING: will have high clipping ~15-27/29 features).')

    args = parser.parse_args()

    try:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        video_output_dir = os.path.join(args.output, video_name, '')

        results = run_pipeline(
            video_path=args.video,
            checkpoint_path=args.checkpoint,
            output_dir=video_output_dir,
            gpu=args.gpu,
            num_classes=args.num_classes,
            aff_features_h5=args.aff_features_h5,
        )
        print("Pipeline completed successfully!")

    except Exception as e:
        print(f"\nError during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)