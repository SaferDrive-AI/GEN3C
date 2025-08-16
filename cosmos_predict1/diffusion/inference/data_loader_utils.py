# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data loading utilities for the distributed format:
- RGB from mp4
- Depth from float16 numpy
- Camera data from float32 numpy
"""

import os
import numpy as np
import torch
import cv2
from pathlib import Path


def load_rgb_from_mp4(video_path):
    """
    Load RGB video from mp4 file and convert to tensor.
    
    Args:
        video_path: str, path to the mp4 file
        
    Returns:
        torch.Tensor: RGB tensor of shape [T, C, H, W] with range [-1, 1]
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")
    
    # Convert to numpy array and then tensor
    frames_np = np.stack(frames, axis=0)  # [T, H, W, C]
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()  # [T, C, H, W]
    
    # Convert from [0, 255] to [-1, 1]
    frames_tensor = (frames_tensor / 127.5) - 1.0
    
    return frames_tensor


def load_depth_from_numpy(depth_path):
    """
    Load depth data from compressed NPZ file.
    
    Args:
        depth_path: str, path to the NPZ file
        
    Returns:
        torch.Tensor: Depth tensor of shape [T, 1, H, W]
    """
    data = np.load(depth_path)
    depth_np = data['depth']  # [T, H, W]
    depth_tensor = torch.from_numpy(depth_np.astype(np.float32))
    
    # Add channel dimension: [T, H, W] -> [T, 1, H, W]
    depth_tensor = depth_tensor.unsqueeze(1)
    
    return depth_tensor


def load_mask_from_numpy(mask_path):
    """
    Load mask data from compressed NPZ file.
    
    Args:
        mask_path: str, path to the NPZ file
        
    Returns:
        torch.Tensor: Mask tensor of shape [T, 1, H, W]
    """
    data = np.load(mask_path)
    mask_np = data['mask']  # [T, H, W] as bool
    mask_tensor = torch.from_numpy(mask_np.astype(np.float32))  # Convert bool to float32
    
    # Add channel dimension: [T, H, W] -> [T, 1, H, W]
    mask_tensor = mask_tensor.unsqueeze(1)
    
    return mask_tensor


def load_camera_from_numpy(data_dir):
    """
    Load camera parameters from compressed NPZ file.
    
    Args:
        data_dir: str, directory containing camera.npz
        
    Returns:
        tuple: (w2c_tensor, intrinsics_tensor)
            - w2c_tensor: torch.Tensor of shape [T, 4, 4]
            - intrinsics_tensor: torch.Tensor of shape [T, 3, 3]
    """
    camera_path = os.path.join(data_dir, "camera.npz")
    
    if not os.path.exists(camera_path):
        raise FileNotFoundError(f"camera file not found: {camera_path}")
    
    data = np.load(camera_path)
    w2c_np = data['w2c']
    intrinsics_np = data['intrinsics']
    
    w2c_tensor = torch.from_numpy(w2c_np)
    intrinsics_tensor = torch.from_numpy(intrinsics_np)
    
    return w2c_tensor, intrinsics_tensor


def load_data_distributed_format(data_dir):
    """Load data from distributed format (mp4 + numpy files)"""
    data_path = Path(data_dir)
    
    # Load RGB from mp4
    cap = cv2.VideoCapture(str(data_path / "rgb.mp4"))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    frames_np = np.stack(frames, axis=0)
    image_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    image_tensor = (image_tensor / 127.5) - 1.0  # [0,255] -> [-1,1]
    
    # Load depth and mask
    depth_tensor = torch.from_numpy(np.load(data_path / "depth.npz")['depth'].astype(np.float32)).unsqueeze(1)
    mask_tensor = torch.from_numpy(np.load(data_path / "mask.npz")['mask'].astype(np.float32)).unsqueeze(1)
    
    # Load camera data
    camera_data = np.load(data_path / "camera.npz")
    w2c_tensor = torch.from_numpy(camera_data['w2c'])
    intrinsics_tensor = torch.from_numpy(camera_data['intrinsics'])
    
    return image_tensor, depth_tensor, mask_tensor, w2c_tensor, intrinsics_tensor


def load_data_packaged_format(pt_path):
    """
    Load data from the packaged pt format for backward compatibility.
    
    Args:
        pt_path: str, path to the pt file
        
    Returns:
        tuple: (image_tensor, depth_tensor, mask_tensor, w2c_tensor, intrinsics_tensor)
    """
    data = torch.load(pt_path)
    
    if len(data) != 5:
        raise ValueError(f"Expected 5 tensors in pt file, got {len(data)}")
    
    return data


def load_data_vipe_format(data_dir):
    """Load data from ViPE output format"""
    data_path = Path(data_dir)
    
    # Load RGB from mp4
    rgb_dir = data_path / "rgb"
    rgb_files = list(rgb_dir.glob("*.mp4"))
    if not rgb_files:
        raise ValueError(f"No MP4 file found in {rgb_dir}")
    
    cap = cv2.VideoCapture(str(rgb_files[0]))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    if not frames:
        raise ValueError(f"No frames found in video {rgb_files[0]}")
    
    frames_np = np.stack(frames, axis=0)
    image_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
    image_tensor = (image_tensor / 127.5) - 1.0  # [0,255] -> [-1,1]
    
    # For now, create dummy depth and mask data
    # TODO: Extract and load actual depth/mask data from zip files
    num_frames = len(frames)
    H, W = frames[0].shape[:2]
    depth_tensor = torch.ones(num_frames, 1, H, W) * 10.0  # Dummy depth
    mask_tensor = torch.ones(num_frames, 1, H, W)  # Dummy mask
    
    # Load camera data - ViPE format uses separate intrinsics and pose files
    intrinsics_files = list((data_path / "intrinsics").glob("*.npz"))
    pose_files = list((data_path / "pose").glob("*.npz"))
    
    if not intrinsics_files or not pose_files:
        raise ValueError("Missing intrinsics or pose files")
    
    # Load intrinsics
    intrinsics_data = np.load(intrinsics_files[0])
    intrinsics_raw = intrinsics_data['data']  # Shape: (N, 4) - fx, fy, cx, cy
    intrinsics_inds = intrinsics_data['inds']
    
    # Convert to 3x3 intrinsics matrices
    intrinsics_matrices = []
    for i in range(num_frames):
        if i < len(intrinsics_inds):
            fx, fy, cx, cy = intrinsics_raw[intrinsics_inds[i]]
        else:
            # Use last available intrinsics
            fx, fy, cx, cy = intrinsics_raw[intrinsics_inds[-1]]
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        intrinsics_matrices.append(K)
    
    intrinsics_tensor = torch.from_numpy(np.stack(intrinsics_matrices, axis=0))
    
    # Load poses (world-to-camera matrices)
    pose_data = np.load(pose_files[0])
    poses_raw = pose_data['data']  # Shape: (N, 4, 4)
    pose_inds = pose_data['inds']
    
    poses_matrices = []
    for i in range(num_frames):
        if i < len(pose_inds):
            w2c = poses_raw[pose_inds[i]]
        else:
            # Use last available pose
            w2c = poses_raw[pose_inds[-1]]
        poses_matrices.append(w2c)
    
    w2c_tensor = torch.from_numpy(np.stack(poses_matrices, axis=0).astype(np.float32))
    
    return image_tensor, depth_tensor, mask_tensor, w2c_tensor, intrinsics_tensor


def load_data_auto_detect(input_path):
    """Auto-detect format and load data"""
    input_path = Path(input_path)
    
    if input_path.is_file() and input_path.suffix == '.pt':
        return load_data_packaged_format(input_path)
    elif input_path.is_dir():
        # Check if it's ViPE format (has rgb/, depth/, etc. subdirectories)
        if (input_path / "rgb").exists() and (input_path / "intrinsics").exists():
            return load_data_vipe_format(input_path)
        else:
            return load_data_distributed_format(input_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}") 