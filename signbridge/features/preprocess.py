from dataclasses import dataclass
from typing import Any

import numpy as np

from signbridge.common.config import FeatureConfig
from signbridge.common.landmarks import (
    FACE,
    LEFT_HAND,
    LIPS,
    POSE,
    POSE_MIRROR_PAIRS,
    POSE_UPPER_BODY,
    RIGHT_HAND,
)


@dataclass
class Features:
    x: np.ndarray
    mask: np.ndarray
    meta: dict[str, Any]

def trim(seq: np.ndarray) -> np.ndarray:
    lh_nan = np.isnan(seq[:, LEFT_HAND[0]:LEFT_HAND[1], 0]).all(axis=1)
    rh_nan = np.isnan(seq[:, RIGHT_HAND[0]:RIGHT_HAND[1], 0]).all(axis=1)
    both_nan = lh_nan & rh_nan
    if both_nan.all():
        return seq
    first = np.argmax(~both_nan)
    last = len(both_nan) - np.argmax(~both_nan[::-1])
    return seq[first:last]

def normalize_global(seq: np.ndarray) -> np.ndarray:
    seq = seq.copy()
    l_shoulder_idx, r_shoulder_idx = POSE[0] + 11, POSE[0] + 12
    l_sh = seq[:, l_shoulder_idx]
    r_sh = seq[:, r_shoulder_idx]
    
    midpoints = (l_sh + r_sh) / 2.0
    widths = np.linalg.norm(l_sh[:, :2] - r_sh[:, :2], axis=1)
    
    valid_mask = ~np.isnan(widths) & (widths > 1e-5)
    if not valid_mask.any():
        return seq
        
    global_mid = np.nanmedian(midpoints[valid_mask], axis=0)
    global_width = np.nanmedian(widths[valid_mask])
    
    midpoints[~valid_mask] = global_mid
    widths[~valid_mask] = global_width
    
    # Subtract midpoint (x, y, z) and scale by width
    seq = (seq - midpoints[:, np.newaxis, :]) / widths[:, np.newaxis, np.newaxis]
    return seq

def canonicalize_handedness(seq: np.ndarray) -> tuple[np.ndarray, bool]:
    lh_valid = ~np.isnan(seq[:, LEFT_HAND[0]:LEFT_HAND[1], 0]).all(axis=1)
    rh_valid = ~np.isnan(seq[:, RIGHT_HAND[0]:RIGHT_HAND[1], 0]).all(axis=1)
    
    left_count = lh_valid.sum()
    right_count = rh_valid.sum()
    is_left_dominant = left_count > right_count
    
    if not is_left_dominant:
        return seq, False
        
    seq = seq.copy()
    seq[..., 0] = -seq[..., 0]  # Flip X
    
    # Swap hands
    lh_data = seq[:, LEFT_HAND[0]:LEFT_HAND[1]].copy()
    seq[:, LEFT_HAND[0]:LEFT_HAND[1]] = seq[:, RIGHT_HAND[0]:RIGHT_HAND[1]]
    seq[:, RIGHT_HAND[0]:RIGHT_HAND[1]] = lh_data
    
    # Swap pose pairs
    for i, j in POSE_MIRROR_PAIRS:
        pi, pj = POSE[0] + i, POSE[0] + j
        pi_data = seq[:, pi].copy()
        seq[:, pi] = seq[:, pj]
        seq[:, pj] = pi_data
        
    # Swap lip X (simplification for tests, accurate symmetric swap needs lip pair map)
    # Using full LIPS mirrored coordinates.
    # In practice we just mirror X, indices aren't strictly paired symmetrically here unless defined.
    return seq, True

def build_features(seq: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    dims = 3 if cfg.use_z else 2
    coords = []
    flags = []
    
    # Select Landmarks
    if cfg.use_left_hand:
        h = seq[:, LEFT_HAND[0]:LEFT_HAND[1], :dims]
        coords.append(h.reshape(seq.shape[0], -1))
        flags.append((~np.isnan(h[:, 0, 0])).astype(np.float32))
    if cfg.use_right_hand:
        h = seq[:, RIGHT_HAND[0]:RIGHT_HAND[1], :dims]
        coords.append(h.reshape(seq.shape[0], -1))
        flags.append((~np.isnan(h[:, 0, 0])).astype(np.float32))
    if cfg.use_pose_upper:
        p_idx = [POSE[0] + i for i in POSE_UPPER_BODY]
        h = seq[:, p_idx, :dims]
        coords.append(h.reshape(seq.shape[0], -1))
        flags.append((~np.isnan(h[:, 0, 0])).astype(np.float32))
    if cfg.use_lips:
        l_idx = [FACE[0] + i for i in LIPS]
        h = seq[:, l_idx, :dims]
        coords.append(h.reshape(seq.shape[0], -1))
        flags.append((~np.isnan(h[:, 0, 0])).astype(np.float32))
        
    # Hand Shape Features
    for hand_slice in [(LEFT_HAND, cfg.use_left_hand), (RIGHT_HAND, cfg.use_right_hand)]:
        sl, use = hand_slice
        if use:
            pts = seq[:, sl[0]:sl[1], :dims]
            wrist = pts[:, 0:1, :]
            shape = pts - wrist
            # bbox diag
            min_pts = np.nanmin(pts, axis=1, keepdims=True)
            max_pts = np.nanmax(pts, axis=1, keepdims=True)
            diag = np.linalg.norm(max_pts - min_pts, axis=2, keepdims=True)
            diag[diag < 1e-5] = 1.0
            shape = shape / diag
            coords.append(shape.reshape(seq.shape[0], -1))

    x = np.concatenate(coords, axis=1)
    
    # Motion
    if cfg.use_motion:
        motion = np.zeros_like(x)
        motion[1:] = x[1:] - x[:-1]
        x = np.concatenate([x, motion], axis=1)
        
    # Fill and Mask
    x = np.nan_to_num(x, nan=0.0)
    if flags:
        flag_arr = np.stack(flags, axis=1)
        x = np.concatenate([x, flag_arr], axis=1)
        
    return x, flag_arr

def resample(x: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    T, D = x.shape
    if T == 0:
        return np.zeros((n_frames, D), dtype=np.float32), np.zeros(n_frames, dtype=bool)
    
    mask = np.ones(n_frames, dtype=bool)
    if T == n_frames:
        return x.astype(np.float32), mask
        
    out = np.zeros((n_frames, D), dtype=np.float32)
    
    if T < n_frames:
        out[:T] = x
        mask[T:] = False
    else:
        # T > n_frames: linear interpolate
        idx_in = np.linspace(0, T - 1, n_frames)
        idx_orig = np.arange(T)
        for d in range(D):
            out[:, d] = np.interp(idx_in, idx_orig, x[:, d])
            
    return out, mask

def preprocess(seq: np.ndarray, cfg: FeatureConfig) -> Features:
    seq = trim(seq)
    seq = normalize_global(seq)
    seq, is_mirrored = canonicalize_handedness(seq)
    
    x_raw, _ = build_features(seq, cfg)
    x, mask = resample(x_raw, cfg.n_frames)
    
    return Features(x=x.astype(np.float32), mask=mask, meta={"mirrored": is_mirrored})
