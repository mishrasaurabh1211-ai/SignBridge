import os

files = {
    # -------------------------------------------------------------
    # 1. Config Updates (adding FeatureConfig)
    # -------------------------------------------------------------
    "configs/features.yaml": """use_z: false
use_left_hand: true
use_right_hand: true
use_pose_upper: true
use_lips: true
use_motion: true
n_frames: 64
""",
    
    "signbridge/common/config.py": """from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

class PathConfig(BaseModel):
    data_root: str = "data/raw"
    cache_root: str = "data/cache"
    artifacts_root: str = "artifacts"
    dictionary_root: str = "data/dictionary"
    features_dir: str = "data/features"

class DataConfig(BaseModel):
    n_folds: int = 5
    num_labels: int = 250
    fps: int = 30

class FeatureConfig(BaseModel):
    use_z: bool = False
    use_left_hand: bool = True
    use_right_hand: bool = True
    use_pose_upper: bool = True
    use_lips: bool = True
    use_motion: bool = True
    n_frames: int = 64

    @property
    def feature_dim(self) -> int:
        from signbridge.common.landmarks import POSE_UPPER_BODY, LIPS
        dims = 3 if self.use_z else 2
        
        # Base coordinates
        n_points = 0
        if self.use_left_hand: n_points += 21
        if self.use_right_hand: n_points += 21
        if self.use_pose_upper: n_points += len(POSE_UPPER_BODY)
        if self.use_lips: n_points += len(LIPS)
        
        dim = n_points * dims
        
        # Hand shape (21 coords each)
        if self.use_left_hand: dim += 21 * dims
        if self.use_right_hand: dim += 21 * dims
        
        # Motion (first difference duplicates current feature dim)
        if self.use_motion:
            dim *= 2
            
        # Missing value indicators (1 per enabled part)
        dim += sum([self.use_left_hand, self.use_right_hand, self.use_pose_upper, self.use_lips])
        return dim

class BaseConfig(BaseModel):
    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    seed: int = 42

def load_config(path: str | Path, model_cls: type[T]) -> T:
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return model_cls.model_validate(data)
""",

    # -------------------------------------------------------------
    # 2. signbridge/features/preprocess.py
    # -------------------------------------------------------------
    "signbridge/features/preprocess.py": """from dataclasses import dataclass
import numpy as np
from signbridge.common.config import FeatureConfig
from signbridge.common.landmarks import (
    FACE, LEFT_HAND, POSE, RIGHT_HAND, POSE_UPPER_BODY, LIPS, POSE_MIRROR_PAIRS
)

@dataclass
class Features:
    x: np.ndarray
    mask: np.ndarray
    meta: dict

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
""",

    # -------------------------------------------------------------
    # 3. signbridge/features/augment.py
    # -------------------------------------------------------------
    "signbridge/features/augment.py": """import numpy as np
from signbridge.common.landmarks import LEFT_HAND, RIGHT_HAND

def augment_affine(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seq = seq.copy()
    scale = rng.uniform(0.8, 1.2)
    angle = np.deg2rad(rng.uniform(-15, 15))
    shift_x = rng.uniform(-0.1, 0.1)
    shift_y = rng.uniform(-0.1, 0.1)
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Applied only to x, y
    x, y = seq[..., 0], seq[..., 1]
    
    x_new = (x * cos_a - y * sin_a) * scale + shift_x
    y_new = (x * sin_a + y * cos_a) * scale + shift_y
    
    seq[..., 0] = x_new
    seq[..., 1] = y_new
    return seq

def augment_speed(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    T = len(seq)
    if T < 2: return seq
    rate = rng.uniform(0.7, 1.3)
    new_T = max(2, int(T * rate))
    
    idx_in = np.linspace(0, T - 1, new_T)
    idx_orig = np.arange(T)
    out = np.zeros((new_T, *seq.shape[1:]), dtype=seq.dtype)
    
    for v in range(seq.shape[1]):
        for d in range(seq.shape[2]):
            out[:, v, d] = np.interp(idx_in, idx_orig, seq[:, v, d])
    return out

def augment_drop_frames(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    T = len(seq)
    if T < 5: return seq
    drop_pct = rng.uniform(0, 0.1)
    keep_n = max(2, int(T * (1.0 - drop_pct)))
    keep_idx = np.sort(rng.choice(T, keep_n, replace=False))
    return seq[keep_idx]

def augment_hand_dropout(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() > 0.1:
        return seq
    seq = seq.copy()
    T = len(seq)
    if T < 3: return seq
    
    span = rng.integers(1, max(2, T // 2))
    start = rng.integers(0, T - span)
    
    hand = LEFT_HAND if rng.random() < 0.5 else RIGHT_HAND
    seq[start:start+span, hand[0]:hand[1]] = np.nan
    return seq

def apply_augmentations(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    seq = augment_affine(seq, rng)
    seq = augment_speed(seq, rng)
    seq = augment_drop_frames(seq, rng)
    seq = augment_hand_dropout(seq, rng)
    return seq
""",

    # -------------------------------------------------------------
    # 4. signbridge/features/dataset.py
    # -------------------------------------------------------------
    "signbridge/features/dataset.py": """import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from signbridge.common.config import FeatureConfig, load_config, BaseConfig
from signbridge.data.cache import load_cached
from signbridge.features.preprocess import preprocess
from signbridge.features.augment import apply_augmentations

class SignDataset(Dataset):
    def __init__(
        self,
        index_df: pd.DataFrame,
        cache_root: str | Path,
        cfg: FeatureConfig,
        mode: str = "val",
        features_dir: str | Path | None = None,
        seed: int = 42
    ):
        self.df = index_df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.cfg = cfg
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        
        self.memmap_x = None
        self.memmap_mask = None
        if self.mode == "val" and features_dir is not None:
            fd = Path(features_dir)
            self.memmap_x = np.memmap(fd / "features.dat", dtype=np.float32, mode="r", 
                                      shape=(len(self.df), cfg.n_frames, cfg.feature_dim))
            self.memmap_mask = np.memmap(fd / "masks.dat", dtype=bool, mode="r",
                                         shape=(len(self.df), cfg.n_frames))

    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        seq_id = row["sequence_id"]
        label = row["label"]
        
        if self.mode == "val" and self.memmap_x is not None and self.memmap_mask is not None:
            x = np.array(self.memmap_x[idx])
            mask = np.array(self.memmap_mask[idx])
        else:
            arr = load_cached(self.cache_root, seq_id)
            if self.mode == "train":
                arr = apply_augmentations(arr, self.rng)
            feats = preprocess(arr, self.cfg)
            x, mask = feats.x, feats.mask
            
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long)
        }

def build_features(cache_root: str | Path, cfg: FeatureConfig, out_dir: str | Path) -> None:
    cache_root = Path(cache_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(cache_root / "index.parquet")
    N = len(df)
    
    x_map = np.memmap(out_dir / "features.dat", dtype=np.float32, mode="w+", shape=(N, cfg.n_frames, cfg.feature_dim))
    m_map = np.memmap(out_dir / "masks.dat", dtype=bool, mode="w+", shape=(N, cfg.n_frames))
    
    labels = np.zeros(N, dtype=np.int64)
    seq_ids = np.zeros(N, dtype=np.int64)
    
    for i, row in tqdm(df.iterrows(), total=N, desc="Building memmap"):
        arr = load_cached(cache_root, row["sequence_id"])
        feat = preprocess(arr, cfg)
        x_map[i] = feat.x
        m_map[i] = feat.mask
        labels[i] = row["label"]
        seq_ids[i] = row["sequence_id"]
        
    x_map.flush()
    m_map.flush()
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "sequence_ids.npy", seq_ids)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    
    if args.build:
        base_cfg = load_config("configs/base.yaml", BaseConfig)
        with open("configs/features.yaml") as f:
            import yaml
            f_cfg = FeatureConfig(**yaml.safe_load(f))
        
        t0 = time.time()
        build_features("data/cache", f_cfg, "data/features")
        elapsed = time.time() - t0
        
        feat_path = Path("data/features/features.dat")
        size_mb = feat_path.stat().st_size / (1024 * 1024)
        
        print(f"Wall time: {elapsed:.2f}s")
        print(f"Memmap size (features.dat): {size_mb:.2f} MB")
        
        labels = np.load("data/features/labels.npy")
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Class Balance: Min={counts.min()}, Median={np.median(counts)}, Max={counts.max()}")
""",

    # -------------------------------------------------------------
    # 5. tests/test_features.py
    # -------------------------------------------------------------
    "tests/test_features.py": """import numpy as np
import pytest
from signbridge.common.config import FeatureConfig
from signbridge.features.preprocess import (
    trim, normalize_global, canonicalize_handedness, resample, preprocess
)
from signbridge.common.landmarks import ROWS_PER_FRAME, LEFT_HAND, RIGHT_HAND, POSE, POSE_MIRROR_PAIRS

@pytest.fixture
def cfg():
    return FeatureConfig(
        use_z=False, use_left_hand=True, use_right_hand=True, 
        use_pose_upper=True, use_lips=True, use_motion=True, n_frames=64
    )

def test_trim():
    seq = np.random.rand(10, ROWS_PER_FRAME, 3)
    seq[:2, LEFT_HAND[0]:LEFT_HAND[1]] = np.nan
    seq[:2, RIGHT_HAND[0]:RIGHT_HAND[1]] = np.nan
    seq[-2:, LEFT_HAND[0]:LEFT_HAND[1]] = np.nan
    seq[-2:, RIGHT_HAND[0]:RIGHT_HAND[1]] = np.nan
    trimmed = trim(seq)
    assert len(trimmed) == 6

def test_normalize_global_invariant_idempotent():
    seq = np.random.rand(5, ROWS_PER_FRAME, 3)
    # Ensure no nans on shoulders
    seq[:, POSE[0]+11] = 1.0
    seq[:, POSE[0]+12] = 2.0
    
    norm1 = normalize_global(seq)
    norm2 = normalize_global(norm1)
    np.testing.assert_allclose(norm1, norm2, atol=1e-5)
    
    # Scale and Translate
    seq_trans = seq * 5.0 + 10.0
    norm3 = normalize_global(seq_trans)
    np.testing.assert_allclose(norm1, norm3, atol=1e-5)

def test_handedness_parity():
    # Left dominant
    seq = np.random.rand(10, ROWS_PER_FRAME, 3)
    seq[:, RIGHT_HAND[0]:RIGHT_HAND[1]] = np.nan
    seq_canon, is_mirrored = canonicalize_handedness(seq)
    assert is_mirrored
    
    # Check X flipped
    np.testing.assert_allclose(seq_canon[:, LEFT_HAND[0]:LEFT_HAND[1], 0], -seq[:, RIGHT_HAND[0]:RIGHT_HAND[1], 0])

def test_resample_identity():
    x = np.random.rand(64, 10).astype(np.float32)
    x_res, mask = resample(x, 64)
    np.testing.assert_allclose(x, x_res)
    assert mask.all()
    
    x_short = np.random.rand(30, 10).astype(np.float32)
    x_res2, mask2 = resample(x_short, 64)
    assert mask2[:30].all()
    assert not mask2[30:].any()

def test_feature_dim(cfg):
    seq = np.random.rand(10, ROWS_PER_FRAME, 3)
    feat = preprocess(seq, cfg)
    assert feat.x.shape[1] == cfg.feature_dim
""",
    
    # -------------------------------------------------------------
    # 6. docs/FEATURES.md
    # -------------------------------------------------------------
    "docs/FEATURES.md": """# Feature Pipeline

This document defines the exact data preprocessing contract for the model. The web/TypeScript port MUST implement these steps identically.

## Preprocessing Steps (Ordered)
1. **Trim:** Drop leading/trailing frames where both hands are entirely NaN.
2. **Normalize Global:** Subtract shoulder midpoint, divide by shoulder width (computed via `POSE` landmarks 11, 12). Falls back to sequence median if missing.
3. **Canonicalize Handedness:** Count valid hand frames. If Left > Right, flip X coordinates, swap left/right hand landmarks, swap pose mirror pairs. (Model only sees right-dominant signers).
4. **Select Landmarks:** Keep Left Hand, Right Hand, Pose Upper Body, Lips. Drop Z axis.
5. **Hand Shape Features:** Append `(point - wrist) / bbox_diagonal` for each hand.
6. **Motion Features:** Append first-difference over time (`t - t_minus_1`), padded with 0 at `t=0`.
7. **Fill and Mask:** Convert remaining NaNs to `0.0`. Append boolean presence flags (1=present, 0=missing) for each of the 4 parts per frame.
8. **Resample:** Linearly interpolate to exactly 64 frames (if `T > 64`), or pad with `mask=False` (if `T < 64`).

## Channel Layout (Base Configuration)
*Note: Configurable via `FeatureConfig`. The exact dimensions will be verified dynamically.*

| Feature Block | Details |
|---|---|
| Handedness Base | x, y coordinates for all enabled parts |
| Shape | Wrist-relative coordinates normalized by bounding box |
| Motion | Time derivatives (size matches Base + Shape) |
| Presence Flags | 1 channel per part (e.g. 4 channels total) |
"""
}

os.makedirs("configs", exist_ok=True)
os.makedirs("signbridge/common", exist_ok=True)
os.makedirs("signbridge/features", exist_ok=True)
os.makedirs("tests", exist_ok=True)
os.makedirs("docs", exist_ok=True)

for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Phase 3 feature layer files generated successfully.")