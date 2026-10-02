import numpy as np

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
