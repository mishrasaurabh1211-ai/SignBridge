import numpy as np
import pytest

from signbridge.common.config import FeatureConfig
from signbridge.common.landmarks import (
    LEFT_HAND,
    POSE,
    RIGHT_HAND,
    ROWS_PER_FRAME,
)
from signbridge.features.preprocess import (
    canonicalize_handedness,
    normalize_global,
    preprocess,
    resample,
    trim,
)


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
    _, mask2 = resample(x_short, 64)
    assert mask2[:30].all()
    assert not mask2[30:].any()

def test_feature_dim(cfg):
    seq = np.random.rand(10, ROWS_PER_FRAME, 3)
    feat = preprocess(seq, cfg)
    assert feat.x.shape[1] == cfg.feature_dim
