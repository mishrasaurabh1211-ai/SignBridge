import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from signbridge.common.landmarks import ROWS_PER_FRAME
from signbridge.data.cache import build_cache, load_cached
from signbridge.data.io import load_metadata, load_sequence
from signbridge.data.splits import make_folds


@pytest.fixture
def synthetic_data_env(tmp_path: Path) -> dict[str, Any]:
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    train_files_dir = raw_dir / "train_landmark_files" / "18"
    train_files_dir.mkdir(parents=True, exist_ok=True)

    # Create 2 sequences, 3 frames each, 543 rows/frame = 1629 rows
    n_frames = 3
    total_rows = n_frames * ROWS_PER_FRAME

    # Sequence 1: Left hand (landmarks 468..488) completely NaN
    frames = np.repeat(np.arange(n_frames), ROWS_PER_FRAME)
    landmark_indices = np.tile(np.arange(ROWS_PER_FRAME), n_frames)
    x1 = np.full(total_rows, 0.5, dtype=np.float32)
    y1 = np.full(total_rows, 0.5, dtype=np.float32)
    z1 = np.full(total_rows, 0.5, dtype=np.float32)
    # Mask left hand as NaN
    left_hand_mask = (landmark_indices >= 468) & (landmark_indices < 489)
    x1[left_hand_mask] = np.nan
    y1[left_hand_mask] = np.nan
    z1[left_hand_mask] = np.nan

    table1 = pa.Table.from_arrays(
        [
            pa.array(frames),
            pa.array([f"{f}-{idx}" for f, idx in zip(frames, landmark_indices)]),
            pa.array(["type"] * total_rows),
            pa.array(landmark_indices),
            pa.array(x1),
            pa.array(y1),
            pa.array(z1),
        ],
        names=["frame", "row_id", "type", "landmark_index", "x", "y", "z"],
    )
    seq1_path = train_files_dir / "101.parquet"
    pq.write_table(table1, seq1_path)

    # Sequence 2: fully non-NaN
    x2 = np.full(total_rows, 0.25, dtype=np.float32)
    y2 = np.full(total_rows, 0.25, dtype=np.float32)
    z2 = np.full(total_rows, 0.25, dtype=np.float32)
    table2 = pa.Table.from_arrays(
        [
            pa.array(frames),
            pa.array([f"{f}-{idx}" for f, idx in zip(frames, landmark_indices)]),
            pa.array(["type"] * total_rows),
            pa.array(landmark_indices),
            pa.array(x2),
            pa.array(y2),
            pa.array(z2),
        ],
        names=["frame", "row_id", "type", "landmark_index", "x", "y", "z"],
    )
    seq2_path = train_files_dir / "102.parquet"
    pq.write_table(table2, seq2_path)

    # 250 labels map ensuring 'thankyou' and 'tv' are kept
    final_sign_map = {"thankyou": 0, "tv": 1}
    for i in range(248):
        final_sign_map[f"sign_{i}"] = i + 2

    map_path = raw_dir / "sign_to_prediction_index_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(final_sign_map, f)

    # Metadata train.csv
    train_df = pd.DataFrame({
        "path": [
            "train_landmark_files/18/101.parquet",
            "train_landmark_files/18/102.parquet",
        ],
        "participant_id": [18, 18],
        "sequence_id": [101, 102],
        "sign": ["thankyou", "tv"],
    })
    train_df.to_csv(raw_dir / "train.csv", index=False)

    return {
        "raw_dir": raw_dir,
        "cache_dir": cache_dir,
        "seq1_path": seq1_path,
        "seq2_path": seq2_path,
        "map_path": map_path,
    }


def test_io_load_sequence_roundtrip(synthetic_data_env: dict[str, Any]) -> None:
    arr = load_sequence(synthetic_data_env["seq1_path"])
    assert arr.shape == (3, 543, 3)
    assert arr.dtype == np.float32
    # Left hand is NaN
    assert np.isnan(arr[:, 468:489, :]).all()
    # Face is not NaN
    assert not np.isnan(arr[:, 0:468, :]).any()


def test_io_invariant_violation_raises(synthetic_data_env: dict[str, Any]) -> None:
    bad_path = synthetic_data_env["raw_dir"] / "bad.parquet"
    # Create invalid parquet with 500 rows (not divisible by 543)
    table = pa.Table.from_arrays(
        [
            pa.array(np.zeros(500, dtype=np.float32)),
            pa.array(np.zeros(500, dtype=np.float32)),
            pa.array(np.zeros(500, dtype=np.float32)),
        ],
        names=["x", "y", "z"],
    )
    pq.write_table(table, bad_path)

    with pytest.raises(ValueError) as excinfo:
        load_sequence(bad_path)
    assert "Invariant violation" in str(excinfo.value)
    assert str(bad_path) in str(excinfo.value)


def test_metadata_assertions(synthetic_data_env: dict[str, Any]) -> None:
    meta = load_metadata(synthetic_data_env["raw_dir"])
    assert list(meta.columns) == ["path", "participant_id", "sequence_id", "sign", "label"]
    assert len(meta) == 2
    assert meta["label"].tolist() == [0, 1]

    # Test failure on sign map length != 250
    map_p = synthetic_data_env["map_path"]
    with open(map_p, "w", encoding="utf-8") as f:
        json.dump({"sign_0": 0}, f)
    with pytest.raises(AssertionError):
        load_metadata(synthetic_data_env["raw_dir"])


def test_cache_roundtrip(synthetic_data_env: dict[str, Any]) -> None:
    index_df = build_cache(
        data_root=synthetic_data_env["raw_dir"],
        cache_root=synthetic_data_env["cache_dir"],
        workers=1,
    )
    assert len(index_df) == 2
    assert (synthetic_data_env["cache_dir"] / "index.parquet").exists()
    assert (synthetic_data_env["cache_dir"] / "101.npy").exists()

    seq1_meta = index_df[index_df["sequence_id"] == 101].iloc[0]
    assert seq1_meta["nan_rate_left_hand"] == 1.0
    assert seq1_meta["nan_rate_face"] == 0.0

    loaded = load_cached(synthetic_data_env["cache_dir"], 101)
    assert loaded.shape == (3, 543, 3)
    assert loaded.dtype == np.float32


def test_splits_group_kfold_disjoint() -> None:
    # 5 participants, 15 sequences
    df = pd.DataFrame({
        "sequence_id": list(range(100, 115)),
        "participant_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5],
    })
    folds_df = make_folds(df, n_folds=5, seed=42)

    assert len(folds_df) == 15
    assert set(folds_df["fold"]) == {0, 1, 2, 3, 4}

    # Verify no participant appears in both train and validation of any fold
    merged = df.merge(folds_df, on="sequence_id")
    for f in range(5):
        val_parts = set(merged[merged["fold"] == f]["participant_id"])
        train_parts = set(merged[merged["fold"] != f]["participant_id"])
        assert val_parts.isdisjoint(train_parts)
