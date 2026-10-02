import os

files = {
    # -------------------------------------------------------------
    # 1. pyproject.toml (updated with matplotlib & mypy overrides)
    # -------------------------------------------------------------
    "pyproject.toml": """[project]
name = "signbridge"
version = "0.1.0"
description = "Real-time, bidirectional ASL<->English translator"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "fastapi",
    "uvicorn",
    "pydantic",
    "pyyaml",
    "onnxruntime",
    "tqdm",
    "matplotlib"
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
    "mypy"
]

[tool.setuptools]
packages = ["signbridge"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.mypy]
strict = true
packages = ["signbridge"]

[[tool.mypy.overrides]]
module = [
    "yaml.*",
    "pyarrow.*",
    "scipy.*",
    "sklearn.*",
    "matplotlib.*"
]
ignore_missing_imports = true
""",

    # -------------------------------------------------------------
    # 2. Config updates (tunables per Rule 3)
    # -------------------------------------------------------------
    "configs/base.yaml": """paths:
  data_root: data/raw
  cache_root: data/cache
  artifacts_root: artifacts
  dictionary_root: data/dictionary
seed: 42
data:
  n_folds: 5
  num_labels: 250
  fps: 30
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


class DataConfig(BaseModel):
    n_folds: int = 5
    num_labels: int = 250
    fps: int = 30


class BaseConfig(BaseModel):
    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    seed: int = 42


def load_config(path: str | Path, model_cls: type[T]) -> T:
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return model_cls.model_validate(data)
""",

    # -------------------------------------------------------------
    # 3. signbridge/data/io.py
    # -------------------------------------------------------------
    "signbridge/data/io.py": """import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from signbridge.common.landmarks import ROWS_PER_FRAME


def load_sequence(path: str | Path) -> np.ndarray[Any, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    table = pq.read_table(path, columns=["x", "y", "z"])
    num_rows = table.num_rows

    if num_rows % ROWS_PER_FRAME != 0:
        raise ValueError(
            f"Invariant violation: row count {num_rows} in '{path}' "
            f"is not divisible by ROWS_PER_FRAME ({ROWS_PER_FRAME})"
        )

    x = table.column("x").to_numpy(zero_copy_only=False).astype(np.float32)
    y = table.column("y").to_numpy(zero_copy_only=False).astype(np.float32)
    z = table.column("z").to_numpy(zero_copy_only=False).astype(np.float32)

    arr = np.stack([x, y, z], axis=-1)
    num_frames = num_rows // ROWS_PER_FRAME
    return arr.reshape((num_frames, ROWS_PER_FRAME, 3))


def load_metadata(data_root: str | Path) -> pd.DataFrame:
    data_root = Path(data_root)
    train_csv_path = data_root / "train.csv"
    map_json_path = data_root / "sign_to_prediction_index_map.json"

    if not train_csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")
    if not map_json_path.exists():
        raise FileNotFoundError(f"Sign map JSON not found at {map_json_path}")

    with open(map_json_path, "r", encoding="utf-8") as f:
        sign_map: dict[str, int] = json.load(f)

    if len(sign_map) != 250:
        raise AssertionError(
            f"Expected exactly 250 labels in {map_json_path}, got {len(sign_map)}"
        )

    df = pd.read_csv(train_csv_path)

    missing_signs = set(df["sign"]) - set(sign_map.keys())
    if missing_signs:
        raise AssertionError(
            f"Signs in train.csv missing from sign map: {missing_signs}"
        )

    df["label"] = df["sign"].map(sign_map).astype(int)
    required_cols = ["path", "participant_id", "sequence_id", "sign", "label"]
    return df[required_cols]
""",

    # -------------------------------------------------------------
    # 4. signbridge/data/splits.py
    # -------------------------------------------------------------
    "signbridge/data/splits.py": """from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def make_folds(
    index_df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    if len(index_df) == 0:
        empty_df = pd.DataFrame(columns=["sequence_id", "fold"])
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(output_path, index=False)
        return empty_df

    unique_participants = np.array(sorted(index_df["participant_id"].unique()))
    if len(unique_participants) < n_folds:
        raise ValueError(
            f"Number of participants ({len(unique_participants)}) must be >= n_folds ({n_folds})"
        )

    rng = np.random.default_rng(seed)
    shuffled_participants = unique_participants.copy()
    rng.shuffle(shuffled_participants)

    participant_to_group = {p: i for i, p in enumerate(shuffled_participants)}
    groups = index_df["participant_id"].map(participant_to_group).to_numpy()

    gkf = GroupKFold(n_splits=n_folds)
    fold_col = np.zeros(len(index_df), dtype=int)

    for fold_idx, (_, val_idx) in enumerate(gkf.split(index_df, groups=groups)):
        fold_col[val_idx] = fold_idx

    result_df = pd.DataFrame({
        "sequence_id": index_df["sequence_id"],
        "fold": fold_col,
    }).sort_values("sequence_id").reset_index(drop=True)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(out_p, index=False)

    return result_df
""",

    # -------------------------------------------------------------
    # 5. signbridge/data/cache.py
    # -------------------------------------------------------------
    "signbridge/data/cache.py": """import argparse
import multiprocessing
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from signbridge.common.landmarks import FACE, LEFT_HAND, POSE, RIGHT_HAND
from signbridge.data.io import load_metadata, load_sequence
from signbridge.data.splits import make_folds


def _process_single_sequence(
    task: tuple[int, int, str, int, str, str, str]
) -> dict[str, Any]:
    seq_id, part_id, sign, label, rel_path, data_root_str, cache_root_str = task
    cache_root = Path(cache_root_str)
    npy_path = cache_root / f"{seq_id}.npy"

    if npy_path.exists():
        arr_f16 = np.load(npy_path)
        arr = arr_f16.astype(np.float32)
    else:
        data_root = Path(data_root_str)
        parquet_path = data_root / rel_path
        arr = load_sequence(parquet_path)
        np.save(npy_path, arr.astype(np.float16))

    n_frames = int(arr.shape[0])
    if n_frames > 0:
        nan_rate_face = float(np.isnan(arr[:, FACE[0]:FACE[1]]).mean())
        nan_rate_left_hand = float(np.isnan(arr[:, LEFT_HAND[0]:LEFT_HAND[1]]).mean())
        nan_rate_pose = float(np.isnan(arr[:, POSE[0]:POSE[1]]).mean())
        nan_rate_right_hand = float(np.isnan(arr[:, RIGHT_HAND[0]:RIGHT_HAND[1]]).mean())
    else:
        nan_rate_face = 1.0
        nan_rate_left_hand = 1.0
        nan_rate_pose = 1.0
        nan_rate_right_hand = 1.0

    return {
        "sequence_id": seq_id,
        "participant_id": part_id,
        "sign": sign,
        "label": label,
        "n_frames": n_frames,
        "nan_rate_face": nan_rate_face,
        "nan_rate_left_hand": nan_rate_left_hand,
        "nan_rate_pose": nan_rate_pose,
        "nan_rate_right_hand": nan_rate_right_hand,
    }


def build_cache(
    data_root: str | Path,
    cache_root: str | Path,
    workers: int | None = None,
    make_folds_flag: bool = False,
) -> pd.DataFrame:
    data_root = Path(data_root)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    meta_df = load_metadata(data_root)
    tasks = [
        (
            int(row["sequence_id"]),
            int(row["participant_id"]),
            str(row["sign"]),
            int(row["label"]),
            str(row["path"]),
            str(data_root),
            str(cache_root),
        )
        for _, row in meta_df.iterrows()
    ]

    results: list[dict[str, Any]] = []
    if workers == 1:
        for t in tqdm(tasks, desc="Caching sequences"):
            results.append(_process_single_sequence(t))
    else:
        num_procs = workers if workers is not None else multiprocessing.cpu_count()
        chunksize = max(1, len(tasks) // (num_procs * 8))
        with multiprocessing.Pool(processes=workers) as pool:
            for res in tqdm(
                pool.imap_unordered(_process_single_sequence, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Caching sequences",
            ):
                results.append(res)

    index_df = pd.DataFrame(results).sort_values("sequence_id").reset_index(drop=True)
    index_df.to_parquet(cache_root / "index.parquet", index=False)

    if make_folds_flag:
        make_folds(index_df, output_path=cache_root / "folds.csv")

    return index_df


def load_cached(cache_root: str | Path, sequence_id: int | str) -> np.ndarray[Any, Any]:
    cache_root = Path(cache_root)
    p = cache_root / f"{sequence_id}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Cached sequence '{sequence_id}' not found in '{cache_root}'")
    arr: np.ndarray[Any, Any] = np.load(p)
    return arr.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache ASL sequence files.")
    parser.add_argument("--data-root", default="data/raw", help="Path to raw dataset.")
    parser.add_argument("--cache-root", default="data/cache", help="Path to cache directory.")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes count.")
    parser.add_argument("--folds", action="store_true", help="Generate folds.csv after caching.")
    args = parser.parse_args()

    t0 = time.time()
    df = build_cache(
        data_root=args.data_root,
        cache_root=args.cache_root,
        workers=args.workers,
        make_folds_flag=args.folds,
    )
    elapsed = time.time() - t0

    cache_path = Path(args.cache_root)
    total_bytes = sum(f.stat().st_size for f in cache_path.glob("*.npy"))
    size_mb = total_bytes / (1024 * 1024)
    size_gb = size_mb / 1024

    print(f"Wall time: {elapsed:.2f}s")
    print(f"Sequence count: {len(df)}")
    if size_gb >= 1.0:
        print(f"Cache size on disk: {size_gb:.2f} GB")
    else:
        print(f"Cache size on disk: {size_mb:.2f} MB")
""",

    # -------------------------------------------------------------
    # 6. scripts/eda.py
    # -------------------------------------------------------------
    "scripts/eda.py": """import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_eda(index_path: str | Path, output_dir: str | Path) -> Path:
    index_path = Path(index_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(index_path)

    lengths = df["n_frames"].to_numpy()
    p1 = float(np.percentile(lengths, 1))
    p10 = float(np.percentile(lengths, 10))
    p50 = float(np.percentile(lengths, 50))
    p90 = float(np.percentile(lengths, 90))
    p99 = float(np.percentile(lengths, 99))
    max_len = int(lengths.max())

    nan_rates = {
        "face": float(df["nan_rate_face"].mean()),
        "left_hand": float(df["nan_rate_left_hand"].mean()),
        "pose": float(df["nan_rate_pose"].mean()),
        "right_hand": float(df["nan_rate_right_hand"].mean()),
    }

    sign_counts = df["sign"].value_counts()
    seq_per_sign = {
        "min": int(sign_counts.min()),
        "median": float(sign_counts.median()),
        "max": int(sign_counts.max()),
    }

    part_counts = df["participant_id"].value_counts().to_dict()
    seq_per_participant = {str(k): int(v) for k, v in part_counts.items()}

    summary = {
        "frame_length_percentiles": {
            "p1": p1,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "p99": p99,
            "max": max_len,
        },
        "nan_rates": nan_rates,
        "sequences_per_sign": seq_per_sign,
        "sequences_per_participant": seq_per_participant,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Render Histogram PNGs
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50, edgecolor="black")
    plt.title("Sequence Frame Length Distribution")
    plt.xlabel("Number of Frames")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "length_distribution.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(list(nan_rates.keys()), list(nan_rates.values()))
    plt.title("Mean NaN Rate per Landmark Part")
    plt.xlabel("Part")
    plt.ylabel("Mean NaN Rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "nan_rates.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(sign_counts.values, bins=30, edgecolor="black")
    plt.title("Sequences per Sign Distribution")
    plt.xlabel("Sequences Count")
    plt.ylabel("Signs Count")
    plt.tight_layout()
    plt.savefig(output_dir / "sequences_per_sign.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(list(seq_per_participant.keys()), list(seq_per_participant.values()))
    plt.title("Sequences per Participant")
    plt.xlabel("Participant ID")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "sequences_per_participant.png")
    plt.close()

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-path", default="data/cache/index.parquet")
    parser.add_argument("--output-dir", default="artifacts/eda")
    args = parser.parse_args()

    written = run_eda(args.index_path, args.output_dir)
    print(str(written).replace("\\\\", "/"))
""",

    # -------------------------------------------------------------
    # 7. tests/test_data.py
    # -------------------------------------------------------------
    "tests/test_data.py": """import json
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

    # 250 labels map
    sign_map = {f"sign_{i}": i for i in range(250)}
    sign_map["thankyou"] = 0
    sign_map["tv"] = 1
    # ensure total keys is exactly 250
    final_sign_map = {f"sign_{i}": i for i in range(250)}
    final_sign_map["thankyou"] = 0
    final_sign_map["tv"] = 1
    if len(final_sign_map) > 250:
        final_sign_map = dict(list(final_sign_map.items())[:250])

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
""",

    # -------------------------------------------------------------
    # 8. docs/DATA.md (template ready for summary numbers)
    # -------------------------------------------------------------
    "docs/DATA.md": """# Dataset Statistics

*Note: All values below are generated directly by `python scripts/eda.py` and sourced from `artifacts/eda/summary.json`.*

## Frame Length Percentiles
| Metric | Value |
|---|---|
| p1 | <P1> |
| p10 | <P10> |
| p50 (median) | <P50> |
| p90 | <P90> |
| p99 | <P99> |
| Max | <MAX> |

## Landmark NaN Rates
| Part | NaN Rate |
|---|---|
| Face | <FACE_NAN> |
| Left Hand | <LEFT_HAND_NAN> |
| Pose | <POSE_NAN> |
| Right Hand | <RIGHT_HAND_NAN> |

## Class Balance (Sequences per Sign)
| Metric | Value |
|---|---|
| Min | <MIN_SIGNS> |
| Median | <MEDIAN_SIGNS> |
| Max | <MAX_SIGNS> |

## Participant Distribution
| Participant ID | Sequence Count |
|---|---|
<!-- PARTICIPANTS_ROWS -->
"""
}

# Ensure directories exist
os.makedirs("scripts", exist_ok=True)
os.makedirs("docs", exist_ok=True)
os.makedirs("configs", exist_ok=True)
os.makedirs("signbridge/data", exist_ok=True)
os.makedirs("signbridge/common", exist_ok=True)
os.makedirs("tests", exist_ok=True)

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

print("Phase 2 data layer files generated successfully.")