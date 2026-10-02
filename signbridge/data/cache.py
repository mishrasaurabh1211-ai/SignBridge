import argparse
import multiprocessing
import time
from pathlib import Path
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
            results.extend(tqdm(
                pool.imap_unordered(_process_single_sequence, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Caching sequences",
            ))

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
