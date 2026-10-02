import json
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
