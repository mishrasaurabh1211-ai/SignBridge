from pathlib import Path

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
