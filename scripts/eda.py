import argparse
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
    print(str(written).replace("\\", "/"))
