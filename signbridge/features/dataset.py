import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from signbridge.common.config import BaseConfig, FeatureConfig, load_config
from signbridge.data.cache import load_cached
from signbridge.features.augment import apply_augmentations
from signbridge.features.preprocess import preprocess


class SignDataset(Dataset[dict[str, torch.Tensor]]):
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
