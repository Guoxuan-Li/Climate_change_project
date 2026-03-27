"""
Multimodal dataset combining Data1D sequences, Env-Data sequences, and Data3D grids.
Used for Stage 5 (fusion model).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.config import (
    DATA_ROOT, DATA1D_ROOT, SEQ_LEN, DATA1D_COLS, DATA1D_FEATURE_COLS,
)
from src.data.utils import env_dict_to_vector


class MultimodalDataset(Dataset):
    """Combined dataset returning all three modalities per sample.

    Returns: (data1d_seq, env_seq, data3d_tensor, target)
        data1d_seq: (seq_len, 4)
        env_seq: (seq_len, 92)
        data3d_tensor: (13, 81, 81)
        target: int
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                 seq_len: int = SEQ_LEN):
        self.data_root = data_root
        self.seq_len = seq_len

        valid = index_df[
            (index_df["future_direction24"] >= 0) &
            (index_df["data3d_exists"] == True)
        ].copy()
        valid = valid.sort_values(["year", "storm_name", "timestamp"]).reset_index(drop=True)

        # Pre-load Data1D tracks per storm
        self._track_cache = {}

        # Build samples: for each storm, create windows that have all 3 modalities
        self.samples = []
        for (year, storm, d1d_file, split, basin), group in valid.groupby(
            ["year", "storm_name", "data1d_file", "split", "basin"]
        ):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < seq_len:
                continue

            # Load Data1D track
            d1d_path = DATA1D_ROOT / basin / split / d1d_file
            if not d1d_path.exists():
                continue
            track_df = pd.read_csv(
                d1d_path, delimiter='\t', header=None, names=DATA1D_COLS,
                dtype={"timestamp": str}
            )
            track_df["timestamp"] = track_df["timestamp"].astype(str).str.strip()
            track_features = track_df[DATA1D_FEATURE_COLS].values.astype(np.float32)
            track_timestamps = track_df["timestamp"].values
            ts_to_idx = {ts: i for i, ts in enumerate(track_timestamps)}

            # Build windows
            for i in range(len(group) - seq_len + 1):
                window = group.iloc[i:i + seq_len]
                last_row = window.iloc[-1]

                # Check all timestamps exist in Data1D
                window_ts = window["timestamp"].astype(str).values
                track_indices = [ts_to_idx.get(ts) for ts in window_ts]
                if None in track_indices:
                    continue

                # Check Data1D indices are consecutive
                if track_indices != list(range(track_indices[0], track_indices[0] + seq_len)):
                    continue

                self.samples.append({
                    "data1d_features": track_features[track_indices[0]:track_indices[0] + seq_len].copy(),
                    "env_paths": window["env_path"].tolist(),
                    "data3d_path": last_row["data3d_path"],
                    "target": int(last_row["future_direction24"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Data1D sequence
        d1d = torch.tensor(s["data1d_features"], dtype=torch.float32)

        # Env-Data sequence
        env_seq = []
        for env_rel in s["env_paths"]:
            env_dict = np.load(str(self.data_root / env_rel), allow_pickle=True).item()
            env_seq.append(env_dict_to_vector(env_dict))
        env = torch.tensor(np.stack(env_seq), dtype=torch.float32)

        # Data3D (single timestep — the last in the window)
        d3d = self._load_data3d(self.data_root / s["data3d_path"])

        target = torch.tensor(s["target"], dtype=torch.long)

        return d1d, env, d3d, target

    def _load_data3d(self, path: Path) -> torch.Tensor:
        import xarray as xr
        ds = xr.open_dataset(str(path))
        u = ds["u"].values[0]
        v = ds["v"].values[0]
        z = ds["z"].values[0] / 10000.0
        sst_raw = ds["sst"].values
        sst = np.where(np.isfinite(sst_raw) & (np.abs(sst_raw) < 1e10), sst_raw, np.nan)
        sst = np.nan_to_num(sst, nan=0.0)
        sst = (sst - 290.0) / 20.0
        ds.close()
        channels = np.concatenate([u, v, z, sst[np.newaxis]], axis=0)
        return torch.tensor(channels, dtype=torch.float32)
