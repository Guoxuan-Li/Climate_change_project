"""
PyTorch Dataset for Data1D (tabular track) sequences — Stage 2.

Creates sliding windows of SEQ_LEN consecutive timesteps from each storm,
with the target being future_direction24 at the last timestep.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.config import (
    DATA_ROOT, DATA1D_ROOT, SEQ_LEN, DATA1D_COLS, DATA1D_FEATURE_COLS,
)


class Data1DSequenceDataset(Dataset):
    """Sliding-window dataset over Data1D track sequences.

    Each sample: (SEQ_LEN, 4) tensor of [LONG, LAT, PRES, WND] + label.
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                 seq_len: int = SEQ_LEN):
        self.data_root = data_root
        self.seq_len = seq_len

        valid_df = index_df[index_df["future_direction24"] >= 0].copy()
        valid_df = valid_df.sort_values(["year", "storm_name", "timestamp"]).reset_index(drop=True)

        # Build timestamp->label lookup from valid_df
        self.ts_to_label = {}
        for _, row in valid_df.iterrows():
            key = (row["year"], row["storm_name"], str(row["timestamp"]))
            self.ts_to_label[key] = int(row["future_direction24"])

        # Group by storm, load Data1D, build windows
        self.samples = []
        storm_groups = valid_df.groupby(["year", "storm_name", "data1d_file", "split"])

        for (year, storm_name, d1d_file, split), group in storm_groups:
            # Find the Data1D file
            d1d_path = DATA1D_ROOT / group.iloc[0]["basin"] / split / d1d_file
            if not d1d_path.exists():
                continue

            # Parse track
            track_df = pd.read_csv(
                d1d_path, delimiter='\t', header=None, names=DATA1D_COLS,
                dtype={"timestamp": str}
            )
            track_df["timestamp"] = track_df["timestamp"].astype(str).str.strip()

            # Extract features as array
            features = track_df[DATA1D_FEATURE_COLS].values.astype(np.float32)
            timestamps = track_df["timestamp"].values

            # Build sliding windows
            for i in range(len(track_df) - seq_len + 1):
                last_ts = timestamps[i + seq_len - 1]
                key = (year, storm_name, last_ts)
                if key in self.ts_to_label:
                    self.samples.append({
                        "features": features[i:i + seq_len].copy(),  # (seq_len, 4)
                        "target": self.ts_to_label[key],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.tensor(s["features"], dtype=torch.float32),
                torch.tensor(s["target"], dtype=torch.long))
