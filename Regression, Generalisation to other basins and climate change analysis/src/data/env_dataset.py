"""
PyTorch Datasets for Env-Data features.

EnvSingleDataset: single-timestep (92-dim vector) — for Stage 1 MLP baseline.
EnvSequenceDataset: sequence of 8 timesteps (8x92) — for Stage 3 temporal model.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.config import DATA_ROOT, SEQ_LEN, ENV_FEATURE_DIM
from src.data.utils import env_dict_to_vector


class EnvSingleDataset(Dataset):
    """Single-timestep Env-Data dataset for MLP baseline.

    Each sample is a 92-dim feature vector + direction class label.
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT):
        # Filter to valid labels only
        self.df = index_df[index_df["future_direction24"] >= 0].reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        env_path = self.data_root / row["env_path"]
        env_dict = np.load(str(env_path), allow_pickle=True).item()

        features = env_dict_to_vector(env_dict)
        target = int(row["future_direction24"])

        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.long)


class EnvSequenceDataset(Dataset):
    """Sequence Env-Data dataset for temporal models (Stage 3).

    For each storm, creates sliding windows of SEQ_LEN consecutive timesteps.
    Each sample: (SEQ_LEN, 92) features + direction class label at last timestep.
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                 seq_len: int = SEQ_LEN):
        self.data_root = data_root
        self.seq_len = seq_len

        # Filter to valid labels
        valid_df = index_df[index_df["future_direction24"] >= 0].copy()
        valid_df = valid_df.sort_values(["storm_name", "year", "timestamp"]).reset_index(drop=True)

        # Group by storm, build windows
        self.samples = []
        for (year, storm), group in valid_df.groupby(["year", "storm_name"]):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < seq_len:
                continue
            for i in range(len(group) - seq_len + 1):
                window = group.iloc[i:i + seq_len]
                last_row = window.iloc[-1]
                self.samples.append({
                    "env_paths": window["env_path"].tolist(),
                    "target": int(last_row["future_direction24"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = []
        for env_rel_path in sample["env_paths"]:
            env_path = self.data_root / env_rel_path
            env_dict = np.load(str(env_path), allow_pickle=True).item()
            seq.append(env_dict_to_vector(env_dict))

        features = np.stack(seq, axis=0)  # (seq_len, 92)
        target = sample["target"]

        return (torch.tensor(features, dtype=torch.float32),
                torch.tensor(target, dtype=torch.long))
