"""
PyTorch Datasets for intensity prediction.

Classification targets: future_inte_change24 (int 0-3, filter out -1)
  0 = Strengthening, 1 = Str-then-Weak, 2 = Weakening, 3 = Maintaining

Regression targets: delta_wnd_norm (float scalar, 24h change in normalized wind)
  Physical conversion: delta_wind_ms = delta_wnd_norm * 25  (from wnd_norm = (wnd-40)/25)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.config import (
    DATA_ROOT, DATA1D_ROOT, SEQ_LEN, ENV_FEATURE_DIM,
    DATA1D_COLS, DATA1D_FEATURE_COLS,
)
from src.data.utils import env_dict_to_vector


# ── Filtering helpers ─────────────────────────────────────────────────────

def _filter_intensity_cls_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with valid intensity class labels (0-3, not -1)."""
    return df[df["future_inte_change24"] >= 0].copy()


def _filter_intensity_reg_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with valid delta_wnd_norm regression targets."""
    return df[df["delta_wnd_norm"].notna()].copy()


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION DATASETS  (target: int 0-3)
# ═══════════════════════════════════════════════════════════════════════════

class IntensityClsEnvSingleDataset(Dataset):
    """Single-timestep Env-Data dataset for intensity classification.

    Each sample: (92,) feature vector + int label (0-3).
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT):
        self.df = _filter_intensity_cls_valid(index_df).reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        env_path = self.data_root / row["env_path"]
        env_dict = np.load(str(env_path), allow_pickle=True).item()

        features = env_dict_to_vector(env_dict)
        target = int(row["future_inte_change24"])

        return (torch.tensor(features, dtype=torch.float32),
                torch.tensor(target, dtype=torch.long))


class IntensityClsEnvSequenceDataset(Dataset):
    """Sequence Env-Data dataset for intensity classification.

    Each sample: (seq_len, 92) features + int label at last timestep.
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                 seq_len: int = SEQ_LEN):
        self.data_root = data_root
        self.seq_len = seq_len

        valid_df = _filter_intensity_cls_valid(index_df)
        valid_df = valid_df.sort_values(
            ["storm_name", "year", "timestamp"]
        ).reset_index(drop=True)

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
                    "target": int(last_row["future_inte_change24"]),
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


class IntensityClsData3DDataset(Dataset):
    """Single-timestep Data3D dataset for intensity classification.

    Each sample: (13, 81, 81) tensor + int label (0-3).
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                 center_crop: int | None = None):
        self.data_root = data_root
        self.center_crop = center_crop

        valid = _filter_intensity_cls_valid(index_df)
        valid = valid[valid["data3d_exists"] == True]
        self.df = valid.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nc_path = self.data_root / row["data3d_path"]
        target = int(row["future_inte_change24"])

        tensor = self._load_nc(nc_path)

        if self.center_crop is not None:
            tensor = self._crop_center(tensor, self.center_crop)

        return tensor, torch.tensor(target, dtype=torch.long)

    def _load_nc(self, path: Path) -> torch.Tensor:
        """Load a NetCDF file into a (13, H, W) float32 tensor."""
        import xarray as xr

        ds = xr.open_dataset(str(path))
        u = ds["u"].values[0]
        v = ds["v"].values[0]
        z = ds["z"].values[0] / 10000.0
        sst = ds["sst"].values

        ds.close()

        sst = np.where(np.isfinite(sst) & (np.abs(sst) < 1e10), sst, np.nan)
        sst = np.nan_to_num(sst, nan=0.0)
        sst = (sst - 290.0) / 20.0

        channels = np.concatenate([u, v, z, sst[np.newaxis]], axis=0)
        return torch.tensor(channels, dtype=torch.float32)

    @staticmethod
    def _crop_center(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
        _, h, w = tensor.shape
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        return tensor[:, y0:y0 + crop_size, x0:x0 + crop_size]


# ═══════════════════════════════════════════════════════════════════════════
#  REGRESSION DATASETS  (target: float scalar delta_wnd_norm)
# ═══════════════════════════════════════════════════════════════════════════

class IntensityRegEnvSingleDataset(Dataset):
    """Single-timestep Env-Data dataset for intensity regression.

    Each sample: (92,) feature vector + (1,) float target (delta_wnd_norm).
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT):
        self.df = _filter_intensity_reg_valid(index_df).reset_index(drop=True)
        self.data_root = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        env_path = self.data_root / row["env_path"]
        env_dict = np.load(str(env_path), allow_pickle=True).item()

        features = env_dict_to_vector(env_dict)
        target = np.float32(row["delta_wnd_norm"])

        return (torch.tensor(features, dtype=torch.float32),
                torch.tensor(target, dtype=torch.float32).unsqueeze(0))


class IntensityRegData1DSequenceDataset(Dataset):
    """Sliding-window Data1D dataset for intensity regression.

    Each sample: (seq_len, 4) tensor + (1,) float target (delta_wnd_norm).
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                 seq_len: int = SEQ_LEN):
        self.data_root = data_root
        self.seq_len = seq_len

        valid_df = _filter_intensity_reg_valid(index_df)
        valid_df = valid_df.sort_values(
            ["year", "storm_name", "timestamp"]
        ).reset_index(drop=True)

        # Build timestamp -> target lookup
        self.ts_to_target = {}
        for _, row in valid_df.iterrows():
            key = (row["year"], row["storm_name"], str(row["timestamp"]))
            self.ts_to_target[key] = np.float32(row["delta_wnd_norm"])

        # Group by storm, load Data1D, build windows
        self.samples = []
        storm_groups = valid_df.groupby(
            ["year", "storm_name", "data1d_file", "split", "basin"]
        )

        for (year, storm_name, d1d_file, split, basin), group in storm_groups:
            d1d_path = DATA1D_ROOT / basin / split / d1d_file
            if not d1d_path.exists():
                continue

            track_df = pd.read_csv(
                d1d_path, delimiter='\t', header=None, names=DATA1D_COLS,
                dtype={"timestamp": str}
            )
            track_df["timestamp"] = track_df["timestamp"].astype(str).str.strip()

            features = track_df[DATA1D_FEATURE_COLS].values.astype(np.float32)
            timestamps = track_df["timestamp"].values

            for i in range(len(track_df) - seq_len + 1):
                last_ts = timestamps[i + seq_len - 1]
                key = (year, storm_name, last_ts)
                if key in self.ts_to_target:
                    self.samples.append({
                        "features": features[i:i + seq_len].copy(),
                        "target": self.ts_to_target[key],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (torch.tensor(s["features"], dtype=torch.float32),
                torch.tensor(s["target"], dtype=torch.float32).unsqueeze(0))


class IntensityRegData3DDataset(Dataset):
    """Single-timestep Data3D dataset for intensity regression.

    Each sample: (13, 81, 81) tensor + (1,) float target (delta_wnd_norm).
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                 center_crop: int | None = None):
        self.data_root = data_root
        self.center_crop = center_crop

        valid = _filter_intensity_reg_valid(index_df)
        valid = valid[valid["data3d_exists"] == True]
        self.df = valid.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nc_path = self.data_root / row["data3d_path"]
        target = np.float32(row["delta_wnd_norm"])

        tensor = self._load_nc(nc_path)

        if self.center_crop is not None:
            tensor = self._crop_center(tensor, self.center_crop)

        return tensor, torch.tensor(target, dtype=torch.float32).unsqueeze(0)

    def _load_nc(self, path: Path) -> torch.Tensor:
        """Load a NetCDF file into a (13, H, W) float32 tensor."""
        import xarray as xr

        ds = xr.open_dataset(str(path))
        u = ds["u"].values[0]
        v = ds["v"].values[0]
        z = ds["z"].values[0] / 10000.0
        sst_raw = ds["sst"].values
        sst = np.where(np.isfinite(sst_raw) & (np.abs(sst_raw) < 1e10),
                        sst_raw, np.nan)
        sst = np.nan_to_num(sst, nan=0.0)
        sst = (sst - 290.0) / 20.0
        ds.close()

        channels = np.concatenate([u, v, z, sst[np.newaxis]], axis=0)
        return torch.tensor(channels, dtype=torch.float32)

    @staticmethod
    def _crop_center(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
        _, h, w = tensor.shape
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        return tensor[:, y0:y0 + crop_size, x0:x0 + crop_size]
