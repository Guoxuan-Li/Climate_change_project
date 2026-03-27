"""
PyTorch Dataset for Data3D (gridded atmospheric fields) — Stage 4.

Loads NetCDF files lazily and stacks u, v, z (4 levels each) + sst into
a 13-channel (81x81) tensor.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.config import DATA_ROOT, DATA3D_CHANNELS, DATA3D_GRID_SIZE


class Data3DDataset(Dataset):
    """Single-timestep Data3D dataset.

    Each sample: (13, 81, 81) tensor + direction class label.
    Channels: u@200, u@500, u@850, u@925, v@200..925, z@200..925, sst
    """

    def __init__(self, index_df: pd.DataFrame, data_root: Path = DATA_ROOT,
                 center_crop: int | None = None):
        """
        Args:
            index_df: master index filtered to desired split
            data_root: root data directory
            center_crop: if set, crop to NxN centered patch (e.g., 41)
        """
        self.data_root = data_root
        self.center_crop = center_crop

        valid = index_df[(index_df["future_direction24"] >= 0) &
                         (index_df["data3d_exists"] == True)].copy()
        self.df = valid.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nc_path = self.data_root / row["data3d_path"]
        target = int(row["future_direction24"])

        tensor = self._load_nc(nc_path)

        if self.center_crop is not None:
            tensor = self._crop_center(tensor, self.center_crop)

        return tensor, torch.tensor(target, dtype=torch.long)

    def _load_nc(self, path: Path) -> torch.Tensor:
        """Load a NetCDF file into a (13, H, W) float32 tensor."""
        import xarray as xr

        ds = xr.open_dataset(str(path))

        # u, v, z: (1, 4, 81, 81) -> (4, 81, 81)
        u = ds["u"].values[0]  # (4, 81, 81)
        v = ds["v"].values[0]
        z = ds["z"].values[0]
        sst = ds["sst"].values  # (81, 81)

        ds.close()

        # Normalize z (geopotential) to similar scale as u/v
        z = z / 10000.0  # rough scaling to O(1)

        # Handle NaN in SST (land pixels)
        sst = np.where(np.isfinite(sst) & (np.abs(sst) < 1e10), sst, np.nan)
        sst = np.nan_to_num(sst, nan=0.0)
        # Normalize SST to O(1): typical range 270-310 K, center at 290
        sst = (sst - 290.0) / 20.0

        # Stack: u(4) + v(4) + z(4) + sst(1) = 13 channels
        channels = np.concatenate([u, v, z, sst[np.newaxis]], axis=0)  # (13, 81, 81)

        return torch.tensor(channels, dtype=torch.float32)

    @staticmethod
    def _crop_center(tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
        """Center-crop a (C, H, W) tensor."""
        _, h, w = tensor.shape
        y0 = (h - crop_size) // 2
        x0 = (w - crop_size) // 2
        return tensor[:, y0:y0 + crop_size, x0:x0 + crop_size]
