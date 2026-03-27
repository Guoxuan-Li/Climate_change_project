"""
Build a master index CSV for the TropiCycloneNet dataset.

Scans the Env-Data directory tree for a given basin, loads each .npy sample,
extracts labels and metadata, determines train/val/test split, and constructs
paths to all three data pillars. The output CSV becomes the single source of
truth for all data loading downstream.

Usage:
    python -m src.data.build_index              # defaults to WP
    python -m src.data.build_index --basin EP
    python -m src.data.build_index --all-basins
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    DATA_ROOT, DATA1D_ROOT, DATA3D_ROOT, ENV_DATA_ROOT,
    INDEX_DIR, BASIN,
)


def _build_split_lookup(basin: str) -> dict[str, str]:
    """Map storm file stems to their split (train/val/test).

    Data1D files are named like WP2018BSTMANGKHUT.txt.
    We extract the storm name and year, and map to the split.
    Returns: { (year, STORM_NAME_UPPER): split }
    """
    lookup = {}
    for split in ["train", "val", "test"]:
        split_dir = DATA1D_ROOT / basin / split
        if not split_dir.exists():
            continue
        for f in split_dir.iterdir():
            if not f.name.endswith(".txt"):
                continue
            # Format: {basin}{year}BST{NAME}.txt
            stem = f.stem  # e.g. WP2018BSTMANGKHUT
            bst_idx = stem.find("BST")
            if bst_idx == -1:
                continue
            year = stem[len(basin):bst_idx]
            name = stem[bst_idx + 3:]
            lookup[(year, name.upper())] = split
    return lookup


def _data1d_filename(basin: str, year: str, storm_name: str) -> str:
    """Construct the Data1D filename for a storm."""
    return f"{basin}{year}BST{storm_name.upper()}.txt"


def _data3d_path(basin: str, year: str, storm_name: str, timestamp: str) -> str:
    """Construct relative path to the Data3D NetCDF file."""
    return str(Path("Data3D") / basin / year / storm_name /
               f"TCND_{storm_name}_{timestamp}_sst_z_u_v.nc")


def _load_data1d_lookup(basin: str, year: str, storm_name: str,
                        split: str) -> dict[str, dict]:
    """Load a Data1D file and return a timestamp -> row dict.

    Each value is a dict with keys: long_norm, lat_norm, pres_norm, wnd_norm
    and an 'idx' key for the row position (used for look-ahead).
    """
    from src.config import DATA1D_COLS, DATA1D_FEATURE_COLS

    data1d_file = _data1d_filename(basin, year, storm_name)
    d1d_path = DATA1D_ROOT / basin / split / data1d_file
    if not d1d_path.exists():
        return {}

    track_df = pd.read_csv(
        d1d_path, delimiter='\t', header=None,
        names=DATA1D_COLS, dtype={"timestamp": str}
    )
    track_df["timestamp"] = track_df["timestamp"].astype(str).str.strip()

    lookup = {}
    for i, row in track_df.iterrows():
        ts = row["timestamp"]
        lookup[ts] = {
            "long_norm": float(row["long_norm"]),
            "lat_norm": float(row["lat_norm"]),
            "idx": int(i),
        }
    # Also store the full arrays for look-ahead
    lookup["_long_norm_arr"] = track_df["long_norm"].values.astype(float)
    lookup["_lat_norm_arr"] = track_df["lat_norm"].values.astype(float)
    lookup["_wnd_norm_arr"] = track_df["wnd_norm"].values.astype(float)
    lookup["_n_rows"] = len(track_df)
    return lookup


def build_index_for_basin(basin: str) -> pd.DataFrame:
    """Scan Env-Data for a basin and build the index DataFrame."""
    env_basin_dir = ENV_DATA_ROOT / basin
    if not env_basin_dir.exists():
        print(f"WARNING: {env_basin_dir} does not exist, skipping.")
        return pd.DataFrame()

    split_lookup = _build_split_lookup(basin)

    rows = []
    years = sorted([d.name for d in env_basin_dir.iterdir() if d.is_dir()])

    # Cache for Data1D lookups: (year, storm_name) -> lookup dict
    data1d_cache = {}

    for year in tqdm(years, desc=f"Scanning {basin}"):
        year_dir = env_basin_dir / year
        storms = sorted([d.name for d in year_dir.iterdir() if d.is_dir()])
        for storm_name in storms:
            storm_dir = year_dir / storm_name
            npy_files = sorted(storm_dir.glob("*.npy"))

            # Determine split
            split = split_lookup.get((year, storm_name.upper()), None)
            if split is None:
                # Try matching with original case
                split = split_lookup.get((year, storm_name), None)
            if split is None:
                # Storm not in Data1D — skip (or mark as unknown)
                split = "unknown"

            data1d_file = _data1d_filename(basin, year, storm_name)

            # Load Data1D lookup for this storm (once per storm)
            cache_key = (year, storm_name, split)
            if cache_key not in data1d_cache:
                if split != "unknown":
                    data1d_cache[cache_key] = _load_data1d_lookup(
                        basin, year, storm_name, split)
                else:
                    data1d_cache[cache_key] = {}
            d1d_lookup = data1d_cache[cache_key]

            for npy_path in npy_files:
                timestamp = npy_path.stem  # e.g. 2018091312
                try:
                    data = np.load(str(npy_path), allow_pickle=True).item()
                except Exception as e:
                    print(f"  ERROR loading {npy_path}: {e}")
                    continue

                future_dir24 = data.get("future_direction24", -1)
                future_int24 = data.get("future_inte_change24", -1)

                # Convert numpy types to Python int
                if hasattr(future_dir24, "item"):
                    future_dir24 = int(future_dir24.item())
                else:
                    future_dir24 = int(future_dir24)
                if hasattr(future_int24, "item"):
                    future_int24 = int(future_int24.item())
                else:
                    future_int24 = int(future_int24)

                # Check if history is available
                hist_dir24 = data.get("history_direction24", -1)
                has_history24 = not (isinstance(hist_dir24, (int, float)) and hist_dir24 == -1)

                hist_dir12 = data.get("history_direction12", -1)
                has_history12 = not (isinstance(hist_dir12, (int, float)) and hist_dir12 == -1)

                # Construct Data3D path (relative to DATA_ROOT)
                d3d_rel = _data3d_path(basin, year, storm_name, timestamp)
                d3d_exists = (DATA_ROOT / d3d_rel).exists()

                # Compute regression targets: delta_lon_norm, delta_lat_norm, delta_wnd_norm
                delta_lon_norm = np.nan
                delta_lat_norm = np.nan
                delta_wnd_norm = np.nan
                ts_str = str(timestamp).strip()
                if d1d_lookup and ts_str in d1d_lookup:
                    ts_info = d1d_lookup[ts_str]
                    cur_idx = ts_info["idx"]
                    future_idx = cur_idx + 4  # 4 timesteps = 24h ahead
                    n_rows = d1d_lookup["_n_rows"]
                    if future_idx < n_rows:
                        lon_arr = d1d_lookup["_long_norm_arr"]
                        lat_arr = d1d_lookup["_lat_norm_arr"]
                        wnd_arr = d1d_lookup["_wnd_norm_arr"]
                        delta_lon_norm = float(lon_arr[future_idx] - lon_arr[cur_idx])
                        delta_lat_norm = float(lat_arr[future_idx] - lat_arr[cur_idx])
                        delta_wnd_norm = float(wnd_arr[future_idx] - wnd_arr[cur_idx])

                rows.append({
                    "basin": basin,
                    "year": int(year),
                    "storm_name": storm_name,
                    "timestamp": timestamp,
                    "data1d_file": data1d_file,
                    "env_path": str(npy_path.relative_to(DATA_ROOT)),
                    "data3d_path": d3d_rel if d3d_exists else "",
                    "data3d_exists": d3d_exists,
                    "split": split,
                    "future_direction24": future_dir24,
                    "future_inte_change24": future_int24,
                    "has_history12": has_history12,
                    "has_history24": has_history24,
                    "delta_lon_norm": delta_lon_norm,
                    "delta_lat_norm": delta_lat_norm,
                    "delta_wnd_norm": delta_wnd_norm,
                })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build master index CSV")
    parser.add_argument("--basin", type=str, default=BASIN,
                        help="Basin to index (default: WP)")
    parser.add_argument("--all-basins", action="store_true",
                        help="Index all 6 basins")
    parser.add_argument("--output-dir", type=str, default=str(INDEX_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    basins = ["EP", "NA", "NI", "SI", "SP", "WP"] if args.all_basins else [args.basin]

    for basin in basins:
        print(f"\n{'='*60}")
        print(f"Building index for basin: {basin}")
        print(f"{'='*60}")

        df = build_index_for_basin(basin)
        if df.empty:
            print(f"  No data found for {basin}.")
            continue

        out_path = output_dir / f"master_index_{basin}.csv"
        df.to_csv(out_path, index=False)

        # Statistics
        total = len(df)
        valid = (df["future_direction24"] >= 0).sum()
        invalid = total - valid
        print(f"\n  Total samples:   {total:,}")
        print(f"  Valid (label>=0): {valid:,}")
        print(f"  Invalid (label=-1): {invalid:,}")
        print(f"  Data3D available: {df['data3d_exists'].sum():,} / {total:,}")
        print(f"\n  Split distribution:")
        print(f"  {df['split'].value_counts().to_dict()}")

        if valid > 0:
            valid_df = df[df["future_direction24"] >= 0]
            print(f"\n  Direction class distribution (valid samples):")
            dist = valid_df["future_direction24"].value_counts().sort_index()
            labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
            for cls_id, count in dist.items():
                pct = count / valid * 100
                print(f"    {cls_id} ({labels[cls_id]:>2s}): {count:>6,} ({pct:5.1f}%)")

        # Regression target statistics
        reg_valid = df["delta_lon_norm"].notna().sum()
        print(f"\n  Regression targets (delta_lon/lat_norm):")
        print(f"    Valid: {reg_valid:,} / {total:,}")
        if reg_valid > 0:
            reg_df = df[df["delta_lon_norm"].notna()]
            print(f"    delta_lon_norm: mean={reg_df['delta_lon_norm'].mean():.4f}, "
                  f"std={reg_df['delta_lon_norm'].std():.4f}")
            print(f"    delta_lat_norm: mean={reg_df['delta_lat_norm'].mean():.4f}, "
                  f"std={reg_df['delta_lat_norm'].std():.4f}")
            wnd_valid = df["delta_wnd_norm"].notna().sum()
            if wnd_valid > 0:
                wnd_df = df[df["delta_wnd_norm"].notna()]
                print(f"    delta_wnd_norm: mean={wnd_df['delta_wnd_norm'].mean():.4f}, "
                      f"std={wnd_df['delta_wnd_norm'].std():.4f}  "
                      f"(valid: {wnd_valid:,})")

        print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
