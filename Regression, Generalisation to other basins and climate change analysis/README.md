# Tropical Cyclone Trajectory & Intensity Prediction

**Course**: ELEC70127 — Machine Learning for Tackling Climate Change, Imperial College London
**Team**: Guglielmo Cimolai, Daniel Budina, Guoxuan Li
**Dataset**: [TropiCycloneNet](https://github.com/xiaochengfuhuo/TropiCycloneNet-Dataset) (Huang et al., Nature Communications, 2025)

A multimodal deep learning pipeline for predicting tropical cyclone trajectories and intensity using 70 years of atmospheric reanalysis data across 6 ocean basins. Includes progressive model ablation (MLP → LSTM → Transformer → CNN → Fusion), a novel regression formulation, cross-basin and temporal generalization experiments, and climate change intensity analysis.

## Key Results

| Task | Best Model | Metric | Value |
|------|-----------|--------|-------|
| **Track prediction** | Regression CNN (Data3D) | Mean 24h error | **138.3 km** |
| | | R² (longitude) | 0.917 |
| | | Derived direction accuracy | 71.7% |
| **Intensity prediction** | Regression LSTM (Data1D) | MAE | **4.67 m/s** |
| **Cross-basin transfer** | Reg CNN (WP → EP) | Mean 24h error | **136.8 km** |
| **Climate change** | Statistical analysis | Cat 4+ fraction increase | **17% → 26% (p=0.034)** |

## Setup

### Requirements

```bash
pip install torch torchvision numpy pandas xarray netCDF4 matplotlib cartopy scikit-learn seaborn scipy tqdm
```

### Dataset

Download the TropiCycloneNet dataset (~25.7 GB full, or 3.34 GB test subset) and extract to `Data/`:

```
Data/
    Data1D/     # Tabular track records (TSV)
    Data3D/     # Gridded atmospheric fields (NetCDF)
    Env-Data/   # Pre-computed environmental features (numpy)
```

See the [TropiCycloneNet repository](https://github.com/xiaochengfuhuo/TropiCycloneNet-Dataset) for download instructions.

## Reproducing All Results

### Step 1: Build master indexes

```bash
python -m src.data.build_index --all-basins
```

Scans the dataset and produces `index/master_index_{basin}.csv` for all 6 basins (WP, EP, NA, NI, SI, SP). These CSVs contain paths, labels, and regression targets for every timestep.

### Step 2: Train track prediction models

```bash
# Classification + regression, all 5 stages (MLP, LSTM, Transformer, CNN, Fusion)
# Stages 1-3: ~30 min | Stages 4-5: ~4 hours each
python -m src.scripts.run_all --epochs 50
```

Outputs to `results/<timestamp>/`: model checkpoints, comparison tables, confusion matrices, training curves, and trajectory visualizations.

To skip the heavy Data3D stages:
```bash
python -m src.scripts.run_all --stages 1 2 3 --skip-stage4 --skip-stage5 --epochs 50
```

To run regression only with all 5 stages:
```bash
python -m src.scripts.run_all --regression-only --reg-stages 1 2 3 4 5 --epochs 50
```

### Step 3: Train intensity prediction models

```bash
python -m src.scripts.run_intensity --epochs 50
```

Trains 4 classification + 4 regression intensity models. Outputs to `results/intensity_<timestamp>/`.

### Step 4: Generalization experiments

```bash
# Cross-basin: evaluate WP-trained models on all 6 basins (instant, uses existing checkpoints)
python -m src.scripts.run_generalization --basin-only

# Temporal: retrain on 1950-1990/2000/2010/2016, test on 2017-2021 (~6-8 hours)
python -m src.scripts.run_generalization --temporal-only --epochs 50

# Intensity temporal generalization
python -m src.scripts.run_intensity_temporal --epochs 50
```

### Step 5: Generate intensity visualizations

```bash
python -m src.scripts.generate_intensity_viz --intensity-dir results/intensity_<timestamp>
```

Produces per-storm intensity lifecycle plots with model predictions, animated GIFs, and decadal statistics.

### Step 6: Comprehensive post-training analysis

```bash
python -m src.scripts.run_analysis \
    --results-dir results/<timestamp> \
    --intensity-dir results/intensity_<timestamp> \
    --output-dir results/analysis \
    --num-storms 20
```

Produces:
- Track predictions for 20 test storms (all models + best-2 regression)
- Regression residual analysis (bias detection, error distributions)
- Actual vs predicted scatter plots (longitude and latitude components)
- Cross-basin decadal intensity trends (Cat 4+ and RI for all 6 basins)
- Climate change statistical analysis (p-values, trend tests, 3-period comparisons)
- Per-storm summary table

## Project Structure

```
src/
    config.py                          # Paths, hyperparameters, normalization constants
    data/
        build_index.py                 # Dataset scanner → master CSV index
        data1d_dataset.py              # Track sequence dataset (sliding windows)
        data3d_dataset.py              # Atmospheric grid dataset (13-channel NetCDF)
        env_dataset.py                 # Environmental feature dataset (92-dim)
        multimodal_dataset.py          # Combined 3-pillar dataset
        regression_dataset.py          # Regression variants (continuous displacement targets)
        intensity_dataset.py           # Intensity prediction datasets
        utils.py                       # Feature extraction, normalization helpers
    models/
        baseline_mlp.py                # Stage 1: MLP on Env-Data (57K params)
        lstm_1d.py                     # Stage 2: LSTM on track coordinates (202K params)
        env_temporal.py                # Stage 3: Transformer on Env-Data sequences (410K params)
        cnn_3d.py                      # Stage 4: ResNet-18 on atmospheric grids (11.3M params)
        fusion_model.py                # Stage 5: Late fusion (11.9M params)
        regression_models.py           # Regression variants (output continuous displacement)
        intensity_models.py            # Intensity classification and regression models
        dual_head.py                   # Joint track + intensity (architecture only)
    training/
        trainer.py                     # Training loop with early stopping (classification + regression)
        losses.py                      # Focal Loss, Haversine Loss
        evaluate.py                    # Classification, regression, and intensity metrics
    visualization/
        trajectory_plots.py            # Geographic track plots, error plots, multi-storm grids
        intensity_plots.py             # Intensity lifecycle plots, animations, decadal statistics
    scripts/
        train.py                       # Single-stage training CLI
        run_all.py                     # Full track prediction pipeline
        run_intensity.py               # Intensity prediction pipeline
        run_generalization.py          # Cross-basin + temporal track experiments
        run_intensity_temporal.py      # Intensity temporal generalization
        generate_intensity_viz.py      # Intensity visualizations from checkpoints
        run_analysis.py                # Comprehensive post-training analysis
```

## Dataset Details

### Three Data Modalities

| Modality | Format | Resolution | Content |
|----------|--------|------------|---------|
| **Data1D** | TSV | 6-hourly | Longitude, latitude, pressure, wind speed (normalized) |
| **Data3D** | NetCDF | 0.25° / 6-hourly | U/V wind, geopotential height at 200/500/850/925 hPa, SST |
| **Env-Data** | NumPy | 6-hourly | Basin, intensity class, velocity, month, location, directional history |

### Data1D Normalization

| Field | Formula | Physical units |
|-------|---------|---------------|
| Longitude | `lon_deg = (long_norm * 50 + 1800) / 10` | degrees E |
| Latitude | `lat_deg = lat_norm * 50 / 10` | degrees N |
| Pressure | `pres_hPa = pres_norm * 50 + 960` | hPa |
| Wind | `wind_ms = wnd_norm * 25 + 40` | m/s |

### Train/Val/Test Split

| Split | Period | WP Samples |
|-------|--------|------------|
| Train | 1950–2016 (80%) | 37,894 |
| Val | 1950–2016 (20%) | 10,062 |
| Test | 2017–2021 | 3,974 |

## Key Findings

1. **Regression outperforms classification** by +7–26% across all architectures. Continuous displacement prediction preserves directional information lost by 8-class binning.

2. **Atmospheric spatial fields (Data3D) halve track error**: 281 km (persistence) → 138 km (CNN). The 500 hPa geopotential height encodes the subtropical high steering flow.

3. **Fusion doesn't help**: the CNN alone (138 km) beats 3-branch fusion (153 km). The spatial signal already contains the essential information.

4. **CNN generalizes across basins**: zero-shot transfer from WP to EP achieves 137 km (identical to in-domain). LSTM fails because normalized coordinates are basin-specific.

5. **Track prediction is temporally robust**: training on 1950–1990 gives 150 km — only 6% worse than full data. Steering physics hasn't shifted detectably.

6. **Intensity shows a climate change signal**: Cat 4+ storms increased from 17% to 26% (p=0.034). RI events tripled. Consistent with warming SSTs shifting the intensity distribution rightward.

## Known Issues

- **SST fill values**: ERA5 uses `9.97e+36` instead of NaN for land pixels in SST. All data loaders mask `abs(sst) > 1e10`. Without this fix, CNN training produces NaN loss.
- **Data1D column count**: Files have 8 tab-separated columns, but the official `read_TCND.py` uses 7 column names. Column 1 (always 1.0) is a flag that can be safely ignored.
- **Southern Hemisphere transfer**: WP-trained CNN fails on SI/SP basins (350–388 km) due to mirrored cyclone dynamics. Hemisphere-aware preprocessing would be needed.

## References

- Huang, C. et al. "Benchmark dataset and deep learning method for global tropical cyclone forecasting." *Nature Communications* 16, 5923 (2025).
- TropiCycloneNet Dataset: [github.com/xiaochengfuhuo/TropiCycloneNet-Dataset](https://github.com/xiaochengfuhuo/TropiCycloneNet-Dataset)

## License

This project was developed for academic purposes as part of ELEC70127 at Imperial College London.
