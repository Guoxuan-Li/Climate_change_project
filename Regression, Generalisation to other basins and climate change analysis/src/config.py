"""
Central configuration for the Cyclone Trajectory Prediction project.
All paths, hyperparameters, and constants in one place.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"
DATA1D_ROOT = DATA_ROOT / "Data1D"
DATA3D_ROOT = DATA_ROOT / "Data3D"
ENV_DATA_ROOT = DATA_ROOT / "Env-Data"
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_DIR.mkdir(exist_ok=True)
MASTER_INDEX_PATH = INDEX_DIR / "master_index_WP.csv"

# ── Dataset constants ──────────────────────────────────────────────────────
BASIN = "WP"
BASINS = ["EP", "NA", "NI", "SI", "SP", "WP"]  # order matches area one-hot

SEQ_LEN = 8          # 48 h history  (8 x 6 h steps)
PRED_HORIZON = 4     # 24 h future   (4 x 6 h steps)

NUM_DIRECTION_CLASSES = 8
NUM_INTENSITY_CLASSES = 4

DIRECTION_LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
INTENSITY_LABELS = ["Strengthening", "Str-then-Weak", "Weakening", "Maintaining"]

# ── Data1D column indices (8-column TSV, no header) ────────────────────────
# Col 0: ID (timestep), Col 1: unknown flag (always 1.0), Col 2: LONG_norm,
# Col 3: LAT_norm, Col 4: PRES_norm, Col 5: WND_norm, Col 6: YYYYMMDDHH,
# Col 7: Name
DATA1D_COLS = ["id", "flag", "long_norm", "lat_norm", "pres_norm", "wnd_norm",
               "timestamp", "name"]
DATA1D_FEATURE_COLS = ["long_norm", "lat_norm", "pres_norm", "wnd_norm"]
DATA1D_NUM_FEATURES = len(DATA1D_FEATURE_COLS)  # 4

# ── Normalization constants (from original paper / env_data.py) ────────────
# Data1D: val_norm = (val_raw - OFFSET) / SCALE
NORM_LONG = {"offset": 1800, "scale": 50}   # raw in 0.1 deg E
NORM_LAT  = {"offset": 0,    "scale": 50}   # raw in 0.1 deg N
NORM_PRES = {"offset": 960,  "scale": 50}   # raw in hPa
NORM_WND  = {"offset": 40,   "scale": 25}   # raw in m/s

# Env-Data normalization
WIND_NORM_FACTOR = 110.0        # wind / 110
VELOCITY_NORM_FACTOR = 1219.8387650082498  # km, velocity / this

# ── Env-Data feature dimensions ────────────────────────────────────────────
ENV_FEATURE_DIM = 92  # 6+1+6+1+12+36+12+8+8+2 (incl. 2 missing-history flags)

# ── Training hyperparameters ───────────────────────────────────────────────
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 50
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
PATIENCE = 7         # early stopping patience (epochs)
SEED = 42
DEVICE = "cuda"      # will fall back to cpu in trainer if unavailable

# ── Model dimensions ───────────────────────────────────────────────────────
HIDDEN_DIM = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.2
MLP_DROPOUT = 0.3
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2

# ── Data3D ─────────────────────────────────────────────────────────────────
DATA3D_CHANNELS = 13       # u(4) + v(4) + z(4) + sst(1)
DATA3D_GRID_SIZE = 81      # 81 x 81
DATA3D_PRESSURE_LEVELS = [200, 500, 850, 925]
