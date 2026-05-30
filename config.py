"""
config.py — Central configuration for the AEA Segmentation project.

All paths, hyperparameters, and constants live here.
Change DATA_ROOT to match your local dataset location before running.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

# Root of the raw dataset (the folder containing CROP1 … CROP7)
DATA_ROOT = Path("../dateArteraEtimoidala")

PROJECT_ROOT   = Path(__file__).parent.resolve()
DATA_DIR       = PROJECT_ROOT / "data"
RAW_DIR        = DATA_DIR / "raw"          # symlink or copy of DATA_ROOT
PROCESSED_DIR  = DATA_DIR / "processed"   # NIfTI images + masks after preprocessing
SPLITS_DIR     = DATA_DIR / "splits"      # JSON files for train/val/test splits
MODELS_DIR     = PROJECT_ROOT / "models"
SWINUNETR_DIR  = MODELS_DIR / "swinunetr"
LOGS_DIR       = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for _dir in [PROCESSED_DIR, SPLITS_DIR, SWINUNETR_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Dataset constants ──────────────────────────────────────────────────────────

# Folder names mapping to case ranges (matches the zip archive naming)
CROP_FOLDERS = ["CROP1", "CROP2", "CROP3", "CROP4", "CROP5", "CROP6", "CROP7"]

# Label values in the NRRD segmentation masks
LABEL_BACKGROUND = 0
LABEL_AEA_LEFT   = 1
LABEL_AEA_RIGHT  = 2
NUM_CLASSES      = 3
CLASS_NAMES      = ["Background", "AEA Left", "AEA Right"]

# Target voxel spacing after resampling (mm) — already isotropic in this dataset
TARGET_SPACING = (0.4, 0.4, 0.4)

# ── Data split ─────────────────────────────────────────────────────────────────

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.80   # ~104 cases
VAL_RATIO    = 0.10   # ~13 cases
TEST_RATIO   = 0.10   # ~13 cases

# ── Model architecture ─────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "img_size"    : (96, 96, 96),   # 3D patch size fed to SwinUNETR
    "in_channels" : 1,              # Single-channel CBCT (grayscale)
    "out_channels": NUM_CLASSES,    # 3-class output
    "feature_size": 48,             # Embedding dimension (base config)
    "use_checkpoint": True,         # Gradient checkpointing — saves VRAM on Colab
}

# ── Training hyperparameters ───────────────────────────────────────────────────

TRAIN_CONFIG = {
    # Patch sampling
    "patch_size"       : (96, 96, 96),
    "num_samples"      : 2,           # 2 patches per volume — safe VRAM usage at batch_size=2
    "pos_sample_ratio" : 2,           # 2:1 foreground-to-background — AEA is tiny, force more artery patches
    "neg_sample_ratio" : 1,

    # DataLoader
    "batch_size"       : 2,           # batch 2 + num_samples 2 targets ~16-17 GB VRAM
    "num_workers"      : 6,           # PersistentDataset cache is on disk — 6 workers is safe and fast

    # Optimiser
    "learning_rate"    : 1e-4,
    "weight_decay"     : 1e-5,

    # Scheduler
    "max_epochs"       : 800,         # More epochs to compensate for fewer patches per epoch
    "warmup_epochs"    : 10,          # Linear warmup before cosine annealing

    # Early stopping
    "patience"         : 50,         # Stop if val Dice doesn't improve for 50 epochs

    # Validation frequency
    "val_every"        : 10,          # Increased 5→10: validation is expensive, run less often

    # Loss
    "dice_weight"      : 1.0,        # Weight for Dice component of DiceCELoss
    "ce_weight"        : 1.0,        # Weight for Cross-Entropy component

    # Checkpointing
    "checkpoint_dir"   : str(SWINUNETR_DIR),
    "best_model_name"  : "swinunetr_best.pth",
    "last_model_name"  : "swinunetr_last.pth",

    # Fine-tuning from a trained checkpoint
    # Lower LR avoids disrupting already-learned features while still
    # allowing the network to adapt to the improved loss and transforms.
    "finetune_lr"      : 1e-5,
}

# ── Inference ──────────────────────────────────────────────────────────────────

INFERENCE_CONFIG = {
    "sw_batch_size"  : 4,           # Validation only (no gradients) — 4 is safe at this VRAM level
    "overlap"        : 0.5,         # Overlap fraction between adjacent windows
    "mode"           : "gaussian",  # Blending mode for overlapping predictions
    "roi_size"       : (96, 96, 96),
}

# ── Intensity normalisation ────────────────────────────────────────────────────

# CBCT Hounsfield Unit window for bone/soft tissue in sinuses
HU_MIN  = -1000   # Air
HU_MAX  =  3000   # Dense bone

# ── Agent ──────────────────────────────────────────────────────────────────────

AGENT_CONFIG = {
    "model_name"    : "llama3.1",   # Ollama model tag
    "base_url"      : "http://localhost:11434",
    "temperature"   : 0.0,          # Deterministic tool calling
    "max_iterations": 10,           # Max ReAct loop steps
}

# ── Evaluation ─────────────────────────────────────────────────────────────────

EVAL_CONFIG = {
    "include_background": False,    # Exclude background from Dice/IoU computation
    "percentile"        : 95,       # HD95 percentile
    "metric_names"      : ["dice_aeal", "dice_aear", "dice_mean",
                           "iou_aeal",  "iou_aear",  "iou_mean",
                           "hd95_aeal", "hd95_aear", "hd95_mean"],
}
