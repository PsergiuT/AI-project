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
    "feature_size": 48,             # Embedding dimension — must match pretrained weights (f48)
    "use_checkpoint": True,         # Gradient checkpointing — saves VRAM on Colab
}

# ── Training hyperparameters ───────────────────────────────────────────────────

TRAIN_CONFIG = {
    # Patch sampling
    "patch_size"       : (96, 96, 96),
    "num_samples"      : 2,           # 2 patches per volume — safe VRAM at batch_size=2
    "pos_sample_ratio" : 4,           # 4:1 foreground-to-background — AEA is ~5 voxels wide,
                                      # aggressively oversample foreground so model sees artery voxels
                                      # in nearly every patch
    "neg_sample_ratio" : 1,

    # DataLoader
    "batch_size"       : 2,           # batch 2 + num_samples 2 targets ~16-17 GB VRAM
    "num_workers"      : 6,           # PersistentDataset cache is on disk — 6 workers is safe

    # Optimiser
    "learning_rate"    : 1e-4,        # Standard starting LR for fine-tuning from pretrained weights
    "weight_decay"     : 1e-5,

    # Scheduler
    "max_epochs"       : 1000,        # More epochs — AEA is hard, needs time to converge
    "warmup_epochs"    : 20,          # Longer warmup protects pretrained encoder from large early updates

    # Early stopping
    "patience"         : 100,         # With val_every=10 this means 10 validation checks (100 epochs)
                                      # without improvement before stopping — gives model room to plateau
                                      # and recover before giving up

    # Validation frequency
    "val_every"        : 10,          # Run validation every 10 epochs

    # Loss
    "dice_weight"      : 2.0,         # Prioritise Dice loss — directly optimises the metric we care about
    "ce_weight"        : 1.0,         # CE guides class boundaries, keep at 1.0

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
    "sw_batch_size"  : 4,           # Validation only (no gradients) — 4 is safe
    "overlap"        : 0.5,          # Overlap fraction between adjacent windows
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
