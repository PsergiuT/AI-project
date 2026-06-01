import json
import random
import numpy as np
from pathlib import Path
from loguru import logger


# ── Reproducibility ───

def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info(f"Random seed set to {seed}")


def save_json(data: dict | list, path: Path) -> None:
    """Save a Python dict or list as a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON → {path}")


def load_json(path: Path) -> dict | list:
    """Load a JSON file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def build_manifest(processed_dir: Path) -> list[dict]:
    """
    Scan the processed data directory and build a list of {image, mask} pairs.

    Expected structure after preprocessing:
        processed/
            case_001/
                image.nii.gz
                mask.nii.gz
            case_002/
                ...

    Returns:
        List of dicts: [{"image": str, "mask": str, "case_id": str}, ...]
    """
    manifest = []
    processed_dir = Path(processed_dir)

    for case_dir in sorted(processed_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        image_path = case_dir / "image.nii.gz"
        mask_path  = case_dir / "mask.nii.gz"
        if image_path.exists() and mask_path.exists():
            manifest.append({
                "case_id": case_dir.name,
                "image"  : str(image_path),
                "mask"   : str(mask_path),
            })
        else:
            logger.warning(f"Incomplete case skipped: {case_dir.name}")

    logger.info(f"Manifest built: {len(manifest)} cases found in {processed_dir}")
    return manifest


def split_manifest(
    manifest    : list[dict],
    train_ratio : float = 0.80,
    val_ratio   : float = 0.10,
    seed        : int   = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split the manifest into train, validation, and test sets.

    Uses a deterministic shuffle so the split is reproducible
    across machines as long as the seed is the same.

    Args:
        manifest:    Full list of case dicts.
        train_ratio: Fraction for training (default 0.80 → ~104 cases).
        val_ratio:   Fraction for validation (default 0.10 → ~13 cases).
        seed:        Random seed for the shuffle.

    Returns:
        (train_manifest, val_manifest, test_manifest)
    """
    rng = random.Random(seed)
    shuffled = manifest.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)

    train = shuffled[:n_train]
    val   = shuffled[n_train : n_train + n_val]
    test  = shuffled[n_train + n_val :]

    logger.info(
        f"Split: {len(train)} train / {len(val)} val / {len(test)} test "
        f"(seed={seed})"
    )
    return train, val, test


def setup_logger(log_dir: Path, log_level: str = "INFO") -> None:
    """Configure loguru to write to both console and a log file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training.log"

    logger.remove()  # Remove default handler
    logger.add(
        sink   = log_file,
        level  = log_level,
        format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
        rotation = "10 MB",
    )
    logger.add(
        sink   = lambda msg: print(msg, end=""),
        level  = log_level,
        colorize = True,
        format = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    logger.info(f"Logger initialised — writing to {log_file}")
