"""
run.py — Entry point for the AEA Segmentation web application.

Usage:
    python run.py                  # Start UI on localhost:7860
    python run.py --port 8080      # Custom port
    python run.py --share          # Create a public Gradio link (useful for demos)

Before running:
    1. Install dependencies:       pip install -r requirements.txt
    2. Install Ollama:             https://ollama.com
    3. Pull the LLM model:         ollama pull llama3.1
    4. Start Ollama server:        ollama serve   (in a separate terminal)
    5. Place trained model at:     models/swinunetr/swinunetr_best.pth
    6. Run this script:            python run.py
"""

import sys
import argparse
from pathlib import Path

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from config import LOGS_DIR
from src.utils import setup_logger


def check_prerequisites() -> bool:
    """
    Verify that all prerequisites are met before launching the UI.
    Returns True if all checks pass, False otherwise.
    """
    all_ok = True

    # 1. Check trained model exists
    from config import SWINUNETR_DIR, TRAIN_CONFIG
    model_path = SWINUNETR_DIR / TRAIN_CONFIG["best_model_name"]
    if not model_path.exists():
        logger.error(
            f"Trained model not found: {model_path}\n"
            "  → Train the model first using notebooks/training.ipynb on Google Colab\n"
            "  → Then copy models/swinunetr/swinunetr_best.pth to this directory"
        )
        all_ok = False
    else:
        logger.info(f"✓ Trained model found: {model_path}")

    # 2. Check Ollama is reachable
    import urllib.request
    from config import AGENT_CONFIG
    try:
        with urllib.request.urlopen(
            f"{AGENT_CONFIG['base_url']}/api/tags", timeout=3
        ) as resp:
            import json
            data   = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            tag    = AGENT_CONFIG["model_name"]
            if any(tag in n for n in models):
                logger.info(f"✓ Ollama running — model '{tag}' available")
            else:
                logger.warning(
                    f"⚠ Ollama running but model '{tag}' not found.\n"
                    f"  Available models: {models}\n"
                    f"  → Run: ollama pull {tag}"
                )
    except Exception:
        logger.error(
            f"✗ Cannot reach Ollama at {AGENT_CONFIG['base_url']}\n"
            "  → Install Ollama: https://ollama.com\n"
            "  → Start the server: ollama serve\n"
            "  → Pull the model:  ollama pull llama3.1"
        )
        all_ok = False

    # 3. Check data splits exist (warn only — UI can still run for inference)
    from config import SPLITS_DIR
    if not (SPLITS_DIR / "train.json").exists():
        logger.warning(
            "⚠ Data splits not found. Run preprocessing before training:\n"
            "  python src/preprocessing.py --data_root <path_to_dataset>"
        )

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="AEA Segmentation — Web UI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the Gradio server on (default: 7860)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public Gradio share URL (useful for remote demos)",
    )
    parser.add_argument(
        "--skip_checks", action="store_true",
        help="Skip prerequisite checks and launch directly",
    )
    args = parser.parse_args()

    setup_logger(LOGS_DIR)
    logger.info("=" * 55)
    logger.info("  AEA Segmentation — Preoperative Planning Tool")
    logger.info("=" * 55)

    if not args.skip_checks:
        logger.info("Checking prerequisites...")
        if not check_prerequisites():
            logger.warning(
                "Some prerequisites are not met. "
                "The UI may not work correctly. "
                "Use --skip_checks to launch anyway."
            )

    from ui.app import launch
    launch(share=args.share, port=args.port)


if __name__ == "__main__":
    main()
