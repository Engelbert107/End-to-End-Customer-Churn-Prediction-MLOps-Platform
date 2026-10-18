
"""Command-line entry point for model training."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_and_compare


if __name__ == "__main__":
    train_and_compare()
