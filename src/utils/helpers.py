"""
Utility helpers: logging, seeding, device selection, metrics formatting.
"""
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ─── Add project root to sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import LOG_LEVEL, LOG_FORMAT, LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(ch)
        # File handler
        fh = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(fh)
    return logger


def seed_everything(seed: int = 42) -> None:
    """Reproducibility helper."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def format_number(n: float) -> str:
    """Format large numbers for display (1.2M, 45.3K, etc.)."""
    if abs(n) >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:.0f}"


def virality_label(views: float, thresholds: dict) -> str:
    """Assign virality label based on view count."""
    for label, (lo, hi) in thresholds.items():
        if lo <= views < hi:
            return label
    return list(thresholds.keys())[-1]
