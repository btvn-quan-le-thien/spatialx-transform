"""Logging configuration for spatialx_transform."""

import logging
from pathlib import Path


def setup_logging(
    name: str = "spatialx_transform",
    level: int = logging.DEBUG,
    log_dir: str = "logging",
) -> logging.Logger:
    """Configure file-only logging. Call once before using warp_transform.

    Creates a log file at <log_dir>/warp.log relative to the current
    working directory.
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(log_path / "warp.log")
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(fh)
    return logger
