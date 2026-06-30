"""Raw data loading with date parsing and basic existence/schema guards."""
from __future__ import annotations

import os

import pandas as pd

from src import config
from src.utils.logger import get_logger

logger = get_logger("data.load")


def load_data(file_path: str | None = None, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load the raw churn CSV into a DataFrame.

    Args:
        file_path: Path to the CSV. Defaults to ``config.RAW_DATA_PATH``.
        parse_dates: Parse the known date columns when present.

    Returns:
        The raw DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = file_path or config.RAW_DATA_PATH
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    date_cols = config.DATE_COLS if parse_dates else None
    # Only parse date columns that are actually present in the file header.
    if date_cols:
        header = pd.read_csv(file_path, nrows=0).columns
        date_cols = [c for c in date_cols if c in header]

    df = pd.read_csv(file_path, parse_dates=date_cols or None)
    logger.info("Loaded %s | %d rows x %d cols", file_path, df.shape[0], df.shape[1])
    return df
