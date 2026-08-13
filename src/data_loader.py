"""
data_loader.py
Handles loading and basic validation of the job roles / skills dataset.
"""

import pandas as pd
from pathlib import Path


def load_skills_data(path: str = "data/raw_skills.csv") -> pd.DataFrame:
    """
    Load the job_role -> skills dataset.

    Args:
        path: relative or absolute path to the CSV file.

    Returns:
        DataFrame with columns ['job_role', 'skills'].

    Raises:
        FileNotFoundError: if the CSV doesn't exist at the given path.
        ValueError: if required columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    df = pd.read_csv(file_path)

    required_cols = {"job_role", "skills"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Drop any rows with missing values in critical columns
    df = df.dropna(subset=["job_role", "skills"]).reset_index(drop=True)

    return df