from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_MAX_ROWS = 100
DEFAULT_RANDOM_STATE = 42
DATA_PATHS: dict[str, Path] = {
    "mgnrega_mp": Path("data/mgnrega_mp.csv"),
    "mgnrega_cg": Path("data/mgnrega_cg.csv"),
    "pmgsy": Path("data/pmgsy.csv"),
    "sanitation": Path("data/sanitation.csv"),
    "nfhs": Path("data/nfhs_5_factsheets_data.xls"),
}


def resolve_data_path(path: str | Path) -> Path:
    """Resolve a relative project path to an absolute file path."""
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def ensure_file_exists(path: str | Path) -> Path:
    """Validate that a dataset file exists before attempting to load it."""
    resolved_path = resolve_data_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")
    return resolved_path


def sample_dataframe(
    df: pd.DataFrame,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Sample a dataframe with optional group balancing while preserving at most max_rows."""
    if df.empty or len(df) <= max_rows:
        return df.reset_index(drop=True)
    if not group_column or group_column not in df.columns:
        return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    grouped_samples: list[pd.DataFrame] = []
    groups = list(df.groupby(group_column, dropna=False))
    base_quota = max(1, max_rows // len(groups))

    for _, group in groups:
        grouped_samples.append(group.sample(n=min(len(group), base_quota), random_state=random_state))

    sampled = pd.concat(grouped_samples).drop_duplicates()
    remaining = max_rows - len(sampled)
    if remaining > 0:
        leftover = df.drop(sampled.index, errors="ignore")
        if not leftover.empty:
            sampled = pd.concat(
                [sampled, leftover.sample(n=min(remaining, len(leftover)), random_state=random_state)]
            )

    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.reset_index(drop=True)


def load_tabular_dataset(
    path: str | Path,
    reader: Callable[..., pd.DataFrame],
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Load and sample a tabular dataset from CSV or Excel."""
    resolved_path = ensure_file_exists(path)
    LOGGER.info("Loading dataset from %s", resolved_path)
    df = reader(resolved_path)
    return sample_dataframe(df, max_rows=max_rows, random_state=random_state, group_column=group_column)


def load_mgnrega(
    path: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Load and sample an MGNREGA dataset at district level."""
    return load_tabular_dataset(
        path,
        pd.read_csv,
        max_rows=max_rows,
        random_state=random_state,
        group_column="district_name",
    )


def load_pmgsy(
    path: str | Path = DATA_PATHS["pmgsy"],
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Load and sample the PMGSY infrastructure dataset."""
    return load_tabular_dataset(
        path,
        pd.read_csv,
        max_rows=max_rows,
        random_state=random_state,
        group_column="PMGSY_SCHEME",
    )


def load_sanitation(
    path: str | Path = DATA_PATHS["sanitation"],
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Load and sample the sanitation coverage dataset."""
    return load_tabular_dataset(
        path,
        pd.read_csv,
        max_rows=max_rows,
        random_state=random_state,
        group_column="StateName",
    )


def load_nfhs(
    path: str | Path = DATA_PATHS["nfhs"],
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Load and sample the NFHS factsheet dataset."""
    return load_tabular_dataset(
        path,
        pd.read_excel,
        max_rows=max_rows,
        random_state=random_state,
        group_column="Area",
    )


def load_sampled_datasets(
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Load all source datasets with consistent sampling settings."""
    return {
        "mgnrega_mp": load_mgnrega(DATA_PATHS["mgnrega_mp"], max_rows=max_rows, random_state=random_state),
        "mgnrega_cg": load_mgnrega(DATA_PATHS["mgnrega_cg"], max_rows=max_rows, random_state=random_state),
        "pmgsy": load_pmgsy(max_rows=max_rows, random_state=random_state),
        "sanitation": load_sanitation(max_rows=max_rows, random_state=random_state),
        "nfhs": load_nfhs(max_rows=max_rows, random_state=random_state),
    }
