from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data_loader import (
    DATA_PATHS,
    DEFAULT_MAX_ROWS,
    DEFAULT_RANDOM_STATE,
    load_mgnrega,
    load_nfhs,
    load_pmgsy,
    load_sanitation,
    resolve_data_path,
)
from text_generators import (
    generate_mgnrega_text,
    generate_nfhs_text,
    generate_pmgsy_text,
    generate_sanitation_text,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


ML_COLUMNS = ["confidence", "predicted_category"]
OUTPUT_COLUMNS = [
    "id",
    "state",
    "district",
    "region_level",
    "text",
    "summary",
    "category",
    "issues",
    "issues_text",
    "num_issues",
    "severity",
    "priority",
    "source",
    "confidence",
    "predicted_category",
]


def build_record(
    record_id: str,
    state: str,
    district: str,
    generated: dict[str, Any],
    region_level: str,
) -> dict[str, Any]:
    """Build a normalized output record for the final NLP dataset."""
    issues = list(generated.get("issues", []))
    return {
        "id": record_id,
        "state": str(state).strip().title() or "Unknown",
        "district": str(district).strip().title() or "Unknown",
        "region_level": region_level,
        "text": str(generated.get("text", "")).strip(),
        "summary": str(generated.get("summary", "")).strip(),
        "category": str(generated.get("category", "")).strip(),
        "issues": issues,
        "issues_text": " | ".join(issues) if issues else "none",
        "num_issues": len(issues),
        "severity": str(generated.get("severity", "")).strip(),
        "priority": str(generated.get("priority", "")).strip(),
        "source": str(generated.get("source", "")).strip(),
        "confidence": None,
        "predicted_category": None,
    }


def records_from_dataframe(
    df: pd.DataFrame,
    generator: Callable[[pd.Series], dict[str, Any]],
    *,
    state_col: str,
    district_col: str,
    id_prefix: str,
    region_level: str,
) -> list[dict[str, Any]]:
    """Generate structured records for every row in a dataframe."""
    return [
        build_record(
            f"{id_prefix}_{index + 1:03d}",
            str(row.get(state_col, "Unknown")),
            str(row.get(district_col, "Unknown")),
            generator(row),
            region_level,
        )
        for index, (_, row) in enumerate(df.iterrows())
    ]


def validate_non_empty_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Filter rows with empty required values for the specified columns."""
    if df.empty:
        return df
    non_empty_mask = df[columns].astype(str).apply(lambda col: col.str.strip() != "").all(axis=1)
    return df.loc[non_empty_mask].copy()


def create_final_dataset(
    output_path: str | Path = "rural_nlp_dataset.csv",
    max_rows_per_source: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Create the final combined NLP-ready dataset from all source datasets."""
    LOGGER.info("Creating final dataset with up to %s rows per source", max_rows_per_source)
    half_mgnrega = max(1, max_rows_per_source // 2)
    mgnrega_mp = load_mgnrega(DATA_PATHS["mgnrega_mp"], max_rows=half_mgnrega, random_state=random_state)
    mgnrega_cg = load_mgnrega(DATA_PATHS["mgnrega_cg"], max_rows=half_mgnrega, random_state=random_state + 1)
    pmgsy = load_pmgsy(max_rows=max_rows_per_source, random_state=random_state)
    sanitation = load_sanitation(max_rows=max_rows_per_source, random_state=random_state)
    nfhs = load_nfhs(max_rows=max_rows_per_source, random_state=random_state)

    records = (
        records_from_dataframe(
            pd.concat([mgnrega_mp, mgnrega_cg], ignore_index=True),
            generate_mgnrega_text,
            state_col="state_name",
            district_col="district_name",
            id_prefix="MGNREGA",
            region_level="district",
        )
        + records_from_dataframe(
            pmgsy,
            generate_pmgsy_text,
            state_col="STATE_NAME",
            district_col="DISTRICT_NAME",
            id_prefix="PMGSY",
            region_level="district",
        )
        + records_from_dataframe(
            sanitation,
            generate_sanitation_text,
            state_col="StateName",
            district_col="DistrictName",
            id_prefix="SANITATION",
            region_level="block",
        )
        + records_from_dataframe(
            nfhs,
            generate_nfhs_text,
            state_col="States/UTs",
            district_col="Area",
            id_prefix="NFHS",
            region_level="state",
        )
    )

    final_df = pd.DataFrame(records)
    if final_df.empty:
        raise ValueError("No records were generated for the final dataset.")

    final_df = final_df.drop_duplicates(subset=["state", "district", "text", "summary", "source"]).reset_index(drop=True)
    final_df = final_df[OUTPUT_COLUMNS]

    core_columns = [col for col in final_df.columns if col not in ML_COLUMNS]
    final_df[core_columns] = final_df[core_columns].fillna("Unknown")
    final_df = validate_non_empty_columns(final_df, core_columns)

    if final_df.empty:
        raise ValueError("All generated rows were filtered out during final dataset validation.")

    source_counts = final_df["source"].value_counts()
    min_count = int(source_counts.min())
    balanced_df = (
        final_df.groupby("source", group_keys=False, sort=False)
        .apply(lambda group: group.sample(n=min_count, random_state=random_state))
        .reset_index(drop=True)
    )
    balanced_df["id"] = [f"{row.source}_{index + 1:03d}" for index, row in enumerate(balanced_df.itertuples())]

    output_file = resolve_data_path(output_path)
    balanced_df.to_csv(output_file, index=False)
    LOGGER.info("Saved %s balanced rows to %s", len(balanced_df), output_file)
    return balanced_df
