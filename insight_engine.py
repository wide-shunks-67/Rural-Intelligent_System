from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


INPUT_FILE = Path("rural_nlp_dataset_with_predictions.csv")
TEXT_COLUMNS = [
    "state",
    "district",
    "region_level",
    "text",
    "summary",
    "category",
    "predicted_category",
    "issues_text",
    "severity",
    "priority",
    "source",
]
REQUIRED_COLUMNS = ["state", "predicted_category", "severity", "priority", "issues"]


def load_dataset(csv_path: str | Path = INPUT_FILE) -> pd.DataFrame:
    """Load and clean the prediction dataset for downstream insights."""
    resolved_path = Path(csv_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Insight dataset not found: {resolved_path}")
    LOGGER.info("Loading insight dataset from %s", resolved_path)
    return clean_dataset(pd.read_csv(resolved_path))


def parse_issues(value: Any) -> list[str]:
    """Parse the serialized issues field into a normalized list."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "unknown"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            LOGGER.warning("Unable to literal-eval issues value: %s", text)

    if "|" in text:
        return [item.strip() for item in text.split("|") if item.strip()]
    return [text]


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error when required columns are missing."""
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")


def _empty_state_summary(state_name: str) -> dict[str, Any]:
    """Return a default state summary for missing or empty state data."""
    return {
        "state": state_name,
        "key_issues": [],
        "severity_distribution": {"high": 0, "moderate": 0, "low": 0},
        "summary": "No records found for this state.",
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing values, issue lists, and confidence values."""
    cleaned = df.copy()
    for column in TEXT_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip()

    if "confidence" in cleaned.columns:
        cleaned["confidence"] = pd.to_numeric(cleaned["confidence"], errors="coerce")

    cleaned["issues"] = cleaned["issues"].apply(parse_issues) if "issues" in cleaned.columns else [[] for _ in range(len(cleaned))]

    if "predicted_category" in cleaned.columns and "category" in cleaned.columns:
        fallback_mask = cleaned["predicted_category"].isin(["", "Unknown", "nan"])
        cleaned.loc[fallback_mask, "predicted_category"] = cleaned.loc[fallback_mask, "category"]

    return cleaned


def _explode_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Explode issue lists and drop empty issue values."""
    if df.empty:
        return pd.DataFrame(columns=list(df.columns) + ["issues"])
    exploded = df.explode("issues")
    return exploded.dropna(subset=["issues"]).query("issues != ''")


def build_state_insights(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build aggregated state-level insights from the prediction dataset."""
    _validate_columns(df, REQUIRED_COLUMNS + ["confidence"])
    if df.empty:
        return {}

    exploded_issues = _explode_issues(df)
    issue_counts = (
        exploded_issues.groupby(["state", "issues"])
        .size()
        .reset_index(name="count")
        .sort_values(["state", "count", "issues"], ascending=[True, False, True])
    )
    top_issue_per_state = (
        issue_counts.drop_duplicates(subset=["state"]).set_index("state")["issues"]
        if not issue_counts.empty
        else pd.Series(dtype="object")
    )

    state_summary = (
        df.groupby("state", dropna=False)
        .agg(
            total_records=("state", "size"),
            most_common_category=("predicted_category", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
            average_confidence=("confidence", "mean"),
            high_severity=("severity", lambda s: int((s == "high").sum())),
            moderate_severity=("severity", lambda s: int((s == "moderate").sum())),
            low_severity=("severity", lambda s: int((s == "low").sum())),
        )
        .reset_index()
    )
    state_summary["average_confidence"] = state_summary["average_confidence"].fillna(0).round(4)
    state_summary["most_common_issue"] = state_summary["state"].map(top_issue_per_state).fillna("None")
    state_summary["high_severity_pct"] = ((state_summary["high_severity"] / state_summary["total_records"]) * 100).round(2)
    state_summary["moderate_severity_pct"] = (
        (state_summary["moderate_severity"] / state_summary["total_records"]) * 100
    ).round(2)
    state_summary["low_severity_pct"] = ((state_summary["low_severity"] / state_summary["total_records"]) * 100).round(2)

    ordered = state_summary.sort_values(
        ["high_severity", "moderate_severity", "total_records", "state"],
        ascending=[False, False, False, True],
    )
    return {row["state"]: {key: row[key] for key in ordered.columns if key != "state"} for _, row in ordered.iterrows()}


def get_top_districts(df: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """Return districts with the highest number of high-severity records."""
    _validate_columns(df, ["state", "district", "severity"])
    if df.empty:
        return []
    district_counts = (
        df[df["severity"] == "high"]
        .groupby(["state", "district"], dropna=False)
        .size()
        .reset_index(name="high_severity_records")
        .sort_values(["high_severity_records", "state", "district"], ascending=[False, True, True])
        .head(top_n)
    )
    return district_counts.to_dict(orient="records")


def get_top_states_by_priority(df: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """Return states with the highest count of immediate-attention records."""
    _validate_columns(df, ["state", "priority"])
    if df.empty:
        return []
    state_counts = (
        df[df["priority"] == "Immediate Attention Required"]
        .groupby("state", dropna=False)
        .size()
        .reset_index(name="immediate_attention_records")
        .sort_values(["immediate_attention_records", "state"], ascending=[False, True])
        .head(top_n)
    )
    return state_counts.to_dict(orient="records")


def get_category_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Return the distribution of predicted categories across the dataset."""
    _validate_columns(df, ["predicted_category"])
    if df.empty:
        return {}
    counts = df["predicted_category"].fillna("Unknown").astype(str).value_counts(dropna=False).sort_values(ascending=False)
    return {str(index): int(value) for index, value in counts.items()}


def get_top_issues(df: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    """Return the most frequent issues across the dataset."""
    _validate_columns(df, ["issues"])
    if df.empty:
        return []
    issue_counts = (
        _explode_issues(df)
        .groupby("issues")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "issues"], ascending=[False, True])
        .head(top_n)
    )
    return issue_counts.rename(columns={"issues": "issue"}).to_dict(orient="records")


def get_global_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a high-level summary across the full dataset."""
    _validate_columns(df, ["predicted_category", "severity", "issues"])
    total_records = int(len(df))
    if total_records == 0:
        return {
            "total_records": 0,
            "most_common_category": "Unknown",
            "top_issue": "None",
            "percentage_high_severity": 0.0,
        }

    most_common_category = (
        df["predicted_category"].mode().iloc[0]
        if not df["predicted_category"].mode().empty
        else "Unknown"
    )
    top_issues = get_top_issues(df, top_n=1)
    top_issue = top_issues[0]["issue"] if top_issues else "None"
    percentage_high_severity = round(((df["severity"] == "high").sum() / total_records) * 100, 2)
    return {
        "total_records": total_records,
        "most_common_category": most_common_category,
        "top_issue": top_issue,
        "percentage_high_severity": percentage_high_severity,
    }


def build_insight_output(df: pd.DataFrame) -> dict[str, Any]:
    """Build the complete structured insight payload used by the dashboard and reports."""
    return {
        "global_summary": get_global_summary(df),
        "state_insights": build_state_insights(df),
        "top_districts": get_top_districts(df),
        "top_states": get_top_states_by_priority(df),
        "category_distribution": get_category_distribution(df),
        "top_issues": get_top_issues(df),
    }


def get_state_summary(df: pd.DataFrame, state_name: str) -> dict[str, Any]:
    """Return a compact summary view for a single state."""
    _validate_columns(df, ["state", "predicted_category", "severity", "issues"])
    if not state_name.strip():
        return _empty_state_summary(state_name)

    state_df = df[df["state"].str.casefold() == state_name.strip().casefold()].copy()
    if state_df.empty:
        return _empty_state_summary(state_name)

    top_issues = get_top_issues(state_df, top_n=5)
    severity_distribution = {
        "high": int((state_df["severity"] == "high").sum()),
        "moderate": int((state_df["severity"] == "moderate").sum()),
        "low": int((state_df["severity"] == "low").sum()),
    }
    dominant_category = (
        state_df["predicted_category"].mode().iloc[0]
        if not state_df["predicted_category"].mode().empty
        else "Unknown"
    )
    leading_issue = top_issues[0]["issue"] if top_issues else "no dominant issue"
    summary = (
        f"{state_df['state'].iloc[0]} has {len(state_df)} records, with {dominant_category} emerging as the most common predicted category. "
        f"The leading issue is {leading_issue}, and the severity mix is high={severity_distribution['high']}, "
        f"moderate={severity_distribution['moderate']}, low={severity_distribution['low']}."
    )
    return {
        "state": state_df["state"].iloc[0],
        "key_issues": top_issues,
        "severity_distribution": severity_distribution,
        "summary": summary,
    }


def main() -> None:
    """Run a sample insight-engine workflow from the command line."""
    dataset = load_dataset(INPUT_FILE)
    insights = build_insight_output(dataset)

    LOGGER.info("Category distribution:\n%s", json.dumps(insights["category_distribution"], indent=2))
    LOGGER.info("Top districts:\n%s", json.dumps(insights["top_districts"], indent=2))
    LOGGER.info("Top issues:\n%s", json.dumps(insights["top_issues"], indent=2))

    sample_state = dataset["state"].iloc[0] if not dataset.empty else "Madhya Pradesh"
    LOGGER.info("State summary for %s:\n%s", sample_state, json.dumps(get_state_summary(dataset, sample_state), indent=2))


if __name__ == "__main__":
    main()
