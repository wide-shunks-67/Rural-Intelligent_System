from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_MAX_ROWS = 100
DEFAULT_RANDOM_STATE = 42
TEXT_LIMIT = 500
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
DATA_PATHS: dict[str, Path] = {
    "mgnrega_mp": Path("data/mgnrega_mp.csv"),
    "mgnrega_cg": Path("data/mgnrega_cg.csv"),
    "pmgsy": Path("data/pmgsy.csv"),
    "sanitation": Path("data/sanitation.csv"),
    "nfhs": Path("data/nfhs_5_factsheets_data.xls"),
}


def _resolve_data_path(path: str | Path) -> Path:
    """Resolve a relative project path to an absolute file path."""
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def _ensure_file_exists(path: str | Path) -> Path:
    """Validate that a dataset file exists before attempting to load it."""
    resolved_path = _resolve_data_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved_path}")
    return resolved_path


def _sample_dataframe(
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


def _load_tabular_dataset(
    path: str | Path,
    reader: Callable[..., pd.DataFrame],
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Load and sample a tabular dataset from CSV or Excel."""
    resolved_path = _ensure_file_exists(path)
    LOGGER.info("Loading dataset from %s", resolved_path)
    df = reader(resolved_path)
    return _sample_dataframe(df, max_rows=max_rows, random_state=random_state, group_column=group_column)


def load_mgnrega(
    path: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Load and sample an MGNREGA dataset at district level."""
    return _load_tabular_dataset(
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
    return _load_tabular_dataset(
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
    return _load_tabular_dataset(
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
    return _load_tabular_dataset(
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


def _to_float(value: Any) -> float | None:
    """Convert a raw dataset value into float when possible."""
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", "")
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_text(value: Any) -> str:
    """Format a numeric value as an integer string for text output."""
    number = _to_float(value)
    return "NA" if number is None else f"{int(round(number)):,}"


def _to_decimal_text(value: Any, digits: int = 1) -> str:
    """Format a numeric value as a decimal string for text output."""
    number = _to_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    """Return a percentage ratio when both values are available."""
    num = _to_float(numerator)
    den = _to_float(denominator)
    if num is None or den in (None, 0):
        return None
    return (num / den) * 100


def _qualitative_band(value: float | None, good_cutoff: float, moderate_cutoff: float) -> str:
    """Map a numeric value to a qualitative band."""
    if value is None:
        return "unclear"
    if value >= good_cutoff:
        return "good"
    if value >= moderate_cutoff:
        return "moderate"
    return "low"


def _severity_from_score(score: int) -> str:
    """Convert a severity score into a categorical severity label."""
    if score >= 4:
        return "high"
    if score >= 2:
        return "moderate"
    return "low"


def _build_summary(issues: list[str], severity: str) -> str:
    """Generate a short human-readable issue summary."""
    if not issues:
        if severity == "low":
            return "Conditions are stable"
        if severity == "moderate":
            return "Mixed conditions require monitoring"
        return "Serious gaps need attention"

    cleaned = [issue.replace("moderate ", "").strip() for issue in issues[:2]]
    if len(cleaned) == 1:
        return cleaned[0].capitalize()
    return f"{cleaned[0].capitalize()} and {cleaned[1]}"


def _build_result(
    text: str,
    category: str,
    issues: list[str],
    severity: str,
    source: str,
) -> dict[str, Any]:
    """Create a standard structured output for generated dataset text."""
    priority_map = {
        "high": "Immediate Attention Required",
        "moderate": "Needs Monitoring",
        "low": "Stable",
    }
    safe_text = text if len(text) <= TEXT_LIMIT else f"{text[:TEXT_LIMIT]}..."
    return {
        "text": safe_text,
        "category": category,
        "source": source,
        "issues": issues,
        "summary": _build_summary(issues, severity),
        "severity": severity,
        "priority": priority_map[severity],
    }


def generate_mgnrega_text(row: pd.Series) -> dict[str, Any]:
    """Generate structured employment text for a single MGNREGA row."""
    district = row.get("district_name", "Unknown district")
    state = row.get("state_name", "Unknown state")
    month = row.get("month", "NA")
    year = row.get("fin_year", "NA")

    avg_days = _to_float(row.get("Average_days_of_employment_provided_per_Household"))
    wage_rate = _to_float(row.get("Average_Wage_rate_per_day_per_person"))
    women_persondays = _to_float(row.get("Women_Persondays"))
    total_persondays = _to_float(row.get("Persondays_of_Central_Liability_so_far"))
    women_share = _ratio(women_persondays, total_persondays)
    sc_share = _ratio(row.get("SC_persondays"), total_persondays)
    st_share = _ratio(row.get("ST_persondays"), total_persondays)
    payment_timeliness = _to_float(row.get("percentage_payments_gererated_within_15_days"))

    days_status = _qualitative_band(avg_days, good_cutoff=45, moderate_cutoff=30)
    wage_status = _qualitative_band(wage_rate, good_cutoff=240, moderate_cutoff=210)
    women_status = _qualitative_band(women_share, good_cutoff=45, moderate_cutoff=33)
    issues: list[str] = []
    severity_score = 0

    if days_status == "good" and wage_status == "good":
        employment_summary = "Employment conditions appear relatively strong"
    elif days_status == "low" or wage_status == "low":
        employment_summary = "Employment conditions look comparatively weak"
    else:
        employment_summary = "Employment conditions appear mixed"

    participation_parts: list[str] = []
    if women_share is not None:
        participation_parts.append(f"women contributed {women_share:.1f}% of recorded persondays")
    if sc_share is not None:
        participation_parts.append(f"SC persondays accounted for {sc_share:.1f}%")
    if st_share is not None:
        participation_parts.append(f"ST persondays accounted for {st_share:.1f}%")
    participation_text = "; ".join(participation_parts) if participation_parts else "participation details are incomplete"

    payment_text = ""
    if payment_timeliness is not None:
        if payment_timeliness >= 90:
            payment_text = f" Payment generation was timely, with {payment_timeliness:.1f}% processed within 15 days."
        else:
            payment_text = f" Payment timeliness was weaker, with only {payment_timeliness:.1f}% generated within 15 days."
            issues.append("delayed wage payments")
            severity_score += 1

    if avg_days is not None and avg_days < 30:
        issues.append("low employment days")
        severity_score += 2
    elif avg_days is not None and avg_days < 45:
        issues.append("moderate employment availability")
        severity_score += 1

    if wage_rate is not None and wage_rate < 210:
        issues.append("low wage rate")
        severity_score += 2
    elif wage_rate is not None and wage_rate < 240:
        issues.append("moderate wage support")
        severity_score += 1

    if women_share is not None and women_share < 33:
        issues.append("low women participation")
        severity_score += 1
    if sc_share is not None and sc_share < 10:
        issues.append("low SC participation")
        severity_score += 1
    if st_share is not None and st_share < 10:
        issues.append("low ST participation")
        severity_score += 1

    severity = _severity_from_score(severity_score)
    if severity == "high":
        interpretation = "This indicates serious livelihood stress and uneven programme reach."
    elif severity == "moderate":
        interpretation = "This suggests moderate employment support, but access and participation remain uneven."
    else:
        interpretation = "This points to relatively stable employment support in the district."

    issue_text = f" Key concerns include {', '.join(issues)}." if issues else ""
    text = (
        f"In {district}, {state}, during {month} {year}, {employment_summary.lower()} under MGNREGA. "
        f"Households received an average of {_to_decimal_text(avg_days, 0)} days of employment, "
        f"while the average wage rate was Rs. {_to_decimal_text(wage_rate, 2)} per day, indicating {days_status} work availability and {wage_status} wage support. "
        f"Participation patterns show that {participation_text}, which suggests {women_status} inclusion of women in the programme."
        f"{payment_text}{issue_text} {interpretation}"
    )
    return _build_result(text, "Employment", issues, severity, "MGNREGA")


def generate_pmgsy_text(row: pd.Series) -> dict[str, Any]:
    """Generate structured infrastructure text for a single PMGSY row."""
    district = row.get("DISTRICT_NAME", "Unknown district")
    state = row.get("STATE_NAME", "Unknown state")
    scheme = row.get("PMGSY_SCHEME", "PMGSY")

    sanctioned = _to_float(row.get("NO_OF_ROAD_WORK_SANCTIONED"))
    completed = _to_float(row.get("NO_OF_ROAD_WORKS_COMPLETED"))
    balance = _to_float(row.get("NO_OF_ROAD_WORKS_BALANCE"))
    sanctioned_km = _to_float(row.get("LENGTH_OF_ROAD_WORK_SANCTIONED_KM"))
    completed_km = _to_float(row.get("LENGTH_OF_ROAD_WORK_COMPLETED_KM"))
    completion_rate = _ratio(completed, sanctioned)
    completion_band = _qualitative_band(completion_rate, good_cutoff=70, moderate_cutoff=35)

    issues: list[str] = []
    severity_score = 0
    if completion_band == "good":
        connectivity = "rural connectivity is improving at a visible pace"
    elif completion_band == "low":
        connectivity = "road connectivity still appears constrained"
    else:
        connectivity = "connectivity is improving, but progress remains incomplete"

    length_sentence = ""
    if sanctioned_km is not None and completed_km is not None:
        length_sentence = (
            f" Against {_to_decimal_text(sanctioned_km, 3)} km sanctioned, "
            f"{_to_decimal_text(completed_km, 3)} km has been completed."
        )

    if completion_rate is not None and completion_rate < 35:
        issues.append("poor road completion")
        severity_score += 2
    elif completion_rate is not None and completion_rate < 70:
        issues.append("incomplete road completion")
        severity_score += 1

    if balance is not None and balance > 0:
        issues.append("pending road works")
        severity_score += 1

    km_completion_rate = _ratio(completed_km, sanctioned_km)
    if km_completion_rate is not None and km_completion_rate < 35:
        issues.append("slow physical connectivity expansion")
        severity_score += 1

    severity = _severity_from_score(severity_score)
    if severity == "high":
        interpretation = "This indicates serious infrastructure gaps and delayed connectivity benefits."
    elif severity == "moderate":
        interpretation = "This suggests moderate infrastructure development, but coverage is still incomplete."
    else:
        interpretation = "This indicates that road infrastructure delivery is broadly moving in the right direction."

    issue_text = f" Major issues include {', '.join(issues)}." if issues else ""
    text = (
        f"In {district}, {state}, under {scheme}, {_to_int_text(sanctioned)} road works were sanctioned and "
        f"{_to_int_text(completed)} have been completed, leaving {_to_int_text(balance)} still pending. "
        f"This places road-work completion at {_to_decimal_text(completion_rate, 1)}%, suggesting that {connectivity}."
        f"{length_sentence}{issue_text} {interpretation}"
    )
    return _build_result(text, "Infrastructure", issues, severity, "PMGSY")


def generate_sanitation_text(row: pd.Series) -> dict[str, Any]:
    """Generate structured sanitation text for a single sanitation row."""
    block = row.get("blockName", "Unknown block")
    district = row.get("DistrictName", "Unknown district")
    state = row.get("StateName", "Unknown state")
    planned = _to_float(row.get("IHHLTotalAsPerDetails"))
    achieved = _to_float(row.get("IHHLTotalAch"))
    achievement_rate = _ratio(achieved, planned)
    quality = _qualitative_band(achievement_rate, good_cutoff=95, moderate_cutoff=70)
    date = row.get("Date", "NA")

    issues: list[str] = []
    severity_score = 0
    if quality == "good":
        status_text = "sanitation coverage appears strong"
    elif quality == "low":
        status_text = "sanitation progress appears weak"
    else:
        status_text = "sanitation progress is partial"

    if achievement_rate is not None and achievement_rate < 70:
        issues.append("low sanitation coverage")
        severity_score += 2
    elif achievement_rate is not None and achievement_rate < 95:
        issues.append("incomplete sanitation coverage")
        severity_score += 1

    if planned is not None and achieved is not None and achieved < planned:
        issues.append("toilet construction gap")
        severity_score += 1

    severity = _severity_from_score(severity_score)
    if severity == "high":
        interpretation = "This indicates poor sanitation delivery and a likely gap in household coverage."
    elif severity == "moderate":
        interpretation = "This suggests sanitation access has improved, but full coverage has not yet been secured."
    else:
        interpretation = "This points to strong sanitation achievement at the block level."

    issue_text = f" Key issues include {', '.join(issues)}." if issues else ""
    text = (
        f"As of {date}, {block} block in {district}, {state} reported {_to_int_text(planned)} household toilets planned "
        f"and {_to_int_text(achieved)} achieved. This corresponds to {_to_decimal_text(achievement_rate, 1)}% achievement, "
        f"which indicates that {status_text} in this block.{issue_text} {interpretation}"
    )
    return _build_result(text, "Infrastructure", issues, severity, "Sanitation")


def generate_nfhs_text(row: pd.Series) -> dict[str, Any]:
    """Generate structured health text for a single NFHS row."""
    geography = row.get("States/UTs", "Unknown region")
    area = row.get("Area", "Total")

    sanitation = _to_float(row.get("Population living in households that use an improved sanitation facility2 (%)"))
    drinking_water = _to_float(row.get("Population living in households with an improved drinking-water source1 (%)"))
    female_literacy = _to_float(row.get("Women (age 15-49) who are literate4 (%)"))
    male_literacy = _to_float(row.get("Men (age 15-49) who are literate4 (%)"))
    vaccination = _to_float(
        row.get(
            "Children age 12-23 months fully vaccinated based on information from either vaccination card or mother's recall11 (%)"
        )
    )
    stunting = _to_float(row.get("Children under 5 years who are stunted (height-for-age)18 (%)"))
    underweight = _to_float(row.get("Children under 5 years who are underweight (weight-for-age)18 (%)"))

    sanitation_band = _qualitative_band(sanitation, good_cutoff=80, moderate_cutoff=60)
    water_band = _qualitative_band(drinking_water, good_cutoff=90, moderate_cutoff=75)
    vaccination_band = _qualitative_band(vaccination, good_cutoff=85, moderate_cutoff=65)

    issues: list[str] = []
    severity_score = 0

    malnutrition_note = "malnutrition remains a concern"
    if stunting is not None and underweight is not None:
        if stunting < 20 and underweight < 20:
            malnutrition_note = "child malnutrition levels are comparatively lower"
        elif stunting >= 30 or underweight >= 30:
            malnutrition_note = "child malnutrition remains serious"

    overall_health = "overall health conditions appear mixed"
    if sanitation_band == "good" and water_band == "good" and vaccination_band == "good" and "lower" in malnutrition_note:
        overall_health = "overall health conditions appear relatively strong"
    elif sanitation_band == "low" or vaccination_band == "low" or "serious" in malnutrition_note:
        overall_health = "overall health conditions appear under stress"

    literacy_text = "literacy levels are not fully available"
    if female_literacy is not None and male_literacy is not None:
        literacy_text = (
            f"female literacy is {_to_decimal_text(female_literacy, 2)}% and male literacy is {_to_decimal_text(male_literacy, 2)}%"
        )

    if sanitation is not None and sanitation < 60:
        issues.append("low sanitation access")
        severity_score += 2
    elif sanitation is not None and sanitation < 80:
        issues.append("moderate sanitation coverage")
        severity_score += 1

    if drinking_water is not None and drinking_water < 75:
        issues.append("low drinking water access")
        severity_score += 2
    elif drinking_water is not None and drinking_water < 90:
        issues.append("moderate drinking water coverage")
        severity_score += 1

    if female_literacy is not None and male_literacy is not None:
        literacy_gap = male_literacy - female_literacy
        if female_literacy < 60:
            issues.append("low female literacy")
            severity_score += 1
        if literacy_gap > 15:
            issues.append("wide gender literacy gap")
            severity_score += 1

    if vaccination is not None and vaccination < 65:
        issues.append("low vaccination coverage")
        severity_score += 2
    elif vaccination is not None and vaccination < 85:
        issues.append("incomplete vaccination coverage")
        severity_score += 1

    if stunting is not None and stunting >= 30:
        issues.append("high child stunting")
        severity_score += 2
    elif stunting is not None and stunting >= 20:
        issues.append("moderate child stunting")
        severity_score += 1

    if underweight is not None and underweight >= 30:
        issues.append("high child underweight burden")
        severity_score += 2
    elif underweight is not None and underweight >= 20:
        issues.append("moderate child underweight burden")
        severity_score += 1

    severity = _severity_from_score(severity_score)
    if severity == "high":
        interpretation = "This indicates serious public health and nutrition gaps that need focused intervention."
    elif severity == "moderate":
        interpretation = "This suggests moderate development progress, but essential health coverage remains uneven."
    else:
        interpretation = "This points to comparatively stable health conditions with fewer immediate vulnerabilities."

    issue_text = f" Key concerns include {', '.join(issues)}." if issues else ""
    text = (
        f"In {geography} ({area}), NFHS indicators show that {_to_decimal_text(sanitation, 2)}% of the population uses improved sanitation, "
        f"{_to_decimal_text(drinking_water, 2)}% has access to improved drinking water, and {literacy_text}. "
        f"Full vaccination among children aged 12-23 months stands at {_to_decimal_text(vaccination, 2)}%. "
        f"At the same time, {_to_decimal_text(stunting, 2)}% of children under five are stunted and {_to_decimal_text(underweight, 2)}% are underweight, "
        f"which means {malnutrition_note}.{issue_text} Taken together, these values suggest that {overall_health}. {interpretation}"
    )
    return _build_result(text, "Health", issues, severity, "NFHS")


def _build_record(
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


def _records_from_dataframe(
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
        _build_record(
            f"{id_prefix}_{index + 1:03d}",
            str(row.get(state_col, "Unknown")),
            str(row.get(district_col, "Unknown")),
            generator(row),
            region_level,
        )
        for index, (_, row) in enumerate(df.iterrows())
    ]


def _validate_non_empty_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
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
        _records_from_dataframe(
            pd.concat([mgnrega_mp, mgnrega_cg], ignore_index=True),
            generate_mgnrega_text,
            state_col="state_name",
            district_col="district_name",
            id_prefix="MGNREGA",
            region_level="district",
        )
        + _records_from_dataframe(
            pmgsy,
            generate_pmgsy_text,
            state_col="STATE_NAME",
            district_col="DISTRICT_NAME",
            id_prefix="PMGSY",
            region_level="district",
        )
        + _records_from_dataframe(
            sanitation,
            generate_sanitation_text,
            state_col="StateName",
            district_col="DistrictName",
            id_prefix="SANITATION",
            region_level="block",
        )
        + _records_from_dataframe(
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
    final_df = _validate_non_empty_columns(final_df, core_columns)

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

    output_file = _resolve_data_path(output_path)
    balanced_df.to_csv(output_file, index=False)
    LOGGER.info("Saved %s balanced rows to %s", len(balanced_df), output_file)
    return balanced_df


if __name__ == "__main__":
    mgnrega_mp_df = load_mgnrega(DATA_PATHS["mgnrega_mp"])
    pmgsy_df = load_pmgsy()
    sanitation_df = load_sanitation()
    nfhs_df = load_nfhs()
    final_dataset_df = create_final_dataset()

    LOGGER.info("MGNREGA sample: %s", generate_mgnrega_text(mgnrega_mp_df.iloc[0]))
    LOGGER.info("PMGSY sample: %s", generate_pmgsy_text(pmgsy_df.iloc[0]))
    LOGGER.info("Sanitation sample: %s", generate_sanitation_text(sanitation_df.iloc[0]))
    LOGGER.info("NFHS sample: %s", generate_nfhs_text(nfhs_df.iloc[1]))
    LOGGER.info("Final dataset preview: %s", final_dataset_df.head(5).to_dict(orient="records"))
    LOGGER.info("Saved rows: %s", len(final_dataset_df))
