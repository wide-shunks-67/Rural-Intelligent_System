from __future__ import annotations

from typing import Any

import pandas as pd


TEXT_LIMIT = 500


def to_float(value: Any) -> float | None:
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


def to_int_text(value: Any) -> str:
    """Format a numeric value as an integer string for text output."""
    number = to_float(value)
    return "NA" if number is None else f"{int(round(number)):,}"


def to_decimal_text(value: Any, digits: int = 1) -> str:
    """Format a numeric value as a decimal string for text output."""
    number = to_float(value)
    return "NA" if number is None else f"{number:.{digits}f}"


def ratio(numerator: Any, denominator: Any) -> float | None:
    """Return a percentage ratio when both values are available."""
    num = to_float(numerator)
    den = to_float(denominator)
    if num is None or den in (None, 0):
        return None
    return (num / den) * 100


def qualitative_band(value: float | None, good_cutoff: float, moderate_cutoff: float) -> str:
    """Map a numeric value to a qualitative band."""
    if value is None:
        return "unclear"
    if value >= good_cutoff:
        return "good"
    if value >= moderate_cutoff:
        return "moderate"
    return "low"


def severity_from_score(score: int) -> str:
    """Convert a severity score into a categorical severity label."""
    if score >= 4:
        return "high"
    if score >= 2:
        return "moderate"
    return "low"


def build_summary(issues: list[str], severity: str) -> str:
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


def build_result(
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
        "summary": build_summary(issues, severity),
        "severity": severity,
        "priority": priority_map[severity],
    }
