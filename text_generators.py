from __future__ import annotations

from typing import Any

import pandas as pd

from utils import (
    build_result,
    qualitative_band,
    ratio,
    severity_from_score,
    to_decimal_text,
    to_float,
    to_int_text,
)


def generate_mgnrega_text(row: pd.Series) -> dict[str, Any]:
    """Generate structured employment text for a single MGNREGA row."""
    district = row.get("district_name", "Unknown district")
    state = row.get("state_name", "Unknown state")
    month = row.get("month", "NA")
    year = row.get("fin_year", "NA")

    avg_days = to_float(row.get("Average_days_of_employment_provided_per_Household"))
    wage_rate = to_float(row.get("Average_Wage_rate_per_day_per_person"))
    women_persondays = to_float(row.get("Women_Persondays"))
    total_persondays = to_float(row.get("Persondays_of_Central_Liability_so_far"))
    women_share = ratio(women_persondays, total_persondays)
    sc_share = ratio(row.get("SC_persondays"), total_persondays)
    st_share = ratio(row.get("ST_persondays"), total_persondays)
    payment_timeliness = to_float(row.get("percentage_payments_gererated_within_15_days"))

    days_status = qualitative_band(avg_days, good_cutoff=45, moderate_cutoff=30)
    wage_status = qualitative_band(wage_rate, good_cutoff=240, moderate_cutoff=210)
    women_status = qualitative_band(women_share, good_cutoff=45, moderate_cutoff=33)
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

    severity = severity_from_score(severity_score)
    if severity == "high":
        interpretation = "This indicates serious livelihood stress and uneven programme reach."
    elif severity == "moderate":
        interpretation = "This suggests moderate employment support, but access and participation remain uneven."
    else:
        interpretation = "This points to relatively stable employment support in the district."

    issue_text = f" Key concerns include {', '.join(issues)}." if issues else ""
    text = (
        f"In {district}, {state}, during {month} {year}, {employment_summary.lower()} under MGNREGA. "
        f"Households received an average of {to_decimal_text(avg_days, 0)} days of employment, "
        f"while the average wage rate was Rs. {to_decimal_text(wage_rate, 2)} per day, indicating {days_status} work availability and {wage_status} wage support. "
        f"Participation patterns show that {participation_text}, which suggests {women_status} inclusion of women in the programme."
        f"{payment_text}{issue_text} {interpretation}"
    )
    return build_result(text, "Employment", issues, severity, "MGNREGA")


def generate_pmgsy_text(row: pd.Series) -> dict[str, Any]:
    """Generate structured infrastructure text for a single PMGSY row."""
    district = row.get("DISTRICT_NAME", "Unknown district")
    state = row.get("STATE_NAME", "Unknown state")
    scheme = row.get("PMGSY_SCHEME", "PMGSY")

    sanctioned = to_float(row.get("NO_OF_ROAD_WORK_SANCTIONED"))
    completed = to_float(row.get("NO_OF_ROAD_WORKS_COMPLETED"))
    balance = to_float(row.get("NO_OF_ROAD_WORKS_BALANCE"))
    sanctioned_km = to_float(row.get("LENGTH_OF_ROAD_WORK_SANCTIONED_KM"))
    completed_km = to_float(row.get("LENGTH_OF_ROAD_WORK_COMPLETED_KM"))
    completion_rate = ratio(completed, sanctioned)
    completion_band = qualitative_band(completion_rate, good_cutoff=70, moderate_cutoff=35)

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
            f" Against {to_decimal_text(sanctioned_km, 3)} km sanctioned, "
            f"{to_decimal_text(completed_km, 3)} km has been completed."
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

    km_completion_rate = ratio(completed_km, sanctioned_km)
    if km_completion_rate is not None and km_completion_rate < 35:
        issues.append("slow physical connectivity expansion")
        severity_score += 1

    severity = severity_from_score(severity_score)
    if severity == "high":
        interpretation = "This indicates serious infrastructure gaps and delayed connectivity benefits."
    elif severity == "moderate":
        interpretation = "This suggests moderate infrastructure development, but coverage is still incomplete."
    else:
        interpretation = "This indicates that road infrastructure delivery is broadly moving in the right direction."

    issue_text = f" Major issues include {', '.join(issues)}." if issues else ""
    text = (
        f"In {district}, {state}, under {scheme}, {to_int_text(sanctioned)} road works were sanctioned and "
        f"{to_int_text(completed)} have been completed, leaving {to_int_text(balance)} still pending. "
        f"This places road-work completion at {to_decimal_text(completion_rate, 1)}%, suggesting that {connectivity}."
        f"{length_sentence}{issue_text} {interpretation}"
    )
    return build_result(text, "Infrastructure", issues, severity, "PMGSY")


def generate_sanitation_text(row: pd.Series) -> dict[str, Any]:
    """Generate structured sanitation text for a single sanitation row."""
    block = row.get("blockName", "Unknown block")
    district = row.get("DistrictName", "Unknown district")
    state = row.get("StateName", "Unknown state")
    planned = to_float(row.get("IHHLTotalAsPerDetails"))
    achieved = to_float(row.get("IHHLTotalAch"))
    achievement_rate = ratio(achieved, planned)
    quality = qualitative_band(achievement_rate, good_cutoff=95, moderate_cutoff=70)
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

    severity = severity_from_score(severity_score)
    if severity == "high":
        interpretation = "This indicates poor sanitation delivery and a likely gap in household coverage."
    elif severity == "moderate":
        interpretation = "This suggests sanitation access has improved, but full coverage has not yet been secured."
    else:
        interpretation = "This points to strong sanitation achievement at the block level."

    issue_text = f" Key issues include {', '.join(issues)}." if issues else ""
    text = (
        f"As of {date}, {block} block in {district}, {state} reported {to_int_text(planned)} household toilets planned "
        f"and {to_int_text(achieved)} achieved. This corresponds to {to_decimal_text(achievement_rate, 1)}% achievement, "
        f"which indicates that {status_text} in this block.{issue_text} {interpretation}"
    )
    return build_result(text, "Infrastructure", issues, severity, "Sanitation")


def generate_nfhs_text(row: pd.Series) -> dict[str, Any]:
    """Generate structured health text for a single NFHS row."""
    geography = row.get("States/UTs", "Unknown region")
    area = row.get("Area", "Total")

    sanitation = to_float(row.get("Population living in households that use an improved sanitation facility2 (%)"))
    drinking_water = to_float(row.get("Population living in households with an improved drinking-water source1 (%)"))
    female_literacy = to_float(row.get("Women (age 15-49) who are literate4 (%)"))
    male_literacy = to_float(row.get("Men (age 15-49) who are literate4 (%)"))
    vaccination = to_float(
        row.get(
            "Children age 12-23 months fully vaccinated based on information from either vaccination card or mother's recall11 (%)"
        )
    )
    stunting = to_float(row.get("Children under 5 years who are stunted (height-for-age)18 (%)"))
    underweight = to_float(row.get("Children under 5 years who are underweight (weight-for-age)18 (%)"))

    sanitation_band = qualitative_band(sanitation, good_cutoff=80, moderate_cutoff=60)
    water_band = qualitative_band(drinking_water, good_cutoff=90, moderate_cutoff=75)
    vaccination_band = qualitative_band(vaccination, good_cutoff=85, moderate_cutoff=65)

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
            f"female literacy is {to_decimal_text(female_literacy, 2)}% and male literacy is {to_decimal_text(male_literacy, 2)}%"
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

    severity = severity_from_score(severity_score)
    if severity == "high":
        interpretation = "This indicates serious public health and nutrition gaps that need focused intervention."
    elif severity == "moderate":
        interpretation = "This suggests moderate development progress, but essential health coverage remains uneven."
    else:
        interpretation = "This points to comparatively stable health conditions with fewer immediate vulnerabilities."

    issue_text = f" Key concerns include {', '.join(issues)}." if issues else ""
    text = (
        f"In {geography} ({area}), NFHS indicators show that {to_decimal_text(sanitation, 2)}% of the population uses improved sanitation, "
        f"{to_decimal_text(drinking_water, 2)}% has access to improved drinking water, and {literacy_text}. "
        f"Full vaccination among children aged 12-23 months stands at {to_decimal_text(vaccination, 2)}%. "
        f"At the same time, {to_decimal_text(stunting, 2)}% of children under five are stunted and {to_decimal_text(underweight, 2)}% are underweight, "
        f"which means {malnutrition_note}.{issue_text} Taken together, these values suggest that {overall_health}. {interpretation}"
    )
    return build_result(text, "Health", issues, severity, "NFHS")
