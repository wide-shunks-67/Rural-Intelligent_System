from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from insight_engine import build_insight_output, get_state_summary, load_dataset


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


DATA_PATH = Path("rural_nlp_dataset_with_predictions.csv")
CLASSIFIER_MODEL = "facebook/bart-large-mnli"
CLASSIFIER_LABELS = [
    "Health",
    "Education",
    "Agriculture",
    "Infrastructure",
    "Employment",
    "Social",
]
HYPOTHESIS_TEMPLATE = "This text is about {}."
TEXT_ANALYZER_LIMIT = 1000
SEVERITY_COLORS = {
    "high": "#d1495b",
    "moderate": "#edae49",
    "low": "#66a182",
}


st.set_page_config(page_title="Rural Intelligence System", page_icon="RI", layout="wide")


@st.cache_data(show_spinner=False)
def get_data(csv_path: str | Path) -> pd.DataFrame:
    """Load dashboard data once per file path."""
    return load_dataset(csv_path)


@st.cache_data(show_spinner=False)
def get_insights(df: pd.DataFrame) -> dict[str, Any]:
    """Cache global insight output for the dashboard."""
    return build_insight_output(df)


@st.cache_resource(show_spinner=True)
def get_text_classifier() -> Any:
    """Load and cache the HuggingFace zero-shot classifier used by the text analyzer."""
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    model_kwargs = {"torch_dtype": torch.float16} if device == 0 else None
    LOGGER.info("Loading dashboard text classifier on %s", "cuda" if device == 0 else "cpu")
    return pipeline(
        task="zero-shot-classification",
        model=CLASSIFIER_MODEL,
        device=device,
        model_kwargs=model_kwargs,
    )


def classify_single_text(text: str) -> tuple[str, float]:
    """Classify a single free-text input using the dashboard model."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return "Unknown", 0.0

    classifier = get_text_classifier()
    import torch

    with torch.no_grad():
        result = classifier(
            sequences=cleaned_text[:TEXT_ANALYZER_LIMIT],
            candidate_labels=CLASSIFIER_LABELS,
            multi_label=False,
            truncation=True,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
        )

    return result["labels"][0], round(float(result["scores"][0]), 4)


def render_metrics(global_summary: dict[str, Any]) -> None:
    """Render the top-level overview metrics."""
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total Records", f"{global_summary['total_records']:,}")
    metric_columns[1].metric("Most Common Category", global_summary["most_common_category"])
    metric_columns[2].metric("% High Severity", f"{global_summary['percentage_high_severity']}%")
    metric_columns[3].metric("Top Issue", global_summary["top_issue"])


def render_overview_page(df: pd.DataFrame, insights: dict[str, Any]) -> None:
    """Render the high-level overview dashboard page."""
    st.title("AI-Powered Rural Intelligence System")
    st.caption("Overview dashboard for rural signals, issue patterns, and severity trends.")
    render_metrics(insights["global_summary"])

    chart_col1, chart_col2 = st.columns([1.4, 1])

    with chart_col1:
        st.subheader("Category Distribution")
        category_df = (
            pd.DataFrame(insights["category_distribution"].items(), columns=["predicted_category", "count"])
            .sort_values("count", ascending=False)
        )
        category_chart = px.bar(
            category_df,
            x="predicted_category",
            y="count",
            color="predicted_category",
            text="count",
        )
        category_chart.update_layout(showlegend=False, xaxis_title="Category", yaxis_title="Records")
        st.plotly_chart(category_chart, use_container_width=True)

    with chart_col2:
        st.subheader("Severity Distribution")
        severity_df = df["severity"].value_counts().rename_axis("severity").reset_index(name="count")
        severity_chart = px.pie(
            severity_df,
            names="severity",
            values="count",
            color="severity",
            color_discrete_map=SEVERITY_COLORS,
        )
        severity_chart.update_layout(margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(severity_chart, use_container_width=True)

    st.subheader("Top 10 Issues")
    issues_df = pd.DataFrame(insights["top_issues"])
    if issues_df.empty:
        st.info("No issues available to display.")
        return

    issues_chart = px.bar(
        issues_df.sort_values("count", ascending=True),
        x="count",
        y="issue",
        orientation="h",
        text="count",
        color="count",
        color_continuous_scale="Sunset",
    )
    issues_chart.update_layout(coloraxis_showscale=False, xaxis_title="Frequency", yaxis_title="Issue")
    st.plotly_chart(issues_chart, use_container_width=True)


def render_state_analysis_page(df: pd.DataFrame) -> None:
    """Render the state analysis page with summary and charts."""
    st.title("State Analysis")
    if df.empty:
        st.warning("No data available for state analysis.")
        return

    states = sorted(df["state"].dropna().astype(str).unique().tolist())
    selected_state = st.selectbox("Select State", states)
    state_summary = get_state_summary(df, selected_state)
    state_df = df[df["state"] == selected_state].copy()

    left_col, right_col = st.columns([1.2, 1])
    with left_col:
        st.subheader("State Summary")
        st.write(state_summary["summary"])
        summary_metrics = st.columns(3)
        summary_metrics[0].metric("High Severity", state_summary["severity_distribution"]["high"])
        summary_metrics[1].metric("Moderate Severity", state_summary["severity_distribution"]["moderate"])
        summary_metrics[2].metric("Low Severity", state_summary["severity_distribution"]["low"])

        st.subheader("Top Issues")
        top_issue_df = pd.DataFrame(state_summary["key_issues"])
        if top_issue_df.empty:
            st.info("No issue records available for this state.")
        else:
            st.dataframe(top_issue_df, use_container_width=True, hide_index=True)

    with right_col:
        st.subheader("Severity Distribution")
        severity_df = pd.DataFrame(
            {
                "severity": list(state_summary["severity_distribution"].keys()),
                "count": list(state_summary["severity_distribution"].values()),
            }
        )
        severity_chart = px.bar(
            severity_df,
            x="severity",
            y="count",
            color="severity",
            text="count",
            color_discrete_map=SEVERITY_COLORS,
        )
        severity_chart.update_layout(showlegend=False, xaxis_title="Severity", yaxis_title="Records")
        st.plotly_chart(severity_chart, use_container_width=True)
        st.subheader("State Records")
        st.caption(f"{len(state_df):,} records for {selected_state}")


def render_data_explorer_page(df: pd.DataFrame) -> None:
    """Render the interactive filtered data explorer."""
    st.title("Data Explorer")
    if df.empty:
        st.warning("No records available to explore.")
        return

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    selected_states = filter_col1.multiselect("State", sorted(df["state"].dropna().astype(str).unique().tolist()))
    selected_categories = filter_col2.multiselect(
        "Category", sorted(df["predicted_category"].dropna().astype(str).unique().tolist())
    )
    selected_severities = filter_col3.multiselect(
        "Severity", sorted(df["severity"].dropna().astype(str).unique().tolist())
    )

    filtered_df = df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]
    if selected_categories:
        filtered_df = filtered_df[filtered_df["predicted_category"].isin(selected_categories)]
    if selected_severities:
        filtered_df = filtered_df[filtered_df["severity"].isin(selected_severities)]

    st.caption(f"Showing {len(filtered_df):,} records")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)


def render_text_analyzer_page() -> None:
    """Render the free-text classifier page."""
    st.title("Text Analyzer")
    st.caption("Classify new rural text using the same HuggingFace zero-shot model.")

    input_text = st.text_area(
        "Enter text to analyze",
        height=180,
        placeholder="Paste a rural development observation, survey note, or district description here...",
    )

    if st.button("Analyze Text", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text before running analysis.")
            return

        try:
            with st.spinner("Running zero-shot classification..."):
                predicted_category, confidence = classify_single_text(input_text)
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Predicted Category", predicted_category)
            metric_col2.metric("Confidence", f"{confidence:.4f}")
        except Exception as exc:
            LOGGER.error("Text classification failed: %s", exc)
            st.error(f"Text classification failed: {exc}")
            st.info("Make sure the HuggingFace model dependencies are installed and available.")


def main() -> None:
    """Run the Streamlit dashboard application."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview Dashboard", "State Analysis", "Data Explorer", "Text Analyzer"],
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        st.info("Generate or place `rural_nlp_dataset_with_predictions.csv` in the project folder.")
        return

    try:
        dataset = get_data(DATA_PATH)
        insights = get_insights(dataset)
    except Exception as exc:
        LOGGER.error("Failed to load dashboard data: %s", exc)
        st.error(f"Failed to load dashboard data: {exc}")
        return

    if page == "Overview Dashboard":
        render_overview_page(dataset, insights)
    elif page == "State Analysis":
        render_state_analysis_page(dataset)
    elif page == "Data Explorer":
        render_data_explorer_page(dataset)
    else:
        render_text_analyzer_page()


if __name__ == "__main__":
    main()
