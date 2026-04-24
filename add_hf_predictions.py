from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

try:
    import torch
    from transformers import pipeline
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Install 'transformers' and 'torch' before running this script."
    ) from exc


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


MODEL_NAME = "facebook/bart-large-mnli"
LABELS = [
    "Health",
    "Education",
    "Agriculture",
    "Infrastructure",
    "Employment",
    "Social",
]
INPUT_FILE = Path("rural_nlp_dataset.csv")
OUTPUT_FILE = Path("rural_nlp_dataset_with_predictions.csv")
CPU_BATCH_SIZE = 4
GPU_BATCH_SIZE = 8
MAX_TEXT_CHARS = 1000
HYPOTHESIS_TEMPLATE = "This text is about {}."
TEXT_COLUMN = "text"
PREDICTED_CATEGORY_COLUMN = "predicted_category"
CONFIDENCE_COLUMN = "confidence"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load the input dataset used for classification."""
    resolved_path = Path(csv_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {resolved_path}")
    LOGGER.info("Loading dataset from %s", resolved_path)
    return pd.read_csv(resolved_path)


def prepare_texts(texts: Iterable[object], max_chars: int = MAX_TEXT_CHARS) -> list[str]:
    """Normalize and truncate text inputs before inference."""
    prepared: list[str] = []
    for text in texts:
        value = "" if pd.isna(text) else str(text).strip()
        prepared.append(value[:max_chars] if len(value) > max_chars else value)
    return prepared


def _build_pipeline(model_name: str, device: int):
    """Build a zero-shot classification pipeline for the requested device."""
    pipeline_kwargs: dict[str, Any] = {
        "task": "zero-shot-classification",
        "model": model_name,
        "device": device,
    }
    if device == 0:
        pipeline_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    return pipeline(**pipeline_kwargs)


def build_classifier(model_name: str = MODEL_NAME) -> tuple[Any, int]:
    """Build the primary classifier and choose an adaptive batch size."""
    device = 0 if torch.cuda.is_available() else -1
    batch_size = GPU_BATCH_SIZE if device == 0 else CPU_BATCH_SIZE
    LOGGER.info("Loading zero-shot classifier '%s' on %s", model_name, "cuda" if device == 0 else "cpu")
    return _build_pipeline(model_name, device), batch_size


def build_cpu_classifier(model_name: str = MODEL_NAME) -> Any:
    """Build a CPU fallback classifier."""
    LOGGER.warning("Retrying classification on CPU.")
    return _build_pipeline(model_name, -1)


def classify_batch(classifier: Any, texts: list[str], labels: list[str]) -> tuple[list[str], list[float]]:
    """Run zero-shot classification for a batch of texts."""
    with torch.no_grad():
        results = classifier(
            sequences=texts,
            candidate_labels=labels,
            multi_label=False,
            truncation=True,
            hypothesis_template=HYPOTHESIS_TEMPLATE,
        )

    result_list = [results] if isinstance(results, dict) else results
    predicted_categories = [result["labels"][0] for result in result_list]
    confidences = [round(float(result["scores"][0]), 4) for result in result_list]
    return predicted_categories, confidences


def add_predictions(
    df: pd.DataFrame,
    classifier: Any,
    labels: list[str],
    batch_size: int,
) -> pd.DataFrame:
    """Add predicted category and confidence columns to the dataset."""
    if TEXT_COLUMN not in df.columns:
        raise KeyError(f"Required column missing: {TEXT_COLUMN}")
    if df.empty:
        LOGGER.warning("Received an empty dataset. Returning without inference.")
        empty_df = df.copy()
        empty_df[PREDICTED_CATEGORY_COLUMN] = pd.Series(dtype="object")
        empty_df[CONFIDENCE_COLUMN] = pd.Series(dtype="float64")
        return empty_df

    updated_df = df.copy()
    texts = prepare_texts(updated_df[TEXT_COLUMN].tolist())
    predicted_categories = ["Unknown"] * len(texts)
    confidences = [0.0] * len(texts)

    for start in tqdm(range(0, len(texts), batch_size), desc="Classifying", unit="batch"):
        end = min(start + batch_size, len(texts))
        indexed_batch = [(row_idx, texts[row_idx]) for row_idx in range(start, end) if texts[row_idx]]
        if not indexed_batch:
            continue

        batch_input = [text for _, text in indexed_batch]
        try:
            batch_predictions, batch_confidences = classify_batch(classifier, batch_input, labels)
        except Exception as exc:
            LOGGER.error("Batch %s-%s failed on current device: %s", start + 1, end, exc)
            if torch.cuda.is_available():
                try:
                    classifier = build_cpu_classifier()
                    batch_predictions, batch_confidences = classify_batch(classifier, batch_input, labels)
                except Exception as cpu_exc:
                    LOGGER.error("CPU retry failed for batch %s-%s: %s", start + 1, end, cpu_exc)
                    continue
            else:
                continue

        for (row_idx, _), prediction, confidence in zip(indexed_batch, batch_predictions, batch_confidences):
            predicted_categories[row_idx] = prediction
            confidences[row_idx] = confidence

    updated_df[PREDICTED_CATEGORY_COLUMN] = predicted_categories
    updated_df[CONFIDENCE_COLUMN] = confidences
    return updated_df


def save_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist the dataset with classifier outputs to disk."""
    LOGGER.info("Saving predictions to %s", output_path)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run batch zero-shot classification over the rural NLP dataset."""
    dataset = load_dataset(INPUT_FILE)
    classifier, batch_size = build_classifier()
    predicted_df = add_predictions(dataset, classifier, LABELS, batch_size=batch_size)
    save_dataset(predicted_df, OUTPUT_FILE)

    LOGGER.info("Prediction run complete.")
    LOGGER.info(
        "Sample predictions: %s",
        predicted_df[[TEXT_COLUMN, "category", PREDICTED_CATEGORY_COLUMN, CONFIDENCE_COLUMN]].head(5).to_dict(
            orient="records"
        ),
    )


if __name__ == "__main__":
    main()
