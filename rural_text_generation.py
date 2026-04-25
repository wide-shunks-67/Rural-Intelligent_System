from __future__ import annotations

import logging

from data_loader import DATA_PATHS, load_mgnrega, load_nfhs, load_pmgsy, load_sanitation
from dataset_builder import create_final_dataset
from text_generators import (
    generate_mgnrega_text,
    generate_nfhs_text,
    generate_pmgsy_text,
    generate_sanitation_text,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Orchestrate sample generation and final dataset creation."""
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


if __name__ == "__main__":
    main()
