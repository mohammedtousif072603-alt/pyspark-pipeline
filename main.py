"""Generate synthetic e-commerce data and save it as Parquet files."""

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src import config
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format a file size in bytes as a readable unit."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def save_datasets(datasets: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    """Save generated pandas DataFrames to individual Parquet files.

    Args:
        datasets: Mapping of dataset names to pandas DataFrames.
        output_dir: Destination directory for the Parquet files.

    Returns:
        Mapping from dataset names to saved Parquet paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for name in ("customers", "products", "orders"):
        if name not in datasets:
            raise KeyError(f"Generated dataset is missing the required key: {name}")
        path = output_dir / f"{name}.parquet"
        datasets[name].to_parquet(path, index=False)
        output_paths[name] = path
    return output_paths


def main() -> int:
    """Generate all datasets, save them, and report time and file sizes."""
    started_at = time.perf_counter()
    try:
        generator = SyntheticDataGenerator(seed=config.DEFAULT_RANDOM_SEED)
        datasets = generator.generate_all()
        output_paths = save_datasets(datasets, config.RAW_DATA_DIR)
        elapsed_seconds = time.perf_counter() - started_at

        print(f"Data generation and Parquet export completed in {elapsed_seconds:.2f} seconds.")
        for name, path in output_paths.items():
            print(f"{name.capitalize()}: {path} ({format_file_size(path.stat().st_size)})")
        LOGGER.info("Successfully generated and saved all datasets to %s", config.RAW_DATA_DIR)
        return 0
    except Exception as error:
        LOGGER.exception("Failed to generate or save synthetic e-commerce data")
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
