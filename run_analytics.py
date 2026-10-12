"""Run and time the e-commerce PySpark analytics."""

import logging
import time

from pyspark.sql import DataFrame

from src import config
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def _display_timed_result(operation_name: str, result: DataFrame) -> float:
    """Display a Spark DataFrame and return the elapsed action time.

    Args:
        operation_name: Human-readable name printed before the result.
        result: Spark DataFrame to display.

    Returns:
        Seconds spent evaluating and displaying the DataFrame.
    """
    started_at = time.perf_counter()
    print(f"\n{operation_name}")
    result.show(truncate=False)
    elapsed_seconds = time.perf_counter() - started_at
    print(f"{operation_name} execution time: {elapsed_seconds:.2f} seconds")
    return elapsed_seconds


def main() -> int:
    """Load the Parquet datasets, run all analytics, and stop Spark."""
    analytics = SalesAnalytics()
    try:
        analytics.create_spark_session()

        load_started_at = time.perf_counter()
        customers_df = analytics.load_parquet(config.RAW_DATA_DIR / "customers.parquet")
        products_df = analytics.load_parquet(config.RAW_DATA_DIR / "products.parquet")
        orders_df = analytics.load_parquet(config.RAW_DATA_DIR / "orders.parquet")
        print(f"Loaded all Parquet datasets in {time.perf_counter() - load_started_at:.2f} seconds")
        LOGGER.info("Loaded customers, products, and orders Parquet datasets")
        # The customer dataset is loaded here as requested; sales aggregations use order and product rows.
        LOGGER.info("Customer dataset schema: %s", customers_df.schema.simpleString())

        _display_timed_result(
            "Top customers by revenue",
            analytics.top_customers_by_revenue(orders_df, products_df),
        )
        _display_timed_result(
            "Sales by category",
            analytics.sales_by_category(orders_df, products_df),
        )
        _display_timed_result(
            "Monthly revenue trends",
            analytics.monthly_trends(orders_df, products_df),
        )
        return 0
    except Exception:
        LOGGER.exception("Analytics run failed")
        return 1
    finally:
        analytics.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
