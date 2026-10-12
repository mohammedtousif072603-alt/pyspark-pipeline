"""Compare Pandas and PySpark sales aggregation performance on Parquet data."""

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src import config
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def _format_seconds(seconds: float) -> str:
    """Format elapsed seconds for a comparison table."""
    return f"{seconds:.3f} s"


def _get_price_column(products: pd.DataFrame) -> str:
    """Return the product price column supported by this input dataset."""
    if "price" in products.columns:
        return "price"
    if "unit_price" in products.columns:
        return "unit_price"
    raise ValueError("products.parquet must contain a 'price' or 'unit_price' column")


def _format_table(rows: list[tuple[str, float, float]]) -> str:
    """Render operation timings as an aligned text table."""
    headers = ("Operation", "Pandas", "PySpark")
    rendered_rows = [
        (operation, _format_seconds(pandas_time), _format_seconds(spark_time))
        for operation, pandas_time, spark_time in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered_rows))
        for index in range(len(headers))
    ]
    lines = [" | ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(
        " | ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in rendered_rows
    )
    return "\n".join(lines)


def _load_pandas(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Load order and product Parquet files into Pandas DataFrames."""
    started_at = time.perf_counter()
    orders = pd.read_parquet(raw_dir / "orders.parquet")
    products = pd.read_parquet(raw_dir / "products.parquet")
    elapsed = time.perf_counter() - started_at
    LOGGER.info("Pandas loaded %s orders and %s products", f"{len(orders):,}", f"{len(products):,}")
    return orders, products, elapsed


def _load_spark(
    spark: SparkSession,
    raw_dir: Path,
) -> tuple[DataFrame, DataFrame, float]:
    """Load and cache order and product Parquet data in Spark."""
    started_at = time.perf_counter()
    orders = spark.read.parquet(str(raw_dir / "orders.parquet")).cache()
    products = spark.read.parquet(str(raw_dir / "products.parquet")).cache()
    # Spark reads lazily, so count forces file scans inside the measured load step.
    order_count = orders.count()
    product_count = products.count()
    elapsed = time.perf_counter() - started_at
    LOGGER.info("PySpark loaded %s orders and %s products", f"{order_count:,}", f"{product_count:,}")
    return orders, products, elapsed


def _pandas_join_and_revenue(
    orders: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Join Pandas orders to products and calculate line revenue."""
    price_column = _get_price_column(products)
    started_at = time.perf_counter()
    joined = orders.merge(products[["product_id", price_column]], on="product_id", how="inner")
    joined["revenue"] = joined["quantity"] * joined[price_column]
    elapsed = time.perf_counter() - started_at
    LOGGER.info("Pandas joined data into %s rows", f"{len(joined):,}")
    return joined, elapsed


def _spark_join_and_revenue(
    orders: DataFrame,
    products: DataFrame,
) -> tuple[DataFrame, float]:
    """Join Spark orders to products and force revenue computation."""
    price_column = "price" if "price" in products.columns else "unit_price"
    if price_column not in products.columns:
        raise ValueError("products.parquet must contain a 'price' or 'unit_price' column")
    started_at = time.perf_counter()
    joined = (
        orders.join(products.select("product_id", price_column), on="product_id", how="inner")
        .withColumn("revenue", F.col("quantity") * F.col(price_column))
        .cache()
    )
    # Aggregate revenue to force Spark to execute and materialize the join.
    joined.agg(F.sum("revenue").alias("revenue_check")).collect()
    elapsed = time.perf_counter() - started_at
    LOGGER.info("PySpark joined data and materialized the revenue calculation")
    return joined, elapsed


def _pandas_top_customers(joined: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Group Pandas rows by customer and return the top ten by revenue."""
    started_at = time.perf_counter()
    top_customers = (
        joined.groupby("customer_id", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .nlargest(10, "total_revenue")
        .reset_index(drop=True)
    )
    elapsed = time.perf_counter() - started_at
    return top_customers, elapsed


def _spark_top_customers(joined: DataFrame) -> tuple[list[Any], float]:
    """Group Spark rows by customer and collect the top ten by revenue."""
    started_at = time.perf_counter()
    top_customers = (
        joined.groupBy("customer_id")
        .agg(F.sum("revenue").alias("total_revenue"))
        .orderBy(F.desc("total_revenue"))
        .limit(10)
        .collect()
    )
    elapsed = time.perf_counter() - started_at
    return top_customers, elapsed


def run_comparison(raw_dir: Path = config.RAW_DATA_DIR) -> None:
    """Run the Pandas and PySpark benchmarks and print their timing table."""
    analytics = SalesAnalytics()
    try:
        analytics.create_spark_session(app_name="pandas-vs-pyspark-benchmark")
        if analytics.spark is None:
            raise RuntimeError("Spark session could not be initialized")
        spark_orders, spark_products, spark_load_time = _load_spark(analytics.spark, raw_dir)
        pandas_orders, pandas_products, pandas_load_time = _load_pandas(raw_dir)

        pandas_joined, pandas_join_time = _pandas_join_and_revenue(pandas_orders, pandas_products)
        spark_joined, spark_join_time = _spark_join_and_revenue(spark_orders, spark_products)

        pandas_top, pandas_aggregate_time = _pandas_top_customers(pandas_joined)
        spark_top, spark_aggregate_time = _spark_top_customers(spark_joined)

        expected_rows = 1_000_000
        actual_rows = len(pandas_orders)
        print(f"\nOrders benchmarked: {actual_rows:,} (target: {expected_rows:,})")
        if actual_rows != expected_rows:
            print("Note: the input order row count differs from the requested 1M-row benchmark.")

        timing_rows = [
            ("Load Parquet", pandas_load_time, spark_load_time),
            ("Join + calculate revenue", pandas_join_time, spark_join_time),
            ("Group by customer + top 10", pandas_aggregate_time, spark_aggregate_time),
            (
                "Total measured time",
                pandas_load_time + pandas_join_time + pandas_aggregate_time,
                spark_load_time + spark_join_time + spark_aggregate_time,
            ),
        ]
        print("\nPandas vs PySpark performance")
        print(_format_table(timing_rows))
        print("\nTop 10 customers (Pandas):")
        print(pandas_top.to_string(index=False))
        print("\nTop 10 customers (PySpark):")
        for row in spark_top:
            print(f"{row['customer_id']}  {row['total_revenue']:.2f}")
    finally:
        analytics.stop()


def main() -> int:
    """Run the benchmark and report failures with a nonzero exit status."""
    try:
        run_comparison()
        return 0
    except Exception:
        LOGGER.exception("Pandas vs PySpark benchmark failed")
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
