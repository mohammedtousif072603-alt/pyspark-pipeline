"""PySpark analytics for synthetic e-commerce sales data."""

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src import config

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Load Parquet data and calculate e-commerce sales summaries with Spark."""

    def __init__(self, spark: SparkSession | None = None) -> None:
        """Initialize analytics, optionally reusing an existing Spark session."""
        self.spark = spark

    def create_spark_session(self, app_name: str = "ecommerce-sales-analytics") -> SparkSession:
        """Create or reuse a local Spark session with 4 GB memory and Kryo.

        Adaptive query execution is enabled to let Spark optimize query plans
        using runtime statistics. The memory settings configure the local driver
        and executor processes.

        Args:
            app_name: Name displayed for this Spark application.

        Returns:
            The configured SparkSession.
        """
        self.spark = (
            SparkSession.builder.appName(app_name)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        LOGGER.info("Created Spark session with local mode, 4 GB memory, AQE, and Kryo serialization")
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet file or directory into a Spark DataFrame.

        Args:
            path: Local path or Spark-supported URI to the Parquet source.

        Returns:
            DataFrame loaded from the Parquet source.
        """
        if self.spark is None:
            self.create_spark_session()
        source_path = str(path)
        LOGGER.info("Loading Parquet data from %s", source_path)
        return self.spark.read.parquet(source_path)

    @staticmethod
    def _revenue_column(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join order lines to products and add a revenue column."""
        price_column = "price" if "price" in products_df.columns else "unit_price"
        if price_column not in products_df.columns:
            raise ValueError("products_df must contain either a 'price' or 'unit_price' column")
        required_order_columns = {"product_id", "quantity"}
        missing_order_columns = required_order_columns.difference(orders_df.columns)
        if missing_order_columns:
            raise ValueError(f"orders_df is missing required columns: {sorted(missing_order_columns)}")
        if "product_id" not in products_df.columns:
            raise ValueError("products_df must contain a 'product_id' column")
        return (
            orders_df.join(products_df.select("product_id", price_column, *(["category"] if "category" in products_df.columns else [])),
                           on="product_id", how="inner")
            .withColumn("revenue", F.col(price_column) * F.col("quantity"))
        )

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return customers with the highest total order revenue.

        Args:
            orders_df: Orders with customer_id, product_id, and quantity.
            products_df: Products with product_id and price (or unit_price).
            n: Maximum number of customers to return.

        Returns:
            DataFrame with customer_id, total_revenue, and order_count.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")
        if "customer_id" not in orders_df.columns:
            raise ValueError("orders_df must contain a 'customer_id' column")
        revenue_orders = self._revenue_column(orders_df, products_df)
        return (
            revenue_orders.groupBy("customer_id")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.countDistinct("order_id").alias("order_count")
                if "order_id" in revenue_orders.columns
                else F.count(F.lit(1)).alias("order_count"),
            )
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Aggregate revenue and units sold for each product category.

        Args:
            orders_df: Orders with product_id and quantity.
            products_df: Products with product_id, category, and price.

        Returns:
            DataFrame with category, total_revenue, and units_sold.
        """
        if "category" not in products_df.columns:
            raise ValueError("products_df must contain a 'category' column")
        revenue_orders = self._revenue_column(orders_df, products_df)
        return (
            revenue_orders.groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate monthly revenue and month-over-month revenue growth.

        Args:
            orders_df: Orders with product_id, quantity, and order_date.
            products_df: Products with product_id and price (or unit_price).

        Returns:
            DataFrame with month, revenue, previous_month_revenue, and
            growth_percentage. Growth is null when the prior month is absent or
            has zero revenue.
        """
        if "order_date" not in orders_df.columns:
            raise ValueError("orders_df must contain an 'order_date' column")
        revenue_orders = self._revenue_column(orders_df, products_df)
        monthly = (
            revenue_orders.withColumn("month_start", F.trunc(F.to_date("order_date"), "month"))
            .groupBy("month_start")
            .agg(F.sum("revenue").alias("revenue"))
        )
        month_window = Window.orderBy("month_start")
        with_previous = monthly.withColumn(
            "previous_month_revenue", F.lag("revenue").over(month_window)
        )
        growth = F.when(
            F.col("previous_month_revenue").isNotNull() & (F.col("previous_month_revenue") != 0),
            (F.col("revenue") - F.col("previous_month_revenue"))
            / F.col("previous_month_revenue")
            * F.lit(100.0),
        )
        return (
            with_previous.withColumn("growth_percentage", F.round(growth, 2))
            .withColumn("month", F.date_format("month_start", "yyyy-MM"))
            .select(
                "month",
                F.round("revenue", 2).alias("revenue"),
                F.round("previous_month_revenue", 2).alias("previous_month_revenue"),
                "growth_percentage",
            )
            .orderBy("month")
        )

    def stop(self) -> None:
        """Stop this instance's Spark session if one has been created."""
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
            LOGGER.info("Stopped Spark session")


def main() -> None:
    """Load generated Parquet files and display the requested sales summaries."""
    analytics = SalesAnalytics()
    try:
        analytics.create_spark_session()
        orders_df = analytics.load_parquet(config.RAW_DATA_DIR / "orders.parquet")
        products_df = analytics.load_parquet(config.RAW_DATA_DIR / "products.parquet")
        LOGGER.info("Top customers by revenue")
        analytics.top_customers_by_revenue(orders_df, products_df).show(truncate=False)
        LOGGER.info("Sales by category")
        analytics.sales_by_category(orders_df, products_df).show(truncate=False)
        LOGGER.info("Monthly sales trends")
        analytics.monthly_trends(orders_df, products_df).show(truncate=False)
    finally:
        analytics.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
