"""Shared paths and defaults for the e-commerce data pipeline."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DATA_DIR = Path(
    os.getenv("PROCESSED_DATA_DIR", PROJECT_ROOT / "data" / "processed")
)
CUSTOMERS_FILE = RAW_DATA_DIR / "customers.csv"
PRODUCTS_FILE = RAW_DATA_DIR / "products.csv"
ORDERS_FILE = RAW_DATA_DIR / "orders.csv"
DEFAULT_CUSTOMER_COUNT = 200
DEFAULT_PRODUCT_COUNT = 50
DEFAULT_ORDER_COUNT = 1_000
DEFAULT_RANDOM_SEED = 42
SPARK_APP_NAME = "ecommerce-data-pipeline"
