# E-commerce Data Pipeline

A small Python project that generates linked fake customer, product, and order data, then uses PySpark to produce business summaries. It is intended for learning, local development, and testing; it does not connect to a live store.

## Project structure

```text
.
├── data/
│   ├── raw/                 # Generated CSV input (created when run)
│   └── processed/           # Spark CSV output (created when run)
├── notebooks/               # Space for exploratory Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   └── test_data_generator.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Use Python 3.10 or newer and a working Java installation supported by your chosen PySpark release. From the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the pipeline

Generate 1,000 orders (and their linked customers/products):

```bash
python -m src.data_generator --orders 1000 --customers 200 --products 50 --seed 42
```

Analyze the generated CSV files with local Spark:

```bash
python -m src.spark_analytics
```

Spark writes these CSV output directories under `data/processed/`: `monthly_revenue/`, `category_revenue/`, `top_products/`, and `customer_spend/`. Each is a Spark output directory containing one or more part files. The pipeline can be rerun; Spark replaces existing output directories. To choose a different raw or processed location, set `RAW_DATA_DIR` or `PROCESSED_DATA_DIR` before running.

The order dataset has one product per order, with quantity, order date, and status. Revenue summaries include completed orders only. The generated records are synthetic and use a fixed seed by default for repeatability.

## Run the generator checks

```bash
python -m unittest discover -s tests -v
```

## Dependencies

See [requirements.txt](requirements.txt). PySpark requires Java; install and configure Java separately for your operating system if it is not already available.
