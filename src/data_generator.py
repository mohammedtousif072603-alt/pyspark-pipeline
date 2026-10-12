"""Create synthetic e-commerce data as pandas DataFrames."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm

from src import config

LOGGER = logging.getLogger(__name__)
PRODUCT_CATEGORIES = ("Electronics", "Clothing", "Home", "Sports", "Books")


class SyntheticDataGenerator:
    """Generate synthetic customers, products, and orders for e-commerce analysis.

    The order generator assigns 80% of orders to a Pareto-weighted group
    containing 20% of customers, then distributes the remaining orders over
    the other customers. A seeded NumPy generator makes results reproducible.
    """

    def __init__(
        self,
        num_customers: int = 100_000,
        num_products: int = 10_000,
        num_orders: int = 1_000_000,
        seed: int = 42,
        progress: bool = True,
    ) -> None:
        """Initialize generator sizes, random sources, and progress settings.

        Args:
            num_customers: Number of customers to generate; must be positive.
            num_products: Number of products to generate; must be positive.
            num_orders: Number of orders to generate; cannot be negative.
            seed: Seed for NumPy and Faker random generation.
            progress: Whether to display tqdm progress bars.
        """
        if num_customers <= 0 or num_products <= 0 or num_orders < 0:
            raise ValueError("Customer and product counts must be positive; order count cannot be negative.")
        self.num_customers = num_customers
        self.num_products = num_products
        self.num_orders = num_orders
        self.seed = seed
        self.progress = progress
        self.rng = np.random.default_rng(seed)
        self.fake = Faker()
        self.fake.seed_instance(seed)
        LOGGER.info(
            "Initialized generator: %d customers, %d products, %d orders, seed=%d",
            num_customers,
            num_products,
            num_orders,
            seed,
        )

    def generate_customers(self) -> pd.DataFrame:
        """Generate customers with normally distributed ages centered near 35.

        Returns:
            DataFrame with customer_id, name, email, age, city, country, and
            registration_date columns.
        """
        indexes = range(self.num_customers)
        iterator = tqdm(indexes, desc="Generating customers", disable=not self.progress)
        names: list[str] = []
        emails: list[str] = []
        cities: list[str] = []
        countries: list[str] = []
        dates: list[str] = []
        for _ in iterator:
            names.append(self.fake.name())
            emails.append(self.fake.unique.email())
            cities.append(self.fake.city())
            countries.append(self.fake.country())
            dates.append(self.fake.date_between(start_date="-5y", end_date="today").isoformat())

        ages = np.clip(np.rint(self.rng.normal(loc=35, scale=12, size=self.num_customers)), 18, 90).astype(int)
        customers = pd.DataFrame(
            {
                "customer_id": [f"C{index:07d}" for index in range(1, self.num_customers + 1)],
                "name": names,
                "email": emails,
                "age": ages,
                "city": cities,
                "country": countries,
                "registration_date": pd.to_datetime(dates),
            }
        )
        LOGGER.info("Generated %d customers", len(customers))
        return customers

    def generate_products(self) -> pd.DataFrame:
        """Generate products with categories, prices, stock, and ratings.

        Returns:
            DataFrame with product_id, name, category, price, stock, and rating
            columns.
        """
        iterator = tqdm(range(self.num_products), desc="Generating products", disable=not self.progress)
        names = [f"{self.fake.word().title()} {self.fake.word().title()}" for _ in iterator]
        products = pd.DataFrame(
            {
                "product_id": [f"P{index:06d}" for index in range(1, self.num_products + 1)],
                "name": names,
                "category": self.rng.choice(PRODUCT_CATEGORIES, size=self.num_products),
                "price": np.round(self.rng.uniform(10, 500, size=self.num_products), 2),
                "stock": self.rng.integers(0, 1_001, size=self.num_products),
                "rating": np.round(self.rng.uniform(1, 5, size=self.num_products), 1),
            }
        )
        LOGGER.info("Generated %d products", len(products))
        return products

    def _sample_customer_indexes(self) -> np.ndarray:
        """Sample order owners with an 80/20 Pareto-weighted customer split."""
        if self.num_orders == 0:
            return np.array([], dtype=np.int64)
        top_count = min(self.num_customers, max(1, int(np.ceil(self.num_customers * 0.2))))
        top_orders = min(self.num_orders, int(round(self.num_orders * 0.8)))
        top_weights = self.rng.pareto(a=1.16, size=top_count) + 1.0
        top_probabilities = top_weights / top_weights.sum()
        top_indexes = self.rng.choice(top_count, size=top_orders, p=top_probabilities)

        tail_count = self.num_customers - top_count
        tail_orders = self.num_orders - top_orders
        if tail_count == 0:
            return top_indexes
        tail_weights = self.rng.pareto(a=1.16, size=tail_count) + 1.0
        tail_probabilities = tail_weights / tail_weights.sum()
        tail_indexes = self.rng.choice(
            np.arange(top_count, self.num_customers),
            size=tail_orders,
            p=tail_probabilities,
        )
        customer_indexes = np.concatenate((top_indexes, tail_indexes))
        self.rng.shuffle(customer_indexes)
        return customer_indexes

    def generate_orders(self) -> pd.DataFrame:
        """Generate orders linked to generated customer and product IDs.

        Returns:
            DataFrame with order_id, customer_id, product_id, quantity, and
            order_date columns.
        """
        customer_indexes = self._sample_customer_indexes()
        order_indexes = np.arange(1, self.num_orders + 1)
        day_offsets = self.rng.integers(0, 5 * 365 + 1, size=self.num_orders)
        product_indexes = self.rng.integers(1, self.num_products + 1, size=self.num_orders)
        customer_ids = np.empty(self.num_orders, dtype=object)
        product_ids = np.empty(self.num_orders, dtype=object)
        order_ids = np.empty(self.num_orders, dtype=object)
        chunk_size = 100_000
        chunks = range(0, self.num_orders, chunk_size)
        for start in tqdm(chunks, desc="Formatting orders", disable=not self.progress):
            stop = min(start + chunk_size, self.num_orders)
            customer_ids[start:stop] = [f"C{index + 1:07d}" for index in customer_indexes[start:stop]]
            product_ids[start:stop] = [f"P{index:06d}" for index in product_indexes[start:stop]]
            order_ids[start:stop] = [f"O{index:09d}" for index in order_indexes[start:stop]]
        start_date = np.datetime64(pd.Timestamp.today().normalize() - pd.Timedelta(days=5 * 365), "D")
        orders = pd.DataFrame(
            {
                "order_id": order_ids,
                "customer_id": customer_ids,
                "product_id": product_ids,
                "quantity": self.rng.integers(1, 11, size=self.num_orders),
                "order_date": pd.to_datetime(start_date + day_offsets.astype("timedelta64[D]")),
            }
        )
        LOGGER.info("Generated %d orders", len(orders))
        return orders

    def generate_all(self) -> dict[str, pd.DataFrame]:
        """Generate and return all three datasets as pandas DataFrames.

        Returns:
            Dictionary with ``customers``, ``products``, and ``orders`` keys.
        """
        LOGGER.info("Starting synthetic e-commerce data generation")
        datasets = {
            "customers": self.generate_customers(),
            "products": self.generate_products(),
            "orders": self.generate_orders(),
        }
        LOGGER.info("Finished generating all synthetic datasets")
        return datasets


def generate_data(
    customer_count: int = config.DEFAULT_CUSTOMER_COUNT,
    product_count: int = config.DEFAULT_PRODUCT_COUNT,
    order_count: int = config.DEFAULT_ORDER_COUNT,
    seed: int = config.DEFAULT_RANDOM_SEED,
    output_dir: Path = config.RAW_DATA_DIR,
) -> dict[str, Path]:
    """Generate CSV files using the class while preserving the project CLI API.

    The CSV shape is adapted to the existing Spark analytics module.

    Args:
        customer_count: Number of customers to create.
        product_count: Number of products to create.
        order_count: Number of orders to create.
        seed: Random seed.
        output_dir: Output directory for CSV files.

    Returns:
        Mapping from dataset names to generated CSV paths.
    """
    generator = SyntheticDataGenerator(customer_count, product_count, order_count, seed)
    datasets = generator.generate_all()
    customers = datasets["customers"].copy()
    name_parts = customers["name"].str.split(" ", n=1, expand=True)
    customers["first_name"] = name_parts[0]
    customers["last_name"] = name_parts[1].fillna("")
    customers = customers.rename(columns={"registration_date": "signup_date"})[
        ["customer_id", "first_name", "last_name", "email", "city", "country", "signup_date"]
    ]
    products = datasets["products"].rename(columns={"name": "product_name", "price": "unit_price"})[
        ["product_id", "product_name", "category", "unit_price"]
    ]
    orders = datasets["orders"].copy()
    orders["status"] = "completed"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "customers": output_dir / "customers.csv",
        "products": output_dir / "products.csv",
        "orders": output_dir / "orders.csv",
    }
    customers.to_csv(paths["customers"], index=False)
    products.to_csv(paths["products"], index=False)
    orders.to_csv(paths["orders"], index=False)
    for name, path in paths.items():
        LOGGER.info("Wrote %s dataset to %s", name, path)
    return paths


def main() -> None:
    """Parse command-line arguments and write generated CSV datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customers", type=int, default=config.DEFAULT_CUSTOMER_COUNT)
    parser.add_argument("--products", type=int, default=config.DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--orders", type=int, default=config.DEFAULT_ORDER_COUNT)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=config.RAW_DATA_DIR)
    args = parser.parse_args()
    generate_data(args.customers, args.products, args.orders, args.seed, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
