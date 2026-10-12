"""Basic tests for synthetic dataset generation."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.data_generator import generate_data


class DataGeneratorTests(unittest.TestCase):
    """Check generated files, row counts, and referential integrity."""

    def test_generates_linked_csv_files(self) -> None:
        """Generate a small dataset and check order references."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = generate_data(3, 2, 8, seed=7, output_dir=Path(temporary_directory))
            with paths["customers"].open(encoding="utf-8", newline="") as source:
                customers = list(csv.DictReader(source))
            with paths["products"].open(encoding="utf-8", newline="") as source:
                products = list(csv.DictReader(source))
            with paths["orders"].open(encoding="utf-8", newline="") as source:
                orders = list(csv.DictReader(source))

            self.assertEqual(len(customers), 3)
            self.assertEqual(len(products), 2)
            self.assertEqual(len(orders), 8)
            self.assertTrue({order["customer_id"] for order in orders} <= {row["customer_id"] for row in customers})
            self.assertTrue({order["product_id"] for order in orders} <= {row["product_id"] for row in products})


if __name__ == "__main__":
    unittest.main()
