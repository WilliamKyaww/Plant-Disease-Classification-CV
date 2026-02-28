import csv
import tempfile
import unittest

from src.datasets import PlantDiseaseDataset


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestDatasetSchema(unittest.TestCase):
    def test_missing_label_column_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = f"{tmp}/data.csv"
            _write_csv(
                csv_path,
                fieldnames=["image_path", "binary_label"],
                rows=[{"image_path": "Datasets/a.jpg", "binary_label": 1}],
            )

            with self.assertRaises(ValueError):
                PlantDiseaseDataset(csv_path, label_column="class_label")

    def test_valid_label_column_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = f"{tmp}/data.csv"
            _write_csv(
                csv_path,
                fieldnames=["image_path", "class_label"],
                rows=[{"image_path": "Datasets/a.jpg", "class_label": 2}],
            )

            ds = PlantDiseaseDataset(csv_path, label_column="class_label")
            self.assertEqual(len(ds), 1)


if __name__ == "__main__":
    unittest.main()
