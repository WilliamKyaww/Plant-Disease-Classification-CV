import unittest

import pandas as pd

from src.prepare_splits import verify_no_leakage


class TestSplitLeakage(unittest.TestCase):
    def test_no_overlap_returns_true(self):
        train_df = pd.DataFrame({"image_path": ["Datasets/a.jpg", "Datasets/b.jpg"]})
        val_df = pd.DataFrame({"image_path": ["Datasets/c.jpg"]})
        test_df = pd.DataFrame({"image_path": ["Datasets/d.jpg"]})

        self.assertTrue(verify_no_leakage(train_df, val_df, test_df))

    def test_overlap_returns_false(self):
        train_df = pd.DataFrame({"image_path": ["Datasets/a.jpg", "Datasets/b.jpg"]})
        val_df = pd.DataFrame({"image_path": ["Datasets/b.jpg"]})
        test_df = pd.DataFrame({"image_path": ["Datasets/d.jpg"]})

        self.assertFalse(verify_no_leakage(train_df, val_df, test_df))


if __name__ == "__main__":
    unittest.main()
