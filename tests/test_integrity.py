import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src import integrity


def _save_image(path: Path, color):
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path)


class TestIntegrity(unittest.TestCase):
    def test_missing_folder_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            present = "ClassA"
            missing = "ClassB"
            (tmp_path / present).mkdir(parents=True, exist_ok=True)
            _save_image(tmp_path / present / "img1.jpg", color=(255, 255, 255))

            fake_metadata = {present: {}, missing: {}}
            with patch.object(integrity, "DATASETS_DIR", str(tmp_path)):
                with patch.object(integrity, "FOLDER_METADATA", fake_metadata):
                    issues = integrity.check_missing_folders()

            self.assertTrue(any(missing in issue for issue in issues))

    def test_near_duplicate_detection_finds_cross_class_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            class_a = "ClassA"
            class_b = "ClassB"
            (tmp_path / class_a).mkdir(parents=True, exist_ok=True)
            (tmp_path / class_b).mkdir(parents=True, exist_ok=True)

            # Identical images across classes should be detected as near-duplicates.
            _save_image(tmp_path / class_a / "img_a.jpg", color=(20, 180, 20))
            _save_image(tmp_path / class_b / "img_b.jpg", color=(20, 180, 20))

            fake_metadata = {class_a: {}, class_b: {}}
            with patch.object(integrity, "DATASETS_DIR", str(tmp_path)):
                with patch.object(integrity, "FOLDER_METADATA", fake_metadata):
                    result = integrity.check_near_duplicates_across_classes(
                        max_distance=0,
                        hash_size=8,
                    )

            self.assertGreaterEqual(result["total_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
