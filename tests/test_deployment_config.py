import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import prediction


class ModelSeedingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.bundled_model = self.root / "bundled" / "demand_model.joblib"
        self.bundled_model.parent.mkdir()
        shutil.copyfile(prediction.BUNDLED_MODEL_PATH, self.bundled_model)
        self.persistent_model = self.root / "volume" / "models" / "demand_model.joblib"
        self.data_dir = self.root / "volume"

        self.patches = [
            patch.object(prediction, "BUNDLED_MODEL_PATH", self.bundled_model),
            patch.object(prediction, "MODEL_PATH", self.persistent_model),
            patch.object(prediction, "DATA_DIR", self.data_dir),
        ]
        for item in self.patches:
            item.start()
        prediction.clear_model_cache()

    def tearDown(self):
        prediction.clear_model_cache()
        for item in reversed(self.patches):
            item.stop()
        self.temp_dir.cleanup()

    def test_seed_model_is_copied_atomically_to_persistent_data_dir(self):
        self.assertTrue(prediction.ensure_model_available())
        self.assertTrue(self.persistent_model.is_file())
        self.assertEqual(
            self.persistent_model.read_bytes(),
            self.bundled_model.read_bytes(),
        )
        self.assertEqual(
            prediction.get_model_bundle()["model_config"]["type"],
            "MLPRegressor",
        )

    def test_existing_persistent_model_is_not_overwritten(self):
        self.persistent_model.parent.mkdir(parents=True)
        self.persistent_model.write_bytes(b"existing-model")

        self.assertTrue(prediction.ensure_model_available())
        self.assertEqual(self.persistent_model.read_bytes(), b"existing-model")


if __name__ == "__main__":
    unittest.main()
