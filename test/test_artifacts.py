import json
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation import MODEL_FILES, evaluate_model, read_csv, validate_data


ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_readme_matches_reported_metrics(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        metadata = json.loads((ROOT / "model" / "metadata.json").read_text())
        for result in metadata["metrics"]:
            row = "| " + result["ML Model Name"] + " | " + " | ".join(
                f"{result[metric]:.4f}" for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
            ) + " |"
            self.assertIn(row, readme)

    def test_saved_models_reproduce_metrics(self):
        metadata = json.loads((ROOT / "model" / "metadata.json").read_text())
        features, target = validate_data(read_csv(ROOT / "test_data.csv"))
        for result in metadata["metrics"]:
            name = result["ML Model Name"]
            with self.subTest(model=name):
                model = joblib.load(ROOT / "model" / MODEL_FILES[name])
                metrics, _, probabilities = evaluate_model(model, features, target)
                self.assertEqual(list(model.feature_names_in_), list(features.columns))
                self.assertEqual(probabilities.shape, (272, 6))
                for metric, value in metrics.items():
                    self.assertAlmostEqual(value, result[metric], places=12)

    def test_stratified_test_split_is_disjoint(self):
        dataset = read_csv(ROOT / "wine+quality" / "winequality-red.csv").drop_duplicates().reset_index(drop=True)
        train_data, test_data = train_test_split(
            dataset, test_size=0.2, random_state=42, stratify=dataset["quality"]
        )
        self.assertTrue(set(map(tuple, train_data.to_numpy())).isdisjoint(
            set(map(tuple, test_data.to_numpy()))
        ))
        exported = read_csv(ROOT / "test_data.csv")
        pd.testing.assert_frame_equal(test_data.reset_index(drop=True), exported)


if __name__ == "__main__":
    unittest.main()