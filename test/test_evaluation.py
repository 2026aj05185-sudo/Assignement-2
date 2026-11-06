import unittest

import numpy as np
import pandas as pd

from evaluation import FEATURES, evaluate_model, read_csv, validate_data


class FixedModel:
    classes_ = np.array([3, 4, 5])

    def predict(self, features):
        return np.array([3, 4, 5])[:len(features)]

    def predict_proba(self, features):
        return np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])[:len(features)]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({feature: [1.0, 2.0, 3.0] for feature in FEATURES})
        self.frame["quality"] = [3, 4, 5]

    def test_perfect_predictions(self):
        features, target = validate_data(self.frame)
        metrics, _, _ = evaluate_model(FixedModel(), features, target)
        self.assertTrue(all(value == 1.0 for value in metrics.values()))

    def test_auc_unavailable_with_missing_class(self):
        features, target = validate_data(self.frame.iloc[:2])
        metrics, _, _ = evaluate_model(FixedModel(), features, target)
        self.assertIsNone(metrics["AUC"])

    def test_rejects_missing_column(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_data(self.frame.drop(columns=[FEATURES[0]]))

    def test_rejects_nan_and_infinity(self):
        for value in [np.nan, np.inf, -np.inf]:
            frame = self.frame.copy()
            frame.loc[0, FEATURES[0]] = value
            with self.assertRaisesRegex(ValueError, "NaN"):
                validate_data(frame)

    def test_rejects_non_numeric(self):
        frame = self.frame.astype(object)
        frame.loc[0, FEATURES[0]] = "bad"
        with self.assertRaisesRegex(ValueError, "numeric"):
            validate_data(frame)

    def test_rejects_invalid_quality(self):
        for value in [2.5, 11, -1]:
            frame = self.frame.astype(float)
            frame.loc[0, "quality"] = value
            with self.assertRaisesRegex(ValueError, "whole-number"):
                validate_data(frame)

    def test_rejects_unseen_class(self):
        with self.assertRaisesRegex(ValueError, "not trained"):
            validate_data(self.frame, classes=[4, 5])

    def test_rejects_empty_data(self):
        with self.assertRaisesRegex(ValueError, "no data"):
            validate_data(self.frame.iloc[:0])

    def test_comma_and_semicolon_csv(self):
        for delimiter in [",", ";"]:
            parsed = read_csv(self.frame.to_csv(index=False, sep=delimiter).encode())
            pd.testing.assert_frame_equal(parsed, self.frame)


if __name__ == "__main__":
    unittest.main()