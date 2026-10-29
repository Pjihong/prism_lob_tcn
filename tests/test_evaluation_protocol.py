import unittest

import numpy as np

from lob_xai.evaluation_protocol import select_top_confidence_alerts


class ExplanationCohortTests(unittest.TestCase):
    def test_selects_correct_target_alerts_by_confidence(self):
        probs = np.array([
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y = np.array([1, 0, 1, 1])
        got = select_top_confidence_alerts(probs, y, 1, 2)
        np.testing.assert_array_equal(got, [0, 2])

    def test_prediction_based_mode_is_explicit(self):
        probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])
        y = np.array([0, 1, 1])
        got = select_top_confidence_alerts(
            probs, y, 1, None, require_correct=False
        )
        np.testing.assert_array_equal(got, [0, 1])

    def test_ties_preserve_dataset_order(self):
        probs = np.array([[0.2, 0.8], [0.2, 0.8], [0.1, 0.9]])
        y = np.ones(3, dtype=int)
        got = select_top_confidence_alerts(probs, y, 1, None)
        np.testing.assert_array_equal(got, [2, 0, 1])


if __name__ == "__main__":
    unittest.main()
