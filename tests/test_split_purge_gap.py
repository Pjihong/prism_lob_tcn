"""Tests for the opt-in validation purge gap in split_within_segments.

The default (purge_gap=0) must reproduce the frozen-protocol split exactly;
a positive gap must only remove rows from the head of each validation part.
"""
from __future__ import annotations

import unittest

import numpy as np

from lob_xai.microstructure_labels import split_within_segments


class TestSplitPurgeGap(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(200, dtype=np.float64).reshape(100, 2)
        self.seg = np.zeros(100, dtype=np.int32)

    def test_default_matches_frozen_protocol(self):
        a = split_within_segments(self.X, self.seg, val_ratio=0.2)
        b = split_within_segments(self.X, self.seg, val_ratio=0.2, purge_gap=0)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(len(a[0]), 80)
        self.assertEqual(len(a[2]), 20)

    def test_gap_removes_head_of_validation_only(self):
        X_tr, _, X_val, seg_val = split_within_segments(
            self.X, self.seg, val_ratio=0.2, purge_gap=5
        )
        self.assertEqual(len(X_tr), 80)          # train side unchanged
        self.assertEqual(len(X_val), 15)         # 20 - 5
        np.testing.assert_array_equal(X_val[0], self.X[85])
        np.testing.assert_array_equal(X_val[-1], self.X[99])
        self.assertEqual(len(seg_val), 15)

    def test_gap_larger_than_validation_empties_it(self):
        _, _, X_val, _ = split_within_segments(
            self.X, self.seg, val_ratio=0.2, purge_gap=999
        )
        self.assertEqual(len(X_val), 0)

    def test_two_segments_each_purged(self):
        seg = np.repeat([0, 1], 50).astype(np.int32)
        _, _, X_val, seg_val = split_within_segments(
            self.X, seg, val_ratio=0.2, purge_gap=3
        )
        self.assertEqual(len(X_val), 2 * (10 - 3))
        self.assertEqual(list(np.bincount(seg_val)), [7, 7])

    def test_negative_gap_rejected(self):
        with self.assertRaises(ValueError):
            split_within_segments(self.X, self.seg, 0.2, purge_gap=-1)


if __name__ == "__main__":
    unittest.main()
