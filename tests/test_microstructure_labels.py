import unittest

import numpy as np

from lob_xai.microstructure_labels import (
    compute_toxic_labels,
    compute_vacuum_labels,
    make_segment_id,
    segment_aware_starts,
)


class MicrostructureLabelTests(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros((12, 40), dtype=np.float32)

    def test_toxic_up_and_down_use_future_top_of_book(self):
        self.x[3, 0] = 0.01
        self.x[4, 2] = -0.01
        y, valid = compute_toxic_labels(self.x, k=3, tick_size=0.001, n_ticks=3)
        self.assertTrue(valid[0])
        self.assertEqual(int(y[0]), 1)
        self.assertEqual(int(y[1]), 2)

    def test_vacuum_requires_volume_drop_and_spread_widening(self):
        self.x[0, 1] = 2.0
        self.x[2, 1] = 0.0
        self.x[2, 0] = 0.01
        y, _ = compute_vacuum_labels(
            self.x, k=2, tick_size=0.001, vol_drop=1.5, n_ticks_spread=1
        )
        self.assertEqual(int(y[0]), 1)
        self.x[2, 0] = 0.0
        y, _ = compute_vacuum_labels(
            self.x, k=2, tick_size=0.001, vol_drop=1.5, n_ticks_spread=1
        )
        self.assertEqual(int(y[0]), 0)

    def test_future_labels_never_cross_segments(self):
        seg = make_segment_id(len(self.x), [6])
        self.x[6, 0] = 1.0
        y, valid = compute_toxic_labels(
            self.x, k=2, tick_size=0.001, n_ticks=3, segment_id=seg
        )
        self.assertFalse(valid[4])
        self.assertEqual(int(y[4]), 0)

    def test_windows_never_cross_segments(self):
        seg = make_segment_id(12, [6])
        valid = np.ones(12, dtype=bool)
        starts = segment_aware_starts(seg, valid, seq_len=4, stride=1)
        self.assertNotIn(3, starts.tolist())
        self.assertTrue(all(seg[s] == seg[s + 3] for s in starts))


if __name__ == "__main__":
    unittest.main()
