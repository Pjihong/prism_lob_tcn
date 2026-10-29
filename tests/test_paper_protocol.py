import unittest

from lob_xai.paper_protocol import (
    COUNTERFACTUAL,
    DATA,
    TRAINING,
    assert_paper_split,
    bin_ctabl_parameter_count,
    branch_variant_parameter_count,
    deeplob_parameter_count,
    ours_parameter_count,
    parse_boundaries,
    tlob_parameter_count,
)


class PaperProtocolTests(unittest.TestCase):
    def test_split_math(self):
        self.assertAlmostEqual(DATA.validation_fraction_within_development_file, 3 / 17)
        # canonical 70/15/15 shape still satisfies the within-development rule
        assert_paper_split(700, 150, 150)
        # the actual provided FI-2010 files (dev 362,400 + test 31,937 rows)
        assert_paper_split(298_448, 63_952, 31_937)
        # wrong within-development validation share must fail
        with self.assertRaises(ValueError):
            assert_paper_split(560, 140, 300)
        # an empty held-out test file must fail
        with self.assertRaises(ValueError):
            assert_paper_split(700, 150, 0)

    def test_boundary_parser_is_sorted_and_unique(self):
        self.assertEqual(parse_boundaries("300,100,300,200"), [100, 200, 300])

    def test_manuscript_defaults(self):
        self.assertEqual(TRAINING.seeds, (1, 2, 3))
        self.assertEqual(COUNTERFACTUAL.deletion_grid[0], 0.05)
        self.assertEqual(COUNTERFACTUAL.deletion_grid[-1], 0.50)
        self.assertGreaterEqual(COUNTERFACTUAL.neighbor_candidates,
                                COUNTERFACTUAL.mc_samples)
        self.assertEqual(TRAINING.branch_temporal_receptive_field, 29)
        self.assertEqual(ours_parameter_count(3), 141_955)
        self.assertEqual(ours_parameter_count(2), 141_890)
        self.assertEqual(deeplob_parameter_count(3), 143_907)
        self.assertEqual(deeplob_parameter_count(2), 143_842)
        self.assertEqual(tlob_parameter_count(3), 1_798_866)
        self.assertEqual(tlob_parameter_count(2), 1_798_815)
        self.assertEqual(bin_ctabl_parameter_count(3), 12_526)
        self.assertEqual(bin_ctabl_parameter_count(2), 12_405)

    def test_branch_upgrade_parameter_counts(self):
        self.assertEqual(branch_variant_parameter_count(3), 141_955)
        self.assertEqual(
            branch_variant_parameter_count(3, pool="multi_scale"), 141_967
        )
        self.assertEqual(
            branch_variant_parameter_count(3, pool="positional"), 142_355
        )
        self.assertEqual(
            branch_variant_parameter_count(2, pool="multi_scale"), 141_902
        )
        self.assertEqual(
            branch_variant_parameter_count(2, pool="positional"), 142_290
        )
        self.assertEqual(
            branch_variant_parameter_count(3, tcn_norm="layer"), 144_003
        )
        self.assertEqual(
            branch_variant_parameter_count(
                3, tcn_norm="layer", pool="avg_last"
            ),
            160_387,
        )
        self.assertEqual(
            branch_variant_parameter_count(
                3, tcn_norm="layer", pool="avg_last",
                interaction_channels=64,
            ),
            185_155,
        )
        self.assertEqual(
            branch_variant_parameter_count(
                3, channels=(32, 32, 64, 64), tcn_norm="layer",
                pool="avg_last", interaction_channels=64,
            ),
            284_995,
        )
        with self.assertRaises(ValueError):
            branch_variant_parameter_count(
                3, pool="multi_scale", interaction_channels=8
            )


if __name__ == "__main__":
    unittest.main()
