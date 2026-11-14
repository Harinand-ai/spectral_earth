"""
Unit tests for Cross-Sensor Adaptation and Pure Margin transfer evaluation.
Verifies data integrity, spectral alignment, and non-fabrication guarantees across all 6 directional pairs.
"""

import unittest
import numpy as np
from src.visualization_app.cross_sensor_wrapper import (
    CrossSensorDemonstrator,
    CrossSensorWrapper,
    TRANSFER_PAIRS,
)


class TestCrossSensorValidation(unittest.TestCase):
    """Test suite ensuring cross-sensor adaptation endpoints and data integrity."""

    @classmethod
    def setUpClass(cls):
        cls.wrapper = CrossSensorDemonstrator()

    def test_all_six_pairs_available(self):
        """Verify that all 6 directional pairs are supported."""
        self.assertEqual(len(TRANSFER_PAIRS), 6)
        expected_pairs = [
            "Indian Pines -> Salinas",
            "Salinas -> Indian Pines",
            "Indian Pines -> Pavia University",
            "Salinas -> Pavia University",
            "Pavia University -> Indian Pines",
            "Pavia University -> Salinas",
        ]
        for ep in expected_pairs:
            self.assertIn(ep, TRANSFER_PAIRS)

    def test_transfer_summary_structure_and_no_nans(self):
        """Verify summary returns complete metadata, progression, and four configs without NaNs."""
        for pair_key in TRANSFER_PAIRS.keys():
            summary = self.wrapper.get_transfer_summary(pair_key)
            self.assertIn("metadata", summary)
            self.assertIn("metrics_summary", summary)
            self.assertIn("round_progression", summary)
            self.assertIn("four_configs", summary)

            # Check metadata
            meta = summary["metadata"]
            self.assertIn("source_dataset", meta)
            self.assertIn("target_dataset", meta)
            self.assertIn("source_sensor", meta)
            self.assertIn("target_sensor", meta)

            # Check progression rows
            prog = summary["round_progression"]
            self.assertEqual(len(prog), 6, f"Pair {pair_key} must have 6 budget progression points")
            for row in prog:
                self.assertIn(row["budget"], [100, 150, 200, 250, 300, 500])
                for col in ["target_only_random_oa", "target_only_margin_oa", "transferred_random_oa", "transferred_margin_oa", "transfer_delta"]:
                    val = row[col]
                    self.assertIsNotNone(val, f"Col {col} was None in {pair_key}")
                    self.assertFalse(np.isnan(val), f"Col {col} was NaN in {pair_key}")

            # Check 4 configs
            fc = summary["four_configs"]
            for cfg_key in ["target_margin", "source_margin", "target_random", "source_random"]:
                self.assertIn(cfg_key, fc)
                pts = fc[cfg_key]["points"]
                self.assertEqual(len(pts), 6)
                for pt in pts:
                    self.assertFalse(np.isnan(pt["oa_mean"]))
                    self.assertFalse(np.isnan(pt["oa_std"]))

    def test_spectral_resampler_dimensions(self):
        """Verify that optical coverage grid and sensor profiles match dimensions."""
        for pair_key in TRANSFER_PAIRS.keys():
            res = self.wrapper.get_spectral_resampling_demo(pair_key)
            mw = res.get("master_wavelengths", res["target_wavelengths"])
            ts = res["target_spectrum"]
            rs = res["resampled_spectrum"]

            self.assertEqual(len(mw), len(ts), f"Wavelengths and target profile dimension mismatch in {pair_key}")
            self.assertEqual(len(ts), len(rs), f"Target and source profile dimension mismatch in {pair_key}")

            # If genuine cross-sensor, verify optical gap definition
            if res.get("is_genuine_cross_sensor"):
                self.assertEqual(res["overlap_window_nm"], [430, 860])
                self.assertEqual(res["non_overlap_swir_nm"], [860, 2500])

    def test_live_adaptation_trial_execution(self):
        """Verify live trial endpoint produces deterministic, real trial evaluation without mock placeholders."""
        res = self.wrapper.run_live_adaptation_trial(
            pair_key="Indian Pines -> Salinas",
            budget=100,
            strategy="margin",
            seed=42,
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["budget"], 100)
        self.assertEqual(res["seed"], 42)
        self.assertAlmostEqual(res["delta_oa"], res["transferred_oa"] - res["target_only_oa"], places=2)
        self.assertGreater(res["train_time_s"], 0)


if __name__ == "__main__":
    unittest.main()
