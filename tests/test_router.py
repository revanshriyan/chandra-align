import unittest
from chandra_align.pipeline.router import (
    SensorMeta,
    RoutingDecision,
    MatcherStrategy,
    compute_azimuth_delta,
    evaluate_route,
)


class TestRouter(unittest.TestCase):
    """Tests for the illumination & scale router."""

    def test_compute_azimuth_delta_basic(self):
        """Test basic azimuth delta computation."""
        self.assertAlmostEqual(compute_azimuth_delta(10.0, 20.0), 10.0)
        self.assertAlmostEqual(compute_azimuth_delta(350.0, 10.0), 20.0)
        self.assertAlmostEqual(compute_azimuth_delta(0.0, 180.0), 180.0)
        self.assertAlmostEqual(compute_azimuth_delta(0.0, 181.0), 179.0)
        self.assertAlmostEqual(compute_azimuth_delta(90.0, 270.0), 180.0)
        self.assertAlmostEqual(compute_azimuth_delta(45.0, 225.0), 180.0)

    def test_rule1_swir_cross_modal(self):
        """SWIR inputs route to the validated LightGlue primary."""
        source = SensorMeta("OHRC", "SWIR", 0.25, 45.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 45.0, 30.0)
        decision = evaluate_route(source, ref)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.LIGHTGLUE_ALIKED)
        self.assertEqual(decision.fallback_strategy, MatcherStrategy.MUTUAL_INFORMATION)
        self.assertIn("Cross-modal", decision.routing_reason)

    def test_rule1_swir_on_reference(self):
        """Rule 1: SWIR reference also triggers cross-modal route."""
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 45.0, 30.0)
        ref = SensorMeta("NAC", "SWIR", 0.5, 45.0, 30.0)
        decision = evaluate_route(source, ref)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.LIGHTGLUE_ALIKED)
        self.assertEqual(decision.fallback_strategy, MatcherStrategy.MUTUAL_INFORMATION)

    def test_rule2_high_azimuth_shift(self):
        """High azimuth shift keeps LightGlue primary and SIFT fallback."""
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 0.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 90.0, 30.0)  # 90° delta
        decision = evaluate_route(source, ref)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.LIGHTGLUE_ALIKED)
        self.assertEqual(decision.fallback_strategy, MatcherStrategy.SIFT_RANSAC)
        self.assertGreaterEqual(decision.delta_azimuth_deg, 60.0)
        self.assertIn("High solar azimuth shift", decision.routing_reason)

    def test_rule2_azimuth_wraparound(self):
        """Rule 2: Azimuth delta >= 60° with wraparound (350° vs 30° = 40°, no trigger)."""
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 350.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 30.0, 30.0)  # 40° delta (wraparound)
        decision = evaluate_route(source, ref)
        self.assertLess(decision.delta_azimuth_deg, 60.0)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.LIGHTGLUE_ALIKED)

    def test_rule2_azimuth_wraparound_trigger(self):
        """Rule 2: Azimuth delta >= 60° with wraparound (350° vs 130° = 80°, trigger)."""
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 350.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 130.0, 30.0)  # 80° delta (wraparound)
        decision = evaluate_route(source, ref)
        self.assertGreaterEqual(decision.delta_azimuth_deg, 60.0)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.LIGHTGLUE_ALIKED)

    def test_rule3_standard_optical(self):
        """Rule 3: Standard regime (low azimuth, same modality) -> LIGHTGLUE_ALIKED."""
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 45.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 50.0, 32.0)  # Small deltas
        decision = evaluate_route(source, ref)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.LIGHTGLUE_ALIKED)
        self.assertEqual(decision.fallback_strategy, MatcherStrategy.SIFT_RANSAC)
        self.assertIn("Standard optical", decision.routing_reason)

    def test_default_optical_route_never_selects_rift2(self):
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 10.0, 30.0)
        reference = SensorMeta("TMC-2", "PANCHROMATIC", 4.47, 12.0, 31.0)
        decision = evaluate_route(source, reference)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.LIGHTGLUE_ALIKED)
        self.assertEqual(decision.fallback_strategy, MatcherStrategy.SIFT_RANSAC)
        self.assertNotEqual(decision.primary_strategy, MatcherStrategy.RIFT2_PHASE_CONGRUENCY)

    def test_rule4_scale_resampling_needed(self):
        """Rule 4: Scale ratio > 4.0 or < 0.25 triggers pre-resampling flag."""
        # Scale ratio > 4.0 (source much coarser)
        source = SensorMeta("OHRC", "PANCHROMATIC", 2.1, 45.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 45.0, 30.0)  # ratio = 4.2
        decision = evaluate_route(source, ref)
        self.assertTrue(decision.needs_pre_resampling)
        self.assertEqual(decision.target_resample_gsd, 0.5)

        # Scale ratio < 0.25 (source much finer)
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.1, 45.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 45.0, 30.0)  # ratio = 0.2
        decision = evaluate_route(source, ref)
        self.assertTrue(decision.needs_pre_resampling)
        self.assertEqual(decision.target_resample_gsd, 0.5)

    def test_scale_ratio_no_resampling(self):
        """Scale ratio within [0.25, 4.0] -> no resampling needed."""
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 45.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 45.0, 30.0)  # ratio = 0.5
        decision = evaluate_route(source, ref)
        self.assertFalse(decision.needs_pre_resampling)
        self.assertIsNone(decision.target_resample_gsd)

    def test_zero_gsd_protection(self):
        """Division by zero protection for reference GSD."""
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 45.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.0, 45.0, 30.0)
        decision = evaluate_route(source, ref)
        self.assertEqual(decision.scale_ratio, 1.0)
        self.assertFalse(decision.needs_pre_resampling)

    def test_rule1_sar_cross_modal_source(self):
        """Rule 1: SAR source triggers SAR_OPTICAL_CROSSMODAL primary."""
        source = SensorMeta("DF-SAR", "PANCHROMATIC", 10.0, 45.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 45.0, 30.0)
        decision = evaluate_route(source, ref)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.SAR_OPTICAL_CROSSMODAL)
        self.assertEqual(decision.fallback_strategy, MatcherStrategy.MUTUAL_INFORMATION)
        self.assertIn("SAR/Radar cross-modal", decision.routing_reason)

    def test_rule1_sar_cross_modal_reference(self):
        """Rule 1: SAR reference also triggers cross-modal route."""
        source = SensorMeta("OHRC", "PANCHROMATIC", 0.25, 45.0, 30.0)
        ref = SensorMeta("LROC-SAR", "PANCHROMATIC", 0.5, 45.0, 30.0)
        decision = evaluate_route(source, ref)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.SAR_OPTICAL_CROSSMODAL)
        self.assertEqual(decision.fallback_strategy, MatcherStrategy.MUTUAL_INFORMATION)

    def test_sar_various_keywords(self):
        """Test various SAR keyword detections."""
        sar_names = ["SAR", "RADAR", "DF-SAR", "L-BAND", "S-BAND", "df-sar", "radar"]
        for sar_name in sar_names:
            source = SensorMeta(sar_name, "PANCHROMATIC", 10.0, 45.0, 30.0)
            ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 45.0, 30.0)
            decision = evaluate_route(source, ref)
            self.assertEqual(decision.primary_strategy, MatcherStrategy.SAR_OPTICAL_CROSSMODAL,
                           f"Failed for SAR name: {sar_name}")

    def test_sar_priority_over_swir(self):
        """SAR rule should be evaluated before SWIR rule (higher priority)."""
        # Source is SAR, ref is SWIR - SAR rule should take priority
        source = SensorMeta("DF-SAR", "PANCHROMATIC", 10.0, 45.0, 30.0)
        ref = SensorMeta("NAC", "SWIR", 0.5, 45.0, 30.0)
        decision = evaluate_route(source, ref)
        # SAR rule should trigger first
        self.assertEqual(decision.primary_strategy, MatcherStrategy.SAR_OPTICAL_CROSSMODAL)
        self.assertIn("SAR/Radar cross-modal", decision.routing_reason)

    def test_sar_priority_over_azimuth(self):
        """SAR rule should be evaluated before azimuth shift rule (higher priority)."""
        # Source is SAR with high azimuth shift - SAR rule should trigger first
        source = SensorMeta("DF-SAR", "PANCHROMATIC", 10.0, 0.0, 30.0)
        ref = SensorMeta("NAC", "PANCHROMATIC", 0.5, 90.0, 30.0)  # 90° delta
        decision = evaluate_route(source, ref)
        # SAR rule should trigger first, not azimuth rule
        self.assertEqual(decision.primary_strategy, MatcherStrategy.SAR_OPTICAL_CROSSMODAL)
        self.assertIn("SAR/Radar cross-modal", decision.routing_reason)

    def test_sar_lowercase_keywords(self):
        """Test SAR detection works with lowercase keywords."""
        source = SensorMeta("df-sar", "PANCHROMATIC", 10.0, 45.0, 30.0)
        ref = SensorMeta("nac", "PANCHROMATIC", 0.5, 45.0, 30.0)
        decision = evaluate_route(source, ref)
        self.assertEqual(decision.primary_strategy, MatcherStrategy.SAR_OPTICAL_CROSSMODAL)

if __name__ == "__main__":
    unittest.main()
