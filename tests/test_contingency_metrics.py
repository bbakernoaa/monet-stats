"""
Tests for contingency_metrics.py module.

This module tests contingency table-based statistical metrics including
probability of detection, false alarm rate, threat score, etc.
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.contingency_metrics import CSI  # Critical Success Index
from monet_stats.contingency_metrics import ETS  # Equitable Threat Score
from monet_stats.contingency_metrics import FAR  # False Alarm Rate
from monet_stats.contingency_metrics import FBI  # Frequency Bias Index
from monet_stats.contingency_metrics import HSS  # Heidke Skill Score
from monet_stats.contingency_metrics import POD  # Probability of Detection
from monet_stats.contingency_metrics import TSS  # True Skill Statistic
from monet_stats.contingency_metrics import BSS_binary  # Binary Brier Skill Score
from monet_stats.contingency_metrics import scores  # Contingency table function
from monet_stats.contingency_metrics import ETS_max_threshold, FAR_min_threshold, HSS_max_threshold, POD_max_threshold
from monet_stats.test_utils import TestDataGenerator


class TestContingencyMetrics:
    """Test suite for contingency metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_gen = TestDataGenerator()

        # Perfect agreement case
        self.obs_perfect = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        self.mod_perfect = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])

        # Some disagreements
        self.obs_test = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        self.mod_test = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1])

        # All misses (no hits)
        self.obs_hits = np.array([1, 1, 1, 1, 1])
        self.mod_misses = np.array([0, 0, 0, 0, 0])

        # All false alarms
        self.obs_zeros = np.array([0, 0, 0, 0])
        self.mod_alarms = np.array([1, 1, 1, 1])

    def test_pod_perfect_detection(self):
        """Test POD with perfect detection."""
        result = POD(self.obs_perfect, self.mod_perfect, minval=0.5)
        assert result == 1.0, f"Perfect detection should give POD=1.0, got {result}"

    def test_pod_no_detection(self):
        """Test POD with no detection (all misses)."""
        result = POD(self.obs_hits, self.mod_misses, minval=0.5)
        assert result == 0.0, f"No detection should give POD=0.0, got {result}"

    def test_pod_partial_detection(self):
        """Test POD with partial detection."""
        # obs=[1,0,1,1,0,1,0,0,1,0], mod=[1,1,1,0,0,1,1,0,1,1], minval=0.5
        # Hits (a): positions where both obs>=0.5 and mod>=0.5: 0,2,5,8 -> 4 hits
        # Misses (b): positions where obs>=0.5 and mod<0.5: 3 -> 1 miss
        # POD = hits / (hits + misses) = 4 / (4 + 1) = 4/5 = 0.8
        result = POD(self.obs_test, self.mod_test, minval=0.5)
        expected = 4 / 5  # 4 hits out of 5 actual events
        assert abs(result - expected) < 1e-10, f"POD should be {expected}, got {result}"

    def test_far_no_false_alarms(self):
        """Test FAR with no false alarms."""
        result = FAR(self.obs_perfect, self.mod_perfect, minval=0.5)
        assert result == 0.0, f"No false alarms should give FAR=0.0, got {result}"

    def test_far_all_false_alarms(self):
        """Test FAR with all false alarms."""
        result = FAR(self.obs_zeros, self.mod_alarms, minval=0.5)
        assert result == 1.0, f"All false alarms should give FAR=1.0, got {result}"

    def test_csi_perfect_score(self):
        """Test CSI with perfect score."""
        result = CSI(self.obs_perfect, self.mod_perfect, minval=0.5)
        assert result == 1.0, f"Perfect score should give CSI=1.0, got {result}"

    def test_csi_worst_score(self):
        """Test CSI with worst score (no hits)."""
        result = CSI(self.obs_hits, self.mod_misses, minval=0.5)
        assert result == 0.0, f"No hits should give CSI=0.0, got {result}"

    def test_fbi_perfect_bias(self):
        """Test frequency bias with perfect agreement."""
        result = FBI(self.obs_perfect, self.mod_perfect, minval=0.5)
        assert abs(result - 1.0) < 1e-10, f"Perfect agreement should give bias=1.0, got {result}"

    def test_fbi_overprediction(self):
        """Test frequency bias with overprediction."""
        obs = np.array([1, 0, 1, 0, 1])
        mod = np.array([1, 1, 1, 1, 1])  # Model predicts more events
        result = FBI(obs, mod, minval=0.5)
        assert result > 1.0, f"Overprediction should give bias > 1.0, got {result}"

    def test_fbi_underprediction(self):
        """Test frequency bias with underprediction."""
        obs = np.array([1, 1, 1, 1, 1])
        mod = np.array([1, 0, 1, 0, 1])  # Model predicts fewer events
        result = FBI(obs, mod, minval=0.5)
        assert result < 1.0, f"Underprediction should give bias < 1.0, got {result}"

    def test_tss_range_bounds(self):
        """Test that TSS is within valid range [-1, 1]."""
        result = TSS(self.obs_test, self.mod_test, minval=0.5)
        assert -1 <= result <= 1, f"TSS should be in [-1,1], got {result}"

    def test_ets_range_bounds(self):
        """Test that ETS is within valid range [-1, 1]."""
        result = ETS(self.obs_test, self.mod_test, minval=0.5)
        assert -1 <= result <= 1, f"ETS should be in [-1,1], got {result}"

    def test_hss_range_bounds(self):
        """Test that HSS is within valid range [-1, 1]."""
        result = HSS(self.obs_test, self.mod_test, minval=0.5)
        assert -1 <= result <= 1, f"HSS should be in [-1,1], got {result}"

    @pytest.mark.parametrize(
        "metric_func,expected_range",
        [
            (POD, (0, 1)),
            (FAR, (0, 1)),
            (CSI, (0, 1)),
        ],
    )
    def test_probability_metrics_range(self, metric_func, expected_range):
        """Test that probability-based metrics are in [0, 1] range."""
        result = metric_func(self.obs_test, self.mod_test, minval=0.5)
        min_val, max_val = expected_range
        assert min_val <= result <= max_val, f"{metric_func.__name__} should be in {expected_range}, got {result}"

    def test_edge_case_empty_arrays(self):
        """Test behavior with empty arrays."""
        result = POD(np.array([]), np.array([]), minval=0.5)
        assert np.isnan(result), f"Empty arrays should return NaN, got {result}"

    def test_edge_case_single_element(self):
        """Test behavior with single element arrays."""
        result = POD(np.array([1]), np.array([1]), minval=0.5)
        assert result == 1.0, "Single perfect match should give POD=1.0"

        result = POD(np.array([1]), np.array([0]), minval=0.5)
        assert result == 0.0, "Single miss should give POD=0.0"

    def test_edge_case_all_zeros(self):
        """Test behavior with all zero arrays."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)

        # POD with no events should be undefined (NaN or exception)
        result = POD(obs_zeros, mod_zeros, minval=0.5)
        assert np.isnan(result) or result == 0.0, "POD with no events should be NaN or 0"

    def test_edge_case_all_ones(self):
        """Test behavior with all one arrays."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)

        result = POD(obs_ones, mod_ones, minval=0.5)
        assert result == 1.0, "All ones should give perfect POD=1.0"

    @pytest.mark.unit
    def test_contingency_metrics_mathematical_correctness(self):
        """Test mathematical correctness of contingency metrics."""
        # Use known contingency table values
        hits, misses, false_alarms, correct_negatives = 4, 2, 3, 5

        # Calculate expected values manually
        expected_pod = hits / (hits + misses)  # 4/6 = 0.667
        expected_far = false_alarms / (hits + false_alarms)  # 3/7 = 0.429
        expected_csi = hits / (hits + misses + false_alarms)  # 4/9 = 0.444
        expected_bias = (hits + false_alarms) / (hits + misses)  # 7/6 = 1.167

        # Create data that produces these contingency values
        obs = np.array([1] * (hits + misses) + [0] * (false_alarms + correct_negatives))
        mod = np.array([1] * hits + [0] * misses + [1] * false_alarms + [0] * correct_negatives)

        # Test calculations
        assert abs(POD(obs, mod, minval=0.5) - expected_pod) < 1e-10
        assert abs(FAR(obs, mod, minval=0.5) - expected_far) < 1e-10
        assert abs(CSI(obs, mod, minval=0.5) - expected_csi) < 1e-10
        assert abs(FBI(obs, mod, minval=0.5) - expected_bias) < 1e-10

    @pytest.mark.slow
    def test_contingency_metrics_performance(self):
        """Test performance with large datasets."""
        # Generate large test dataset
        large_obs, large_mod = self.data_gen.generate_contingency_data(n_samples=10000)

        import time

        start_time = time.time()
        result = POD(large_obs, large_mod, minval=0.5)
        end_time = time.time()

        # Should complete quickly (adjust threshold as needed)
        assert end_time - start_time < 1.0, "POD should complete in under 1 second"
        assert isinstance(result, (float, np.floating)), "Should return a float"

    def test_bss_binary_perfect_score(self):
        """Test Binary Brier Skill Score with perfect agreement."""
        obs = np.array([1, 0, 1, 0])
        mod = np.array([1, 0, 1, 0])
        result = BSS_binary(obs, mod, threshold=0.5)
        assert result == 1.0, f"Perfect agreement should give BSS=1.0, got {result}"

    def test_bss_binary_worst_score(self):
        """Test Binary Brier Skill Score with worst agreement."""
        obs = np.array([1, 0, 1, 0])
        mod = np.array([0, 1, 0, 1])
        result = BSS_binary(obs, mod, threshold=0.5)
        assert result < 0, f"Opposite predictions should give negative BSS, got {result}"

    def test_scores_function(self):
        """Test the scores function that returns contingency table values."""
        a, b, c, d = scores(self.obs_test, self.mod_test, minval=0.5)
        # From manual calculation:
        # obs=[1,0,1,1,0,1,0,0,1,0], mod=[1,1,1,0,0,1,1,0,1,1]
        # Hits (a): positions 0,2,5,8 -> 4 hits
        # Misses (b): positions 3 -> 1 miss
        # False alarms (c): positions 1,6,9 -> 3 false alarms
        # Correct negatives (d): positions 4,7 -> 2 correct negatives
        assert a == 4  # hits
        assert b == 1  # misses
        assert c == 3  # false alarms
        assert d == 2  # correct negatives

    def test_threshold_optimization_functions(self):
        """Test threshold optimization functions."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        # Test HSS max threshold
        opt_threshold, max_hss = HSS_max_threshold(obs, mod, 1, 5, 0.5)
        assert isinstance(opt_threshold, (int, float)), "Should return threshold value"
        assert isinstance(max_hss, (int, float)), "Should return HSS value"

        # Test ETS max threshold
        opt_threshold, max_ets = ETS_max_threshold(obs, mod, 1, 5, 0.5)
        assert isinstance(opt_threshold, (int, float)), "Should return threshold value"
        assert isinstance(max_ets, (int, float)), "Should return ETS value"

        # Test POD max threshold
        opt_threshold, max_pod = POD_max_threshold(obs, mod, 1, 5, 0.5)
        assert isinstance(opt_threshold, (int, float)), "Should return threshold value"
        assert isinstance(max_pod, (int, float)), "Should return POD value"

        # Test FAR min threshold
        opt_threshold, min_far = FAR_min_threshold(obs, mod, 1, 5, 0.5)
        assert isinstance(opt_threshold, (int, float)), "Should return threshold value"
        assert isinstance(min_far, (int, float)), "Should return FAR value"


class TestContingencyMetricsXarray:
    """Test suite for contingency metrics with xarray inputs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.obs_xr = xr.DataArray([1, 0, 1, 1, 0], dims=["time"])
        self.mod_xr = xr.DataArray([1, 1, 1, 0, 0], dims=["time"])

    def test_POD_xarray(self):
        """Test POD with xarray inputs."""
        result = POD(self.obs_xr, self.mod_xr, minval=0.5)
        assert np.isclose(result, 2 / 3)

    def test_FAR_xarray(self):
        """Test FAR with xarray inputs."""
        result = FAR(self.obs_xr, self.mod_xr, minval=0.5)
        assert np.isclose(result, 1 / 3)

    def test_CSI_xarray(self):
        """Test CSI with xarray inputs."""
        result = CSI(self.obs_xr, self.mod_xr, minval=0.5)
        assert np.isclose(result, 2 / 4)

    def test_FBI_xarray(self):
        """Test FBI with xarray inputs."""
        result = FBI(self.obs_xr, self.mod_xr, minval=0.5)
        assert np.isclose(result, 1.0)
