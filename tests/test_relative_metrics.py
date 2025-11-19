"""
Tests for relative_metrics.py module.

This module tests relative-based statistical metrics including
NMB, NME, FB, FE, and other relative statistics.
"""
import numpy as np
import pytest

from src.monet_stats.relative_metrics import (
    FB,
    FE,
    ME,
    MPE,
    NMB,
    NMB_ABS,
    NME,
    USUTPB,
    USUTPE,
    WDME,
    MdnE,
    MdnPE,
    NMdnB,
    NMdnE,
    NME_m,
    NME_m_ABS,
    WDMdnE,
    WDME_m,
    WDNMB_m,
)
from tests.test_utils import TestDataGenerator


class TestRelativeMetrics:
    """Test suite for relative metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.data_gen = TestDataGenerator()

        # Perfect agreement case
        self.obs_perfect = np.array([1, 2, 3, 4, 5])
        self.mod_perfect = np.array([1, 2, 3, 4, 5])

        # Small systematic bias
        self.obs_biased = np.array([1, 2, 3, 4, 5])
        self.mod_biased = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # 0.1 bias

        # Random errors
        np.random.seed(42)
        self.obs_random = np.random.normal(0, 1, 50)
        self.mod_random = self.obs_random + np.random.normal(0, 0.2, 50)  # Small noise

    def test_nmb_perfect_agreement(self):
        """Test Normalized Mean Bias with perfect agreement."""
        result = NMB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NMB=0.0, got {result}"

    def test_nmb_with_bias(self):
        """Test NMB with systematic bias."""
        result = NMB(self.obs_biased, self.mod_biased)
        # With 0.1 bias and mean of 3, NMB should be (0.1/3)*100 = 3.333... (as percentage)
        expected = (0.1 / np.mean(self.obs_biased)) * 100
        assert abs(result - expected) < 1e-10, f"Expected NMB={expected}, got {result}"

    def test_nme_perfect_agreement(self):
        """Test Normalized Mean Error with perfect agreement."""
        result = NME(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NME=0.0, got {result}"

    def test_fb_perfect_agreement(self):
        """Test Fractional Bias with perfect agreement."""
        result = FB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give FB=0.0, got {result}"

    def test_fe_perfect_agreement(self):
        """Test Fractional Error with perfect agreement."""
        result = FE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give FE=0.0, got {result}"

    def test_me_perfect_agreement(self):
        """Test Mean Error with perfect agreement."""
        result = ME(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give ME=0.0, got {result}"

    def test_mpe_perfect_agreement(self):
        """Test Mean Percentage Error with perfect agreement."""
        result = MPE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MPE=0.0, got {result}"

    def test_mdne_perfect_agreement(self):
        """Test Median Error with perfect agreement."""
        result = MdnE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MdnE=0.0, got {result}"

    def test_mdnpe_perfect_agreement(self):
        """Test Median Prediction Error with perfect agreement."""
        result = MdnPE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MdnPE=0.0, got {result}"

    def test_nmdnb_perfect_agreement(self):
        """Test Normalized Median Bias with perfect agreement."""
        result = NMdnB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NMdnB=0.0, got {result}"

    def test_nmdne_perfect_agreement(self):
        """Test Normalized Median Error with perfect agreement."""
        result = NMdnE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NMdnE=0.0, got {result}"

    def test_nme_m_perfect_agreement(self):
        """Test Modified NME with perfect agreement."""
        result = NME_m(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NME_m=0.0, got {result}"

    def test_nme_m_abs_perfect_agreement(self):
        """Test Modified NME absolute with perfect agreement."""
        result = NME_m_ABS(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NME_m_ABS=0.0, got {result}"

    def test_nmb_abs_perfect_agreement(self):
        """Test NMB absolute with perfect agreement."""
        result = NMB_ABS(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NMB_ABS=0.0, got {result}"

    def test_usutpb_perfect_agreement(self):
        """Test USUTPB with perfect agreement."""
        result = USUTPB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give USUTPB=0.0, got {result}"

    def test_usutpe_perfect_agreement(self):
        """Test USUTPE with perfect agreement."""
        result = USUTPE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give USUTPE=0.0, got {result}"

    def test_wdme_perfect_agreement(self):
        """Test Wind Direction Mean Error with perfect agreement."""
        result = WDME(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give WDME=0.0, got {result}"

    def test_wdmdne_perfect_agreement(self):
        """Test Wind Direction Median Error with perfect agreement."""
        result = WDMdnE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give WDMdnE=0.0, got {result}"

    def test_wdme_m_perfect_agreement(self):
        """Test Modified Wind Direction Mean Error with perfect agreement."""
        result = WDME_m(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give WDME_m=0.0, got {result}"

    def test_wdnmb_m_perfect_agreement(self):
        """Test Modified Wind Direction NMB with perfect agreement."""
        result = WDNMB_m(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give WDNMB_m=0.0, got {result}"

    @pytest.mark.parametrize("metric_func", [
        NMB, NME, FB, FE, ME, MPE, MdnE, MdnPE,
        NMdnB, NMdnE, NME_m, NME_m_ABS, NMB_ABS, USUTPB, USUTPE, WDME,
        WDMdnE, WDME_m, WDNMB_m
    ])
    def test_relative_metrics_output_type(self, metric_func):
        """Test that relative metrics return appropriate values."""
        result = metric_func(self.obs_random, self.mod_random)
        assert isinstance(result, (float, np.floating, int, np.integer)), \
            f"{metric_func.__name__} should return a numeric value, got {type(result)}"

    def test_edge_case_single_element(self):
        """Test behavior with single element arrays."""
        result = NMB(np.array([1.0]), np.array([1.0]))
        assert result == 0.0, "Single perfect match should give NMB=0.0"

        result = NMB(np.array([1.0]), np.array([2.0]))
        assert result == 100.0, "Single element difference should give NMB=100.0"  # ((2-1)/1)*100 = 100

    def test_edge_case_all_zeros(self):
        """Test behavior with all zero arrays (should handle division by zero)."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)

        # NMB should handle division by zero when obs mean is 0
        result = NMB(obs_zeros, mod_zeros)
        # When both obs and mod are zero, the result should be 0/0 which is NaN or 0 depending on implementation
        assert np.isnan(result) or result == 0.0, f"All zeros should give NaN or 0, got {result}"

    def test_edge_case_all_ones(self):
        """Test behavior with all one arrays."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)

        result = NMB(obs_ones, mod_ones)
        assert abs(result - 0.0) < 1e-10, f"Identical arrays should give NMB=0.0, got {result}"

    @pytest.mark.unit
    def test_relative_metrics_mathematical_correctness(self):
        """Test mathematical correctness of relative metrics."""
        # Create data with known properties
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # 0.1 bias

        # NMB should be (0.1 / mean(obs))*100 = (0.1 / 3)*100 = 3.333...
        expected_nmb = (0.1 / np.mean(obs)) * 100
        nmb_result = NMB(obs, mod)
        assert abs(nmb_result - expected_nmb) < 1e-10, f"Expected NMB={expected_nmb}, got {nmb_result}"

        # ME should be the mean of differences = 0.1
        expected_me = 0.1
        me_result = ME(obs, mod)
        assert abs(me_result - expected_me) < 1e-10, f"Expected ME={expected_me}, got {me_result}"

    @pytest.mark.slow
    def test_relative_metrics_performance(self):
        """Test performance with large datasets."""
        # Generate large test dataset
        large_obs = np.random.normal(0, 1, 10000)
        large_mod = large_obs + np.random.normal(0, 0.1, 10000)  # Small noise - same size as large_obs

        import time
        start_time = time.time()
        result = NMB(large_obs, large_mod)
        end_time = time.time()

        # Should complete quickly (adjust threshold as needed)
        assert end_time - start_time < 1.0, "NMB should complete in under 1 second"
        assert isinstance(result, (float, np.floating)), "Should return a float"

    def test_nmb_with_negative_values(self):
        """Test NMB with negative values."""
        obs = np.array([-2, -1, 0, 1, 2])
        mod = np.array([-1.9, -0.9, 0.1, 1.1, 2.1])  # Small errors
        result = NMB(obs, mod)
        # This should handle the negative values appropriately
        assert not np.isnan(result), f"NMB with negative values should not be NaN, got {result}"

    def test_nme_with_positive_values(self):
        """Test NME with positive values."""
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Small errors
        result = NME(obs, mod)
        assert result >= 0, f"NME should be non-negative, got {result}"
