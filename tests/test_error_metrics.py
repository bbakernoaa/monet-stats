"""
Tests for error_metrics.py module.

This module tests error-based statistical metrics including
MAE, RMSE, MB, STDO, STDP, and other error statistics.
"""
import numpy as np
import pytest

from src.monet_stats.error_metrics import (
    MAE, MAE_m, MB, MNE, MNB, MO, NOP, NP, NRMSE, RM, RMdn, STDO,
    STDP, WDMB, WDMB_m, WDMdnB, MdnB, MdnNB, MdnNE, MdnO, MdnP,
    NMdnGE, MedAE, MedAE_m, RMSE_m, IOA_m, NSE_alpha, NSE_beta
)
from tests.test_utils import TestDataGenerator, validate_metric_output


class TestErrorMetrics:
    """Test suite for error metrics."""
    
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
    
    def test_mae_perfect_agreement(self):
        """Test Mean Absolute Error with perfect agreement."""
        result = MAE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MAE=0.0, got {result}"
    
    def test_mae_positive_values(self):
        """Test that MAE is always positive."""
        result = MAE(self.obs_random, self.mod_random)
        assert result >= 0, f"MAE should be non-negative, got {result}"
    
    def test_mb_perfect_agreement(self):
        """Test Mean Bias with perfect agreement."""
        result = MB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MB=0.0, got {result}"
    
    def test_mb_systematic_bias(self):
        """Test Mean Bias with systematic bias."""
        result = MB(self.obs_biased, self.mod_biased)
        expected = -0.1  # Mean bias should be -0.1 (obs - mod = -0.1 for each element)
        assert abs(result - expected) < 1e-10, f"Expected MB={expected}, got {result}"
    
    def test_rm_perfect_agreement(self):
        """Test RM (Root Mean) with perfect agreement."""
        result = RM(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give RM=0.0, got {result}"
    
    def test_rmdn_perfect_agreement(self):
        """Test Root Median with perfect agreement."""
        result = RMdn(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give RMdn=0.0, got {result}"
    
    def test_stdo_perfect_agreement(self):
        """Test STDO with perfect agreement."""
        result = STDO(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give STDO=0.0, got {result}"
    
    def test_stdp_perfect_agreement(self):
        """Test STDP with perfect agreement."""
        result = STDP(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give STDP=0.0, got {result}"
    
    def test_nrmse_perfect_agreement(self):
        """Test Normalized RMSE with perfect agreement."""
        result = NRMSE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NRMSE=0.0, got {result}"
    
    def test_wdmb_perfect_agreement(self):
        """Test Wind Direction Mean Bias with perfect agreement."""
        result = WDMB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give WDMB=0.0, got {result}"
    
    def test_wdmb_m_perfect_agreement(self):
        """Test Wind Direction Mean Bias modified with perfect agreement."""
        result = WDMB_m(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give WDMB_m=0.0, got {result}"
    
    def test_mdnb_perfect_agreement(self):
        """Test Median Bias with perfect agreement."""
        result = MdnB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MdnB=0.0, got {result}"
    
    def test_medae_perfect_agreement(self):
        """Test Median Absolute Error with perfect agreement."""
        result = MedAE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MedAE=0.0, got {result}"
    
    def test_rmse_m_perfect_agreement(self):
        """Test RMSE modified with perfect agreement."""
        result = RMSE_m(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give RMSE_m=0.0, got {result}"
    
    def test_ioa_m_perfect_agreement(self):
        """Test IOA modified with perfect agreement."""
        result = IOA_m(self.obs_perfect, self.mod_perfect)
        assert abs(result - 1.0) < 1e-10, f"Perfect agreement should give IOA_m=1.0, got {result}"
    
    def test_mne_perfect_agreement(self):
        """Test Mean Normalized Error with perfect agreement."""
        result = MNE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MNE=0.0, got {result}"
    
    def test_mnb_perfect_agreement(self):
        """Test Mean Normalized Bias with perfect agreement."""
        result = MNB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MNB=0.0, got {result}"
    
    def test_mo_perfect_agreement(self):
        """Test Mean Observed with perfect agreement."""
        result = MO(self.obs_perfect, self.mod_perfect)
        expected = np.mean(np.abs(self.obs_perfect - self.mod_perfect))
        assert abs(result - expected) < 1e-10, f"MO should match expected value, got {result}"
    
    def test_nop_perfect_agreement(self):
        """Test Number of Pairs with perfect agreement."""
        result = NOP(self.obs_perfect, self.mod_perfect)
        expected = len(self.obs_perfect)  # Should return length of arrays
        assert result == expected, f"NOP should return array length, got {result}"
    
    def test_np_perfect_agreement(self):
        """Test Number of Pairs (alternative) with perfect agreement."""
        result = NP(self.obs_perfect, self.mod_perfect)
        expected = len(self.obs_perfect)  # Should return length of arrays
        assert result == expected, f"NP should return array length, got {result}"
    
    def test_mdnne_perfect_agreement(self):
        """Test Median Normalized Error with perfect agreement."""
        result = MdnNE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MdnNE=0.0, got {result}"
    
    def test_mdno_perfect_agreement(self):
        """Test Median Observed with perfect agreement."""
        result = MdnO(self.obs_perfect, self.mod_perfect)
        expected = np.median(np.abs(self.obs_perfect - self.mod_perfect))
        assert abs(result - expected) < 1e-10, f"MdnO should match expected value, got {result}"
    
    def test_mdnp_perfect_agreement(self):
        """Test Median Predicted with perfect agreement."""
        result = MdnP(self.obs_perfect, self.mod_perfect)
        expected = np.median(np.abs(self.obs_perfect - self.mod_perfect))
        assert abs(result - expected) < 1e-10, f"MdnP should match expected value, got {result}"
    
    def test_mdnnb_perfect_agreement(self):
        """Test Median Normalized Bias with perfect agreement."""
        result = MdnNB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MdnNB=0.0, got {result}"
    
    def test_nmdnge_perfect_agreement(self):
        """Test Normalized Median GE with perfect agreement."""
        result = NMdnGE(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give NMdnGE=0.0, got {result}"
    
    @pytest.mark.parametrize("metric_func", [
        MAE, MB, MedAE, MNE, MNB, MdnB, MdnNE, MdnO, MdnP, MdnNB, NMdnGE
    ])
    def test_error_metrics_output_type(self, metric_func):
        """Test that error metrics return appropriate values."""
        result = metric_func(self.obs_random, self.mod_random)
        assert isinstance(result, (float, np.floating, int, np.integer)), \
            f"{metric_func.__name__} should return a numeric value, got {type(result)}"
    
    def test_edge_case_single_element(self):
        """Test behavior with single element arrays."""
        result = MAE(np.array([1.0]), np.array([1.0]))
        assert result == 0.0, "Single perfect match should give MAE=0.0"
        
        result = MB(np.array([1.0]), np.array([2.0]))
        assert result == -1.0, "Single element difference should give MB=-1.0"
    
    def test_edge_case_all_zeros(self):
        """Test behavior with all zero arrays."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)
        
        result = MAE(obs_zeros, mod_zeros)
        assert abs(result - 0.0) < 1e-10, f"Identical zeros should give MAE=0.0, got {result}"
        
        result = MB(obs_zeros, mod_zeros)
        assert abs(result - 0.0) < 1e-10, f"Identical zeros should give MB=0.0, got {result}"
    
    def test_edge_case_all_ones(self):
        """Test behavior with all one arrays."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)
        
        result = MAE(obs_ones, mod_ones)
        assert abs(result - 0.0) < 1e-10, f"Identical arrays should give MAE=0.0, got {result}"
    
    @pytest.mark.unit
    def test_error_metrics_mathematical_correctness(self):
        """Test mathematical correctness of error metrics."""
        # Create data with known properties
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # 0.1 bias
        
        # MAE should be 0.1 for this data (absolute error of 0.1 for each element)
        mae_result = MAE(obs, mod)
        assert abs(mae_result - 0.1) < 1e-10, f"Expected MAE=0.1, got {mae_result}"
        
        # MB should be -0.1 for this data (obs - mod = -0.1 for each element)
        mb_result = MB(obs, mod)
        assert abs(mb_result - (-0.1)) < 1e-10, f"Expected MB=-0.1, got {mb_result}"
    
    @pytest.mark.slow
    def test_error_metrics_performance(self):
        """Test performance with large datasets."""
        # Generate large test dataset
        large_obs = np.random.normal(0, 1, 10000)
        large_mod = large_obs + np.random.normal(0, 0.1, 10000)  # Small noise
        
        import time
        start_time = time.time()
        result = MAE(large_obs, large_mod)
        end_time = time.time()
        
        # Should complete quickly (adjust threshold as needed)
        assert end_time - start_time < 1.0, "MAE should complete in under 1 second"
        assert isinstance(result, (float, np.floating)), "Should return a float"
    
    def test_nse_alpha_beta(self):
        """Test NSE alpha and beta metrics."""
        result_alpha = NSE_alpha(self.obs_perfect, self.mod_perfect)
        result_beta = NSE_beta(self.obs_perfect, self.mod_perfect)
        
        # With perfect agreement, both should be close to 1
        assert abs(result_alpha - 1.0) < 1e-10, f"Perfect agreement should give NSE_alpha=1.0, got {result_alpha}"
        assert abs(result_beta - 1.0) < 1e-10, f"Perfect agreement should give NSE_beta=1.0, got {result_beta}"
    
    def test_mae_m_modified(self):
        """Test MAE modified metric."""
        result = MAE_m(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give MAE_m=0.0, got {result}"
    
    def test_wdmdnb_perfect_agreement(self):
        """Test Wind Direction Median Bias with perfect agreement."""
        result = WDMdnB(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10, f"Perfect agreement should give WDMdnB=0.0, got {result}"