"""
Tests for error_metrics.py module.

This module tests error-based statistical metrics including
MAE, RMSE, MB, STDO, STDP, and other error statistics.
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from src.monet_stats.error_metrics import (
    COE,
    CORR_INDEX,
    LOG_ERROR,
    MAE,
    MB,
    MNB,
    MNE,
    MO,
    NMSE,
    NOP,
    NP,
    NRMSE,
    RM,
    RMSE,
    STDO,
    STDP,
    VOLUMETRIC_ERROR,
    WDMB,
    IOA_m,
    MAE_m,
    MAE_norm,
    MAPE_mod,
    MASE_mod,
    MdnB,
    MdnNB,
    MdnNE,
    MdnO,
    MdnP,
    MedAE,
    NMdnGE,
    NSE_alpha,
    NSE_beta,
    RMdn,
    RMSE_m,
    RMSE_norm,
    WDMB_m,
    WDMdnB,
    bias_fraction,
)
from tests.test_utils import TestDataGenerator


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
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MAE=0.0, got {result}"

    def test_mae_positive_values(self):
        """Test that MAE is always positive."""
        result = MAE(self.obs_random, self.mod_random)
        assert result >= 0, f"MAE should be non-negative, got {result}"

    def test_mb_perfect_agreement(self):
        """Test Mean Bias with perfect agreement."""
        result = MB(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MB=0.0, got {result}"

    def test_mb_systematic_bias(self):
        """Test Mean Bias with systematic bias."""
        result = MB(self.obs_biased, self.mod_biased)
        expected = -0.1  # Mean bias should be -0.1 (obs - mod = -0.1 for each element)
        assert abs(result - expected) < 1e-10, f"Expected MB={expected}, got {result}"

    def test_rm_perfect_agreement(self):
        """Test RM (Root Mean) with perfect agreement."""
        result = RM(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give RM=0.0, got {result}"

    def test_rmdn_perfect_agreement(self):
        """Test Root Median with perfect agreement."""
        result = RMdn(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give RMdn=0.0, got {result}"

    def test_stdo_perfect_agreement(self):
        """Test STDO with perfect agreement."""
        result = STDO(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give STDO=0.0, got {result}"

    def test_stdp_perfect_agreement(self):
        """Test STDP with perfect agreement."""
        result = STDP(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give STDP=0.0, got {result}"

    def test_nrmse_perfect_agreement(self):
        """Test Normalized RMSE with perfect agreement."""
        result = NRMSE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give NRMSE=0.0, got {result}"

    def test_wdmb_perfect_agreement(self):
        """Test Wind Direction Mean Bias with perfect agreement."""
        result = WDMB(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give WDMB=0.0, got {result}"

    def test_wdmb_m_perfect_agreement(self):
        """Test Wind Direction Mean Bias modified with perfect agreement."""
        result = WDMB_m(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give WDMB_m=0.0, got {result}"

    def test_mdnb_perfect_agreement(self):
        """Test Median Bias with perfect agreement."""
        result = MdnB(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MdnB=0.0, got {result}"

    def test_medae_perfect_agreement(self):
        """Test Median Absolute Error with perfect agreement."""
        result = MedAE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MedAE=0.0, got {result}"

    def test_rmse_m_perfect_agreement(self):
        """Test RMSE modified with perfect agreement."""
        result = RMSE_m(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give RMSE_m=0.0, got {result}"

    def test_ioa_m_perfect_agreement(self):
        """Test IOA modified with perfect agreement."""
        result = IOA_m(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 1.0) < 1e-10
        ), f"Perfect agreement should give IOA_m=1.0, got {result}"

    def test_mne_perfect_agreement(self):
        """Test Mean Normalized Error with perfect agreement."""
        result = MNE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MNE=0.0, got {result}"

    def test_mnb_perfect_agreement(self):
        """Test Mean Normalized Bias with perfect agreement."""
        result = MNB(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MNB=0.0, got {result}"

    def test_mo_perfect_agreement(self):
        """Test Mean Observed with perfect agreement."""
        result = MO(self.obs_perfect, self.mod_perfect)
        expected = np.mean(np.abs(self.obs_perfect - self.mod_perfect))
        assert (
            abs(result - expected) < 1e-10
        ), f"MO should match expected value, got {result}"

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
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MdnNE=0.0, got {result}"

    def test_mdno_perfect_agreement(self):
        """Test Median Observed with perfect agreement."""
        result = MdnO(self.obs_perfect, self.mod_perfect)
        expected = np.median(np.abs(self.obs_perfect - self.mod_perfect))
        assert (
            abs(result - expected) < 1e-10
        ), f"MdnO should match expected value, got {result}"

    def test_mdnp_perfect_agreement(self):
        """Test Median Predicted with perfect agreement."""
        result = MdnP(self.obs_perfect, self.mod_perfect)
        expected = np.median(np.abs(self.obs_perfect - self.mod_perfect))
        assert (
            abs(result - expected) < 1e-10
        ), f"MdnP should match expected value, got {result}"

    def test_mdnnb_perfect_agreement(self):
        """Test Median Normalized Bias with perfect agreement."""
        result = MdnNB(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MdnNB=0.0, got {result}"

    def test_nmdnge_perfect_agreement(self):
        """Test Normalized Median GE with perfect agreement."""
        result = NMdnGE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give NMdnGE=0.0, got {result}"

    @pytest.mark.parametrize(
        "metric_func",
        [MAE, MB, MedAE, MNE, MNB, MdnB, MdnNE, MdnO, MdnP, MdnNB, NMdnGE],
    )
    def test_error_metrics_output_type(self, metric_func):
        """Test that error metrics return appropriate values."""
        result = metric_func(self.obs_random, self.mod_random)
        assert isinstance(
            result, (float, np.floating, int, np.integer)
        ), f"{metric_func.__name__} should return a numeric value, got {type(result)}"

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
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Identical zeros should give MAE=0.0, got {result}"

        result = MB(obs_zeros, mod_zeros)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Identical zeros should give MB=0.0, got {result}"

    def test_edge_case_all_ones(self):
        """Test behavior with all one arrays."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)

        result = MAE(obs_ones, mod_ones)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Identical arrays should give MAE=0.0, got {result}"

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
        assert (
            abs(result_alpha - 1.0) < 1e-10
        ), f"Perfect agreement should give NSE_alpha=1.0, got {result_alpha}"
        assert (
            abs(result_beta - 1.0) < 1e-10
        ), f"Perfect agreement should give NSE_beta=1.0, got {result_beta}"

    def test_mae_m_modified(self):
        """Test MAE modified metric."""
        result = MAE_m(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MAE_m=0.0, got {result}"

    def test_wdmdnb_perfect_agreement(self):
        """Test Wind Direction Median Bias with perfect agreement."""
        result = WDMdnB(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give WDMdnB=0.0, got {result}"

    # Tests for missing error metric functions
    def test_mape_mod_perfect_agreement(self):
        """Test Modified MAPE with perfect agreement."""
        result = MAPE_mod(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MAPE_mod=0.0, got {result}"

    def test_mase_mod_perfect_agreement(self):
        """Test Modified MASE with perfect agreement."""
        result = MASE_mod(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MASE_mod=0.0, got {result}"

    def test_rmse_norm_perfect_agreement(self):
        """Test Normalized RMSE with perfect agreement."""
        result = RMSE_norm(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give RMSE_norm=0.0, got {result}"

    def test_mae_norm_perfect_agreement(self):
        """Test Normalized MAE with perfect agreement."""
        result = MAE_norm(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give MAE_norm=0.0, got {result}"

    def test_bias_fraction_perfect_agreement(self):
        """Test Bias Fraction with perfect agreement."""
        result = bias_fraction(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give bias_fraction=0.0, got {result}"

    def test_nmse_perfect_agreement(self):
        """Test Normalized MSE with perfect agreement."""
        result = NMSE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give NMSE=0.0, got {result}"

    def test_log_error_perfect_agreement(self):
        """Test Log Error with perfect agreement."""
        # Use positive values to avoid log of zero/negative
        obs_pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mod_pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = LOG_ERROR(obs_pos, mod_pos)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give LOG_ERROR=0.0, got {result}"

    def test_coe_perfect_agreement(self):
        """Test Center of Mass Error with perfect agreement."""
        result = COE(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give COE=0.0, got {result}"

    def test_volumetric_error_perfect_agreement(self):
        """Test Volumetric Error with perfect agreement."""
        result = VOLUMETRIC_ERROR(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - 0.0) < 1e-10
        ), f"Perfect agreement should give VOLUMETRIC_ERROR=0.0, got {result}"

    def test_corr_index_perfect_agreement(self):
        """Test Correlation Index with perfect agreement."""
        result = CORR_INDEX(self.obs_perfect, self.mod_perfect)
        # Convert to float for comparison
        result_float = (
            np.asarray(result).item() if hasattr(result, "item") else float(result)
        )
        assert (
            abs(result_float - 1.0) < 1e-10
        ), f"Perfect agreement should give CORR_INDEX=1.0, got {result_float}"

    @pytest.mark.parametrize(
        "metric_func,expected_value",
        [
            (MAPE_mod, 0.0),
            (MASE_mod, 0.0),
            (RMSE_norm, 0.0),
            (MAE_norm, 0.0),
            (bias_fraction, 0.0),
            (NMSE, 0.0),
            (VOLUMETRIC_ERROR, 0.0),
        ],
    )
    def test_missing_error_functions_perfect_agreement(
        self, metric_func, expected_value
    ):
        """Test perfect agreement for all missing error metric functions."""
        result = metric_func(self.obs_perfect, self.mod_perfect)
        assert (
            abs(result - expected_value) < 1e-10
        ), f"{metric_func.__name__} should give {expected_value} for perfect agreement, got {result}"

    def test_mathematical_correctness_mape_mod(self):
        """Test mathematical correctness of MAPE_mod."""
        obs = np.array([1, 2, 3])
        mod = np.array([2, 2, 4])
        result = MAPE_mod(obs, mod)
        # Manual calculation with epsilon handling
        # Should be close to regular MAPE for non-zero values
        expected = np.mean(np.abs((mod - obs) / obs)) * 100
        assert (
            abs(result - expected) < 1e-10
        ), f"MAPE_mod should match expected calculation, got {result}"

    def test_mathematical_correctness_nmse(self):
        """Test mathematical correctness of NMSE."""
        obs = np.array([1, 2, 3, 4])
        mod = np.array([2, 2, 2, 2])
        result = NMSE(obs, mod)
        # Manual calculation: mse = ((1-2)^2 + (2-2)^2 + (3-2)^2 + (4-2)^2)/4 = (1+0+1+4)/4 = 1.5
        # obs_var = var([1,2,3,4]) = 1.25
        # NMSE = 1.5 / 1.25 = 1.2
        mse = np.mean((mod - obs) ** 2)
        obs_var = np.var(obs)
        expected = mse / obs_var
        assert (
            abs(result - expected) < 1e-10
        ), f"NMSE should be {expected}, got {result}"

    def test_edge_cases_error_metrics(self):
        """Test edge cases for error metrics."""
        # Test with zeros for MAPE_mod (should handle division by zero)
        zeros = np.zeros(5)
        result_mape = MAPE_mod(zeros, zeros)
        assert np.isfinite(result_mape), "MAPE_mod should handle zeros"

        # Test with constants for NMSE
        constants = np.ones(5) * 3
        result_nmse = NMSE(constants, constants)
        assert abs(result_nmse - 0.0) < 1e-10, "NMSE should handle constants"

        # Test with single element
        single_obs = np.array([5.0])
        single_mod = np.array([5.0])
        result_volumetric = VOLUMETRIC_ERROR(single_obs, single_mod)
        assert (
            abs(result_volumetric - 0.0) < 1e-10
        ), "VOLUMETRIC_ERROR should handle single elements"

    def test_error_handling_error_metrics(self):
        """Test error handling for error metrics."""
        # Test with mismatched dimensions
        obs_short = np.array([1, 2])
        mod_long = np.array([1, 2, 3, 4])

        with pytest.raises((ValueError, IndexError)):
            MAPE_mod(obs_short, mod_long)

        # Test with negative values for LOG_ERROR
        obs_neg = np.array([-1, -2, -3])
        mod_neg = np.array([-1, -2, -3])

        # LOG_ERROR should handle negative values with epsilon
        result = LOG_ERROR(obs_neg, mod_neg)
        assert np.isfinite(result), "LOG_ERROR should handle negative values"

    @pytest.mark.unit
    def test_error_metrics_mathematical_properties(self):
        """Test mathematical properties of error metrics."""
        # Test that normalized metrics are scale-independent
        obs1 = np.array([1, 2, 3])
        mod1 = np.array([2, 3, 4])
        obs2 = np.array([10, 20, 30])  # Scaled by 10
        mod2 = np.array([20, 30, 40])  # Scaled by 10

        norm1 = RMSE_norm(obs1, mod1)
        norm2 = RMSE_norm(obs2, mod2)
        assert abs(norm1 - norm2) < 1e-10, "RMSE_norm should be scale-independent"

        # Test that bias_fraction is between 0 and 1
        obs = np.array([1, 2, 3, 4])
        mod = np.array([1.5, 2.5, 3.5, 4.5])
        bf = bias_fraction(obs, mod)
        assert 0.0 <= bf <= 1.0, f"Bias fraction should be in [0,1], got {bf}"

    @given(
        st.lists(
            st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=50,
        ),
        st.lists(
            st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=50,
        ),
    )
    def test_property_based_mape_mod(self, obs_list, mod_list):
        """Property-based test for MAPE_mod function."""
        obs = np.array(obs_list)
        mod = np.array(mod_list)

        # Skip if arrays have different lengths
        if len(obs) != len(mod):
            return

        result = MAPE_mod(obs, mod)

        # MAPE should be >= 0
        assert result >= 0.0, f"MAPE_mod should be >= 0.0, got {result}"

        # MAPE should be finite
        assert np.isfinite(result), f"MAPE_mod should be finite, got {result}"

    @given(
        st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=50,
        ),
        st.lists(
            st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=50,
        ),
    )
    def test_property_based_nmse(self, obs_list, mod_list):
        """Property-based test for NMSE function."""
        obs = np.array(obs_list)
        mod = np.array(mod_list)

        # Skip if arrays have different lengths
        if len(obs) != len(mod):
            return

        result = NMSE(obs, mod)

        # NMSE should be >= 0
        assert result >= 0.0, f"NMSE should be >= 0.0, got {result}"

        # NMSE should be finite
        assert np.isfinite(result), f"NMSE should be finite, got {result}"

    @given(
        st.lists(
            st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=50,
        ),
        st.lists(
            st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
            min_size=2,
            max_size=50,
        ),
    )
    def test_property_based_volumetric_error(self, obs_list, mod_list):
        """Property-based test for VOLUMETRIC_ERROR function."""
        obs = np.array(obs_list)
        mod = np.array(mod_list)

        # Skip if arrays have different lengths
        if len(obs) != len(mod):
            return

        result = VOLUMETRIC_ERROR(obs, mod)

        # Volumetric error should be >= 0
        assert result >= 0.0, f"VOLUMETRIC_ERROR should be >= 0.0, got {result}"

        # Volumetric error should be finite
        assert np.isfinite(result), f"VOLUMETRIC_ERROR should be finite, got {result}"

    def test_xarray_compatibility_error_metrics(self):
        """Test xarray compatibility for missing error metric functions."""
        obs_xr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        mod_xr = xr.DataArray([1.1, 2.1, 3.1, 4.1, 5.1], dims=["time"])

        # Test MAPE_mod with xarray
        result_mape = MAPE_mod(obs_xr, mod_xr)
        assert isinstance(
            result_mape, xr.DataArray
        ), "MAPE_mod should return xarray.DataArray"
        assert np.isfinite(result_mape), "MAPE_mod should return finite value"

        # Test NMSE with xarray
        result_nmse = NMSE(obs_xr, mod_xr)
        assert isinstance(
            result_nmse, xr.DataArray
        ), "NMSE should return xarray.DataArray"
        assert np.isfinite(result_nmse), "NMSE should return finite value"

        # Test VOLUMETRIC_ERROR with xarray
        result_vol = VOLUMETRIC_ERROR(obs_xr, mod_xr)
        assert isinstance(
            result_vol, xr.DataArray
        ), "VOLUMETRIC_ERROR should return xarray.DataArray"
        assert np.isfinite(result_vol), "VOLUMETRIC_ERROR should return finite value"


class TestErrorMetricsXarray:
    """Test suite for error metrics with xarray inputs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.obs_xr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        self.mod_xr = xr.DataArray([1.1, 2.1, 3.1, 4.1, 5.1], dims=["time"])

    def test_MAE_xarray(self):
        """Test MAE with xarray inputs."""
        result = MAE(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, 0.1)

    def test_RMSE_xarray(self):
        """Test RMSE with xarray inputs."""
        result = RMSE(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, 0.1)

    def test_MB_xarray(self):
        """Test MB with xarray inputs."""
        result = MB(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, -0.1)

    def test_MNB_xarray(self):
        """Test MNB with xarray inputs."""
        result = MNB(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, 4.56666667)

    def test_MNE_xarray(self):
        """Test MNE with xarray inputs."""
        result = MNE(self.obs_xr, self.mod_xr)
        assert isinstance(result, xr.DataArray)
        assert np.isclose(result, 4.56666667)
