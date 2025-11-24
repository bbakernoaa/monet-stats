"""
Property-based tests for statistical metrics using Hypothesis.

These tests verify general properties and invariants that should hold across
a wide range of inputs, helping to catch edge cases and unexpected behavior.
"""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from monet_stats.contingency_metrics import CSI, FAR, POD
from monet_stats.correlation_metrics import IOA, R2, RMSE, pearsonr
from monet_stats.efficiency_metrics import MAPE, MSE, NSE
from monet_stats.error_metrics import MAE
from monet_stats.relative_metrics import NMB
from monet_stats.utils_stats import angular_difference, circlebias


class TestPropertyBased:
    """Property-based tests for statistical metrics."""

    # Define strategies for generating test data
    float_array = arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=10),
        elements=st.floats(min_value=-100, max_value=1000, allow_nan=False, allow_infinity=False),
    )

    @given(obs=float_array, mod=float_array)
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_rmse_non_negative(self, obs, mod):
        """Test that RMSE is always non-negative."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        result = RMSE(obs, mod)
        assert result >= 0, f"RMSE should be non-negative, got {result}"

    @given(obs=float_array, mod=float_array)
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_mae_non_negative(self, obs, mod):
        """Test that MAE is always non-negative."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        result = MAE(obs, mod)
        assert result >= 0, f"MAE should be non-negative, got {result}"

    @given(obs=float_array, mod=float_array)
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_perfect_agreement_metrics(self, obs, mod):
        """Test that metrics behave correctly with perfect agreement."""
        # Use the same array for both obs and mod to ensure perfect agreement
        result_rmse = RMSE(obs, obs)
        result_mae = MAE(obs, obs)

        # Both should be very close to 0
        assert result_rmse < 1e-10, f"RMSE with perfect agreement should be ~0, got {result_rmse}"
        assert result_mae < 1e-10, f"MAE with perfect agreement should be ~0, got {result_mae}"

    @given(value=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
    def test_circlebias_range(self, value):
        """Test that circlebias returns values in [-180, 180] range."""
        result = circlebias(np.array([value]))
        assert -180 <= result[0] <= 180, f"circlebias result {result[0]} not in [-180, 180] range"

    @given(
        angle1=st.floats(min_value=0, max_value=360, allow_nan=False, allow_infinity=False),
        angle2=st.floats(min_value=0, max_value=360, allow_nan=False, allow_infinity=False),
    )
    def test_angular_difference_range(self, angle1, angle2):
        """Test that angular_difference returns values in [0, 180] range for degrees."""
        result = angular_difference(angle1, angle2, units="degrees")
        assert 0 <= result <= 180, f"angular_difference result {result} not in [0, 180] range"

    @given(obs=float_array, mod=float_array)
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_r2_upper_bound(self, obs, mod):
        """Test that R2 is generally <= 1.0."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        # Skip if obs has no variance (would cause division by zero)
        if np.var(obs) == 0:
            return

        result = R2(obs, mod)
        # R2 can be less than 0 for very poor models, but should be <= 1
        assert result <= 1.0, f"R2 should be <= 1.0, got {result}"

    @given(obs=float_array, mod=float_array)
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_nse_upper_bound(self, obs, mod):
        """Test that NSE is generally <= 1.0."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        # Skip if obs has no variance (would cause division by zero)
        if np.var(obs) == 0:
            return

        result = NSE(obs, mod)
        # NSE can be less than 0 for very poor models, but should be <= 1
        assert result <= 1.0, f"NSE should be <= 1.0, got {result}"

    @given(
        obs=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=50),
            elements=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
        mod=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=50),
            elements=st.floats(min_value=1, max_value=10, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_mse_vs_rmse_relationship(self, obs, mod):
        """Test that MSE and RMSE are related (RMSE = sqrt(MSE))."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        mse_result = MSE(obs, mod)
        rmse_result = RMSE(obs, mod)

        # RMSE should be sqrt of MSE
        expected_rmse = np.sqrt(mse_result)
        assert abs(rmse_result - expected_rmse) < 1e-10, f"RMSE {rmse_result} should equal sqrt(MSE) {expected_rmse}"

    @given(
        obs=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=-100, max_value=10, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_correlation_range(self, obs):
        """Test that correlation coefficient is in [-1, 1] range."""
        # Create mod as a linear transformation of obs plus noise
        mod = obs + np.random.normal(0, 0.1, size=obs.shape)

        if len(obs) < 2:
            return  # Need at least 2 points for correlation

        # Get correlation (handle potential tuple return)
        corr_result = pearsonr(obs, mod)
        if isinstance(corr_result, tuple):
            corr = corr_result[0]
        else:
            corr = corr_result

        assert -1 <= corr <= 1, f"Correlation {corr} not in [-1, 1] range"

    @given(
        obs=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
        mod=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_contingency_metrics_bounds(self, obs, mod):
        """Test that contingency metrics are in valid ranges."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        # Use median as threshold to ensure we get some events
        threshold = np.median(obs)

        try:
            pod_result = POD(obs, mod, threshold)
            far_result = FAR(obs, mod, threshold)
            csi_result = CSI(obs, mod, threshold)

            # These should all be in [0, 1] range if defined
            if np.isfinite(pod_result):
                assert 0 <= pod_result <= 1, f"POD {pod_result} not in [0, 1] range"
            if np.isfinite(far_result):
                assert 0 <= far_result <= 1, f"FAR {far_result} not in [0, 1] range"
            if np.isfinite(csi_result):
                assert 0 <= csi_result <= 1, f"CSI {csi_result} not in [0, 1] range"
        except ZeroDivisionError:
            # This can happen when there are no events to evaluate
            pass

    @given(
        obs=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
        mod=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_mape_non_negative(self, obs, mod):
        """Test that MAPE is non-negative."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        try:
            result = MAPE(obs, mod)
            if np.isfinite(result):
                assert result >= 0, f"MAPE {result} should be non-negative"
        except ZeroDivisionError:
            # This can happen when obs contains zeros
            pass

    @given(
        obs=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=1, max_value=10, allow_nan=False, allow_infinity=False),
        ),
        mod=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_symmetric_metrics_with_swapped_inputs(self, obs, mod):
        """Test that some metrics behave appropriately when inputs are swapped."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        # RMSE should be the same when inputs are swapped
        rmse_original = RMSE(obs, mod)
        rmse_swapped = RMSE(mod, obs)
        assert (
            abs(rmse_original - rmse_swapped) < 1e-10
        ), f"RMSE should be symmetric, got {rmse_original} vs {rmse_swapped}"

        # MAE should be the same when inputs are swapped
        mae_original = MAE(obs, mod)
        mae_swapped = MAE(mod, obs)
        assert abs(mae_original - mae_swapped) < 1e-10, f"MAE should be symmetric, got {mae_original} vs {mae_swapped}"

    @given(
        obs=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
        mod=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=10, max_value=50),
            elements=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_relative_metrics_with_scaled_inputs(self, obs, mod):
        """Test behavior of relative metrics with scaled inputs."""
        if len(obs) != len(mod):
            # Truncate to same length
            min_len = min(len(obs), len(mod))
            obs = obs[:min_len]
            mod = mod[:min_len]

        # Scale both arrays by the same factor
        scale_factor = 10.0
        scaled_obs = obs * scale_factor
        scaled_mod = mod * scale_factor

        # NMB should be the same with scaled values (relative metric)
        try:
            nmb_original = NMB(obs, mod)
            nmb_scaled = NMB(scaled_obs, scaled_mod)

            if np.isfinite(nmb_original) and np.isfinite(nmb_scaled):
                assert (
                    abs(nmb_original - nmb_scaled) < 1e-10
                ), f"NMB should be scale-invariant, got {nmb_original} vs {nmb_scaled}"
        except ZeroDivisionError:
            # This can happen when mean of obs is 0
            pass

    def test_edge_case_zeros(self):
        """Test behavior with arrays containing zeros."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)

        # Test metrics that should handle zeros appropriately
        rmse_result = RMSE(obs_zeros, mod_zeros)
        assert abs(rmse_result - 0.0) < 1e-10, f"RMSE with zeros should be 0, got {rmse_result}"

        mae_result = MAE(obs_zeros, mod_zeros)
        assert abs(mae_result - 0.0) < 1e-10, f"MAE with zeros should be 0, got {mae_result}"

    def test_edge_case_ones(self):
        """Test behavior with arrays of all ones."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)

        rmse_result = RMSE(obs_ones, mod_ones)
        assert abs(rmse_result - 0.0) < 1e-10, f"RMSE with all ones should be 0, got {rmse_result}"

        nse_result = NSE(obs_ones, mod_ones)
        # With identical arrays, NSE should be 1
        assert abs(nse_result - 1.0) < 1e-10, f"NSE with identical arrays should be 1, got {nse_result}"

    def test_edge_case_constant_arrays(self):
        """Test behavior with constant but different arrays."""
        obs_const = np.ones(10) * 5
        mod_const = np.ones(10) * 6  # Different constant

        rmse_result = RMSE(obs_const, mod_const)
        assert abs(rmse_result - 1.0) < 1e-10, f"RMSE between 5s and 6s should be 1, got {rmse_result}"

        mae_result = MAE(obs_const, mod_const)
        assert abs(mae_result - 1.0) < 1e-10, f"MAE between 5s and 6s should be 1, got {mae_result}"

    @given(single_value=st.floats(min_value=-360, max_value=360, allow_nan=False, allow_infinity=False))
    def test_circlebias_single_values(self, single_value):
        """Test circlebias with single values."""
        result = circlebias(np.array([single_value]))
        # Result should be in [-180, 180] range
        assert -180 <= result[0] <= 180, f"circlebias result {result[0]} not in [-180, 180] range"

    @given(
        arr1=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=20),
            elements=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
        arr2=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=5, max_value=20),
            elements=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=1000)
    def test_ioa_upper_bound(self, arr1, arr2):
        """Test that IOA is generally <= 1.0."""
        if len(arr1) != len(arr2):
            min_len = min(len(arr1), len(arr2))
            arr1 = arr1[:min_len]
            arr2 = arr2[:min_len]

        # Skip if the denominator is zero
        obsmean = arr1.mean()
        denom = ((np.abs(arr2 - obsmean) + np.abs(arr1 - obsmean)) ** 2).sum()
        if denom == 0:
            return

        result = IOA(arr1, arr2)
        # IOA should be <= 1.0
        assert result <= 1.0, f"IOA should be <= 1.0, got {result}"
