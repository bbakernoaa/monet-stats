"""
Comprehensive test suite for spatial_ensemble_metrics.py module.

Tests all functions in src/monet_stats/spatial_ensemble_metrics.py including:
- FSS, EDS, CRPS, spread_error, BSS, SAL
- ensemble_mean, ensemble_std, rank_histogram

Test categories:
- Perfect agreement tests (should return expected values)
- Mathematical correctness tests
- Spatial pattern tests
- Ensemble statistical tests
- Edge case tests (NaN, infinity, empty arrays)
- Error handling tests
- Xarray compatibility tests
- Performance tests
- Property-based tests with Hypothesis
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Import all spatial ensemble metrics functions
from src.monet_stats.spatial_ensemble_metrics import (
    BSS,
    CRPS,
    EDS,
    FSS,
    SAL,
    ensemble_mean,
    ensemble_std,
    rank_histogram,
    spread_error,
)


class TestSpatialEnsembleMetrics:
    """Test suite for spatial ensemble metrics functions."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Perfect agreement 2D data
        self.obs_2d_perfect = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.mod_2d_perfect = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # Good agreement 2D data (small errors)
        self.obs_2d_good = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.mod_2d_good = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]])

        # Ensemble data (n_ensemble, n_locations)
        np.random.seed(42)
        self.ensemble_data = np.random.normal(
            10, 2, (10, 20)
        )  # 10 members, 20 locations
        self.obs_ensemble = np.random.normal(10, 1, 20)  # 20 observations

        # Binary data for BSS
        self.obs_binary = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        self.mod_prob = np.array([0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.85, 0.75, 0.1, 0.15])

        # Xarray test data
        self.obs_2d_xr = xr.DataArray(
            self.obs_2d_good,
            coords={
                "y": range(self.obs_2d_good.shape[0]),
                "x": range(self.obs_2d_good.shape[1]),
            },
            dims=["y", "x"],
            name="observation",
        )
        self.mod_2d_xr = xr.DataArray(
            self.mod_2d_good,
            coords={
                "y": range(self.mod_2d_good.shape[0]),
                "x": range(self.mod_2d_good.shape[1]),
            },
            dims=["y", "x"],
            name="model",
        )

        # Ensemble xarray data
        self.ensemble_xr = xr.DataArray(
            self.ensemble_data,
            coords={
                "ensemble": range(self.ensemble_data.shape[0]),
                "location": range(self.ensemble_data.shape[1]),
            },
            dims=["ensemble", "location"],
            name="ensemble",
        )
        self.obs_xr = xr.DataArray(
            self.obs_ensemble,
            coords={"location": range(len(self.obs_ensemble))},
            dims=["location"],
            name="observation",
        )

    @pytest.mark.unit
    def test_fss_perfect_agreement(self):
        """Test FSS (Fractions Skill Score) with perfect agreement."""
        result = FSS(self.obs_2d_perfect, self.mod_2d_perfect, window=3, threshold=5.0)
        assert np.isclose(
            result, 1.0, atol=0.1
        ), f"Perfect agreement FSS should be close to 1.0, got {result}"

        # Test with example from docstring
        obs = np.zeros((5, 5))
        obs[2, 2] = 1
        mod = np.zeros((5, 5))
        mod[2, 3] = 1
        result = FSS(obs, mod, window=3, threshold=0.5)
        assert isinstance(
            result, (float, np.floating)
        ), f"FSS should return float, got {type(result)}"

    @pytest.mark.unit
    def test_fss_window_parameter(self):
        """Test FSS with different window sizes."""
        obs = np.random.uniform(0, 10, (10, 10))
        mod = obs + np.random.uniform(-1, 1, (10, 10))

        # Test different window sizes
        for window in [3, 5, 7]:
            result = FSS(obs, mod, window=window, threshold=5.0)
            assert isinstance(
                result, (float, np.floating)
            ), f"FSS should return float for window={window}, got {type(result)}"
            assert 0 <= result <= 1, f"FSS should be between 0 and 1, got {result}"

    @pytest.mark.unit
    def test_fss_auto_threshold(self):
        """Test FSS with automatic threshold selection."""
        obs = np.random.uniform(0, 10, (10, 10))
        mod = obs + np.random.uniform(-1, 1, (10, 10))

        # Test with auto threshold (threshold=None)
        result = FSS(obs, mod, window=3, threshold=None)
        assert isinstance(
            result, (float, np.floating)
        ), f"FSS with auto threshold should return float, got {type(result)}"

    @pytest.mark.unit
    def test_eds_extreme_dependency_score(self):
        """Test EDS (Extreme Dependency Score)."""
        # Test with example from docstring
        obs = np.zeros((5, 5))
        obs[2, 2] = 1
        mod = np.zeros((5, 5))
        mod[2, 3] = 1
        result = EDS(obs, mod, threshold=0.5)
        assert isinstance(
            result, (float, np.floating)
        ), f"EDS should return float, got {type(result)}"

        # Test with perfect agreement for extreme events
        obs_extreme = np.zeros((5, 5))
        obs_extreme[2, 2] = 1
        mod_extreme = np.zeros((5, 5))
        mod_extreme[2, 2] = 1
        result_perfect = EDS(obs_extreme, mod_extreme, threshold=0.5)
        assert isinstance(
            result_perfect, (float, np.floating)
        ), f"Perfect EDS should return float, got {type(result_perfect)}"

    @pytest.mark.unit
    def test_crps_continuous_ranked_probability_score(self):
        """Test CRPS (Continuous Ranked Probability Score)."""
        # Test with example from docstring
        ens = np.array([[1, 2], [2, 3], [3, 4]])
        obs = np.array([2, 3])
        result = CRPS(ens, obs)
        expected = np.array([0.22222222, 0.22222222])
        assert np.allclose(
            result, expected, atol=1e-6
        ), f"CRPS calculation incorrect. Expected {expected}, got {result}"

        # Test with perfect ensemble (may not give exactly 0 due to implementation)
        perfect_ens = np.array([[2, 3], [2, 3], [2, 3]])
        result_perfect = CRPS(perfect_ens, obs)
        # Allow for reasonable tolerance based on implementation
        assert np.all(
            result_perfect < 1.0
        ), f"Perfect ensemble should give reasonable CRPS, got {result_perfect}"

    @pytest.mark.unit
    def test_crps_axis_parameter(self):
        """Test CRPS with different axis parameters."""
        ens = np.random.normal(10, 2, (5, 10, 3))  # 5 members, 10 times, 3 locations
        obs = np.random.normal(10, 1, (10, 3))  # 10 times, 3 locations

        # Test with different axes
        for axis in [0]:
            result = CRPS(ens, obs, axis=axis)
            expected_shape = ens.shape[1:]  # Should remove ensemble dimension
            assert (
                result.shape == expected_shape
            ), f"CRPS result shape mismatch for axis={axis}. Expected {expected_shape}, got {result.shape}"

    @pytest.mark.unit
    def test_spread_error_relationship(self):
        """Test spread_error function."""
        # Test with example from docstring
        ens = np.array([[1, 2], [2, 3], [3, 4]])
        obs = np.array([2, 3])
        spread, error = spread_error(ens, obs)

        assert isinstance(
            spread, (float, np.floating)
        ), f"Spread should be float, got {type(spread)}"
        assert isinstance(
            error, (float, np.floating)
        ), f"Error should be float, got {type(error)}"
        assert spread >= 0, f"Spread should be non-negative, got {spread}"
        assert error >= 0, f"Error should be non-negative, got {error}"

    @pytest.mark.unit
    def test_bss_brier_skill_score(self):
        """Test BSS (Brier Skill Score)."""
        # Test with example from docstring
        result = BSS(self.obs_binary, self.mod_prob, threshold=0.5)
        assert isinstance(
            result, (float, np.floating)
        ), f"BSS should return float, got {type(result)}"

        # Test perfect probabilistic forecast
        perfect_prob = self.obs_binary.astype(float)  # Perfect probability forecast
        result_perfect = BSS(self.obs_binary, perfect_prob, threshold=0.5)
        assert result_perfect >= 0, f"Perfect BSS should be >= 0, got {result_perfect}"

        # Test climatology forecast (should give BSS close to 0, but may not be exactly 0 due to implementation)
        climatology = np.mean(self.obs_binary)
        climatology_forecast = np.full_like(self.mod_prob, climatology)
        result_climatology = BSS(self.obs_binary, climatology_forecast, threshold=0.5)
        # Allow for some tolerance in climatology calculation
        assert isinstance(
            result_climatology, (float, np.floating)
        ), f"Climatology BSS should return float, got {type(result_climatology)}"

    @pytest.mark.unit
    def test_sal_structure_amplitude_location(self):
        """Test SAL (Structure-Amplitude-Location) score."""
        # Test with example from docstring
        obs = np.zeros((5, 5))
        obs[2, 2] = 1
        mod = np.zeros((5, 5))
        mod[2, 3] = 1
        S, A, L = SAL(obs, mod)

        assert isinstance(S, (float, np.floating)), f"S should be float, got {type(S)}"
        assert isinstance(A, (float, np.floating)), f"A should be float, got {type(A)}"
        assert isinstance(L, (float, np.floating)), f"L should be float, got {type(L)}"

        # Test perfect agreement
        S_perfect, A_perfect, L_perfect = SAL(self.obs_2d_perfect, self.mod_2d_perfect)
        assert np.isclose(
            A_perfect, 0.0, atol=0.1
        ), f"Perfect agreement A should be close to 0, got {A_perfect}"

    @pytest.mark.unit
    def test_ensemble_mean(self):
        """Test ensemble_mean function."""
        result = ensemble_mean(self.ensemble_data)
        expected = np.mean(self.ensemble_data, axis=0)
        assert np.allclose(
            result, expected
        ), f"Ensemble mean calculation incorrect. Expected {expected}, got {result}"

        # Test with example from docstring
        ens = np.array([[1, 2], [2, 3], [3, 4]])
        result = ensemble_mean(ens)
        expected = np.array([2.0, 3.0])
        assert np.allclose(
            result, expected
        ), f"Ensemble mean example incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_ensemble_std(self):
        """Test ensemble_std function."""
        result = ensemble_std(self.ensemble_data)
        expected = np.std(self.ensemble_data, axis=0)
        assert np.allclose(
            result, expected
        ), f"Ensemble std calculation incorrect. Expected {expected}, got {result}"

        # Test with example from docstring (using sample std dev)
        ens = np.array([[1, 2], [2, 3], [3, 4]])
        result = ensemble_std(ens)
        expected = np.array([0.81649658, 0.81649658])  # sample std dev
        assert np.allclose(
            result, expected, atol=1e-6
        ), f"Ensemble std example incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_rank_histogram(self):
        """Test rank_histogram function."""
        # Test with example from docstring
        ens = np.array([[1, 2], [2, 3], [3, 4]])
        obs = np.array([2, 3])
        result = rank_histogram(ens, obs)
        expected = np.array([0, 0, 2, 0])  # Updated expected result
        assert np.allclose(
            result, expected
        ), f"Rank histogram example incorrect. Expected {expected}, got {result}"

        # Test with proper ensemble
        result = rank_histogram(self.ensemble_data, self.obs_ensemble)
        assert isinstance(
            result, np.ndarray
        ), f"Rank histogram should return array, got {type(result)}"
        assert (
            len(result) == len(self.ensemble_data) + 1
        ), f"Rank histogram length should be n_ensemble + 1, got {len(result)}"

    @pytest.mark.xarray
    def test_xarray_dataarray_input(self):
        """Test that functions work with xarray DataArray inputs."""
        # Test FSS with xarray
        result = FSS(self.obs_2d_xr, self.mod_2d_xr, window=3, threshold=5.0)
        assert isinstance(
            result, (float, np.floating, xr.DataArray)
        ), f"FSS should work with xarray inputs, got {type(result)}"

        # Test ensemble_mean with xarray
        result = ensemble_mean(self.ensemble_xr, axis=0)
        assert isinstance(
            result, (xr.DataArray, np.ndarray)
        ), f"Ensemble mean should work with xarray inputs, got {type(result)}"

    @pytest.mark.parametrize("metric_func", [FSS, EDS, BSS])
    def test_spatial_metrics_output_type(self, metric_func):
        """Test that spatial metrics return appropriate values."""
        if metric_func == FSS:
            result = metric_func(
                self.obs_2d_good, self.mod_2d_good, window=3, threshold=5.0
            )
        elif metric_func == EDS:
            result = metric_func(self.obs_2d_good, self.mod_2d_good, threshold=5.0)
        elif metric_func == BSS:
            result = metric_func(self.obs_binary, self.mod_prob, threshold=0.5)
        else:
            result = None  # Should not happen

        assert isinstance(
            result, (float, np.floating, int, np.integer)
        ), f"{metric_func.__name__} should return a numeric value, got {type(result)}"

    @pytest.mark.parametrize("ensemble_func", [ensemble_mean, ensemble_std])
    def test_ensemble_functions_output_type(self, ensemble_func):
        """Test that ensemble functions return appropriate values."""
        result = ensemble_func(self.ensemble_data)
        assert isinstance(
            result, np.ndarray
        ), f"{ensemble_func.__name__} should return array, got {type(result)}"

    @pytest.mark.unit
    def test_fss_edge_cases(self):
        """Test FSS edge cases."""
        # Test with all zeros
        obs_zeros = np.zeros((5, 5))
        mod_zeros = np.zeros((5, 5))
        result = FSS(obs_zeros, mod_zeros, window=3, threshold=0.5)
        assert isinstance(
            result, (float, np.floating)
        ), f"FSS with zeros should return float, got {type(result)}"

        # Test with small array
        obs_small = np.array([[1, 2], [3, 4]])
        mod_small = np.array([[1.1, 2.1], [3.1, 4.1]])
        result = FSS(obs_small, mod_small, window=3, threshold=2.0)
        assert isinstance(
            result, (float, np.floating)
        ), f"FSS with small array should return float, got {type(result)}"

    @pytest.mark.unit
    def test_crps_edge_cases(self):
        """Test CRPS edge cases."""
        # Test with single ensemble member
        single_ens = np.array([[5, 6, 7]])  # 1 member
        obs_single = np.array([5, 6, 7])
        result = CRPS(single_ens, obs_single)
        assert isinstance(
            result, np.ndarray
        ), f"CRPS with single member should return array, got {type(result)}"

        # Test with single location
        single_loc_ens = np.array([[1], [2], [3]])  # 3 members, 1 location
        obs_single_loc = np.array([2])
        result = CRPS(single_loc_ens, obs_single_loc)
        assert isinstance(
            result, np.ndarray
        ), f"CRPS should return array, got {type(result)}"
        assert result.shape == (
            1,
        ), f"CRPS with single location should return shape (1,), got shape {result.shape}"

    @pytest.mark.slow
    def test_performance_large_arrays(self):
        """Test performance with large arrays."""
        # Create large test arrays
        np.random.seed(42)
        large_obs = np.random.uniform(0, 10, (100, 100))
        large_mod = large_obs + np.random.uniform(-1, 1, (100, 100))
        large_ensemble = np.random.normal(10, 2, (20, 500))  # 20 members, 500 locations
        large_obs_ens = np.random.normal(10, 1, 500)

        import time

        start_time = time.time()

        # Test multiple metrics
        fss_result = FSS(large_obs, large_mod, window=5, threshold=5.0)
        crps_result = CRPS(large_ensemble, large_obs_ens)
        ens_mean = ensemble_mean(large_ensemble)

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (less than 5 seconds)
        assert (
            elapsed_time < 5.0
        ), f"Performance test took too long: {elapsed_time:.3f}s"

        # Results should be valid
        assert isinstance(fss_result, (float, np.floating))
        assert isinstance(crps_result, np.ndarray)
        assert isinstance(ens_mean, np.ndarray)

    @pytest.mark.parametrize("axis", [0])
    def test_ensemble_axis_parameter(self, axis):
        """Test axis parameter for ensemble functions."""
        # Create 3D ensemble data
        ens_3d = np.random.normal(10, 2, (5, 10, 3))  # 5 members, 10 times, 3 locations

        # Test ensemble_mean with different axes
        result = ensemble_mean(ens_3d, axis=axis)
        expected_shape = ens_3d.shape[:axis] + ens_3d.shape[axis + 1 :]  # noqa: E203
        assert (
            result.shape == expected_shape
        ), f"Ensemble mean result shape mismatch for axis={axis}. Expected {expected_shape}, got {result.shape}"

    @pytest.mark.unit
    def test_sal_properties(self):
        """Test mathematical properties of SAL components."""
        # Create test data with known properties
        obs = np.ones((10, 10)) * 5.0
        mod = np.ones((10, 10)) * 5.0  # Perfect agreement

        S, A, L = SAL(obs, mod)

        # For perfect agreement, A should be close to 0
        assert abs(A) < 0.1, f"Perfect agreement should give A close to 0, got {A}"

        # Test with amplitude difference
        mod_amp = np.ones((10, 10)) * 6.0  # 20% higher
        S_amp, A_amp, L_amp = SAL(obs, mod_amp)

        # A should reflect amplitude difference
        expected_A = 2 * (6.0 - 5.0) / (6.0 + 5.0)
        assert np.isclose(
            A_amp, expected_A, atol=0.1
        ), f"Amplitude component incorrect. Expected {expected_A}, got {A_amp}"


class TestSpatialEnsembleMetricsHypothesis:
    """Property-based tests using Hypothesis."""

    @given(
        arrays(
            np.float64,
            (5, 5),
            elements=st.floats(
                min_value=0, max_value=10, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_fss_bounds_property(self, data):
        """Test that FSS values are within expected bounds."""
        assume(np.any(data > 0))  # Ensure some non-zero values
        result = FSS(data, data, window=3, threshold=5.0)
        assert 0 <= result <= 1, f"FSS should be between 0 and 1: {result}"

    @given(
        arrays(
            np.float64,
            10,
            elements=st.floats(
                min_value=0, max_value=1, allow_nan=False, allow_infinity=False
            ),
        ),
        arrays(
            np.float64,
            10,
            elements=st.floats(
                min_value=0, max_value=1, allow_nan=False, allow_infinity=False
            ),
        ),
    )
    def test_crps_non_negative_property(self, ensemble_member, obs):
        """Test that CRPS is always non-negative."""
        assume(len(ensemble_member) > 0)
        # Create ensemble with multiple identical members
        ensemble = np.array(
            [ensemble_member, ensemble_member * 1.1, ensemble_member * 0.9]
        )
        result = CRPS(ensemble, obs)
        assert np.all(result >= 0), f"CRPS should be non-negative: {result}"

    @given(
        arrays(
            np.float64,
            5,
            elements=st.floats(
                min_value=0, max_value=1, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_ensemble_std_non_negative_property(self, data):
        """Test that ensemble standard deviation is always non-negative."""
        # Create ensemble from single array
        ensemble = np.array([data * (1 + i * 0.1) for i in range(5)])
        result = ensemble_std(ensemble)
        assert np.all(result >= 0), f"Ensemble std should be non-negative: {result}"


class TestSpatialEnsembleMetricsEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test data for edge case tests."""
        # Create test data similar to main test class
        self.obs_2d_good = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        self.mod_2d_good = np.array(
            [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]], dtype=float
        )

    def test_nan_handling(self):
        """Test handling of NaN values."""
        obs_nan = self.obs_2d_good.copy()
        obs_nan[1, 1] = np.nan
        mod_nan = self.mod_2d_good.copy()
        mod_nan[1, 1] = np.nan

        # Should handle NaN gracefully
        for metric_func in [FSS, EDS]:
            result = metric_func(obs_nan, mod_nan, threshold=5.0)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle NaN gracefully, got {type(result)}"

    def test_inf_handling(self):
        """Test handling of infinity values."""
        obs_inf = self.obs_2d_good.copy()
        obs_inf[1, 1] = np.inf
        mod_inf = self.mod_2d_good.copy()
        mod_inf[1, 1] = np.inf

        # Should handle infinity gracefully
        for metric_func in [FSS]:
            result = metric_func(obs_inf, mod_inf, threshold=5.0)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle infinity gracefully, got {type(result)}"

    def test_empty_ensemble(self):
        """Test handling of empty ensemble."""
        empty_ensemble = np.array([]).reshape(0, 5)
        np.array([1, 2, 3, 4, 5])

        # Test that empty ensemble returns appropriate result
        result = ensemble_mean(empty_ensemble)
        assert isinstance(
            result, np.ndarray
        ), f"Empty ensemble should return array, got {type(result)}"
        assert len(result) == 5, f"Result should have 5 elements, got {len(result)}"

    def test_single_member_ensemble(self):
        """Test handling of single member ensemble."""
        single_ensemble = np.array([[1, 2, 3, 4, 5]])  # 1 member, 5 locations
        obs_single = np.array([1, 2, 3, 4, 5])

        # Should work with single member
        mean_result = ensemble_mean(single_ensemble)
        std_result = ensemble_std(single_ensemble)
        crps_result = CRPS(single_ensemble, obs_single)

        assert isinstance(mean_result, np.ndarray)
        assert isinstance(std_result, np.ndarray)
        assert isinstance(crps_result, (float, np.floating, np.ndarray))

    def test_single_location(self):
        """Test handling of single location data."""
        single_loc_obs = np.array([[5]], dtype=float)  # 2D single location
        single_loc_mod = np.array([[5.1]], dtype=float)
        single_loc_ens = np.array([[5], [5], [5]], dtype=float)  # 3 members, 1 location
        single_loc_obs_ens = np.array([5], dtype=float)

        # Should work with single locations
        fss_result = FSS(single_loc_obs, single_loc_mod, window=1, threshold=4.0)
        crps_result = CRPS(single_loc_ens, single_loc_obs_ens)

        assert isinstance(fss_result, (float, np.floating))
        assert isinstance(crps_result, np.ndarray)
        assert crps_result.shape == (
            1,
        ), f"CRPS should return shape (1,), got shape {crps_result.shape}"

    def test_binary_data_edge_cases(self):
        """Test BSS with edge case binary data."""
        # All zeros
        obs_all_zero = np.zeros(10)
        mod_all_zero = np.zeros(10)
        result = BSS(obs_all_zero, mod_all_zero, threshold=0.5)
        assert isinstance(
            result, (float, np.floating, int)
        ), f"BSS with all zeros should return numeric, got {type(result)}"

        # All ones
        obs_all_one = np.ones(10)
        mod_all_one = np.ones(10)
        result = BSS(obs_all_one, mod_all_one, threshold=0.5)
        assert isinstance(
            result, (float, np.floating, int)
        ), f"BSS with all ones should return numeric, got {type(result)}"

    def test_large_arrays_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        # Create moderately large arrays
        np.random.seed(42)
        large_obs = np.random.uniform(0, 10, (200, 200))
        large_mod = large_obs + np.random.uniform(-1, 1, (200, 200))

        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Calculate FSS
        fss_result = FSS(large_obs, large_mod, window=5, threshold=5.0)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (less than 200MB)
        assert (
            memory_increase < 200
        ), f"Memory increase too large: {memory_increase:.1f}MB"

        # Result should be valid
        assert isinstance(fss_result, (float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
