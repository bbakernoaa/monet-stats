"""
Comprehensive test suite for relative_metrics.py module.

Tests all functions in src/monet_stats/relative_metrics.py including:
- NMB, WDNMB_m, NMB_ABS, NMdnB, FB
- ME, MdnE, WDME_m, WDME, WDMdnE
- NME_m, NME_m_ABS, NME, NMdnE, FE
- USUTPB, USUTPE, MNPB, MdnNPB, MNPE, MdnNPE
- NMPB, NMdnPB, NMPE, NMdnPE, PSUT wrappers
- MPE, MdnPE

Test categories:
- Perfect agreement tests (should return 0 or expected values)
- Mathematical correctness tests
- Edge case tests (NaN, infinity, zero values)
- Error handling tests
- Xarray compatibility tests
- Wind direction specific tests
- Property-based tests with Hypothesis
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# Import all relative metrics functions
from src.monet_stats.relative_metrics import (
    FB,
    FE,
    ME,
    MNPB,
    MNPE,
    MPE,
    NMB,
    NMB_ABS,
    NME,
    PSUTMNPB,
    PSUTMNPE,
    USUTPB,
    USUTPE,
    WDME,
    MdnE,
    MdnNPB,
    MdnNPE,
    MdnPE,
    NMdnB,
    NMdnE,
    NME_m,
    NME_m_ABS,
    WDMdnE,
    WDME_m,
    WDNMB_m,
)
from src.monet_stats.utils_stats import circlebias, circlebias_m


class TestRelativeMetrics:
    """Test suite for relative metrics functions."""

    def setup_method(self):
        """Set up test data for each test method."""
        # Perfect agreement data
        self.obs_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.mod_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Good agreement data (small errors)
        self.obs_good = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        self.mod_good = np.array([10.5, 20.5, 30.5, 40.5, 50.5])

        # Random positive data for percentage calculations
        np.random.seed(42)
        self.obs_random = np.random.uniform(1, 100, 50)
        self.mod_random = self.obs_random * np.random.uniform(0.8, 1.2, 50)

        # Wind direction test data (degrees)
        self.wind_obs = np.array([350, 10, 20, 45, 90])
        self.wind_mod = np.array([345, 15, 25, 50, 95])

        # 2D data for peak-based metrics
        self.obs_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.mod_2d = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]])

        # Xarray test data
        self.obs_xr = xr.DataArray(
            self.obs_good,
            coords={"time": range(len(self.obs_good))},
            dims=["time"],
            name="observation",
        )
        self.mod_xr = xr.DataArray(
            self.mod_good,
            coords={"time": range(len(self.mod_good))},
            dims=["time"],
            name="model",
        )

        # 2D Xarray data
        self.obs_2d_xr = xr.DataArray(
            self.obs_2d,
            coords={"y": range(self.obs_2d.shape[0]), "x": range(self.obs_2d.shape[1])},
            dims=["y", "x"],
            name="observation",
        )
        self.mod_2d_xr = xr.DataArray(
            self.mod_2d,
            coords={"y": range(self.mod_2d.shape[0]), "x": range(self.mod_2d.shape[1])},
            dims=["y", "x"],
            name="model",
        )

    @pytest.mark.unit
    def test_nmb_normalized_mean_bias(self):
        """Test NMB (Normalized Mean Bias)."""
        result = NMB(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement NMB should be 0%, got {result}"

        # Test with known bias
        obs = np.array([10, 20, 30])
        mod = np.array([11, 22, 33])  # 10% high bias
        result = NMB(obs, mod)
        expected = ((11 - 10) + (22 - 20) + (33 - 30)) / (10 + 20 + 30) * 100
        assert np.isclose(
            result, expected
        ), f"NMB calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_wdnmb_wind_direction_normalized_bias(self):
        """Test WDNMB_m (Wind Direction Normalized Mean Bias)."""
        # Test with perfect agreement
        result = WDNMB_m(self.wind_obs, self.wind_obs)
        assert np.isclose(
            result, 0.0
        ), f"Perfect wind direction agreement should give 0%, got {result}"

        # Test with known wind direction bias
        obs_dir = np.array([0, 90, 180, 270])
        mod_dir = np.array([5, 95, 185, 275])  # 5 degree bias
        result = WDNMB_m(obs_dir, mod_dir)
        assert isinstance(
            result, (float, np.floating)
        ), f"WDNMB_m should return numeric, got {type(result)}"

    @pytest.mark.unit
    def test_nmb_abs_normalized_bias_absolute(self):
        """Test NMB_ABS (Normalized Mean Bias with absolute denominator)."""
        obs = np.array([10, 20, 30])
        mod = np.array([11, 22, 33])  # 10% high bias

        result = NMB_ABS(obs, mod)
        expected = ((11 - 10) + (22 - 20) + (33 - 30)) / abs(10 + 20 + 30) * 100
        assert np.isclose(
            result, expected
        ), f"NMB_ABS calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_nmdnb_normalized_median_bias(self):
        """Test NMdnB (Normalized Median Bias)."""
        result = NMdnB(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement NMdnB should be 0%, got {result}"

        # Test with known median bias
        obs = np.array([10, 20, 30, 40, 50])
        mod = np.array([11, 22, 33, 44, 55])  # 10% high bias
        result = NMdnB(obs, mod)
        expected = (33 - 30) / 30 * 100  # median bias
        assert np.isclose(
            result, expected
        ), f"NMdnB calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_fb_fractional_bias(self):
        """Test FB (Fractional Bias)."""
        result = FB(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement FB should be 0%, got {result}"

        # Test with known fractional bias
        obs = np.array([10, 20])
        mod = np.array([11, 22])  # 10% high bias
        result = FB(obs, mod)
        # FB = 2 * mean((mod-obs)/(mod+obs)) * 100
        expected = 2 * np.mean([(11 - 10) / (11 + 10), (22 - 20) / (22 + 20)]) * 100
        assert np.isclose(
            result, expected
        ), f"FB calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_me_mean_gross_error(self):
        """Test ME (Mean Gross Error)."""
        result = ME(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement ME should be 0, got {result}"

        # Test with known error
        obs = np.array([10, 20, 30])
        mod = np.array([11, 21, 31])  # 1 unit error each
        result = ME(obs, mod)
        expected = np.mean([1, 1, 1])
        assert np.isclose(
            result, expected
        ), f"ME calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_mdne_median_gross_error(self):
        """Test MdnE (Median Gross Error)."""
        result = MdnE(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement MdnE should be 0, got {result}"

        # Test with known median error
        obs = np.array([10, 20, 30])
        mod = np.array([11, 21, 31])  # 1 unit error each
        result = MdnE(obs, mod)
        expected = np.median([1, 1, 1])
        assert np.isclose(
            result, expected
        ), f"MdnE calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_wdme_wind_direction_mean_error(self):
        """Test WDME_m and WDME (Wind Direction Mean Error)."""
        # Perfect agreement
        result_m = WDME_m(self.wind_obs, self.wind_obs)
        result_std = WDME(self.wind_obs, self.wind_obs)
        assert np.isclose(
            result_m, 0.0
        ), f"Perfect wind direction agreement should give 0, got {result_m}"
        assert np.isclose(
            result_std, 0.0
        ), f"Perfect wind direction agreement should give 0, got {result_std}"

        # Test with known wind direction error
        obs_dir = np.array([0, 90])
        mod_dir = np.array([10, 80])  # 10 degree errors
        result = WDME_m(obs_dir, mod_dir)
        assert isinstance(
            result, (float, np.floating)
        ), f"WDME_m should return numeric, got {type(result)}"

    @pytest.mark.unit
    def test_wdmnde_wind_direction_median_error(self):
        """Test WDMdnE (Wind Direction Median Error)."""
        result = WDMdnE(self.wind_obs, self.wind_obs)
        assert np.isclose(
            result, 0.0
        ), f"Perfect wind direction agreement should give 0, got {result}"

    @pytest.mark.unit
    def test_nme_normalized_mean_error(self):
        """Test NME_m, NME_m_ABS, and NME (Normalized Mean Error)."""
        # Perfect agreement
        result_m = NME_m(self.obs_perfect, self.mod_perfect)
        result_abs = NME_m_ABS(self.obs_perfect, self.mod_perfect)
        result_std = NME(self.obs_perfect, self.mod_perfect)

        assert np.isclose(
            result_m, 0.0
        ), f"Perfect agreement NME_m should be 0%, got {result_m}"
        assert np.isclose(
            result_abs, 0.0
        ), f"Perfect agreement NME_m_ABS should be 0%, got {result_abs}"
        assert np.isclose(
            result_std, 0.0
        ), f"Perfect agreement NME should be 0%, got {result_std}"

    @pytest.mark.unit
    def test_nmdne_normalized_median_error(self):
        """Test NMdnE (Normalized Median Error)."""
        result = NMdnE(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement NMdnE should be 0%, got {result}"

    @pytest.mark.unit
    def test_fe_fractional_error(self):
        """Test FE (Fractional Error)."""
        result = FE(self.obs_perfect, self.mod_perfect)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement FE should be 0%, got {result}"

        # Test with known fractional error
        obs = np.array([10, 20])
        mod = np.array([11, 22])  # 10% high bias
        result = FE(obs, mod)
        # FE = 2 * mean(abs(mod-obs)/(mod+obs)) * 100
        expected = (
            2 * np.mean([abs(11 - 10) / (11 + 10), abs(22 - 20) / (22 + 20)]) * 100
        )
        assert np.isclose(
            result, expected
        ), f"FE calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_usutpb_unpaired_space_time_peak_bias(self):
        """Test USUTPB (Unpaired Space/Time Peak Bias)."""
        result = USUTPB(self.obs_2d, self.obs_2d)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement USUTPB should be 0%, got {result}"

        # Test with known peak bias
        obs = np.array([1, 2, 3, 4])
        mod = np.array([1, 2, 3, 5])  # peak bias of (5-4)/4 * 100 = 25%
        result = USUTPB(obs, mod)
        expected = (5 - 4) / 4 * 100
        assert np.isclose(
            result, expected
        ), f"USUTPB calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_usutpe_unpaired_space_time_peak_error(self):
        """Test USUTPE (Unpaired Space/Time Peak Error)."""
        result = USUTPE(self.obs_2d, self.obs_2d)
        assert np.isclose(
            result, 0.0
        ), f"Perfect agreement USUTPE should be 0%, got {result}"

        # Test with known peak error
        obs = np.array([1, 2, 3, 4])
        mod = np.array([1, 2, 3, 6])  # peak error of abs(6-4)/4 * 100 = 50%
        result = USUTPE(obs, mod)
        expected = abs(6 - 4) / 4 * 100
        assert np.isclose(
            result, expected
        ), f"USUTPE calculation incorrect. Expected {expected}, got {result}"

    @pytest.mark.unit
    def test_mnpb_mean_normalized_peak_bias(self):
        """Test MNPB (Mean Normalized Peak Bias)."""
        # Test with 2D data
        result = MNPB(self.obs_2d, self.mod_2d, paxis=1, axis=None)
        assert isinstance(
            result, (float, np.floating)
        ), f"MNPB should return numeric, got {type(result)}"

    @pytest.mark.unit
    def test_mdnnpb_median_normalized_peak_bias(self):
        """Test MdnNPB (Median Normalized Peak Bias)."""
        result = MdnNPB(self.obs_2d, self.mod_2d, paxis=1, axis=None)
        assert isinstance(
            result, (float, np.floating)
        ), f"MdnNPB should return numeric, got {type(result)}"

    @pytest.mark.unit
    def test_mnpe_mean_normalized_peak_error(self):
        """Test MNPE (Mean Normalized Peak Error)."""
        result = MNPE(self.obs_2d, self.mod_2d, paxis=1, axis=None)
        assert isinstance(
            result, (float, np.floating)
        ), f"MNPE should return numeric, got {type(result)}"

    @pytest.mark.unit
    def test_mdnpe_median_normalized_peak_error(self):
        """Test MdnNPE (Median Normalized Peak Error)."""
        result = MdnNPE(self.obs_2d, self.mod_2d, paxis=1, axis=None)
        assert isinstance(
            result, (float, np.floating)
        ), f"MdnNPE should return numeric, got {type(result)}"

    @pytest.mark.unit
    def test_psut_wrapper_functions(self):
        """Test PSUT wrapper functions."""
        # Test a few PSUT wrapper functions
        result_mnpb = PSUTMNPB(self.obs_2d, self.mod_2d)
        result_mnpe = PSUTMNPE(self.obs_2d, self.mod_2d)

        assert isinstance(
            result_mnpb, (float, np.floating)
        ), f"PSUTMNPB should return numeric, got {type(result_mnpb)}"
        assert isinstance(
            result_mnpe, (float, np.floating)
        ), f"PSUTMNPE should return numeric, got {type(result_mnpe)}"

    @pytest.mark.unit
    def test_mpe_mean_peak_error(self):
        """Test MPE (Mean Peak Error)."""
        result = MPE(self.obs_2d, self.mod_2d)
        assert isinstance(
            result, (float, np.floating)
        ), f"MPE should return numeric, got {type(result)}"

    @pytest.mark.unit
    def test_mdnpe_median_peak_error(self):
        """Test MdnPE (Median Peak Error)."""
        result = MdnPE(self.obs_2d, self.mod_2d)
        assert isinstance(
            result, (float, np.floating)
        ), f"MdnPE should return numeric, got {type(result)}"

    @pytest.mark.xarray
    def test_xarray_dataarray_input(self):
        """Test that functions work with xarray DataArray inputs."""
        # Test NMB with xarray
        result = NMB(self.obs_xr, self.mod_xr)
        assert isinstance(
            result, (float, np.floating, xr.DataArray)
        ), f"NMB should work with xarray inputs, got {type(result)}"

        # Test ME with xarray
        result = ME(self.obs_xr, self.mod_xr)
        assert isinstance(
            result, (float, np.floating, xr.DataArray)
        ), f"ME should work with xarray inputs, got {type(result)}"

    @pytest.mark.xarray
    def test_xarray_2d_input(self):
        """Test that 2D functions work with xarray DataArray inputs."""
        # Test MNPB with 2D xarray using dimension names instead of axis numbers
        result = MNPB(self.obs_2d_xr, self.mod_2d_xr, paxis="x", axis=None)
        assert isinstance(
            result, (float, np.floating, xr.DataArray)
        ), f"MNPB should work with 2D xarray inputs, got {type(result)}"

    @pytest.mark.parametrize(
        "metric_func", [NMB, NMB_ABS, NMdnB, FB, ME, MdnE, NME, NMdnE, FE]
    )
    def test_relative_metrics_output_type(self, metric_func):
        """Test that relative metrics return appropriate values."""
        result = metric_func(self.obs_random, self.mod_random)
        assert isinstance(
            result, (float, np.floating, int, np.integer)
        ), f"{metric_func.__name__} should return a numeric value, got {type(result)}"

    @pytest.mark.unit
    def test_circlebias_utility_functions(self):
        """Test circlebias utility functions."""
        # Test with crossing 0/360 boundary
        angles_cross = np.array([350, 10])  # Should give circular difference
        diff = angles_cross[1] - angles_cross[0]  # 10 - 350 = -340
        result_cross = circlebias(diff)
        # circlebias should convert -340 to 20 (going the other way around the circle)
        assert (
            result_cross == 20
        ), f"circlebias should handle 0/360 crossing, got {result_cross}"

        # Test circlebias_m function
        result_m = circlebias_m(diff)
        assert (
            result_m == 20
        ), f"circlebias_m should handle 0/360 crossing, got {result_m}"

    @pytest.mark.unit
    def test_zero_division_handling(self):
        """Test handling of zero division cases."""
        # Test NMB with zero observations
        obs_zero = np.array([0, 0, 0])
        mod_zero = np.array([0, 0, 0])

        # Should handle zero gracefully (may return nan or inf depending on implementation)
        result = NMB(obs_zero, mod_zero)
        assert isinstance(
            result, (float, np.floating)
        ), f"NMB should handle zero observations, got {type(result)}"

    @pytest.mark.unit
    def test_negative_values_handling(self):
        """Test handling of negative values."""
        obs_neg = np.array([-10, -5, 0, 5, 10])
        mod_neg = np.array([-9, -4, 1, 6, 11])

        for metric_func in [NMB, ME, NME]:
            result = metric_func(obs_neg, mod_neg)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle negative values, got {type(result)}"

    @pytest.mark.slow
    def test_performance_large_arrays(self):
        """Test performance with large arrays."""
        # Create large test arrays
        np.random.seed(42)
        large_obs = np.random.uniform(1, 100, 10000)
        large_mod = large_obs * np.random.uniform(0.9, 1.1, 10000)

        import time

        start_time = time.time()

        # Test multiple metrics
        nmb_result = NMB(large_obs, large_mod)
        me_result = ME(large_obs, large_mod)
        nme_result = NME(large_obs, large_mod)

        elapsed_time = time.time() - start_time

        # Should complete in reasonable time (less than 2 seconds)
        assert (
            elapsed_time < 2.0
        ), f"Performance test took too long: {elapsed_time:.3f}s"

        # Results should be valid
        assert isinstance(nmb_result, (float, np.floating))
        assert isinstance(me_result, (float, np.floating))
        assert isinstance(nme_result, (float, np.floating))

    @pytest.mark.parametrize("axis", [None, 0])
    def test_axis_parameter(self, axis):
        """Test axis parameter for functions that support it."""
        # Create 2D data
        obs_2d = np.array([[10, 20, 30], [40, 50, 60]])
        mod_2d = np.array([[11, 21, 31], [41, 51, 61]])

        # Test functions that support axis parameter
        for metric_func in [ME, MdnE]:
            result = metric_func(obs_2d, mod_2d, axis=axis)
            if axis is None:
                assert (
                    np.isscalar(result) or result.shape == ()
                ), f"{metric_func.__name__} with axis=None should return scalar"
            else:
                expected_shape = (
                    obs_2d.shape[:axis] + obs_2d.shape[axis + 1 :]  # noqa: E203
                )  # noqa: E203
                assert (
                    result.shape == expected_shape
                ), f"{metric_func.__name__} result shape mismatch"


class TestRelativeMetricsHypothesis:
    """Property-based tests using Hypothesis."""

    @given(
        arrays(
            np.float64,
            10,
            elements=st.floats(
                min_value=1, max_value=100, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_nmb_zero_for_identical_arrays(self, data):
        """Test that NMB returns 0 for identical arrays."""
        assume(np.sum(data) != 0)  # Avoid division by zero
        result = NMB(data, data)
        assert np.isclose(
            result, 0.0, atol=1e-10
        ), f"Identical arrays should give NMB=0%, got {result}"

    @given(
        arrays(
            np.float64,
            10,
            elements=st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
        arrays(
            np.float64,
            10,
            elements=st.floats(
                min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
    )
    def test_me_non_negative_property(self, obs, mod):
        """Test that ME is always non-negative."""
        assume(len(obs) > 0)
        result = ME(obs, mod)
        assert result >= 0, f"ME should be non-negative, got {result}"

    @given(
        arrays(
            np.float64,
            5,
            elements=st.floats(
                min_value=1, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
        arrays(
            np.float64,
            5,
            elements=st.floats(
                min_value=1, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
    )
    def test_nme_me_relationship_property(self, obs, mod):
        """Test relationship between NME and ME."""
        assume(np.sum(obs) != 0)  # Avoid division by zero
        ME(obs, mod)
        nme = NME(obs, mod)
        # NME should be related to ME scaled by observations
        assert isinstance(nme, (float, np.floating)), f"NME should be finite, got {nme}"


class TestRelativeMetricsEdgeCases:
    """Test edge cases and error conditions."""

    def test_nan_handling(self):
        """Test handling of NaN values."""
        obs_nan = np.array([10.0, 20.0, np.nan, 40.0, 50.0])
        mod_nan = np.array([10.5, 20.5, 30.5, 40.5, 50.5])

        # Should handle NaN gracefully
        for metric_func in [NMB, ME, NME]:
            result = metric_func(obs_nan, mod_nan)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle NaN gracefully, got {type(result)}"

    def test_inf_handling(self):
        """Test handling of infinity values."""
        obs_inf = np.array([10.0, 20.0, np.inf, 40.0, 50.0])
        mod_inf = np.array([10.5, 20.5, 30.5, 40.5, 50.5])

        # Should handle infinity gracefully
        for metric_func in [NMB, ME]:
            result = metric_func(obs_inf, mod_inf)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle infinity gracefully, got {type(result)}"

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        obs_empty = np.array([])
        mod_empty = np.array([])

        for metric_func in [ME, NME]:
            result = metric_func(obs_empty, mod_empty)
            # Empty arrays should return NaN or similar indicator
            assert np.isnan(result) or np.ma.is_masked(
                result
            ), f"{metric_func.__name__} should handle empty arrays gracefully, got {result}"

    def test_wind_direction_boundary_cases(self):
        """Test wind direction calculations at 0/360 boundary."""
        # Test angles crossing 0/360 boundary
        obs_dir = np.array([355, 358, 1, 3])
        mod_dir = np.array([357, 1, 3, 5])

        result_me = WDME_m(obs_dir, mod_dir)
        result_std = WDME(obs_dir, mod_dir)

        assert isinstance(
            result_me, (float, np.floating)
        ), f"WDME_m should handle boundary cases, got {type(result_me)}"
        assert isinstance(
            result_std, (float, np.floating)
        ), f"WDME should handle boundary cases, got {type(result_std)}"

    def test_single_value_arrays(self):
        """Test handling of single value arrays."""
        obs_single = np.array([50.0])
        mod_single = np.array([55.0])

        for metric_func in [NMB, ME, NME]:
            result = metric_func(obs_single, mod_single)
            assert isinstance(
                result, (float, np.floating)
            ), f"{metric_func.__name__} should handle single values, got {type(result)}"

    def test_large_arrays_memory_efficiency(self):
        """Test memory efficiency with large arrays."""
        # Create moderately large arrays
        np.random.seed(42)
        large_obs = np.random.uniform(1, 100, 25000)
        large_mod = large_obs * np.random.uniform(0.9, 1.1, 25000)

        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Calculate metrics
        nmb_result = NMB(large_obs, large_mod)
        me_result = ME(large_obs, large_mod)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (less than 100MB)
        assert (
            memory_increase < 100
        ), f"Memory increase too large: {memory_increase:.1f}MB"

        # Results should be valid
        assert isinstance(nmb_result, (float, np.floating))
        assert isinstance(me_result, (float, np.floating))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
