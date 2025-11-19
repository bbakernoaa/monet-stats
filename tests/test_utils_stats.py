"""
Tests for utils_stats.py module.

This module tests utility functions for statistics including
mask handling, circular statistics, and basic statistical functions.
"""
import numpy as np
import pytest

from src.monet_stats.utils_stats import (
    angular_difference,
    circlebias,
    circlebias_m,
    correlation,
    mae,
    matchedcompressed,
    matchmasks,
    rmse,
)


class TestUtilsStats:
    """Test suite for utility statistics functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Perfect agreement case
        self.obs_perfect = np.array([1, 2, 3, 4, 5])
        self.mod_perfect = np.array([1, 2, 3, 4, 5])

        # Some differences
        self.obs_test = np.array([1, 2, 3, 4, 5])
        self.mod_test = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    def test_matchmasks_basic(self):
        """Test matchmasks with basic arrays."""
        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])

        result1, result2 = matchmasks(a1, a2)

        # Should return the same arrays since there are no masks
        np.testing.assert_array_equal(result1, a1)
        np.testing.assert_array_equal(result2, a2)

    def test_matchmasks_with_masked_arrays(self):
        """Test matchmasks with masked arrays."""
        import numpy.ma as ma

        a1 = ma.array([1, 2, 3, 4], mask=[0, 1, 0, 0])
        a2 = ma.array([4, 5, 6, 7], mask=[0, 0, 1, 0])

        result1, result2 = matchmasks(a1, a2)

        # Combined mask should be [0, 1, 1, 0] (OR of individual masks)
        expected_mask = [False, True, True, False]
        assert np.array_equal(result1.mask, expected_mask)
        assert np.array_equal(result2.mask, expected_mask)

    def test_matchmasks_with_different_masks(self):
        """Test matchmasks with different mask patterns."""
        import numpy.ma as ma

        a1 = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 1, 0])
        a2 = ma.array([4, 5, 6, 7, 8], mask=[1, 0, 1, 0, 0])

        result1, result2 = matchmasks(a1, a2)

        # Combined mask should be [1, 1, 1, 1, 0] (OR of individual masks)
        expected_mask = [True, True, True, True, False]
        assert np.array_equal(result1.mask, expected_mask)
        assert np.array_equal(result2.mask, expected_mask)

    def test_matchedcompressed_basic(self):
        """Test matchedcompressed with basic arrays."""
        a1 = np.array([1, 2, 3])
        a2 = np.array([4, 5, 6])

        result1, result2 = matchedcompressed(a1, a2)

        # Should return the same arrays since there are no masks
        np.testing.assert_array_equal(result1, a1)
        np.testing.assert_array_equal(result2, a2)

    def test_matchedcompressed_with_masked_arrays(self):
        """Test matchedcompressed with masked arrays."""
        import numpy.ma as ma

        a1 = ma.array([1, 2, 3, 4], mask=[0, 1, 0, 0])
        a2 = ma.array([4, 5, 6, 7], mask=[0, 0, 1, 0])

        result1, result2 = matchedcompressed(a1, a2)

        # Only indices where both arrays are valid should remain
        # Index 0: both valid -> [1], [4]
        # Index 1: a1 masked, a2 valid -> excluded
        # Index 2: a1 valid, a2 masked -> excluded
        # Index 3: both valid -> [4], [7]
        expected1 = np.array([1, 4])
        expected2 = np.array([4, 7])

        np.testing.assert_array_equal(result1, expected1)
        np.testing.assert_array_equal(result2, expected2)

    def test_circlebias_basic(self):
        """Test circlebias with basic values."""
        angles = np.array([190, -190, 10, -10])
        result = circlebias(angles)
        expected = np.array([-170, 170, 10, -10])
        np.testing.assert_array_equal(result, expected)

    def test_circlebias_boundary_values(self):
        """Test circlebias with boundary values."""
        # Test values at 0/360 boundaries
        angles = np.array([0, 360, -360, 180, -180])
        result = circlebias(angles)
        # All should be mapped to [-180, 180] range
        expected = np.array([0, 0, 0, -180, -180])
        np.testing.assert_array_equal(result, expected)

    def test_circlebias_m_basic(self):
        """Test circlebias_m with basic values."""
        angles = np.array([190, -190, 10, -10])
        result = circlebias_m(angles)
        expected = np.array([-170, 170, 10, -10])
        np.testing.assert_array_equal(result, expected)

    def test_circlebias_m_with_masked_array(self):
        """Test circlebias_m with masked arrays."""
        import numpy.ma as ma

        angles = ma.array([190, -190, 10, -10], mask=[0, 1, 0, 0])
        result = circlebias_m(angles)

        expected_data = np.array([-170, 170, 10, -10])
        expected_mask = [0, 1, 0, 0]

        assert np.array_equal(result.data, expected_data)
        assert np.array_equal(result.mask, expected_mask)

    def test_angular_difference_basic(self):
        """Test angular_difference with basic values."""
        result = angular_difference(10, 350, units='degrees')
        expected = 20.0
        assert abs(result - expected) < 1e-10

    def test_angular_difference_same_angle(self):
        """Test angular_difference with same angles."""
        result = angular_difference(45, 45, units='degrees')
        expected = 0.0
        assert abs(result - expected) < 1e-10

    def test_angular_difference_opposite_angles(self):
        """Test angular_difference with opposite angles."""
        result = angular_difference(0, 180, units='degrees')
        expected = 180.0
        assert abs(result - expected) < 1e-10

    def test_angular_difference_radians(self):
        """Test angular_difference with radians."""
        result = angular_difference(np.pi/4, 7*np.pi/4, units='radians')
        expected = np.pi/2  # 90 degrees in radians
        assert np.isclose(result, expected)

    def test_angular_difference_invalid_units(self):
        """Test angular_difference with invalid units."""
        with pytest.raises(ValueError):
            angular_difference(10, 20, units='invalid')

    def test_rmse_perfect_agreement(self):
        """Test rmse with perfect agreement."""
        result = rmse(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10

    def test_rmse_with_differences(self):
        """Test rmse with known differences."""
        # Differences: [0.1, 0.1, 0.1, 0.1, 0.1]
        # Squared: [0.01, 0.01, 0.01, 0.01, 0.01]
        # Mean: 0.01
        # RMSE: sqrt(0.01) = 0.1
        result = rmse(self.obs_test, self.mod_test)
        expected = 0.1
        assert abs(result - expected) < 1e-10

    def test_rmse_axis_parameter(self):
        """Test rmse with axis parameter."""
        obs_2d = np.array([[1, 2], [3, 4]])
        mod_2d = np.array([[1.1, 2.1], [3.1, 4.1]])

        # Test axis=0 (column-wise)
        result_axis0 = rmse(obs_2d, mod_2d, axis=0)
        expected_axis0 = np.array([0.1, 0.1])
        np.testing.assert_allclose(result_axis0, expected_axis0)

        # Test axis=1 (row-wise)
        result_axis1 = rmse(obs_2d, mod_2d, axis=1)
        expected_axis1 = np.array([0.1, 0.1])
        np.testing.assert_allclose(result_axis1, expected_axis1)

    def test_mae_perfect_agreement(self):
        """Test mae with perfect agreement."""
        result = mae(self.obs_perfect, self.mod_perfect)
        assert abs(result - 0.0) < 1e-10

    def test_mae_with_differences(self):
        """Test mae with known differences."""
        # Differences: [0.1, 0.1, 0.1, 0.1, 0.1]
        # Mean: 0.1
        result = mae(self.obs_test, self.mod_test)
        expected = 0.1
        assert abs(result - expected) < 1e-10

    def test_mae_axis_parameter(self):
        """Test mae with axis parameter."""
        obs_2d = np.array([[1, 2], [3, 4]])
        mod_2d = np.array([[1.1, 2.1], [3.1, 4.1]])

        # Test axis=0 (column-wise)
        result_axis0 = mae(obs_2d, mod_2d, axis=0)
        expected_axis0 = np.array([0.1, 0.1])
        np.testing.assert_allclose(result_axis0, expected_axis0)

        # Test axis=1 (row-wise)
        result_axis1 = mae(obs_2d, mod_2d, axis=1)
        expected_axis1 = np.array([0.1, 0.1])
        np.testing.assert_allclose(result_axis1, expected_axis1)

    def test_correlation_perfect_positive(self):
        """Test correlation with perfectly correlated data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # Perfect positive correlation

        result = correlation(x, y)
        expected = 1.0
        assert abs(result - expected) < 1e-10

    def test_correlation_perfect_negative(self):
        """Test correlation with perfectly negatively correlated data."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([-2, -4, -6, -8, -10])  # Perfect negative correlation

        result = correlation(x, y)
        expected = -1.0
        assert abs(result - expected) < 1e-10

    def test_correlation_no_correlation(self):
        """Test correlation with no correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])  # Perfect negative correlation
        # Actually this is perfectly negatively correlated, let's use random data

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 3, 2, 5, 4])  # Not perfectly correlated

        result = correlation(x, y)
        # Should be between -1 and 1
        assert -1 <= result <= 1

    def test_correlation_axis_parameter(self):
        """Test correlation with axis parameter."""
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[2, 4], [6, 8]])  # Each column has perfect correlation

        # Test with axis=None (flattened)
        result = correlation(x, y, axis=None)
        expected = 1.0  # Perfect correlation when flattened
        assert abs(result - expected) < 1e-10

    def test_edge_case_empty_arrays(self):
        """Test behavior with empty arrays."""
        with pytest.raises(ValueError):
            correlation(np.array([]), np.array([]))

    def test_edge_case_single_element(self):
        """Test behavior with single element arrays."""
        result = rmse(np.array([1.0]), np.array([1.0]))
        assert abs(result - 0.0) < 1e-10

    def test_edge_case_all_zeros(self):
        """Test behavior with all zero arrays."""
        obs_zeros = np.zeros(10)
        mod_zeros = np.zeros(10)

        result_rmse = rmse(obs_zeros, mod_zeros)
        assert abs(result_rmse - 0.0) < 1e-10

        result_mae = mae(obs_zeros, mod_zeros)
        assert abs(result_mae - 0.0) < 1e-10

    def test_edge_case_all_ones(self):
        """Test behavior with all one arrays."""
        obs_ones = np.ones(10)
        mod_ones = np.ones(10)

        result_rmse = rmse(obs_ones, mod_ones)
        assert abs(result_rmse - 0.0) < 1e-10

        result_mae = mae(obs_ones, mod_ones)
        assert abs(result_mae - 0.0) < 1e-10

    @pytest.mark.unit
    def test_utils_mathematical_correctness(self):
        """Test mathematical correctness of utility functions."""
        # Test RMSE calculation manually
        obs = np.array([1, 2, 3, 4, 5])
        mod = np.array([2, 3, 4, 5, 6])  # Difference of 1 for each element

        # RMSE = sqrt(mean((obs - mod)^2)) = sqrt(mean([1, 1, 1, 1])) = sqrt(1) = 1
        expected_rmse = 1.0
        result_rmse = rmse(obs, mod)
        assert abs(result_rmse - expected_rmse) < 1e-10

        # MAE = mean(abs(obs - mod)) = mean([1, 1, 1, 1, 1]) = 1
        expected_mae = 1.0
        result_mae = mae(obs, mod)
        assert abs(result_mae - expected_mae) < 1e-10

    @pytest.mark.slow
    def test_utils_performance(self):
        """Test performance with large datasets."""
        # Generate large test dataset
        large_obs = np.random.normal(0, 1, 10000)
        large_mod = large_obs + np.random.normal(0, 0.1, 10000)  # Small noise

        import time
        start_time = time.time()
        result = rmse(large_obs, large_mod)
        end_time = time.time()

        # Should complete quickly (adjust threshold as needed)
        assert end_time - start_time < 1.0, "RMSE should complete in under 1 second"
        assert isinstance(result, (float, np.floating)), "Should return a float"

    def test_angular_difference_various_cases(self):
        """Test angular_difference with various cases."""
        # Test case 1: Normal angles
        assert angular_difference(30, 60) == 30

        # Test case 2: Crossing 0/360 boundary (smaller difference)
        assert angular_difference(5, 355) == 10 # 360-355+5 = 10, not 350

        # Test case 3: Crossing 0/360 boundary (other direction)
        assert angular_difference(355, 5) == 10 # Same as above, just reversed

        # Test case 4: Opposite angles
        assert angular_difference(0, 180) == 180
        assert angular_difference(90, 270) == 180

    def test_circlebias_various_cases(self):
        """Test circlebias with various cases."""
        # Test case 1: Normal angles
        result = circlebias(np.array([45, 135, 225, 315]))
        expected = np.array([45, 135, -135, -45])  # All in [-180, 180] range
        np.testing.assert_array_equal(result, expected)

        # Test case 2: Large angles
        result = circlebias(np.array([360, 720, -360, -720]))
        expected = np.array([0, 0, 0, 0])  # All should map to 0
        np.testing.assert_array_equal(result, expected)

        # Test case 3: Angles that need adjustment
        result = circlebias(np.array([200, -200]))
        expected = np.array([-160, 160])  # 200 -> -160, -200 -> 160
        np.testing.assert_array_equal(result, expected)
