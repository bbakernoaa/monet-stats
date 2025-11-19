"""
Regression testing with mathematical accuracy validation.
Tests known values and mathematical correctness.
"""
import numpy as np
import pytest

from monet_stats.test_aliases import (
    coefficient_of_determination,
    critical_success_index,
    equitable_threat_score,
    false_alarm_rate,
    hit_rate,
    index_of_agreement,
    mean_absolute_error,
    mean_bias_error,
    modified_index_of_agreement,
    pearson_correlation,
    root_mean_squared_error,
    spearman_correlation,
)
from tests.test_utils import TestDataGenerator


class TestKnownValues:
    """Test against known analytical values."""

    def test_simple_linear_relationship(self):
        """Test metrics with simple linear relationship where analytical values are known."""
        # Simple case: y = 2x + 1
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 1  # Perfect linear relationship

        # For perfect linear relationship: y = 2x + 1
        # Error metrics should be zero when comparing x with (y-1)/2
        expected_x = (y - 1) / 2

        mae = mean_absolute_error(x, expected_x)
        rmse = root_mean_squared_error(x, expected_x)
        bias = mean_bias_error(x, expected_x)

        assert abs(mae) < 1e-10, f"MAE should be zero for perfect relationship, got {mae}"
        assert abs(rmse) < 1e-10, f"RMSE should be zero for perfect relationship, got {rmse}"
        assert abs(bias) < 1e-10, f"Bias should be zero for perfect relationship, got {bias}"

        # Correlation should be 1.0
        corr = pearson_correlation(x, y)
        r2 = coefficient_of_determination(x, y)

        assert abs(corr - 1.0) < 1e-10, f"Correlation should be 1.0, got {corr}"
        assert abs(r2 - 1.0) < 1e-10, f"R² should be 1.0, got {r2}"

    def test_known_statistical_examples(self):
        """Test with known statistical examples from literature."""
        # Example from standard statistics textbooks
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # For y = 2x, correlation should be 1.0
        corr = pearson_correlation(x, y)
        assert abs(corr - 1.0) < 1e-10, f"Perfect correlation should be 1.0, got {corr}"

        # MAE for this case
        mae = mean_absolute_error(x, y/2)  # y/2 = x
        assert abs(mae) < 1e-10, f"MAE should be zero, got {mae}"

        # Test with some noise
        y_noisy = y + np.array([0.1, -0.2, 0.1, -0.1, 0.2])

        # Should have small but non-zero error
        mae_noisy = mean_absolute_error(x, y_noisy/2)
        assert 0 < mae_noisy < 0.5, f"MAE with noise should be small but positive, got {mae_noisy}"

    def test_contingency_table_known_values(self):
        """Test contingency table metrics with known values."""
        # Classic 2x2 contingency table
        #         Obs
        #        Yes  No
        # Mod Yes 30   10
        #     No  20   40

        # obs: 50 yes, 50 no
        # mod: 40 yes, 60 no
        # hits: 30, false alarms: 10, misses: 20, correct negatives: 40
        obs = np.concatenate([np.ones(30), np.zeros(10), np.ones(20), np.zeros(40)])
        mod = np.concatenate([np.ones(40), np.zeros(60)])

        hr = hit_rate(obs, mod)
        far = false_alarm_rate(obs, mod)
        csi = critical_success_index(obs, mod)

        # Expected values:
        # HR = 30/(30+20) = 0.6
        # FAR = 10/(30+10) = 0.25
        # CSI = 30/(30+20+10) = 0.5

        assert abs(hr - 0.6) < 1e-10, f"Hit rate should be 0.6, got {hr}"
        assert abs(far - 0.25) < 1e-10, f"False alarm rate should be 0.25, got {far}"
        assert abs(csi - 0.5) < 1e-10, f"CSI should be 0.5, got {csi}"


class TestMathematicalIdentities:
    """Test mathematical identities and relationships."""

    def test_jensens_inequality(self):
        """Test Jensen's inequality: RMSE >= MAE."""
        data_gen = TestDataGenerator()

        # Test with various correlation levels
        for correlation in [0.0, 0.5, 0.8, 0.95]:
            obs, mod = data_gen.generate_correlated_data(n_samples=1000, correlation=correlation)

            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)

            # RMSE should be >= MAE (equality when all errors are equal)
            assert rmse >= mae - 1e-10, f"Jensen's inequality violated: RMSE ({rmse}) < MAE ({mae}) for correlation {correlation}"

    def test_correlation_bounds(self):
        """Test that correlation coefficients are within valid bounds."""
        data_gen = TestDataGenerator()

        # Test with various data types
        for _ in range(10):
            obs, mod = data_gen.generate_correlated_data(n_samples=100, correlation=0.7)

            # Add some noise to avoid perfect correlations
            obs += np.random.normal(0, 0.1, len(obs))
            mod += np.random.normal(0, 0.1, len(mod))

            pearson_r = pearson_correlation(obs, mod)
            spearman_r = spearman_correlation(obs, mod)
            r2 = coefficient_of_determination(obs, mod)

            # Check bounds
            assert -1 <= pearson_r <= 1, f"Pearson correlation {pearson_r} outside [-1, 1]"
            assert -1 <= spearman_r <= 1, f"Spearman correlation {spearman_r} outside [-1, 1]"
            assert 0 <= r2 <= 1, f"R² {r2} outside [0, 1]"

            # R² should equal Pearson correlation squared
            assert abs(pearson_r**2 - r2) < 1e-10, f"R² should equal Pearson correlation squared: {pearson_r**2} vs {r2}"

    def test_index_of_agreement_bounds(self):
        """Test that Index of Agreement is within valid bounds."""
        data_gen = TestDataGenerator()

        # Test with various scenarios
        scenarios = [
            (0.9, "high correlation"),
            (0.5, "moderate correlation"),
            (0.1, "low correlation"),
            (0.0, "no correlation")
        ]

        for correlation, description in scenarios:
            obs, mod = data_gen.generate_correlated_data(n_samples=1000, correlation=correlation)

            ioa = index_of_agreement(obs, mod)
            mioa = modified_index_of_agreement(obs, mod)

            # IOA should be in [0, 1]
            assert 0 <= ioa <= 1, f"IOA {ioa} outside [0, 1] for {description}"
            assert 0 <= mioa <= 1, f"Modified IOA {mioa} outside [0, 1] for {description}"

            # Higher correlation should generally give higher IOA
            if correlation > 0.5:
                assert ioa > 0.5, f"IOA should be > 0.5 for high correlation, got {ioa}"

    def test_contingency_metric_bounds(self):
        """Test that contingency metrics are within valid bounds."""
        # Test various contingency scenarios
        scenarios = [
            # (obs, mod, description)
            ([1, 1, 0, 0], [1, 1, 0, 0], "perfect forecast"),
            ([1, 1, 0, 0], [1, 0, 1, 0], "mixed forecast"),
            ([1, 1, 0, 0], [0, 0, 1, 1], "worst forecast"),
        ]

        for obs, mod, description in scenarios:
            obs_array = np.array(obs)
            mod_array = np.array(mod)

            hr = hit_rate(obs_array, mod_array)
            far = false_alarm_rate(obs_array, mod_array)
            csi = critical_success_index(obs_array, mod_array)
            ets = equitable_threat_score(obs_array, mod_array)

            # Check bounds
            assert 0 <= hr <= 1, f"Hit rate {hr} outside [0, 1] for {description}"
            assert 0 <= far <= 1, f"False alarm rate {far} outside [0, 1] for {description}"
            assert 0 <= csi <= 1, f"CSI {csi} outside [0, 1] for {description}"
            assert -1/3 <= ets <= 1, f"ETS {ets} outside [-1/3, 1] for {description}"


class TestScaleAndTranslationInvariance:
    """Test scale and translation invariance properties."""

    def test_correlation_scale_invariance(self):
        """Test that correlation is invariant to linear transformations."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100, correlation=0.8)

        # Test scale invariance
        scale_factors = [0.5, 2.0, 10.0]

        for scale in scale_factors:
            obs_scaled = obs * scale
            mod_scaled = mod * scale

            corr_original = pearson_correlation(obs, mod)
            corr_scaled = pearson_correlation(obs_scaled, mod_scaled)

            assert abs(corr_original - corr_scaled) < 1e-10, f"Correlation not scale-invariant: {corr_original} vs {corr_scaled}"

    def test_correlation_translation_invariance(self):
        """Test that correlation is invariant to translations."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100, correlation=0.8)

        # Test translation invariance
        offsets = [-10, 0, 5, 100]

        for offset in offsets:
            obs_offset = obs + offset
            mod_offset = mod + offset

            corr_original = pearson_correlation(obs, mod)
            corr_offset = pearson_correlation(obs_offset, mod_offset)

            assert abs(corr_original - corr_offset) < 1e-10, f"Correlation not translation-invariant: {corr_original} vs {corr_offset}"


class TestErrorMetricConsistency:
    """Test consistency between different error metrics."""

    def test_error_metric_relationships(self):
        """Test mathematical relationships between error metrics."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=1000, correlation=0.7)

        # Compute various error metrics
        mae = mean_absolute_error(obs, mod)
        rmse = root_mean_squared_error(obs, mod)
        bias = mean_bias_error(obs, mod)

        # Test relationships
        assert rmse >= mae, f"RMSE should be >= MAE: {rmse} vs {mae}"
        assert abs(bias) <= rmse, f"Absolute bias should be <= RMSE: {abs(bias)} vs {rmse}"

        # For non-zero errors, RMSE should be > MAE (strict inequality)
        if mae > 1e-10:
            assert rmse > mae, f"RMSE should be > MAE for non-uniform errors: {rmse} vs {mae}"

    def test_normalized_metrics(self):
        """Test normalized error metrics."""
        data_gen = TestDataGenerator()

        # Test with different magnitude data
        magnitudes = [1, 10, 100, 1000]

        for magnitude in magnitudes:
            obs = np.random.normal(magnitude, magnitude * 0.1, 100)
            mod = obs + np.random.normal(0, magnitude * 0.01, 100)

            mae = mean_absolute_error(obs, mod)
            nmae = mae / np.mean(np.abs(obs))  # Manual NMAE calculation

            # NMAE should be dimensionless and comparable across scales
            assert nmae > 0, f"NMAE should be positive: {nmae}"
            assert nmae < 10, f"NMAE should be reasonable: {nmae}"  # Less than 1000%


class TestReproducibility:
    """Test reproducibility of results."""

    def test_deterministic_results(self):
        """Test that metrics produce deterministic results."""
        data_gen = TestDataGenerator()
        obs, mod = data_gen.generate_correlated_data(n_samples=100, correlation=0.8, seed=42)

        # Compute metrics multiple times
        results = []
        for _ in range(5):
            mae = mean_absolute_error(obs, mod)
            rmse = root_mean_squared_error(obs, mod)
            corr = pearson_correlation(obs, mod)
            results.append((mae, rmse, corr))

        # All results should be identical
        for i in range(1, len(results)):
            assert abs(results[0][0] - results[i][0]) < 1e-15, f"MAE not deterministic: {results[0][0]} vs {results[i][0]}"
            assert abs(results[0][1] - results[i][1]) < 1e-15, f"RMSE not deterministic: {results[0][1]} vs {results[i][1]}"
            assert abs(results[0][2] - results[i][2]) < 1e-15, f"Correlation not deterministic: {results[0][2]} vs {results[i][2]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
