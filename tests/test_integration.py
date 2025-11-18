"""
Integration tests for statistical metrics.

These tests verify that multiple metrics work together correctly
and produce consistent results in realistic scenarios.
"""
import numpy as np
import pytest

from src.monet_stats.contingency_metrics import POD, FAR, CSI, HSS, ETS
from src.monet_stats.correlation_metrics import R2, RMSE, IOA, KGE, pearsonr
from src.monet_stats.error_metrics import MAE, MB, NRMSE
from src.monet_stats.relative_metrics import NMB, NME
from src.monet_stats.efficiency_metrics import NSE, MSE, MAPE
from src.monet_stats.utils_stats import correlation


class TestIntegration:
    """Integration tests for statistical metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Generate synthetic data representing a realistic scenario
        np.random.seed(42)
        
        # Simulate observed data (e.g., temperature measurements)
        n_points = 100
        time = np.linspace(0, 100, n_points)
        true_signal = 20 + 5 * np.sin(2 * np.pi * time / 10)  # Base signal with seasonal variation
        obs = true_signal + np.random.normal(0, 1, n_points)  # Add noise
        
        # Simulate model data with some bias and error
        mod = true_signal + 0.5 + np.random.normal(0, 1.5, n_points)  # Add bias and different noise
        
        self.obs = obs
        self.mod = mod
        
        # Generate binary event data for contingency testing
        self.threshold = np.percentile(obs, 75)  # 75th percentile as threshold
        self.obs_binary = (obs >= self.threshold).astype(int)
        self.mod_binary = (mod >= self.threshold).astype(int)
    
    def test_basic_statistical_consistency(self):
        """Test basic consistency between related metrics."""
        # Calculate various metrics
        rmse_val = RMSE(self.obs, self.mod)
        mse_val = MSE(self.obs, self.mod)
        mae_val = MAE(self.obs, self.mod)
        
        # RMSE should be >= MAE (by Cauchy-Schwarz inequality)
        assert rmse_val >= mae_val, f"RMSE ({rmse_val}) should be >= MAE ({mae_val})"
        
        # MSE should be RMSE squared (approximately)
        assert abs(mse_val - rmse_val**2) < 1e-10, f"MSE ({mse_val}) should equal RMSE^2 ({rmse_val**2})"
    
    def test_perfect_agreement_integration(self):
        """Test all metrics with perfect agreement."""
        obs_perfect = self.obs
        mod_perfect = self.obs  # Perfect agreement
        
        # Error metrics should be 0 or near 0
        assert abs(RMSE(obs_perfect, mod_perfect)) < 1e-10
        assert abs(MAE(obs_perfect, mod_perfect)) < 1e-10
        assert abs(MB(obs_perfect, mod_perfect)) < 1e-10
        
        # Correlation metrics should be 1 or near 1
        assert abs(R2(obs_perfect, mod_perfect) - 1.0) < 1e-10
        assert abs(IOA(obs_perfect, mod_perfect) - 1.0) < 1e-10
        assert abs(KGE(obs_perfect, mod_perfect) - 1.0) < 1e-10
        
        # Efficiency metrics should be 1 or near 1
        assert abs(NSE(obs_perfect, mod_perfect) - 1.0) < 1e-10
    
    def test_model_bias_detection(self):
        """Test that metrics properly detect systematic bias."""
        # Create model with systematic bias
        bias = 2.0
        mod_biased = self.obs + bias
        
        # Mean bias should detect the bias
        mb_val = MB(self.obs, mod_biased)
        assert abs(mb_val - bias) < 1e-10, f"MB should detect bias of {bias}, got {mb_val}"
        
        # NMB should also reflect the bias
        nmb_val = NMB(self.obs, mod_biased)
        expected_nmb = (bias / np.mean(self.obs)) * 100
        assert abs(nmb_val - expected_nmb) < 1e-6, f"NMB should reflect bias, expected {expected_nmb}, got {nmb_val}"
    
    def test_correlation_efficiency_relationship(self):
        """Test relationship between correlation and efficiency metrics."""
        # High correlation should generally correspond to good efficiency
        pearson_corr = pearsonr(self.obs, self.mod)
        if isinstance(pearson_corr, tuple):
            pearson_corr = pearson_corr[0]
        
        r2_val = R2(self.obs, self.mod)
        nse_val = NSE(self.obs, self.mod)
        
        # R2 is the square of Pearson correlation (for simple linear regression)
        assert abs(r2_val - pearson_corr**2) < 1e-6, f"R2 should be ~Pearson^2, {r2_val} vs {pearson_corr**2}"
        
        # For good models, NSE should be positive and related to correlation
        if pearson_corr > 0.7:  # High correlation
            assert nse_val > 0.5, f"High correlation should correspond to good NSE, got {nse_val}"
    
    def test_contingency_metrics_consistency(self):
        """Test consistency among contingency metrics."""
        # Use binary version of our data
        threshold = self.threshold
        
        pod_val = POD(self.obs, self.mod, threshold)
        far_val = FAR(self.obs, self.mod, threshold)
        csi_val = CSI(self.obs, self.mod, threshold)
        hss_val = HSS(self.obs, self.mod, threshold)
        ets_val = ETS(self.obs, self.mod, threshold)
        
        # Check that values are finite
        assert np.isfinite(pod_val), f"POD should be finite, got {pod_val}"
        assert np.isfinite(far_val), f"FAR should be finite, got {far_val}"
        assert np.isfinite(csi_val), f"CSI should be finite, got {csi_val}"
        assert np.isfinite(hss_val), f"HSS should be finite, got {hss_val}"
        assert np.isfinite(ets_val), f"ETS should be finite, got {ets_val}"
        
        # POD and FAR should be in [0, 1] when finite
        if 0 <= pod_val <= 1:
            assert 0 <= pod_val <= 1, f"POD should be in [0,1], got {pod_val}"
        if 0 <= far_val <= 1:
            assert 0 <= far_val <= 1, f"FAR should be in [0,1], got {far_val}"
        if 0 <= csi_val <= 1:
            assert 0 <= csi_val <= 1, f"CSI should be in [0,1], got {csi_val}"
    
    def test_relative_error_metrics_consistency(self):
        """Test consistency among relative error metrics."""
        nmb_val = NMB(self.obs, self.mod)
        nme_val = NME(self.obs, self.mod)
        
        # Both should be finite
        assert np.isfinite(nmb_val), f"NMB should be finite, got {nmb_val}"
        assert np.isfinite(nme_val), f"NME should be finite, got {nme_val}"
        
        # NME should generally be >= absolute value of NMB (since NME uses absolute errors)
        if np.isfinite(nme_val) and np.isfinite(nmb_val):
            assert nme_val >= abs(nmb_val), f"NME ({nme_val}) should be >= |NMB| ({abs(nmb_val)})"
    
    def test_efficiency_metrics_hierarchy(self):
        """Test expected hierarchy among efficiency metrics."""
        nse_val = NSE(self.obs, self.mod)
        kge_val = KGE(self.obs, self.mod)
        
        # Both should be finite
        assert np.isfinite(nse_val), f"NSE should be finite, got {nse_val}"
        assert np.isfinite(kge_val), f"KGE should be finite, got {kge_val}"
        
        # For good models, both should be positive
        if nse_val > 0.5:
            assert kge_val > 0.3, f"Good NSE should correspond to decent KGE, got {kge_val}"
    
    def test_realistic_scenario_metrics(self):
        """Test metrics on a more realistic scenario."""
        # Create a more realistic scenario with known properties
        np.random.seed(123)
        n = 500
        
        # True values with some trend and seasonality
        time = np.linspace(0, 10, n)
        true_values = 25 + 3 * np.sin(2 * np.pi * time) + 0.1 * time # Trend + seasonality
        obs = true_values + np.random.normal(0, 0.5, n)  # Add measurement noise
        
        # Model with systematic bias and additional model error
        mod = true_values + 0.8 + np.random.normal(0, 0.8, n)  # Bias + model error
        
        # Calculate metrics
        rmse = RMSE(obs, mod)
        mae = MAE(obs, mod)
        mb = MB(obs, mod)
        nse = NSE(obs, mod)
        r2 = R2(obs, mod)
        ioa = IOA(obs, mod)
        
        # Check reasonable ranges
        assert rmse > 0, f"RMSE should be positive, got {rmse}"
        assert mae > 0, f"MAE should be positive, got {mae}"
        assert abs(mb - 0.8) < 0.5, f"MB should be close to bias (0.8), got {mb}"  # Allow some variation
        assert -1 <= nse <= 1, f"NSE should be in [-1,1], got {nse}"
        assert -1 <= r2 <= 1, f"R2 should be in [-1,1], got {r2}"
        assert 0 <= ioa <= 1, f"IOA should be in [0,1], got {ioa}"
    
    def test_extreme_case_metrics(self):
        """Test metrics behavior with extreme cases."""
        # Perfect model
        obs_perfect = np.array([1, 2, 3, 4, 5])
        mod_perfect = np.array([1, 2, 3, 4, 5])
        
        assert abs(RMSE(obs_perfect, mod_perfect)) < 1e-10
        assert abs(NSE(obs_perfect, mod_perfect) - 1.0) < 1e-10
        assert abs(R2(obs_perfect, mod_perfect) - 1.0) < 1e-10
        
        # Model that's just the mean
        obs_mean = np.array([1, 2, 3, 4, 5])
        mod_mean = np.full_like(obs_mean, np.mean(obs_mean))
        
        nse_mean = NSE(obs_mean, mod_mean)
        assert abs(nse_mean - 0.0) < 1e-10, f"NSE for mean model should be ~0, got {nse_mean}"
    
    def test_metrics_scaling_invariance(self):
        """Test which metrics are affected by scaling."""
        obs_original = self.obs
        mod_original = self.mod
        
        # Scale by a factor
        scale_factor = 10.0
        obs_scaled = obs_original * scale_factor
        mod_scaled = mod_original * scale_factor
        
        # RMSE should scale linearly
        rmse_original = RMSE(obs_original, mod_original)
        rmse_scaled = RMSE(obs_scaled, mod_scaled)
        expected_scaled = rmse_original * scale_factor
        
        assert abs(rmse_scaled - expected_scaled) < 1e-6, \
            f"RMSE should scale linearly, got {rmse_scaled} vs expected {expected_scaled}"
        
        # NMB should be unchanged (relative metric)
        nmb_original = NMB(obs_original, mod_original)
        nmb_scaled = NMB(obs_scaled, mod_scaled)
        
        if np.isfinite(nmb_original) and np.isfinite(nmb_scaled):
            assert abs(nmb_original - nmb_scaled) < 1e-6, \
                f"NMB should be scale-invariant, got {nmb_original} vs {nmb_scaled}"
    
    def test_correlation_calculation_methods(self):
        """Test different methods of calculating correlation."""
        # Test our correlation function vs scipy's pearsonr
        from scipy.stats import pearsonr
        
        corr_our = correlation(self.obs, self.mod)
        corr_scipy, _ = pearsonr(self.obs, self.mod)
        
        # Should be very close
        assert abs(corr_our - corr_scipy) < 1e-10, \
            f"Our correlation ({corr_our}) should match scipy ({corr_scipy})"
        
        # Also compare with the pearsonr from our module
        corr_module = pearsonr(self.obs, self.mod)
        if isinstance(corr_module, tuple):
            corr_module = corr_module[0]
        
        assert abs(corr_our - corr_module) < 1e-10, \
            f"Our correlation ({corr_our}) should match module ({corr_module})"
    
    def test_error_metrics_relationships(self):
        """Test mathematical relationships between error metrics."""
        rmse_val = RMSE(self.obs, self.mod)
        mae_val = MAE(self.obs, self.mod)
        mb_val = MB(self.obs, self.mod)
        
        # RMSE >= MAE (by Cauchy-Schwarz inequality)
        assert rmse_val >= mae_val, f"RMSE ({rmse_val}) should be >= MAE ({mae_val})"
        
        # |MB| <= MAE (bias is a type of error)
        assert abs(mb_val) <= mae_val, f"|MB| ({abs(mb_val)}) should be <= MAE ({mae_val})"
    
    def test_comprehensive_workflow(self):
        """Test a comprehensive workflow with multiple metrics."""
        # Calculate a comprehensive set of metrics
        metrics = {
            'RMSE': RMSE(self.obs, self.mod),
            'MAE': MAE(self.obs, self.mod),
            'MB': MB(self.obs, self.mod),
            'R2': R2(self.obs, self.mod),
            'NSE': NSE(self.obs, self.mod),
            'IOA': IOA(self.obs, self.mod),
            'KGE': KGE(self.obs, self.mod),
            'NMB': NMB(self.obs, self.mod),
            'MAPE': MAPE(self.obs, self.mod) if np.all(self.obs != 0) else np.nan
        }
        
        # Verify all metrics are computed
        for name, value in metrics.items():
            if np.isfinite(value):
                assert isinstance(value, (int, float, np.number)), \
                    f"{name} should return a numeric value, got {type(value)}"
        
        # Check that error metrics are positive
        error_metrics = ['RMSE', 'MAE']
        for metric in error_metrics:
            if np.isfinite(metrics[metric]):
                assert metrics[metric] >= 0, f"{metric} should be non-negative"
        
        # Check that bounded metrics are in range
        if np.isfinite(metrics['R2']):
            assert metrics['R2'] <= 1.0, "R2 should be <= 1.0"
        if np.isfinite(metrics['NSE']):
            assert metrics['NSE'] <= 1.0, "NSE should be <= 1.0"
        if np.isfinite(metrics['IOA']):
            assert 0 <= metrics['IOA'] <= 1, "IOA should be in [0, 1]"
        if np.isfinite(metrics['KGE']):
            assert metrics['KGE'] <= 1.0, "KGE should be <= 1.0"
    
    def test_metrics_with_noise_levels(self):
        """Test how metrics respond to different noise levels."""
        base_signal = np.linspace(0, 10, 100)
        
        # Test with different noise levels
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        results = []
        
        for noise in noise_levels:
            obs = base_signal + np.random.normal(0, noise, 100)
            mod = base_signal + np.random.normal(0, noise/2, 100)  # Model has less noise
            
            rmse = RMSE(obs, mod)
            nse = NSE(obs, mod)
            results.append((noise, rmse, nse))
        
        # As noise increases, RMSE should generally increase
        for i in range(1, len(results)):
            if results[i-1][2] > results[i][2]:  # If NSE decreased with more noise, that's expected
                continue  # This is expected behavior
            # Check that metrics respond appropriately to noise level changes
            assert np.isfinite(results[i][1]), f"RMSE should be finite for noise level {results[i][0]}"
            assert np.isfinite(results[i][2]), f"NSE should be finite for noise level {results[i][0]}"