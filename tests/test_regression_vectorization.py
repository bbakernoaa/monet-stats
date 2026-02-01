"""
Unit tests for vectorized regression-based error metrics (Aero Protocol).
"""

import numpy as np
from scipy.stats import linregress

from monet_stats.correlation_metrics import RMSEs, RMSEu


def test_rmses_vectorized_parity():
    """Verify RMSEs vectorized matches iterative linregress approach."""
    np.random.seed(42)
    # 3D array: (time, lat, lon)
    obs = np.random.rand(10, 5, 5)
    mod = obs * 0.8 + 0.1 + np.random.randn(10, 5, 5) * 0.05

    # Add some NaNs
    obs[0, 0, 0] = np.nan
    mod[1, 1, 1] = np.nan

    # Compute using vectorized function over time axis (0)
    res_vec = RMSEs(obs, mod, axis=0)

    # Compute manually using loops and linregress
    res_manual = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            o_slice = obs[:, i, j]
            m_slice = mod[:, i, j]
            mask = ~np.isnan(o_slice) & ~np.isnan(m_slice)
            if np.sum(mask) < 2:
                res_manual[i, j] = np.nan
            else:
                # Use ONLY paired indices
                m, c, _, _, _ = linregress(o_slice[mask], m_slice[mask])
                fit = c + m * o_slice[mask]
                # manual RMSE(fit, obs_paired)
                res_manual[i, j] = np.sqrt(np.mean((fit - o_slice[mask]) ** 2))

    np.testing.assert_allclose(res_vec, res_manual, rtol=1e-10)


def test_rmseu_vectorized_parity():
    """Verify RMSEu vectorized matches iterative linregress approach."""
    np.random.seed(42)
    obs = np.random.rand(10, 5, 5)
    mod = obs * 0.8 + 0.1 + np.random.randn(10, 5, 5) * 0.05

    obs[2, 2, 2] = np.nan

    res_vec = RMSEu(obs, mod, axis=0)

    res_manual = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            o_slice = obs[:, i, j]
            m_slice = mod[:, i, j]
            mask = ~np.isnan(o_slice) & ~np.isnan(m_slice)
            if np.sum(mask) < 2:
                res_manual[i, j] = np.nan
            else:
                m, c, _, _, _ = linregress(o_slice[mask], m_slice[mask])
                fit = c + m * o_slice[mask]
                # manual RMSE(fit, mod_paired)
                res_manual[i, j] = np.sqrt(np.mean((fit - m_slice[mask]) ** 2))

    np.testing.assert_allclose(res_vec, res_manual, rtol=1e-10)


def test_regression_metrics_edge_cases():
    """Test with constant values or all NaNs."""
    obs = np.ones(10)
    mod = np.ones(10) * 2

    # Constant values -> ss_xx = 0
    # RMSEs: fit should be mod_hat = c + 0*obs.
    # Since mod is constant, m=0, c=mean(mod)=2.
    # mod_hat = 2. RMSEs = sqrt(mean((2-1)^2)) = 1.0
    assert RMSEs(obs, mod) == 1.0
    # RMSEu: sqrt(mean((2-2)^2)) = 0.0
    assert RMSEu(obs, mod) == 0.0

    # All NaNs
    obs_nan = np.full(10, np.nan)
    assert np.isnan(RMSEs(obs_nan, mod))
    assert np.isnan(RMSEu(obs_nan, mod))
