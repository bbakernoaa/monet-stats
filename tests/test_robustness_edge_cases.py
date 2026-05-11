import numpy as np
import xarray as xr

from monet_stats.correlation_metrics import R2
from monet_stats.efficiency_metrics import NSE, PC, mNSE, rNSE
from monet_stats.error_metrics import (
    CRMSE,
    FAC2,
    IOA,
    MAE,
    MB,
    NRMSE,
    NSC,
    RMSE,
    STDO,
    STDP,
)


def test_empty_numpy():
    obs = np.array([])
    mod = np.array([])

    # These should all return nan for empty inputs
    assert np.isnan(MAE(obs, mod))
    assert np.isnan(NSE(obs, mod))
    assert np.isnan(rNSE(obs, mod))
    assert np.isnan(mNSE(obs, mod))
    assert np.isnan(PC(obs, mod))
    assert np.isnan(MB(obs, mod))
    assert np.isnan(RMSE(obs, mod))
    assert np.isnan(STDO(obs, mod))
    assert np.isnan(STDP(obs, mod))
    assert np.isnan(CRMSE(obs, mod))
    assert np.isnan(NRMSE(obs, mod))
    assert np.isnan(FAC2(obs, mod))
    assert np.isnan(R2(obs, mod))


def test_all_nan_numpy():
    obs = np.array([np.nan, np.nan])
    mod = np.array([1.0, 2.0])

    assert np.isnan(MAE(obs, mod))
    assert np.isnan(RMSE(obs, mod))
    assert np.isnan(STDO(obs, mod))
    assert np.isnan(R2(obs, mod))


def test_zero_variance_r2():
    obs = np.array([1, 1, 1])
    mod = np.array([1, 2, 3])
    assert np.isnan(R2(obs, mod))

    # Perfect match with zero variance (if possible? Usually implies constant data)
    obs = np.array([1, 1, 1])
    mod = np.array([1, 1, 1])
    # For constant data vs constant data, correlation is often undefined
    assert np.isnan(R2(obs, mod))


def test_xarray_robustness():
    obs = xr.DataArray([np.nan, np.nan], dims="x")
    mod = xr.DataArray([1.0, 2.0], dims="x")

    assert np.isnan(MAE(obs, mod).values)
    assert np.isnan(RMSE(obs, mod).values)
    assert np.isnan(R2(obs, mod).values)


def test_nsc_robustness():
    # Perfect agreement but constant data -> denominator is zero
    obs = np.array([1, 1])
    mod = np.array([1, 1])
    assert NSC(obs, mod) == 1.0

    # Not perfect agreement, constant data -> denominator zero
    obs = np.array([1, 1])
    mod = np.array([1.1, 1.1])
    assert NSC(obs, mod) == -np.inf


def test_ioa_robustness():
    obs = np.array([1, 1])
    mod = np.array([1, 1])
    assert IOA(obs, mod) == 1.0

    obs = np.array([1, 1])
    mod = np.array([1.1, 1.1])
    # Constant obs mean and constant mod, if they don't match, IOA might be 0.0 or nan.
    # Current implementation gives 0.0 because num == denom.
    assert IOA(obs, mod) == 0.0


def test_fac2_nan_handling():
    obs = np.array([1, 2, np.nan])
    mod = np.array([1.5, 5, 2.5])
    # Only first pair matches (1.5/1.0 = 1.5, in range [0.5, 2.0])
    # Second pair (5/2 = 2.5, out of range)
    # Third pair has NaN, ignored.
    # Total valid pairs = 2. Hits = 1. Result = 0.5.
    assert FAC2(obs, mod) == 0.5
