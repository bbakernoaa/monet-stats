"""
Tests for error_metrics.py module.

This module tests error-based statistical metrics including
MAE, RMSE, MB, STDO, STDP, and other error statistics.
"""

import numpy as np
import pytest
import xarray as xr
import dask.array as da

from monet_stats.error_metrics import (
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


@pytest.fixture
def perfect_agreement_data():
    """Fixture for perfect agreement data."""
    obs = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], name="obs")
    mod = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], name="mod")
    return obs, mod


@pytest.fixture
def biased_data():
    """Fixture for systematically biased data."""
    obs = xr.DataArray([1, 2, 3, 4, 5], dims=["time"], name="obs")
    mod = xr.DataArray([1.1, 2.1, 3.1, 4.1, 5.1], dims=["time"], name="mod")
    return obs, mod


@pytest.fixture
def random_data():
    """Fixture for data with random errors."""
    np.random.seed(42)
    obs = xr.DataArray(np.random.normal(0, 1, 50), dims=["time"], name="obs")
    mod = obs + np.random.normal(0, 0.2, 50)
    return obs, mod


@pytest.fixture
def dask_random_data(random_data):
    """Fixture for Dask-backed data with random errors."""
    obs, mod = random_data
    return obs.chunk({"time": 10}), mod.chunk({"time": 10})


class TestErrorMetrics:
    """Test suite for error metrics."""

    def test_mae_perfect_agreement(self, perfect_agreement_data):
        """Test MAE with perfect agreement."""
        obs, mod = perfect_agreement_data
        result = MAE(obs, mod)
        assert abs(result.item() - 0.0) < 1e-10

    def test_mb_systematic_bias(self, biased_data):
        """Test Mean Bias with systematic bias."""
        obs, mod = biased_data
        result = MB(obs, mod)
        assert abs(result.item() - (-0.1)) < 1e-10

    @pytest.mark.parametrize(
        "metric_func,expected",
        [
            (MAE, 0.0),
            (MB, 0.0),
            (RM, 0.0),
            (RMdn, 0.0),
            (STDO, 0.0),
            (STDP, 0.0),
            (NRMSE, 0.0),
            (WDMB, 0.0),
            (WDMB_m, 0.0),
            (MdnB, 0.0),
            (MedAE, 0.0),
            (RMSE_m, 0.0),
            (IOA_m, 1.0),
            (MNE, 0.0),
            (MNB, 0.0),
            (MdnNE, 0.0),
            (MdnNB, 0.0),
            (NMdnGE, 0.0),
            (NSE_alpha, 1.0),
            (NSE_beta, 1.0),
            (MAPE_mod, 0.0),
            (MASE_mod, 0.0),
            (RMSE_norm, 0.0),
            (MAE_norm, 0.0),
            (bias_fraction, 0.0),
            (NMSE, 0.0),
            (LOG_ERROR, 0.0),
            (COE, 0.0),
            (VOLUMETRIC_ERROR, 0.0),
            (CORR_INDEX, 1.0),
        ],
    )
    def test_metrics_perfect_agreement(self, perfect_agreement_data, metric_func, expected):
        """Test various metrics with perfect agreement."""
        obs, mod = perfect_agreement_data
        result = metric_func(obs, mod)
        assert np.isclose(result.item(), expected)

    def test_n_metrics(self, perfect_agreement_data):
        """Test metrics that count observations."""
        obs, mod = perfect_agreement_data
        assert NOP(obs, mod).item() == 5
        assert NP(obs, mod).item() == 5

    def test_xarray_input_output(self, random_data):
        """Test that metrics handle xarray inputs and return xarray objects."""
        obs, mod = random_data
        result = MAE(obs, mod)
        assert isinstance(result, xr.DataArray)


class TestErrorMetricsDask:
    """Test suite for error metrics with Dask-backed xarray DataArrays."""

    def test_dask_computation(self, dask_random_data):
        """Test that metrics work with Dask and compute correctly."""
        obs, mod = dask_random_data

        # Test a few key metrics
        mae_dask = MAE(obs, mod)
        mb_dask = MB(obs, mod)
        rmse_dask = RMSE(obs, mod)

        assert isinstance(mae_dask.data, da.Array)
        assert isinstance(mb_dask.data, da.Array)
        assert isinstance(rmse_dask.data, da.Array)

        # Compute the results and compare with numpy-backed computation
        mae_computed = mae_dask.compute()
        mb_computed = mb_dask.compute()
        rmse_computed = rmse_dask.compute()

        obs_np, mod_np = obs.compute(), mod.compute()
        mae_expected = MAE(obs_np, mod_np)
        mb_expected = MB(obs_np, mod_np)
        rmse_expected = RMSE(obs_np, mod_np)

        assert np.isclose(mae_computed, mae_expected)
        assert np.isclose(mb_computed, mb_expected)
        assert np.isclose(rmse_computed, rmse_expected)

    def test_dask_perfect_agreement(self, perfect_agreement_data):
        """Test perfect agreement with Dask arrays."""
        obs, mod = perfect_agreement_data
        obs_dask, mod_dask = obs.chunk(), mod.chunk()

        result = MAE(obs_dask, mod_dask)
        assert np.isclose(result.compute().item(), 0.0)
