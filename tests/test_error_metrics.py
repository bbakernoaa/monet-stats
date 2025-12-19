"""
Tests for error_metrics.py module.
"""

import numpy as np
import pytest
import xarray as xr
import dask.array as da

from monet_stats.error_metrics import (
    MAE, MB, RMSE, STDO, STDP, MNB, MNE, MdnNB, MdnNE, NMdnGE, NO, NOP, NP, MO, MP,
    MdnO, MdnP, RM, RMdn, MdnB, WDMB_m, WDMB, WDMdnB, MedAE, sMAPE_original, CRMSE,
    MAPE, sMAPE, NRMSE, MASE, MASEm, RMSPE, MAPEm, sMAPEm, NSC, NSE_alpha, NSE_beta,
    MAE_m, MedAE_m, RMSE_m, IOA, IOA_m, MAPE_mod, MASE_mod, RMSE_norm, MAE_norm,
    bias_fraction, NMSE, LOG_ERROR, COE, VOLUMETRIC_ERROR, CORR_INDEX
)


@pytest.fixture(params=["numpy", "xarray"])
def factory(request):
    """Provide a factory for creating numpy or xarray arrays."""
    if request.param == "numpy":
        return np
    else:
        return xr


@pytest.fixture
def perfect_agreement_data(factory):
    """Fixture for perfect agreement data."""
    if factory == np:
        obs = factory.array([1, 2, 3, 4, 5])
        mod = factory.array([1, 2, 3, 4, 5])
    else:
        obs = factory.DataArray([1, 2, 3, 4, 5], dims=["time"])
        mod = factory.DataArray([1, 2, 3, 4, 5], dims=["time"])
    return obs, mod


@pytest.fixture
def biased_data(factory):
    """Fixture for systematically biased data."""
    if factory == np:
        obs = factory.array([1, 2, 3, 4, 5])
        mod = factory.array([1.1, 2.1, 3.1, 4.1, 5.1])
    else:
        obs = factory.DataArray([1, 2, 3, 4, 5], dims=["time"])
        mod = factory.DataArray([1.1, 2.1, 3.1, 4.1, 5.1], dims=["time"])
    return obs, mod


class TestErrorMetrics:
    """Test suite for error metrics."""

    @pytest.mark.parametrize(
        "metric,expected",
        [
            (MAE, 0.0), (MB, 0.0), (RMSE, 0.0), (STDO, 0.0), (STDP, 0.0), (MNE, 0.0),
            (MNB, 0.0), (MdnNB, 0.0), (MdnNE, 0.0), (NMdnGE, 0.0), (MO, 0.0),
            (MdnO, 0.0), (MdnP, 0.0), (RM, 0.0), (RMdn, 0.0), (MdnB, 0.0),
            (WDMB_m, 0.0), (WDMB, 0.0), (WDMdnB, 0.0), (MedAE, 0.0), (sMAPE_original, 0.0),
            (CRMSE, 0.0), (MAPE, 0.0), (sMAPE, 0.0), (NRMSE, 0.0), (MASE, 0.0),
            (MASEm, 0.0), (RMSPE, 0.0), (MAPEm, 0.0), (sMAPEm, 0.0), (NSC, 1.0),
            (NSE_alpha, 1.0), (NSE_beta, 1.0), (MAE_m, 0.0), (MedAE_m, 0.0), (RMSE_m, 0.0),
            (IOA, 1.0), (IOA_m, 1.0), (MAPE_mod, 0.0), (MASE_mod, 0.0), (RMSE_norm, 0.0),
            (MAE_norm, 0.0), (bias_fraction, 0.0), (NMSE, 0.0), (LOG_ERROR, 0.0),
            (COE, 0.0), (VOLUMETRIC_ERROR, 0.0), (CORR_INDEX, 1.0),
        ],
    )
    def test_perfect_agreement(self, perfect_agreement_data, metric, expected):
        """Test all metrics with perfect agreement."""
        obs, mod = perfect_agreement_data
        result = metric(obs, mod)
        assert np.isclose(result, expected)

    def test_counting_metrics(self, perfect_agreement_data):
        """Test NOP and NP metrics."""
        obs, mod = perfect_agreement_data
        assert NOP(obs, mod) == 5
        assert NP(obs, mod) == 5

    def test_mp_metric(self, biased_data):
        """Test MP metric."""
        obs, mod = biased_data
        assert np.isclose(MP(obs, mod), 3.1)


class TestErrorMetricsDask:
    """Test suite for Dask-backed xarray DataArrays."""

    def test_dask_computation(self):
        """Test that metrics work with Dask and compute correctly."""
        obs = xr.DataArray([1, 2, 3, 4, 5], dims=["time"]).chunk()
        mod = xr.DataArray([1.1, 2.1, 3.1, 4.1, 5.1], dims=["time"]).chunk()

        mae_dask = MAE(obs, mod)
        assert isinstance(mae_dask.data, da.Array)
        assert np.isclose(mae_dask.compute(), 0.1)
