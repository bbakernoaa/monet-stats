"""
Tests for scientific provenance tracking (Aero Protocol).
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats import contingency_metrics, correlation_metrics, error_metrics


@pytest.fixture
def sample_da():
    lats = np.linspace(30, 50, 10)
    lons = np.linspace(-120, -70, 10)
    data = np.random.rand(10, 10)
    return xr.DataArray(data, coords={"lat": lats, "lon": lons}, dims=("lat", "lon"), name="test")


def test_error_metrics_provenance(sample_da):
    obs = sample_da
    mod = sample_da + 0.1

    # Test a few representative metrics
    metrics_to_test = [
        (error_metrics.MAE, "MAE"),
        (error_metrics.RMSE, "RMSE"),
        (error_metrics.MB, "MB"),
        (error_metrics.MNB, "MNB"),
        (error_metrics.MdnNB, "MdnNB"),
    ]

    for func, name in metrics_to_test:
        res = func(obs, mod)
        assert "history" in res.attrs
        assert f"Calculated {name} using monet-stats" in res.attrs["history"]
        # Standard format includes timestamp in brackets
        assert "[" in res.attrs["history"]
        assert "]" in res.attrs["history"]


def test_correlation_metrics_provenance(sample_da):
    obs = sample_da
    mod = sample_da + 0.1

    metrics_to_test = [
        (correlation_metrics.pearsonr, "pearsonr"),
        (correlation_metrics.R2, "R2"),
        (correlation_metrics.KGE, "KGE"),
        (correlation_metrics.taylor_skill, "taylor_skill"),
    ]

    for func, name in metrics_to_test:
        res = func(obs, mod)
        assert "history" in res.attrs
        assert f"Calculated {name} using monet-stats" in res.attrs["history"]


def test_contingency_metrics_provenance(sample_da):
    obs = sample_da
    mod = sample_da + 0.1

    metrics_to_test = [
        (contingency_metrics.HSS, "Heidke Skill Score (HSS)"),
        (contingency_metrics.ETS, "Equitable Threat Score (ETS)"),
        (contingency_metrics.POD, "Probability of Detection (POD)"),
    ]

    for func, name in metrics_to_test:
        res = func(obs, mod, minval=0.5)
        assert "history" in res.attrs
        assert f"Calculated {name} using monet-stats" in res.attrs["history"]


def test_history_accumulation(sample_da):
    obs = sample_da
    mod = sample_da + 0.1

    # First operation
    res1 = error_metrics.MAE(obs, mod)
    initial_history = res1.attrs["history"]

    # Second operation on the result (if applicable, or just check accumulation manually)
    # Since res1 is a scalar DA, we can't easily chain metrics that expect arrays,
    # but we can check if _update_history preserves existing history.
    from monet_stats.utils_stats import _update_history

    res2 = _update_history(res1, "SecondOp")

    assert initial_history in res2.attrs["history"]
    assert "Calculated SecondOp using monet-stats" in res2.attrs["history"]
    assert res2.attrs["history"].count("Calculated") == 2
