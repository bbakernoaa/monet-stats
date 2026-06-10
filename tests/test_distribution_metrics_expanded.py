import numpy as np
import pytest
import xarray as xr

from monet_stats.distribution_metrics import EnergyDistance, JensenShannonDivergence, SinkhornDistance


def test_js_divergence_numpy():
    obs = np.random.normal(0, 1, 1000)
    mod = np.random.normal(0, 1, 1000)
    res = JensenShannonDivergence(obs, mod, bins=10)
    # JS divergence is symmetric and bounded [0, 1]
    assert 0 <= res <= 1.0

    # Same distribution should have small JS divergence
    res_same = JensenShannonDivergence(obs, obs, bins=10)
    assert res_same < 0.05


def test_energy_distance_numpy():
    obs = np.array([1, 2, 3])
    mod = np.array([1, 2, 3])
    res = EnergyDistance(obs, mod)
    assert np.isclose(res, 0.0)

    mod2 = np.array([2, 3, 4])
    res2 = EnergyDistance(obs, mod2)
    assert res2 > 0


def test_sinkhorn_distance_numpy():
    obs = np.array([1.0, 2.0])
    mod = np.array([1.0, 2.0])
    # For identical distributions, Sinkhorn should be small (depending on epsilon)
    res = SinkhornDistance(obs, mod, epsilon=0.1)
    assert res < 0.1

    mod2 = np.array([2.0, 3.0])
    res2 = SinkhornDistance(obs, mod2, epsilon=0.1)
    assert res2 > 0


def test_distribution_metrics_xarray_eager():
    obs = xr.DataArray(np.random.normal(0, 1, 100), dims="x")
    mod = xr.DataArray(np.random.normal(0.5, 1, 100), dims="x")

    js = JensenShannonDivergence(obs, mod, bins=10)
    energy = EnergyDistance(obs, mod)
    sinkhorn = SinkhornDistance(obs, mod)

    assert isinstance(js, xr.DataArray)
    assert isinstance(energy, xr.DataArray)
    assert isinstance(sinkhorn, xr.DataArray)

    assert "history" in js.attrs
    assert "history" in energy.attrs
    assert "history" in sinkhorn.attrs


def test_distribution_metrics_xarray_lazy():
    pytest.importorskip("dask.array")
    import dask.array as da

    obs_data = np.random.normal(0, 1, 100)
    mod_data = np.random.normal(0.5, 1, 100)
    obs = xr.DataArray(da.from_array(obs_data, chunks=50), dims="x")
    mod = xr.DataArray(da.from_array(mod_data, chunks=50), dims="x")

    js = JensenShannonDivergence(obs, mod, bins=10)
    energy = EnergyDistance(obs, mod)
    sinkhorn = SinkhornDistance(obs, mod)

    # Verify laziness
    assert hasattr(js.data, "chunks")
    assert hasattr(energy.data, "chunks")
    assert hasattr(sinkhorn.data, "chunks")

    # Compute and verify
    assert js.compute() >= 0
    assert energy.compute() >= 0
    assert sinkhorn.compute() >= 0


def test_accessor_distribution_metrics():
    obs = xr.DataArray(np.random.normal(0, 1, 100), dims="x")
    mod = xr.DataArray(np.random.normal(0.5, 1, 100), dims="x")

    js = mod.monet_stats.jensenshannon_divergence(obs, bins=10)
    energy = mod.monet_stats.energy_distance(obs)
    sinkhorn = mod.monet_stats.sinkhorn_distance(obs)

    assert isinstance(js, xr.DataArray)
    assert js >= 0
    assert energy >= 0
    assert sinkhorn >= 0


def test_multi_dim_numpy():
    obs = np.random.normal(0, 1, (2, 100))
    mod = np.random.normal(0.5, 1, (2, 100))

    # Test reduction along axis 1
    js = JensenShannonDivergence(obs, mod, axis=1, bins=10)
    energy = EnergyDistance(obs, mod, axis=1)
    sinkhorn = SinkhornDistance(obs, mod, axis=1)

    assert js.shape == (2,)
    assert energy.shape == (2,)
    assert sinkhorn.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__])
