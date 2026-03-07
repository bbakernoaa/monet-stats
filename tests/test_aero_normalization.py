import dask.array as da
import numpy as np
import xarray as xr

from monet_stats.data_processing import normalize_data


def test_normalize_data_numpy_zscore():
    obs = np.array([[1, 2], [3, 4]], dtype=float)
    mod = np.array([[1, 2], [3, 4]], dtype=float)

    # Global normalization (default)
    o_norm, m_norm = normalize_data(obs, mod, method="zscore")
    assert np.allclose(o_norm.mean(), 0)
    assert np.allclose(o_norm.std(), 1)

    # Normalization along axis 0
    o_norm, m_norm = normalize_data(obs, mod, method="zscore", axis=0)
    assert np.allclose(o_norm.mean(axis=0), 0)
    assert np.allclose(o_norm.std(axis=0), 1)


def test_normalize_data_xarray_zscore():
    data = np.random.rand(10, 5)
    obs = xr.DataArray(data, dims=["lat", "lon"], coords={"lat": np.arange(10), "lon": np.arange(5)})
    mod = obs.copy()

    # Normalize along lat
    o_norm, m_norm = normalize_data(obs, mod, method="zscore", dim="lat")
    assert np.allclose(o_norm.mean(dim="lat"), 0)
    assert np.allclose(o_norm.std(dim="lat"), 1)
    assert "Normalized (zscore) along lat" in o_norm.attrs["history"]


def test_normalize_data_xarray_minmax():
    obs = xr.DataArray([0, 5, 10], dims="x")
    mod = xr.DataArray([0, 5, 10], dims="x")

    o_norm, m_norm = normalize_data(obs, mod, method="minmax")
    assert np.allclose(o_norm.min(), 0)
    assert np.allclose(o_norm.max(), 1)
    assert np.allclose(o_norm.values, [0, 0.5, 1])


def test_normalize_data_robust_dask():
    # Create a lazy dask-backed array
    data = da.random.random((100, 10), chunks=(50, 10))
    obs = xr.DataArray(data, dims=["time", "site"])
    mod = obs.copy()

    # Normalization along 'time' (should trigger ensure_single_chunk)
    o_norm, m_norm = normalize_data(obs, mod, method="robust", dim="time")

    # Verify laziness (should still be dask-backed)
    assert hasattr(o_norm.data, "chunks")

    # Compute and verify
    o_computed = o_norm.compute()
    assert np.allclose(np.median(o_computed, axis=0), 0)


def test_normalize_data_numpy_robust():
    obs = np.array([1, 2, 3, 100], dtype=float)  # 100 is an outlier
    mod = obs.copy()

    o_norm, m_norm = normalize_data(obs, mod, method="robust")

    # Median should be 0
    # Median of [1, 2, 3, 100] is 2.5
    # MAD: abs([1, 2, 3, 100] - 2.5) = [1.5, 0.5, 0.5, 97.5]
    # Median of MAD is 1.0 (mean of 0.5 and 1.5)

    expected_median = np.median(obs)
    expected_mad = np.median(np.abs(obs - expected_median))
    expected = (obs - expected_median) / expected_mad

    assert np.allclose(o_norm.flatten(), expected)
