import dask.array as da
import numpy as np
import xarray as xr

from monet_stats.temporal_metrics import CrossWaveletTransform, DynamicTimeWarping, PhaseError


def test_dtw_numpy():
    # Simple cases
    obs = np.array([1, 2, 3])
    mod = np.array([1, 2, 3])
    assert DynamicTimeWarping(obs, mod) == 0.0

    # Shifted
    obs = np.array([1, 2, 3, 0])
    mod = np.array([0, 1, 2, 3])
    # DTW should find a small distance compared to Euclidean
    dist = DynamicTimeWarping(obs, mod)
    assert dist > 0

    # NaN handling
    obs = np.array([1, np.nan, 3])
    mod = np.array([1, 2, 3])
    # The current implementation drops NaNs, resulting in [1, 3] and [1, 2, 3].
    # Alignment: 1-1, 3-2, 3-3 cost=1
    assert DynamicTimeWarping(obs, mod) == 1.0


def test_dtw_xarray():
    obs = xr.DataArray([1, 2, 3], dims="time", coords={"time": [0, 1, 2]})
    mod = xr.DataArray([1, 2, 3], dims="time", coords={"time": [0, 1, 2]})

    res = DynamicTimeWarping(obs, mod, dim="time")
    assert float(res) == 0.0
    assert "history" in res.attrs
    assert "Dynamic Time Warping (DTW)" in res.attrs["history"]


def test_dtw_dask():
    obs = xr.DataArray(da.from_array([1, 2, 3], chunks=3), dims="time")
    mod = xr.DataArray(da.from_array([1, 2, 3], chunks=3), dims="time")

    res = DynamicTimeWarping(obs, mod, dim="time")
    assert hasattr(res.data, "dask")
    assert float(res.compute()) == 0.0


def test_phase_error_numpy():
    obs = np.array([0, 1, 0])  # peak at index 1
    mod = np.array([0, 0, 1])  # peak at index 2
    assert PhaseError(obs, mod) == 1.0

    # Multidimensional
    obs = np.array([[0, 1, 0], [1, 0, 0]])  # peaks at 1, 0
    mod = np.array([[0, 0, 1], [0, 1, 0]])  # peaks at 2, 1
    res = PhaseError(obs, mod, axis=1)
    # Obs peaks: [1, 0], Mod peaks: [2, 1] -> Phase Error: [2-1, 1-0] = [1, 1]
    np.testing.assert_array_equal(res, [1.0, 1.0])


def test_phase_error_numpy_nans():
    # Test all-NaN slice robust logic
    obs = np.array([[np.nan, np.nan], [0, 1]])
    mod = np.array([[np.nan, np.nan], [0, 1]])
    res = PhaseError(obs, mod, axis=1)
    np.testing.assert_array_equal(res, [np.nan, 0.0])


def test_phase_error_xarray():
    obs = xr.DataArray([[0, 1, 0], [1, 0, 0]], dims=("lat", "time"))
    mod = xr.DataArray([[0, 0, 1], [0, 1, 0]], dims=("lat", "time"))

    res = PhaseError(obs, mod, dim="time")
    assert isinstance(res, xr.DataArray)
    np.testing.assert_array_equal(res.values, [1.0, 1.0])
    assert "history" in res.attrs


def test_phase_error_dask():
    obs = xr.DataArray(da.from_array([[0, 1, 0], [1, 0, 0]], chunks=(2, 3)), dims=("lat", "time"))
    mod = xr.DataArray(da.from_array([[0, 0, 1], [0, 1, 0]], chunks=(2, 3)), dims=("lat", "time"))

    res = PhaseError(obs, mod, dim="time")
    assert hasattr(res.data, "dask")
    np.testing.assert_array_equal(res.compute().values, [1.0, 1.0])


def test_xwt_numpy():
    obs = np.sin(np.linspace(0, 10, 100))
    mod = np.sin(np.linspace(0, 10, 100))
    widths = np.arange(1, 5)
    res = CrossWaveletTransform(obs, mod, widths=widths)
    assert res.shape == (len(widths), 100)
    assert res.dtype == complex


def test_xwt_xarray():
    obs = xr.DataArray(np.sin(np.linspace(0, 10, 100)), dims="time")
    mod = xr.DataArray(np.sin(np.linspace(0, 10, 100)), dims="time")
    widths = np.arange(1, 5)

    res = CrossWaveletTransform(obs, mod, widths=widths, dim="time")
    assert isinstance(res, xr.DataArray)
    assert res.shape == (len(widths), 100)
    assert "scale" in res.coords
    assert "history" in res.attrs


def test_xwt_dask():
    obs = xr.DataArray(da.from_array(np.sin(np.linspace(0, 10, 100)), chunks=50), dims="time")
    mod = xr.DataArray(da.from_array(np.sin(np.linspace(0, 10, 100)), chunks=50), dims="time")
    widths = np.arange(1, 5)

    res = CrossWaveletTransform(obs, mod, widths=widths, dim="time")
    assert hasattr(res.data, "dask")
    computed = res.compute()
    assert computed.shape == (len(widths), 100)
