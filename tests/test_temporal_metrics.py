import numpy as np
import pytest
import xarray as xr

from monet_stats.temporal_metrics import CrossWaveletTransform, DynamicTimeWarping


def test_dtw_xarray_lazy():
    pytest.importorskip("dask.array")
    import dask.array as da

    obs = xr.DataArray(da.from_array(np.array([1, 2, 3, 4, 5]), chunks=5), dims="time")
    mod = xr.DataArray(da.from_array(np.array([0, 1, 2, 3, 4, 5]), chunks=6), dims="time")
    res = DynamicTimeWarping(obs, mod)
    assert hasattr(res.data, "chunks")
    val = res.compute()
    # Distance between [1,2,3,4,5] and [0,1,2,3,4,5] should be 1.0 (matching 1 with 0, then 1-1, 2-2...)
    assert np.isclose(val, 1.0)


def test_xwt_xarray_lazy():
    pytest.importorskip("dask.array")
    import dask.array as da

    t = np.linspace(0, 10, 100)
    obs_data = np.sin(2 * np.pi * t)
    mod_data = np.sin(2 * np.pi * t + 0.5)
    obs = xr.DataArray(da.from_array(obs_data, chunks=100), dims="time", coords={"time": t})
    mod = xr.DataArray(da.from_array(mod_data, chunks=100), dims="time", coords={"time": t})
    widths = np.arange(1, 10)
    res = CrossWaveletTransform(obs, mod, widths=widths)
    assert "scale" in res.dims
    assert res.sizes["scale"] == len(widths)
    assert res.sizes["time"] == 100
    assert hasattr(res.data, "chunks")
    val = res.compute()
    assert np.iscomplexobj(val.values)


def test_dtw_numpy():
    obs = np.array([1, 2, 3, 4, 5])
    mod = np.array([0, 1, 2, 3, 4, 5])
    # DTW should handle the shift [0] at the beginning
    res = DynamicTimeWarping(obs, mod)
    assert res < 5.0


def test_xwt_numpy():
    t = np.linspace(0, 10, 100)
    obs = np.sin(2 * np.pi * t)
    mod = np.sin(2 * np.pi * t + 0.5)
    widths = np.arange(1, 10)
    res = CrossWaveletTransform(obs, mod, widths=widths)
    assert res.shape == (len(widths), len(obs))
    assert np.iscomplexobj(res)


if __name__ == "__main__":
    test_dtw_numpy()
    try:
        test_dtw_xarray_lazy()
    except pytest.skip.Exception:
        pass
    test_xwt_numpy()
    try:
        test_xwt_xarray_lazy()
    except pytest.skip.Exception:
        pass
    print("Temporal metrics tests passed!")
