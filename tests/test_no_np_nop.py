import numpy as np
import pytest
import xarray as xr

from monet_stats.error_metrics import NO, NOP, NP


class TestNO_NP_NOP:
    def test_no_nop_np_nan(self):
        arr = np.array([1, 2, np.nan, 4, 5])
        # Should count only non-NaN as valid
        assert NO(arr, arr) == 4
        assert NOP(arr, arr) == 4
        assert NP(arr, arr) == 4

    def test_no_nop_np_inf(self):
        arr = np.array([1, 2, np.inf, 4, 5])
        # Inf is not masked, so should be counted
        assert NO(arr, arr) == 5
        assert NOP(arr, arr) == 5
        assert NP(arr, arr) == 5

    def test_no_nop_np_mismatched_shapes(self):
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3, 4])
        # Should raise ValueError due to shape mismatch (numpy broadcasting)
        with pytest.raises(ValueError):
            NO(arr1, arr2)
        with pytest.raises(ValueError):
            NOP(arr1, arr2)
        with pytest.raises(ValueError):
            NP(arr1, arr2)

    def test_no_nop_np_non_numeric(self):
        arr = ["a", "b", "c"]
        with pytest.raises(Exception):
            NO(arr, arr)
        with pytest.raises(Exception):
            NOP(arr, arr)
        with pytest.raises(Exception):
            NP(arr, arr)

    def test_no_nop_np_none(self):
        arr = None
        with pytest.raises(Exception):
            NO(arr, arr)
        with pytest.raises(Exception):
            NOP(arr, arr)
        with pytest.raises(Exception):
            NP(arr, arr)

    def test_no_xarray(self):
        arr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        assert NO(arr, arr) == 5
        # xarray axis must be the dimension name, not integer
        assert NO(arr, arr, axis="time") == 5

    def test_no_masked(self):
        arr = np.ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 1])
        assert NO(arr, arr) == 3
        assert NO(arr, arr, axis=0) == 3

    def test_nop_xarray(self):
        arr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        assert NOP(arr, arr) == 5
        assert NOP(arr, arr, axis="time") == 5

    def test_nop_numpy(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert NOP(arr, arr) == 5
        assert NOP(arr, arr, axis=0) == 5

    def test_nop_masked(self):
        arr = np.ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 1])
        assert NOP(arr, arr) == 3
        assert NOP(arr, arr, axis=0) == 3

    def test_np_xarray(self):
        arr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        assert NP(arr, arr) == 5
        assert NP(arr, arr, axis="time") == 5

    def test_np_numpy(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert NP(arr, arr) == 5
        assert NP(arr, arr, axis=0) == 5

    def test_np_masked(self):
        arr = np.ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 1])
        assert NP(arr, arr) == 3
        assert NP(arr, arr, axis=0) == 3

    def test_no_nop_np_empty(self):
        arr = np.array([])
        assert NO(arr, arr) == 0
        assert NOP(arr, arr) == 0
        assert NP(arr, arr) == 0

    def test_no_nop_np_all_masked(self):
        arr = np.ma.array([1, 2, 3], mask=[1, 1, 1])
        assert NO(arr, arr) == 0
        assert NOP(arr, arr) == 0
        assert NP(arr, arr) == 0

    def test_nop_masked(self):
        arr = np.ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 1])
        assert NOP(arr, arr) == 3
        assert NOP(arr, arr, axis=0) == 3

    def test_nop_xarray(self):
        arr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        assert NOP(arr, arr) == 5
        assert NOP(arr, arr, axis="time") == 5

    def test_np_numpy(self):
        arr = np.array([1, 2, 3, 4, 5])
        assert NP(arr, arr) == 5
        assert NP(arr, arr, axis=0) == 5

    def test_np_masked(self):
        arr = np.ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 1])
        assert NP(arr, arr) == 3
        assert NP(arr, arr, axis=0) == 3

    def test_np_xarray(self):
        arr = xr.DataArray([1, 2, 3, 4, 5], dims=["time"])
        assert NP(arr, arr) == 5
        assert NP(arr, arr, axis="time") == 5

    def test_no_nop_np_empty(self):
        arr = np.array([])
        assert NO(arr, arr) == 0
        assert NOP(arr, arr) == 0
        assert NP(arr, arr) == 0

    def test_no_nop_np_all_masked(self):
        arr = np.ma.array([1, 2, 3], mask=[1, 1, 1])
        assert NO(arr, arr) == 0
        assert NOP(arr, arr) == 0
        assert NP(arr, arr) == 0
