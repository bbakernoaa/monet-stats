import numpy as np
import pytest
import xarray as xr

from monet_stats.contingency_metrics import (
    HSS,
    FAR_min_threshold,
    HSS_max_threshold,
)


def test_optimization_equivalence():
    # Synthetic data
    np.random.seed(42)
    obs = np.random.rand(100) * 10
    mod = obs + np.random.randn(100)

    min_r, max_r, step = 1.0, 9.0, 0.5

    # Test HSS
    thresh_hss, val_hss = HSS_max_threshold(obs, mod, min_r, max_r, step)

    # Manual check
    thresholds = np.arange(min_r, max_r, step)
    manual_vals = [HSS(obs, mod, t) for t in thresholds]
    max_idx = np.argmax(manual_vals)

    assert thresh_hss == thresholds[max_idx]
    assert np.isclose(val_hss, manual_vals[max_idx])


def test_xarray_optimization():
    obs = xr.DataArray(np.random.rand(10, 10) * 10, dims=("lat", "lon"))
    mod = obs + 0.5

    thresh, val = HSS_max_threshold(obs, mod, 1.0, 9.0, 1.0)

    assert isinstance(thresh, float)
    assert isinstance(val, float)
    assert val > 0


def test_far_min_threshold():
    obs = np.array([0, 0, 1, 1])
    mod = np.array([0, 1, 1, 1])  # One false alarm at low threshold

    # At threshold 0.5: Mod >= 0.5 is [0, 1, 1, 1]. Obs >= 0.5 is [0, 0, 1, 1].
    # Hits=2, False Alarm=1. FAR = 1/(2+1) = 0.33

    # At threshold 1.5: Mod >= 1.5 is [0, 0, 0, 0]. Obs >= 1.5 is [0, 0, 0, 0].
    # FAR = nan or 0 depending on denom.

    thresh, val = FAR_min_threshold(obs, mod, 0.1, 2.0, 0.5)
    assert thresh >= 0.1


def test_empty_input():
    obs = np.array([])
    mod = np.array([])
    with pytest.raises(ValueError):
        # np.arange might return empty if range is invalid, but here we test empty data
        # _contingency_table will return 0s, HSS will return NaN.
        # nanargmax on all-NaN will raise ValueError
        HSS_max_threshold(obs, mod, 1, 5)


if __name__ == "__main__":
    pytest.main([__file__])
