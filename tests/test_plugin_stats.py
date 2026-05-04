import numpy as np
import pandas as pd
import pytest
import xarray as xr

from monet_stats import stats
from monet_stats.plugin_system import plugin_manager


def test_plugin_decorator_registration():
    """Test that the @register decorator correctly registers a plugin."""

    @plugin_manager.register(name="CustomAdd", description="Adds 1 to the mean of mod")
    def custom_add(obs, mod, axis=None):
        if isinstance(mod, xr.DataArray):
            return mod.mean(dim=axis) + 1
        return np.mean(mod, axis=axis) + 1

    # Verify original function is preserved
    assert callable(custom_add)
    assert custom_add.__name__ == "custom_add"

    assert "CustomAdd" in plugin_manager.list_plugins()
    plugin = plugin_manager.get_plugin("CustomAdd")
    assert plugin.name() == "CustomAdd"
    assert plugin.description() == "Adds 1 to the mean of mod"


def test_stats_with_plugins_pandas():
    """Test that stats() correctly includes plugin results for Pandas DataFrames."""
    df = pd.DataFrame({"Obs": [1, 2, 3], "Mod": [2, 4, 6]})

    # We already have WMAPE registered as a built-in plugin
    results = stats(df, plugins=["WMAPE"])

    assert "WMAPE" in results
    # WMAPE = (sum|mod-obs| / sum|obs|) * 100 = ( (1+2+3) / (1+2+3) ) * 100 = 100.0
    assert np.isclose(results["WMAPE"], 100.0)


def test_stats_with_plugins_xarray():
    """Test that stats() correctly includes plugin results for Xarray Datasets."""
    obs = xr.DataArray([1.0, 2.0, 3.0], dims="x", name="Obs")
    mod = xr.DataArray([2.0, 4.0, 6.0], dims="x", name="Mod")
    ds = xr.Dataset({"Obs": obs, "Mod": mod})

    results = stats(ds, plugins=["WMAPE", "MAPE_Bias"])

    assert "WMAPE" in results
    assert "MAPE_Bias" in results
    assert np.isclose(results["WMAPE"], 100.0)

    # MAPE_Bias: pe = (mod-obs)/obs = [1, 1, 1]. All positive.
    # pos_errors = mean([1, 1, 1]) = 1.0. neg_errors = 0.0. Result = 1.0.
    assert np.isclose(results["MAPE_Bias"], 1.0)


def test_stats_with_plugins_dask():
    """Test that plugins in stats() work with Dask and preserve laziness."""
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    obs = xr.DataArray(da.from_array([1.0, 2.0, 3.0], chunks=2), dims="x", name="Obs")
    mod = xr.DataArray(da.from_array([2.0, 4.0, 6.0], chunks=2), dims="x", name="Mod")
    ds = xr.Dataset({"Obs": obs, "Mod": mod})

    # This should trigger a single compute call
    results = stats(ds, plugins=["WMAPE"])

    assert "WMAPE" in results
    assert np.isclose(results["WMAPE"], 100.0)


def test_stats_with_missing_plugin():
    """Test that stats() handles non-existent plugins gracefully."""
    df = pd.DataFrame({"Obs": [1, 2, 3], "Mod": [2, 4, 6]})
    results = stats(df, plugins=["NonExistentPlugin"])

    assert "NonExistentPlugin" in results
    assert np.isnan(results["NonExistentPlugin"])


def test_plugin_zero_division_handling():
    """Test that refactored plugins handle zero division correctly."""
    # Obs has a zero
    obs = np.array([0.0, 1.0, 2.0])
    mod = np.array([1.0, 2.0, 3.0])

    # WMAPE should handle it (sum|obs| = 3.0, not zero)
    wmape = plugin_manager.get_plugin("WMAPE")
    res = wmape.compute(obs, mod)
    # sum|mod-obs| = 1+1+1 = 3. sum|obs| = 3. Result = 100.0
    assert np.isclose(res, 100.0)

    # MAPE_Bias has pe = (mod-obs)/obs. For the first element, it's 1.0/0.0 -> inf or nan.
    # Our implementation uses .where to put nan there.
    mape_bias = plugin_manager.get_plugin("MAPE_Bias")
    res = mape_bias.compute(obs, mod)
    # pe = [nan, 1.0, 0.5]. Mean of [1.0, 0.5] = 0.75.
    assert np.isclose(res, 0.75)
