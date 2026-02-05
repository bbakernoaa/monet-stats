import numpy as np
import xarray as xr
from monet_stats.analysis import kz_filter
from monet_stats.utils_stats import _update_history

da = xr.DataArray(np.random.rand(50), dims="time")
print("Before history, attrs:", da.attrs)
da_with_history = _update_history(da, "test")
print("After history, attrs:", da_with_history.attrs)

ds = xr.Dataset({"var1": da})
print("Dataset map call...")
res_ds = ds.map(kz_filter, m=3, k=2, dim="time")
print("res_ds.var1 attrs:", res_ds.var1.attrs)
