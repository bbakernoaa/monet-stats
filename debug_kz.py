import numpy as np
import xarray as xr
from monet_stats.analysis import kz_filter

da = xr.DataArray(np.random.rand(50), dims="time")
print("Calling kz_filter on DataArray...")
res_da = kz_filter(da, m=3, k=2, dim="time")
print("res_da attrs:", res_da.attrs)

ds = xr.Dataset({"var1": da})
print("\nCalling kz_filter on Dataset...")
res_ds = kz_filter(ds, m=3, k=2, dim="time")
print("res_ds.var1 attrs:", res_ds.var1.attrs)
