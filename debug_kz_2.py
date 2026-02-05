import numpy as np
import xarray as xr
from monet_stats.analysis import kz_filter

da = xr.DataArray(np.random.rand(50), dims="time")
ds = xr.Dataset({"var1": da})

def my_func(x, **kwargs):
    x.attrs["foo"] = "bar"
    return x

res_ds = ds.map(my_func)
print("res_ds.var1 attrs:", res_ds.var1.attrs)
