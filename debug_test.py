import numpy as np
import xarray as xr
from monet_stats.analysis import kz_filter

ds = xr.Dataset({
    "var1": (("time"), np.random.rand(50)),
    "var2": (("time"), np.random.rand(50))
})

print("Before filter, var1 attrs:", ds.var1.attrs)
result = kz_filter(ds, m=3, k=2, dim="time")
print("After filter, var1 attrs:", result.var1.attrs)
print("After filter, dataset attrs:", result.attrs)
