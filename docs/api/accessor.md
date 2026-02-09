# Xarray Accessors

The MONET Stats package provides first-class Xarray accessors to enable seamless integration with the Pangeo ecosystem. These accessors allow you to perform statistical analyses directly on `xarray.DataArray` and `xarray.Dataset` objects using the `.monet_stats` namespace.

## DataArray Accessor

The `MonetDataArrayAccessor` (available via `da.monet_stats`) provides methods for common time-series and spatial analyses.

::: monet_stats.accessor.MonetDataArrayAccessor

## Dataset Accessor

The `MonetDatasetAccessor` (available via `ds.monet_stats`) provides methods for calculating summary statistics and performing analyses across multiple variables in a dataset.

::: monet_stats.accessor.MonetDatasetAccessor
