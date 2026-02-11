# Xarray Accessors

The MONET Stats package provides first-class Xarray accessors to enable seamless integration with the Pangeo ecosystem. These accessors allow you to perform statistical analyses directly on `xarray.DataArray` and `xarray.Dataset` objects using the `.monet_stats` namespace.

## DataArray Accessor

The `MonetDataArrayAccessor` (available via `da.monet_stats`) provides methods for common time-series and spatial analyses, as well as a comprehensive suite of verification metrics.

### Verification Metrics

The accessor provides direct access to many common verification metrics. These methods typically take an `obs` DataArray as their first argument and an optional `dim` parameter.

Example usage:
```python
rmse = mod_da.monet_stats.rmse(obs_da, dim="time")
pearson_r = mod_da.monet_stats.pearsonr(obs_da)
```

Available metrics include:
- **Error Metrics**: `mae`, `rmse`, `mb`, `ioa`, `crmse`, `mdnb`, `nmse`, `mnb`, `mne`, `nse`
- **Correlation Metrics**: `pearsonr`, `r2`, `kge`, `ccc`
- **Relative Metrics**: `nmb`, `fb`

### Verification Bundle

The `verify` method allows for efficient computation of multiple metrics at once, returning an `xarray.Dataset`.

```python
metrics_ds = mod_da.monet_stats.verify(obs_da, dim="time")
```

The bundle includes: `MAE`, `RMSE`, `MB`, `R` (Pearson), `IOA`, `NMB`, `MNB`, `MNE`, `NSE`, and `R2`.

### Full API Reference

::: monet_stats.accessor.MonetDataArrayAccessor

## Dataset Accessor

The `MonetDatasetAccessor` (available via `ds.monet_stats`) provides methods for calculating summary statistics and performing analyses across multiple variables in a dataset.

::: monet_stats.accessor.MonetDatasetAccessor
