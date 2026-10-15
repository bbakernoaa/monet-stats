---
type: reference
category: reference
tags: [xarray, accessor, pangeo]
---

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
- **Error Metrics**: `mae`, `rmse`, `mb`, `ioa`, `crmse`, `mdnb`, `nmse`, `nmb`, `mnb`, `mne`, `mse`, `mape`, `mase`, `nrmse`, `stdo`, `stdp`, `coe`, `corr_index`, `bias_fraction`, `log_error`, `volumetric_error`, `fac2`, `rmsle`, `nse`, `rnse`, `mnse`, `pc`
- **Correlation Metrics**: `pearsonr`, `r2`, `kge`, `ccc`, `spearmanr`, `kendalltau`
- **Relative Metrics**: `nmb`, `fb`, `mg`, `vg`, `nme`, `mdnnb`, `mdnne`, `mpe`, `nmpe`
- **Categorical Metrics**: `hss`, `ets`, `csi`, `pod`, `far`, `fbi`, `tss`, `bss_binary`, `bs`
- **Distributional Metrics**: `wasserstein_distance`, `kl_divergence`, `jensenshannon_divergence`, `energy_distance`, `sinkhorn_distance`, `mutual_information`
- **Temporal Metrics**: `dtw`, `xwt`, `phase_error`
- **Spatial & Ensemble Metrics**: `fss`, `vets`, `crps`, `sal`, `eds`, `spread_error`, `reliability_diagram`
- **Track Metrics**: `track_error`, `along_track_error`, `cross_track_error`, `translation_speed`, `find_storm_center`, `find_storm_centers`
- **Uncertainty**: `bootstrap` (block bootstrap confidence intervals)
- **Bundled Reports**: `taylor_statistics`, `verify`
- **Analysis Methods**: `climatology`, `resample_data`, `kz_filter`, `diurnal_cycle`, `rolling_mean_8h`, `rolling_mean_24h`, `mda1`, `mda8`, `exceedance_count`, `percentile`, `peak_timing`, `weighted_spatial_mean`, `fft_analysis`, `power_spectrum`, `seasonal_mean`, `monthly_climatology`, `anomalies`, `detrend`
- **Performance Helpers**: `optimize`, `rechunk`

### Verification Bundle

The `verify` method allows for efficient computation of multiple metrics at once, returning an `xarray.Dataset`.

```python
metrics_ds = mod_da.monet_stats.verify(obs_da, dim="time")

# Add categorical scores by supplying an event threshold
metrics_ds = mod_da.monet_stats.verify(obs_da, dim="time", threshold=0.2, weights=weights)
```

The bundle always includes: `MAE`, `RMSE`, `MB`, `R` (Pearson), `IOA`, `NMB`,
`MNB`, `MNE`, `NSE`, `R2`, `FAC2`, `MSE`, `SpearmanR`, and `NME`. When
`threshold=` is provided, the categorical scores `POD`, `FAR`, `HSS`, `CSI`,
`TSS`, `ETS`, `FBI`, and `BSS_binary` are appended. The optional `weights`
argument switches the weighted-metric variants into the same computation.

### Full API Reference

::: monet_stats.accessor.MonetDataArrayAccessor

## Dataset Accessor

The `MonetDatasetAccessor` (available via `ds.monet_stats`) provides methods for calculating summary statistics and performing analyses across multiple variables in a dataset.

::: monet_stats.accessor.MonetDatasetAccessor
