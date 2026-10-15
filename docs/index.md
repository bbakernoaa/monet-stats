<div class="ms-hero" markdown>

# MONET Stats

## Model verification for atmospheric science

Evaluate forecasts against observations with a broad set of deterministic, categorical, spatial, ensemble, and temporal statistics. Work with NumPy, pandas, or labeled Xarray data, and keep large workflows lazy with Dask.

[Get started](getting-started.md){ .md-button .md-button--primary }
[Browse the API](api/overview.md){ .md-button }

</div>

<div class="ms-highlights" markdown>

<div class="ms-highlight">

**149 metrics**
Error, skill, correlation, distribution, and specialized verification methods.

</div>
<div class="ms-highlight">

**Labeled workflows**
Use Xarray dimensions and coordinates to reduce over time, space, or both.

</div>
<div class="ms-highlight">

**Lazy by design**
Keep supported Xarray calculations Dask-backed until you choose to compute.

</div>

</div>

## Start with a few metrics

Install the core package and calculate common scores with NumPy arrays:

```bash
pip install monet-stats
```

```python
import numpy as np
import monet_stats as ms

obs = np.array([1.2, 2.5, 3.7, 4.1, 5.0])
mod = np.array([1.1, 2.6, 3.5, 4.3, 4.8])

scores = {
    "RMSE": ms.RMSE(obs, mod),
    "Mean bias": ms.MB(obs, mod),
    "Correlation": ms.pearsonr(obs, mod),
    "Nash–Sutcliffe efficiency": ms.NSE(obs, mod),
}
```

For larger gridded datasets, install the optional Dask dependencies and open data in chunks:

```bash
pip install 'monet-stats[dask]'
```

```python
import xarray as xr
import monet_stats as ms

obs = xr.open_dataset("observations.nc", chunks={"time": 120})["temperature"]
mod = xr.open_dataset("forecast.nc", chunks={"time": 120})["temperature"]

rmse_map = ms.RMSE(obs, mod, axis="time")  # remains lazy
rmse_map = rmse_map.compute()               # calculate when needed
```

Choose chunks that fit comfortably in worker memory. Some distribution and sequence metrics need all values along their reduction dimensions in one chunk; see the [performance guide](api/performance.md) for details.

## Explore the toolkit

<div class="grid cards" markdown>

-   :material-chart-box:{ .lg .middle } **Model scores**

    ---

    Summarize continuous forecasts with error, bias, efficiency, and correlation metrics.

    [Error metrics](api/error-metrics.md) · [Efficiency](api/efficiency-metrics.md) · [Correlation](api/correlation-metrics.md)

-   :material-check-decagram:{ .lg .middle } **Forecast verification**

    ---

    Evaluate event forecasts, thresholds, probabilistic predictions, and ensembles.

    [Contingency scores](api/contingency-metrics.md) · [Spatial and ensemble](api/spatial-ensemble-metrics.md)

-   :material-earth:{ .lg .middle } **Spatial and temporal analysis**

    ---

    Compare structures, tracks, timing, distributions, and gridded fields.

    [Spatial skill](api/spatial-skill-metrics.md) · [Track metrics](api/track-metrics.md) · [Temporal metrics](api/temporal-metrics.md)

-   :material-code-braces:{ .lg .middle } **Build a workflow**

    ---

    Use the Xarray accessor, performance helpers, plugins, and worked examples.

    [Xarray accessor](api/accessor.md) · [Climate workflow](workflows/climate-data-analysis.md) · [Notebooks](notebooks/README.md)

</div>

## Pick the right entry point

=== "I have arrays"

    Use top-level functions such as `RMSE`, `NSE`, and `pearsonr` for NumPy arrays. Functions accept an `axis` argument when you want a score for each slice.

    [See the API overview](api/overview.md)

=== "I have gridded data"

    Pass Xarray `DataArray` objects to retain coordinates and metadata. Reduce over named dimensions such as `time` or `lat` and `lon`.

    [Read the Xarray guide](api/accessor.md)

=== "I need a complete example"

    Follow the getting started guide, then browse the climate workflow and notebooks for end-to-end patterns.

    [Open the tutorial](getting-started.md)

## About MONET Stats

MONET Stats is an atmospheric-science library in the MONET ecosystem. It brings commonly used verification methods into a consistent Python interface and supports metadata-aware analysis with Xarray.

[Installation](installation.md){ .md-button }
[Mathematical methods](math/overview.md){ .md-button }
[Contributing](contributing.md){ .md-button }
