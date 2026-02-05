# API Reference Overview

The Monet Stats API provides a comprehensive collection of statistical metrics and utilities for atmospheric sciences applications. This reference covers all available functions, their parameters, return values, and use cases.

## API Structure

Monet Stats is organized into several functional modules:

### Core Modules

- **[Contingency Metrics](contingency-metrics.md)**: Binary event verification and categorical forecast evaluation
- **[Correlation Metrics](correlation-metrics.md)**: Statistical correlation and skill score calculations
- **[Error Metrics](error-metrics.md)**: Error analysis and bias quantification
- **[Efficiency Metrics](efficiency-metrics.md)**: Model efficiency and performance measures
- **[Relative Metrics](relative-metrics.md)**: Normalized and relative error measures
- **[Spatial & Ensemble Metrics](spatial-ensemble-metrics.md)**: Spatial verification and ensemble analysis
- **[Utility Functions](utils-stats.md)**: Helper functions and data processing utilities

## Import Conventions

### Standard Imports

```python
# Import entire library
import monet_stats as ms

# Import specific modules
from monet_stats import contingency_metrics, correlation_metrics

# Import specific functions
from monet_stats import R2, RMSE, POD, FAR
```

### Recommended Import Style

```python
import monet_stats as ms
import numpy as np
import xarray as xr
```

## Data Format Support

### NumPy Arrays

```python
import numpy as np

obs = np.array([1, 2, 3, 4, 5])
mod = np.array([1.1, 2.1, 2.9, 4.1, 4.8])

r2 = ms.R2(obs, mod)  # Works with 1D arrays
rmse = ms.RMSE(obs, mod)
```

### Multi-dimensional Arrays

```python
# 2D arrays (e.g., spatial fields)
obs_2d = np.random.normal(20, 2, (50, 50))
mod_2d = obs_2d + np.random.normal(0, 1, (50, 50))

fss = ms.FSS(obs_2d, mod_2d, window=5)
```

### Pandas DataFrames

```python
import pandas as pd

df = pd.DataFrame({
    'observed': np.random.normal(20, 2, 100),
    'modeled': np.random.normal(20.5, 2.5, 100),
    'station': ['A'] * 50 + ['B'] * 50
})

# Apply metrics by group
results = df.groupby('station').apply(
    lambda x: pd.Series({
        'RMSE': ms.RMSE(x['observed'], x['modeled']),
        'R2': ms.R2(x['observed'], x['modeled'])
    })
)
```

### XArray DataArrays

```python
import xarray as xr

obs_da = xr.DataArray(
    np.random.normal(20, 2, (10, 10, 365)),
    dims=['lat', 'lon', 'time'],
    coords={
        'lat': range(10),
        'lon': range(10),
        'time': pd.date_range('2020-01-01', periods=365, freq='D')
    }
)

mod_da = obs_da + xr.DataArray(
    np.random.normal(0, 1, (10, 10, 365)),
    dims=['lat', 'lon', 'time'],
    coords=obs_da.coords
)

# Metrics preserve coordinates and dimensions
skill = ms.R2(obs_da, mod_da)  # Returns DataArray with same coordinates
```

## Common Parameters

### Core Parameters

Most metrics accept these common parameters:

- `obs`: Observed values (array-like)
- `mod`: Modeled/predicted values (array-like)
- `axis`: Axis along which to compute metrics (int, optional)
- `nan_policy`: How to handle NaN values ('omit', 'propagate', 'raise')

### Threshold Parameters

Many metrics use threshold parameters for categorical analysis:

- `minval`: Minimum threshold for event definition
- `maxval`: Maximum threshold for event definition (optional)

### Spatial Parameters

Spatial metrics often include:

- `window`: Size of spatial window (int)
- `threshold`: Event threshold for spatial analysis

## Return Value Types

### Scalar Values

Most metrics return single scalar values:

```python
r2 = ms.R2(obs, mod)  # float
rmse = ms.RMSE(obs, mod)  # float
```

### Arrays

Some metrics return arrays for multi-dimensional input:

```python
# For 2D spatial data
fss = ms.FSS(obs_2d, mod_2d)  # float
```

### DataArrays (xarray)

When using xarray inputs, metrics return DataArrays:

```python
skill = ms.R2(obs_da, mod_da)  # DataArray with coordinates
```

## Error Handling

### Data Shape Validation

```python
try:
    result = ms.R2(obs_1d, mod_2d)  # Will raise ValueError
except ValueError as e:
    print(f"Shape mismatch: {e}")
```

### NaN Handling

```python
# Data with NaN values
obs_with_nan = np.array([1, 2, np.nan, 4])
mod_with_nan = np.array([1.1, 2.1, 3.1, 4.1])

# Functions automatically handle NaN by default
rmse = ms.RMSE(obs_with_nan, mod_with_nan)  # Uses valid pairs only
```

### Type Validation

```python
# Invalid types will raise TypeError
try:
    result = ms.R2("invalid", "data")  # TypeError
except TypeError as e:
    print(f"Invalid data type: {e}")
```

## Performance Considerations

### Vectorized Operations

All metrics use NumPy and Xarray vectorized operations for optimal performance. Loop-free implementations ensure maximum speed on modern hardware.

### Out-of-Core Processing with Dask

For datasets larger than RAM, `monet-stats` is fully compatible with Dask. Most metrics are "lazy-aware" and will preserve the Dask computation graph.

```python
# Open large dataset with chunks (Aero Protocol recommended)
ds = xr.open_dataset("large_data.nc", chunks={"time": "auto", "lat": 100, "lon": 100})
obs = xr.open_dataset("obs_data.nc", chunks={"time": "auto", "lat": 100, "lon": 100})

# Metrics stay lazy and don't trigger loading
skill = ms.RMSE(obs.var, ds.var, axis="time")

# Execution only happens on compute()
result = skill.compute()
```

### Scientific Provenance

When using Xarray DataArrays, `monet-stats` automatically updates the `attrs['history']` to track which statistical operations were applied to the data, ensuring scientific reproducibility.

## Example Usage Patterns

### Basic Error Analysis

```python
import monet_stats as ms
import numpy as np

# Sample data
obs = np.array([1.0, 2.5, 3.2, 4.8, 5.0])
mod = np.array([1.2, 2.3, 3.5, 4.6, 5.2])

# Error metrics
error_analysis = {
    'RMSE': ms.RMSE(obs, mod),
    'MAE': ms.MAE(obs, mod),
    'MB': ms.MB(obs, mod),
    'NMB': ms.NMB(obs, mod),
    'NME': ms.NME(obs, mod)
}
```

### Comprehensive Model Evaluation

```python
def evaluate_model(observed, modeled):
    """Comprehensive model evaluation suite"""

    metrics = {
        # Error measures
        'RMSE': ms.RMSE(observed, modeled),
        'MAE': ms.MAE(observed, modeled),
        'MB': ms.MB(observed, modeled),
        'NMB': ms.NMB(observed, modeled),

        # Skill scores
        'R2': ms.R2(observed, modeled),
        'NSE': ms.NSE(observed, modeled),
        'KGE': ms.KGE(observed, modeled),
        'IOA': ms.IOA(observed, modeled),

        # Relative measures
        'MPE': ms.MPE(observed, modeled),
        'NME': ms.NME(observed, modeled)
    }

    return metrics

# Usage
results = evaluate_model(obs, mod)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
```

### Categorical Event Analysis

```python
# Binary event analysis
obs_events = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
mod_events = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])

# Contingency table metrics
contingency_metrics = {
    'POD': ms.POD(obs_events, mod_events, threshold=0.5),
    'FAR': ms.FAR(obs_events, mod_events, threshold=0.5),
    'CSI': ms.CSI(obs_events, mod_events, threshold=0.5),
    'HSS': ms.HSS(obs_events, mod_events, threshold=0.5),
    'ETS': ms.ETS(obs_events, mod_events, threshold=0.5)
}
```

## API Reference

The following sections provide auto-generated documentation for each core module based on docstrings.

### Contingency Metrics
::: monet_stats.contingency_metrics

### Correlation Metrics
::: monet_stats.correlation_metrics

### Error Metrics
::: monet_stats.error_metrics

### Efficiency Metrics
::: monet_stats.efficiency_metrics

### Relative Metrics
::: monet_stats.relative_metrics

### Spatial & Ensemble Metrics
::: monet_stats.spatial_ensemble_metrics

### Utility Functions
::: monet_stats.utils_stats

## Contributing to API Documentation

If you find issues with the API documentation or would like to suggest improvements:

1. Check the [GitHub Issues](https://github.com/noaa-oar-arl/monet-stats/issues)
2. Submit new issues with clear descriptions
3. Consider contributing improvements via pull requests

For development documentation, see the [Contributing Guide](../contributing.md).
