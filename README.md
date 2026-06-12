# MONET Stats 🍃⚡

A comprehensive statistics and utility library for atmospheric sciences, optimized for the Pangeo ecosystem and fully **Aero Protocol** compliant.

[![CI/CD Pipeline](https://github.com/noaa-oar-arl/monet-stats/actions/workflows/ci.yml/badge.svg)](https://github.com/noaa-oar-arl/monet-stats/actions/workflows/ci.yml)
[![Docs](https://github.com/noaa-oar-arl/monet-stats/actions/workflows/docs.yml/badge.svg)](https://github.com/noaa-oar-arl/monet-stats/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/noaa-oar-arl/monet-stats/branch/main/graph/badge.svg)](https://codecov.io/gh/noaa-oar-arl/monet-stats)
[![Documentation](https://img.shields.io/badge/docs-noaa--oar--arl.github.io-blue.svg)](https://noaa-oar-arl.github.io/monet-stats)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Core Mission

Architect scientific pipelines that balance three competing goals:
- **Speed**: Aggressive vectorization (Numpy/Xarray) and lazy evaluation (Dask/Cubed).
- **Maintainability**: Strictly typed code with NumPy-style docstrings.
- **Provenance**: Automatically track data lineage (what happened to the data) via `attrs['history']`.

## Installation

```bash
pip install monet-stats
```

Optional dependencies for Pangeo stack:
```bash
pip install monet-stats[dask,cubed,docs,test]
```

## Quick Start

### Error Metrics

```python
import xarray as xr
import numpy as np
from monet_stats.error_metrics import MB, RMSE

# Assume all data > RAM. Use dask chunks immediately.
obs = xr.open_dataset('obs.nc', chunks={'time': 100})['variable']
mod = xr.open_dataset('mod.nc', chunks={'time': 100})['variable']

# Compute Mean Bias map over the time dimension
bias = MB(obs, mod, axis='time')

# Automatic provenance tracking
print(bias.attrs['history'])
```

### Categorical Skill Scores

```python
from monet_stats.contingency_metrics import HSS, ETS

# Evaluate Heidke Skill Score at a specific threshold
skill = HSS(obs, mod, minval=50.0)
```

### Xarray Accessor

Chain operations directly on Xarray objects using the `.monet_stats` namespace:

```python
# Compute monthly climatology and area-weighted mean in one go
result = obs.monet_stats.climatology(freq='month').monet_stats.weighted_spatial_mean()
```

## Documentation

Full API documentation and tutorials are available at:
**[https://noaa-oar-arl.github.io/monet-stats](https://noaa-oar-arl.github.io/monet-stats)**

## CI/CD and Code Quality

This project uses a comprehensive CI/CD pipeline with the following quality checks:

- **Testing**: Multi-Python version testing (3.11-3.12) with 60%+ coverage
- **Code Formatting**: Black and Ruff formatting enforcement
- **Linting**: Ruff and Pycodestyle linting
- **Type Checking**: MyPy static type analysis
- **Provenance**: Mandatory `history` attribute updates on all transformations
