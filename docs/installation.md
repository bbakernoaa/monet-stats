---
type: howto
category: how_to
tags: [installation, pip, dependencies]
---

# Installation and Setup Guide

This guide covers the installation process, system requirements, and setup instructions for Monet Stats.

## Requirements

### System Requirements

- **Python**: 3.11 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 512 MB minimum (2GB recommended for large datasets)
- **Storage**: 100 MB for installation

### Software Dependencies

Monet Stats requires the following Python packages (installed automatically by pip):

- **NumPy**: ≥1.20.0
- **Pandas**: ≥2.0.0
- **SciPy**: ≥1.7.0
- **XArray**: ≥2023.5.0

Optional backends are provided as extras: `dask` (Dask + distributed) and
`cubed` (Cubed + cubed-xarray). See [Optional Dependencies](#optional-dependencies).

## Installation Methods

### Option 1: pip Installation (Recommended)

```bash
# Install from PyPI
pip install monet-stats

# Install with development dependencies
pip install monet-stats[dev]

# Install with test dependencies
pip install monet-stats[test]
```

### Option 2: GitHub Installation

```bash
# Clone the repository
git clone https://github.com/noaa-oar-arl/monet-stats.git
cd monet-stats

# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[dev,test]"
```

### Pre-commit Hooks

To maintain code quality standards, this project uses pre-commit hooks. After installing the development dependencies, install the pre-commit hooks:

```bash
pre-commit install
```

This will ensure code formatting, linting, and other quality checks run automatically before each commit.

To run pre-commit checks manually on all files:

```bash
pre-commit run --all-files
```

### Option 3: uv Installation

```bash
# Install with uv
uv pip install monet-stats

# Install from source with all extras
uv pip install -e ".[dev,test,docs]"
```

> **Note:** Monet Stats is not currently published on conda-forge. Install it
> with pip or uv.

## Verification

After installation, verify the installation by running the following Python code:

```python
import importlib.metadata

import monet_stats

# Check version (the package does not define __version__; use package metadata)
print(f"Monet Stats version: {importlib.metadata.version('monet-stats')}")

# Test basic imports
from monet_stats import R2, RMSE, POD
print("✓ All core metrics imported successfully")

# Test with sample data
import numpy as np
obs = np.array([1, 2, 3, 4, 5])
mod = np.array([1.1, 2.1, 2.9, 4.1, 4.8])

r2 = R2(obs, mod)
rmse = RMSE(obs, mod)
print(f"✓ Sample calculation - R²: {r2:.3f}, RMSE: {rmse:.3f}")
```

## Optional Dependencies

### For Advanced Features

#### XArray Support

Xarray is a **required** core dependency — no extra install is needed. For
NetCDF/Zarr file access, install the IO engines separately:

```bash
pip install netcdf4 zarr
```

#### Lazy and Parallel Computation

```bash
# Dask backend
pip install "monet-stats[dask]"

# Cubed backend
pip install "monet-stats[cubed]"
```

#### Visualization

```bash
pip install matplotlib seaborn plotly
```

Visualization packages are not required by Monet Stats itself; they are
convenient for plotting metric results.

## Development Installation

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/noaa-oar-arl/monet-stats.git
cd monet-stats

# Create virtual environment
python -m venv monet-stats-env
source monet-stats-env/bin/activate  # On Windows: monet-stats-env\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The `dev` extra includes:

- **Pre-commit hooks**: `pre-commit`
- **Testing**: `pytest`, `pytest-cov`, `pytest-benchmark`

The `test` extra adds `hypothesis`, `coverage`, and `psutil`. Code formatting
and linting are handled by **ruff** through the pre-commit hooks (see
[Contributing Guide](contributing.md)). Documentation builds use the `docs`
extra (`mkdocs`, `mkdocstrings`, `mkdocs-material`).

## Configuration

Monet Stats has no configuration file or environment variables. All behavior
is controlled through function arguments — for example, `axis`/`dim` for
reduction, `weights` for weighted metrics, and the `threshold`/`minval`/
`maxval` arguments for categorical scores. Lazy computation is activated by
passing chunked (Dask-backed) xarray objects; see the
[Getting Started Guide](getting-started.md) and
[Performance reference](api/performance.md).

## Common Issues and Solutions

### Installation Problems

#### Permission Issues

```bash
# Use user installation
pip install --user monet-stats

# Or use virtual environment
python -m venv myenv
source myenv/bin/activate
pip install monet-stats
```

#### Version Conflicts

```bash
# Upgrade pip
pip install --upgrade pip

# Force reinstallation
pip install --force-reinstall monet-stats

# Clean cache
pip cache purge
```

#### Missing Dependencies

```bash
# Install specific versions (must satisfy the package requirements:
# numpy>=1.20.0, pandas>=2.0.0, scipy>=1.7.0, xarray>=2023.5.0)
pip install "numpy>=1.20.0" "pandas>=2.0.0" "scipy>=1.7.0" "xarray>=2023.5.0"

# Or let pip resolve compatible versions automatically
pip install --force-reinstall monet-stats
```

### Runtime Issues

#### ImportError: No module named 'monet_stats'

```bash
# Verify installation
pip show monet-stats

# Reinstall if needed
pip install --reinstall monet-stats
```

#### Memory Issues with Large Datasets

```python
import monet_stats
import numpy as np

# Process data in chunks
def process_in_chunks(obs, mod, chunk_size=10000):
    n = len(obs)
    results = []

    for i in range(0, n, chunk_size):
        obs_chunk = obs[i:i+chunk_size]
        mod_chunk = mod[i:i+chunk_size]

        # Calculate metrics for chunk
        r2 = monet_stats.R2(obs_chunk, mod_chunk)
        rmse = monet_stats.RMSE(obs_chunk, mod_chunk)

        results.append({'R2': r2, 'RMSE': rmse})

    return results
```

#### XArray Compatibility

```python
import monet_stats as ms
import xarray as xr

# Ensure xarray is installed
try:
    import xarray
except ImportError:
    raise ImportError("xarray is required for DataArray support")

# Use with xarray DataArrays
obs_da = xr.DataArray(obs, dims=['time'])
mod_da = xr.DataArray(mod, dims=['time'])

r2 = ms.R2(obs_da, mod_da)  # Works with xarray
```

## Containerized Deployment

Monet Stats does not ship a `Dockerfile`. To run it in a container, install
the package from PyPI inside any Python ≥3.11 base image:

```dockerfile
FROM python:3.12-slim
RUN pip install monet-stats
```

```bash
docker build -t monet-stats-app .

# Run an analysis script with a mounted data directory
docker run --rm -v $(pwd)/data:/data monet-stats-app python /data/analysis.py
```

## Performance Optimization

### For Large Datasets

For datasets that do not fit in memory, prefer chunked xarray/Dask inputs so
metrics stay lazy and are computed only when `.compute()` is called:

```python
import xarray as xr

obs = xr.open_dataset("obs.nc", chunks={"time": 1000}).temperature
mod = xr.open_dataset("mod.nc", chunks={"time": 1000}).temperature

rmse_lazy = monet_stats.RMSE(obs, mod, axis="time")  # lazy
rmse = rmse_lazy.compute()  # executes only now
```

For plain NumPy arrays, the `monet_stats.performance` helpers provide
chunking utilities:

```python
from monet_stats.performance import get_chunk_recommendation, parallel_compute

# Chunk-size recommendation for a target memory budget
chunks = get_chunk_recommendation(obs, target_mb=100.0)

# Apply one metric function across chunks of a single array
result = parallel_compute(monet_stats.RMSE, obs, chunk_size=1_000_000, axis=0)
```

See the [Performance reference](api/performance.md) for the full API.

## Next Steps

After completing the installation:

1. **Basic Usage**: Start with the [Getting Started Guide](getting-started.md)
2. **API Reference**: Explore the complete [API Documentation](api/overview.md)
3. **Workflows**: Learn specific [Climate Data Workflows](workflows/climate-data-analysis.md)
4. **Examples**: Check out practical [Examples](notebooks/01_basic_statistical_analysis.ipynb)

## Support

If you encounter installation issues:

- 📖 [Documentation](https://noaa-oar-arl.github.io/monet-stats)
- 🐛 [GitHub Issues](https://github.com/noaa-oar-arl/monet-stats/issues)
- 📧 Email: arl.webmaster@noaa.gov
