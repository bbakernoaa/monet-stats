# Installation and Setup Guide

This guide covers the installation process, system requirements, and setup instructions for Monet Stats.

## Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 512 MB minimum (2GB recommended for large datasets)
- **Storage**: 100 MB for installation

### Software Dependencies
Monet Stats requires the following Python packages:
- **NumPy**: ≥1.19.0
- **Pandas**: ≥1.3.0
- **SciPy**: ≥1.7.0
- **Statsmodels**: ≥0.12.0
- **XArray**: ≥0.18.0 (optional, for xarray support)

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

### Option 3: Conda Installation

```bash
# Install with conda-forge
conda install -c conda-forge monet-stats

# Or install from the local environment
conda env create -f environment.yml
conda activate monet-stats
```

## Verification

After installation, verify the installation by running the following Python code:

```python
import monet_stats

# Check version
print(f"Monet Stats version: {monet_stats.__version__}")

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
```bash
pip install xarray dask netcdf4
```

#### Enhanced Statistical Testing
```bash
pip install statsmodels pingouin
```

#### Parallel Processing
```bash
pip install dask joblib
```

#### Visualization
```bash
pip install matplotlib seaborn plotly
```

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
- **Code Quality**: black, flake8, isort, mypy
- **Testing**: pytest, pytest-cov, hypothesis
- **Documentation**: mkdocs, mkdocstrings
- **Build**: setuptools, wheel, build

## Configuration

### Environment Variables

Set `MONET_STATS_CACHE_DIR` to control the cache location:
```bash
export MONET_STATS_CACHE_DIR=/path/to/cache
```

### Configuration File

Create a `.monet-stats.toml` file in your home directory:

```toml
# .monet-stats.toml
[cache]
directory = "~/.cache/monet-stats"
max_size = "1GB"

[performance]
parallel_processing = true
chunk_size = 10000

[output]
decimal_places = 3
scientific_notation = false
```

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
# Install specific versions
pip install numpy==1.21.0 pandas==1.3.0

# Use conda for better dependency resolution
conda install numpy pandas scipy
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

## Docker Installation

### Using Docker
```bash
# Build the Docker image
docker build -t monet-stats .

# Run in container
docker run -it monet-stats

# Use with mounted volume
docker run -v $(pwd)/data:/data monet-stats python -c "
import monet_stats as ms
import numpy as np
# Your analysis code here
"
```

### Docker Compose
```yaml
version: '3.8'
services:
  monet-stats:
    build: .
    volumes:
      - ./data:/data
      - ./output:/output
    environment:
      - PYTHONPATH=/app
```

## Performance Optimization

### For Large Datasets
```python
# Use NumPy arrays for best performance
import numpy as np
obs = np.array(obs_data)  # Convert to NumPy array
mod = np.array(mod_data)

# Batch processing
from monet_stats import batch_metrics

results = batch_metrics(
    obs, mod,
    metrics=['R2', 'RMSE', 'MAE'],
    batch_size=50000
)
```

### Parallel Processing
```python
# Enable parallel processing (if available)
import monet_stats
monet_stats.set_config(parallel_processing=True)

# Process multiple metrics simultaneously
metrics = {
    'correlation': monet_stats.R2,
    'error': monet_stats.RMSE,
    'bias': monet_stats.MB
}

results = monet_stats.parallel_compute(obs, mod, metrics)
```

## Next Steps

After completing the installation:

1. **Basic Usage**: Start with the [Getting Started Guide](getting-started.md)
2. **API Reference**: Explore the complete [API Documentation](api/overview.md)
3. **Workflows**: Learn specific [Climate Data Workflows](workflows/climate-data-analysis.md)
4. **Examples**: Check out practical [Examples](examples/basic-usage.md)

## Support

If you encounter installation issues:

- 📖 [Documentation](https://noaa-oar-arl.github.io/monet-stats)
- 🐛 [GitHub Issues](https://github.com/noaa-oar-arl/monet-stats/issues)
- 📧 Email: arl.webmaster@noaa.gov