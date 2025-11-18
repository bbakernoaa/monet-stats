# FAQ and Troubleshooting Guide

This FAQ addresses common questions and issues encountered when using Monet Stats for atmospheric sciences applications. If you don't find your question here, please check the [GitHub Issues](https://github.com/noaa-oar-arl/monet-stats/issues) or submit a new issue.

## Installation and Setup

### Q: I'm getting ImportError when trying to import monet_stats
**A:** This typically indicates that Monet Stats is not properly installed. Try these steps:

```bash
# Check if package is installed
pip show monet-stats

# If not installed
pip install monet-stats

# Reinstall if needed
pip install --reinstall monet-stats

# For development installation
pip install -e .
```

### Q: I need additional dependencies like xarray or pandas
**A:** Install the optional dependencies:

```bash
# Install with xarray support
pip install monet-stats[xarray]

# Install with all optional dependencies
pip install monet-stats[dev,test]

# Install specific dependencies
pip install xarray pandas scipy matplotlib
```

### Q: How do I set up a development environment?
**A:** Follow these steps for development setup:

```bash
# Clone the repository
git clone https://github.com/noaa-oar-arl/monet-stats.git
cd monet-stats

# Create virtual environment
python -m venv monet-stats-env
source monet-stats-env/bin/activate  # Windows: monet-stats-env\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## Data Format Issues

### Q: My xarray DataArray imports fail
**A:** Ensure you have xarray installed and your DataArrays have compatible dimensions:

```python
import xarray as xr
import monet_stats as ms

# Ensure DataArrays have same coordinates
obs_da = xr.DataArray(obs_data, dims=['time'])
mod_da = xr.DataArray(mod_data, dims=['time'])

# This works
result = ms.R2(obs_da, mod_da)

# This will fail due to dimension mismatch
# mod_da = xr.DataArray(mod_data, dims=['space'])  # Error!
```

### Q: How do I handle NaN values in my data?
**A:** Monet Stats automatically handles NaN values by using pairwise deletion:

```python
import numpy as np
import monet_stats as ms

# Data with NaN values
obs_with_nan = np.array([1, 2, np.nan, 4, 5])
mod_with_nan = np.array([1.1, 2.1, 3.1, np.nan, 5.1])

# Functions automatically use valid pairs only
rmse = ms.RMSE(obs_with_nan, mod_with_nan)  # Uses (1,1.1), (2,2.1), (5,5.1)
```

### Q: My data shapes don't match
**A:** Ensure your observed and modeled arrays have compatible shapes:

```python
# Correct: Same shape
obs = np.array([1, 2, 3, 4, 5])
mod = np.array([1.1, 2.1, 2.9, 4.1, 4.8])

# Error: Different shapes
# obs = np.array([1, 2, 3])  # This will raise ValueError
# mod = np.array([1.1, 2.1, 2.9, 4.1, 4.8])

# Solution: Align your data first
obs, mod = obs[:len(mod)], mod[:len(obs)]  # Truncate to shorter length
```

## Metric Calculation Issues

### Q: Why do I get NaN values for my metrics?
**A:** NaN results typically occur when:

1. **Insufficient valid data pairs**:
```python
# Too few valid pairs
obs = np.array([1, np.nan, np.nan])
mod = np.array([np.nan, 2, np.nan])

# Result: NaN (only one valid pair, which may be insufficient)
rmse = ms.RMSE(obs, mod)
```

2. **Division by zero**:
```python
# Zero variance in observed data
obs = np.array([1, 1, 1, 1])
mod = np.array([1.1, 1.1, 1.1, 1.1])

# Result: NaN (division by zero in R² calculation)
r2 = ms.R2(obs, mod)
```

### Q: My contingency metrics return weird values
**A:** Check your threshold and event definition:

```python
# For precipitation data
obs_precip = np.array([0, 1, 1, 0, 0])  # 0 = no rain, 1 = rain
mod_precip = np.array([0, 1, 0, 0, 1])

# Correct threshold for binary events
pod = ms.POD(obs_precip, mod_precip, threshold=0.5)

# Wrong threshold (should be 0.5 for binary data)
# pod = ms.POD(obs_precip, mod_precip, threshold=10.0)  # Wrong!
```

### Q: Wind direction metrics give strange results
**A:** Wind direction requires special circular statistics:

```python
import monet_stats as ms
import numpy as np

# Wind directions in degrees (0-360)
obs_wind = np.array([10, 20, 350])  # Note the circular nature
mod_wind = np.array([15, 25, 5])    # 350° is close to 5°

# Use circular bias for wind direction
circular_bias = ms.circlebias(mod_wind - obs_wind)

# Standard RMSE for wind direction (may give misleading results)
wind_rmse = ms.RMSE(obs_wind, mod_wind)
```

## Performance Issues

### Q: My calculations are too slow for large datasets
**A:** Optimize performance with these techniques:

```python
import monet_stats as ms
import numpy as np

# For very large arrays (>1M elements), process in chunks
def process_large_data(obs, mod, chunk_size=100000):
    results = []
    for i in range(0, len(obs), chunk_size):
        obs_chunk = obs[i:i+chunk_size]
        mod_chunk = mod[i:i+chunk_size]
        
        result = ms.RMSE(obs_chunk, mod_chunk)
        results.append(result)
    
    return np.mean(results)

# Use NumPy arrays for best performance
obs_np = np.array(obs_data)  # Convert to NumPy array
mod_np = np.array(mod_data)

# Avoid Python lists and loops
# Slow: 
# results = [ms.RMSE(obs[i], mod[i]) for i in range(len(obs))]

# Fast:
results = ms.RMSE(obs_np, mod_np)
```

### Q: How can I reduce memory usage?
**A:** Use memory-efficient data types and processing:

```python
import numpy as np

# Use float32 instead of float64 for large datasets
obs_float32 = obs.astype(np.float32)
mod_float32 = mod.astype(np.float32)

# Process data in chunks
def memory_efficient_analysis(obs, mod, chunk_size=50000):
    total_results = []
    
    for i in range(0, len(obs), chunk_size):
        obs_chunk = obs[i:i+chunk_size]
        mod_chunk = mod[i:i+chunk_size]
        
        # Calculate metrics for chunk
        chunk_results = {
            'RMSE': ms.RMSE(obs_chunk, mod_chunk),
            'R2': ms.R2(obs_chunk, mod_chunk),
            'MAE': ms.MAE(obs_chunk, mod_chunk)
        }
        total_results.append(chunk_results)
    
    return total_results
```

## Statistical Interpretation

### Q: What's the difference between RMSE and MAE?
**A:** RMSE and MAE measure different aspects of error:

```python
import numpy as np
import monet_stats as ms

# Example with different error distributions
obs = np.array([10, 10, 10, 10, 10])

# Case 1: Small errors uniformly
mod1 = np.array([10.1, 10.2, 9.9, 10.0, 10.1])

# Case 2: Large error on one point, small on others
mod2 = np.array([10.0, 10.0, 10.0, 10.0, 15.0])

print("Case 1 - Uniform errors:")
print(f"  RMSE: {ms.RMSE(obs, mod1):.3f}")
print(f"  MAE:  {ms.MAE(obs, mod1):.3f}")

print("\nCase 2 - One large error:")
print(f"  RMSE: {ms.RMSE(obs, mod2):.3f}")
print(f"  MAE:  {ms.MAE(obs, mod2):.3f}")
```

**Key Differences:**
- RMSE squares errors, so large errors are heavily weighted
- MAE treats all errors equally
- RMSE is more sensitive to outliers

### Q: When should I use skill scores vs. raw error metrics?
**A:** Use both for comprehensive evaluation:

```python
import monet_stats as ms
import numpy as np

# Model vs. climatology reference
obs = np.array([15, 16, 14, 17, 18, 16, 15])
mod = np.array([15.5, 15.8, 14.2, 16.9, 17.2, 15.5, 14.8])
climatology = np.mean(obs)  # Simple climatology

# Raw error metrics
rmse_model = ms.RMSE(obs, mod)
rmse_climo = ms.RMSE(obs, [climatology] * len(obs))

# Skill score
skill_score = ms.NSE(obs, mod)

print(f"Model RMSE: {rmse_model:.3f}")
print(f"Climatology RMSE: {rmse_climo:.3f}")
print(f"Model Skill Score: {skill_score:.3f}")

# Interpretation:
# - Raw errors show absolute performance
# - Skill scores show performance relative to reference
```

### Q: How do I interpret negative skill scores?
**A:** Negative skill scores indicate the model performs worse than the reference:

```python
import monet_stats as ms
import numpy as np

# Model that performs worse than climatology
obs = np.array([1, 2, 3, 4, 5])
bad_model = np.array([10, 10, 10, 10, 10])  # Constant prediction

# Compare to climatology reference
climatology = np.mean(obs)

bad_skill = ms.NSE(obs, bad_model)
clim_skill = ms.NSE(obs, [climatology] * len(obs))

print(f"Bad model skill: {bad_skill:.3f}")
print(f"Climatology skill: {clim_skill:.3f}")

# Negative skill means model is worse than climatology
# This often happens with poor forecasts or non-stationary data
```

## Spatial and Ensemble Issues

### Q: My spatial verification metrics fail
**A:** Ensure your spatial data has the correct dimensions:

```python
import numpy as np
import monet_stats as ms

# 2D spatial data (lat, lon)
obs_2d = np.random.normal(20, 2, (10, 10))  # 10x10 grid
mod_2d = obs_2d + np.random.normal(0, 1, (10, 10))

# This works with spatial metrics
fss = ms.FSS(obs_2d, mod_2d, window=5)

# This fails (1D data)
# fss = ms.FSS(obs_1d, mod_1d, window=5)  # Error!
```

### Q: How do I verify ensemble forecasts?
**A:** Use ensemble-specific metrics and proper formatting:

```python
import numpy as np
import monet_stats as ms

# Ensemble data: (n_members, n_times)
n_members = 50
n_times = 100

# Generate ensemble forecasts
ensemble = np.random.normal(20, 2, (n_members, n_times))
observed = np.random.normal(20, 1.5, n_times)

# Calculate ensemble statistics
ensemble_mean = np.mean(ensemble, axis=0)
ensemble_std = np.std(ensemble, axis=0)

# Ensemble metrics
crps = ms.CRPS(ensemble, observed)  # Continuous Ranked Probability Score
bss = ms.BSS(observed > 20, ensemble_mean > 20, threshold=0.5)

print(f"Ensemble CRPS: {crps:.3f}")
print(f"Ensemble BSS: {bss:.3f}")

# Spread-skill relationship
spread_skill_corr = ms.pearsonr(ensemble_std, 1/ensemble_mean)[0]
print(f"Spread-Skill Correlation: {spread_skill_corr:.3f}")
```

## Common Error Messages

### Error: "ValueError: cannot convert float NaN to integer"

**Cause:** Division by zero or insufficient data for integer conversion.

**Solution:**
```python
import numpy as np
import monet_stats as ms

# Check for zero variance
obs = np.array([1, 1, 1, 1])
mod = np.array([1.1, 1.1, 1.1, 1.1])

# Check variance before calculation
if np.std(obs) == 0:
    print("Warning: Zero variance in observed data")
    r2 = 0.0  # Handle appropriately
else:
    r2 = ms.R2(obs, mod)
```

### Error: "RuntimeWarning: invalid value encountered in divide"

**Cause:** Division by very small numbers or zero in calculations.

**Solution:**
```python
import numpy as np
from monet_stats import NMB

def safe_nmb(obs, mod):
    """Safe NMB calculation with error handling"""
    obs_sum = np.sum(obs)
    mod_sum = np.sum(mod)
    
    if abs(obs_sum) < 1e-10:  # Very small denominator
        if abs(mod_sum) < 1e-10:
            return 0.0  # Both sums are effectively zero
        else:
            return np.sign(mod_sum) * np.inf  # Infinite bias
    
    return (mod_sum - obs_sum) / obs_sum

# Usage
nmb = safe_nmb(obs, mod)
```

### Error: "MemoryError: Unable to allocate array"

**Cause:** Trying to process very large arrays in memory.

**Solution:**
```python
import monet_stats as ms
import numpy as np

def process_large_dataset(obs_file, mod_file, chunk_size=100000):
    """Process large datasets in chunks"""
    # Load data in chunks (example with text files)
    obs_chunks = np.genfromtxt(obs_file, max_rows=chunk_size)
    mod_chunks = np.genfromtxt(mod_file, max_rows=chunk_size)
    
    results = []
    while len(obs_chunks) > 0:
        # Calculate metrics for chunk
        chunk_result = {
            'RMSE': ms.RMSE(obs_chunks, mod_chunks),
            'R2': ms.R2(obs_chunks, mod_chunks),
            'MAE': ms.MAE(obs_chunks, mod_chunks)
        }
        results.append(chunk_result)
        
        # Load next chunk
        obs_chunks = np.genfromtxt(obs_file, max_rows=chunk_size, skip_header=len(obs_chunks))
        mod_chunks = np.genfromtxt(mod_file, max_rows=chunk_size, skip_header=len(mod_chunks))
    
    return results
```

## Best Practices

### 1. Data Preparation
```python
import numpy as np
import monet_stats as ms

def prepare_data(obs, mod):
    """Clean and prepare data for analysis"""
    # Remove NaN values using pairwise deletion
    valid_mask = ~np.isnan(obs) & ~np.isnan(mod)
    obs_clean = obs[valid_mask]
    mod_clean = mod[valid_mask]
    
    # Check for sufficient data
    if len(obs_clean) < 10:
        raise ValueError("Insufficient valid data pairs")
    
    return obs_clean, mod_clean

# Usage
obs_clean, mod_clean = prepare_data(observed, modeled)
results = ms.RMSE(obs_clean, mod_clean)
```

### 2. Error Handling
```python
def safe_metric_calculation(obs, mod, metric_func, **kwargs):
    """Safely calculate metrics with error handling"""
    try:
        result = metric_func(obs, mod, **kwargs)
        
        # Check for NaN or infinite results
        if not np.isfinite(result):
            print(f"Warning: Non-finite result for {metric_func.__name__}")
            return np.nan
        
        return result
    
    except Exception as e:
        print(f"Error in {metric_func.__name__}: {e}")
        return np.nan

# Usage
rmse = safe_metric_calculation(obs, mod, ms.RMSE)
r2 = safe_metric_calculation(obs, mod, ms.R2)
```

### 3. Comprehensive Analysis
```python
def comprehensive_verification(obs, mod):
    """Perform comprehensive model verification"""
    results = {}
    
    # Error metrics
    results['RMSE'] = ms.RMSE(obs, mod)
    results['MAE'] = ms.MAE(obs, mod)
    results['MB'] = ms.MB(obs, mod)
    results['NMB'] = ms.NMB(obs, mod)
    
    # Skill scores
    results['R2'] = ms.R2(obs, mod)
    results['NSE'] = ms.NSE(obs, mod)
    results['KGE'] = ms.KGE(obs, mod)
    
    # Relative metrics
    results['MPE'] = ms.MPE(obs, mod)
    results['NME'] = ms.NME(obs, mod)
    
    return results

# Usage
verification_results = comprehensive_verification(observed, modeled)
for metric, value in verification_results.items():
    print(f"{metric}: {value:.4f}")
```

## Getting Help

### Where to Find Help

1. **Documentation**: [Full Documentation](https://noaa-oar-arl.github.io/monet-stats)
2. **GitHub Issues**: [Report bugs or request features](https://github.com/noaa-oar-arl/monet-stats/issues)
3. **Community Discussions**: [GitHub Discussions](https://github.com/noaa-oar-arl/monet/discussions)
4. **Email Support**: arl.webmaster@noaa.gov

### How to Report Issues

When reporting issues, please include:

1. **Environment Information**:
   ```python
   import monet_stats
   print(f"Monet Stats version: {monet_stats.__version__}")
   import sys
   print(f"Python version: {sys.version}")
   ```

2. **Minimal Reproducible Example**:
   ```python
   import numpy as np
   import monet_stats as ms
   
   # Your problematic code here
   obs = np.array([1, 2, 3])
   mod = np.array([1.1, 2.1, 2.9])
   
   # This causes the error
   result = ms.YourMetric(obs, mod)  # Replace with actual call
   ```

3. **Expected vs. Actual Behavior**:
   - What you expected to happen
   - What actually happened
   - Any error messages received

### Contributing

If you'd like to contribute to Monet Stats:

1. Fork the repository on GitHub
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Submit a pull request

See the [Contributing Guide](contributing.md) for detailed instructions.