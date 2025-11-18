# Getting Started with Monet Stats

This guide will help you get up and running with Monet Stats quickly. You'll learn how to import the library, load your data, and perform common statistical analyses used in atmospheric sciences.

## Prerequisites

Before starting, ensure you have Monet Stats installed:

```bash
pip install monet-stats
```

## First Steps

### Basic Import and Setup

```python
import numpy as np
import monet_stats as ms

print(f"Monet Stats version: {ms.__version__}")
```

### Simple Example: Temperature Forecast Evaluation

Let's start with a basic example evaluating temperature forecasts:

```python
import numpy as np
import monet_stats as ms

# Sample observed temperatures (°C)
observed_temps = np.array([
    15.2, 16.8, 14.5, 13.9, 17.2, 18.5, 19.1, 20.3, 18.7, 16.9,
    15.8, 14.2, 13.6, 12.8, 15.4, 17.9, 19.5, 21.2, 20.1, 18.4
])

# Corresponding model predictions
model_temps = np.array([
    15.8, 16.2, 15.1, 13.4, 17.8, 18.9, 19.8, 20.1, 18.3, 17.2,
    15.1, 14.8, 12.9, 13.3, 15.9, 18.2, 20.1, 21.5, 19.8, 18.9
])

# Basic error metrics
rmse = ms.RMSE(observed_temps, model_temps)
mae = ms.MAE(observed_temps, model_temps)
mb = ms.MB(observed_temps, model_temps)

print(f"RMSE: {rmse:.3f}°C")
print(f"MAE: {mae:.3f}°C") 
print(f"Mean Bias: {mb:.3f}°C")
```

## Common Statistical Analyses

### 1. Error Metrics Analysis

```python
# Comprehensive error analysis
error_metrics = {
    'RMSE': ms.RMSE(observed_temps, model_temps),
    'MAE': ms.MAE(observed_temps, model_temps),
    'Mean Bias': ms.MB(observed_temps, model_temps),
    'Normalized Mean Bias': ms.NMB(observed_temps, model_temps),
    'Normalized Mean Error': ms.NME(observed_temps, model_temps)
}

print("Error Metrics:")
for metric, value in error_metrics.items():
    print(f"  {metric}: {value:.3f}")
```

### 2. Correlation Analysis

```python
# Correlation metrics
correlation_metrics = {
    'R²': ms.R2(observed_temps, model_temps),
    'Pearson Correlation': ms.pearsonr(observed_temps, model_temps)[0],
    'Index of Agreement': ms.IOA(observed_temps, model_temps),
    'Kling-Gupta Efficiency': ms.KGE(observed_temps, model_temps)
}

print("\nCorrelation Metrics:")
for metric, value in correlation_metrics.items():
    print(f"  {metric}: {value:.3f}")
```

### 3. Skill Score Analysis

```python
# Skill scores relative to climatology
skill_scores = {
    'Nash-Sutcliffe Efficiency': ms.NSE(observed_temps, model_temps),
    'Modified NSE': ms.NSEm(observed_temps, model_temps),
    'Log NSE': ms.NSElog(observed_temps, model_temps),
    'Mean Absolute Percentage Error': ms.MAPE(observed_temps, model_temps)
}

print("\nSkill Scores:")
for metric, value in skill_scores.items():
    print(f"  {metric}: {value:.3f}")
```

## Working with Different Data Types

### NumPy Arrays

```python
# Using NumPy arrays (most common)
obs = np.random.normal(20, 5, 1000)  # Normal distribution
mod = obs + np.random.normal(0, 2, 1000)  # Add noise

# All metrics work with NumPy arrays
r2 = ms.R2(obs, mod)
rmse = ms.RMSE(obs, mod)
print(f"NumPy arrays - R²: {r2:.3f}, RMSE: {rmse:.3f}")
```

### Pandas DataFrames

```python
import pandas as pd

# Using pandas DataFrames
data = pd.DataFrame({
    'observed': observed_temps,
    'modeled': model_temps,
    'station_id': ['STN1'] * 10 + ['STN2'] * 10,
    'time': pd.date_range('2023-01-01', periods=20, freq='H')
})

# Calculate metrics by station
results = data.groupby('station_id').apply(
    lambda x: pd.Series({
        'RMSE': ms.RMSE(x['observed'], x['modeled']),
        'R2': ms.R2(x['observed'], x['modeled']),
        'MB': ms.MB(x['observed'], x['modeled'])
    })
)

print(results)
```

### XArray DataArrays

```python
import xarray as xr

# Using xarray DataArrays
obs_da = xr.DataArray(
    observed_temps,
    dims=['time'],
    coords={'time': pd.date_range('2023-01-01', periods=20, freq='H')},
    attrs={'units': '°C', 'long_name': 'Observed Temperature'}
)

mod_da = xr.DataArray(
    model_temps,
    dims=['time'],
    coords={'time': pd.date_range('2023-01-01', periods=20, freq='H')},
    attrs={'units': '°C', 'long_name': 'Modeled Temperature'}
)

# Metrics return DataArray with coordinates
r2_da = ms.R2(obs_da, mod_da)
print(f"XArray result with coordinates:\n{r2_da}")
```

## Categorical Event Analysis

### Precipitation Verification

```python
# Binary precipitation data (1 = rain, 0 = no rain)
obs_precip = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1])
mod_precip = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1])

# Contingency table metrics
pod = ms.POD(obs_precip, mod_precip, threshold=0.1)  # Probability of Detection
far = ms.FAR(obs_precip, mod_precip, threshold=0.1)  # False Alarm Rate
csi = ms.CSI(obs_precip, mod_precip, threshold=0.1)  # Critical Success Index
hss = ms.HSS(obs_precip, mod_precip, threshold=0.1)  # Heidke Skill Score

print("\nPrecipitation Verification:")
print(f"  POD: {pod:.3f}")
print(f"  FAR: {far:.3f}")
print(f"  CSI: {csi:.3f}")
print(f"  HSS: {hss:.3f}")
```

### Probabilistic Forecast Evaluation

```python
# Ensemble forecast example
n_ensemble = 50
n_times = 100

# Generate ensemble forecasts
ensemble_forecasts = np.random.normal(20, 3, (n_ensemble, n_times))
observed_values = np.random.normal(20, 2, n_times)

# Calculate ensemble statistics
ensemble_mean = ms.ensemble_mean(ensemble_forecasts)
ensemble_std = ms.ensemble_std(ensemble_forecasts)

# Probabilistic metrics
crps = ms.CRPS(ensemble_forecasts, observed_values)
bss = ms.BSS(observed_values > 20, ensemble_mean > 20, threshold=0.5)

print(f"\nEnsemble Statistics:")
print(f"  Ensemble Mean RMSE: {ms.RMSE(ensemble_mean, observed_values):.3f}")
print(f"  Continuous Rank Probability Score: {crps:.3f}")
print(f"  Brier Skill Score: {bss:.3f}")
```

## Spatial Analysis

### 2D Field Verification

```python
# Create sample 2D fields (e.g., precipitation or temperature)
obs_field = np.random.exponential(5, (20, 20))
mod_field = obs_field + np.random.normal(0, 1, (20, 20))

# Ensure non-negative values
obs_field = np.maximum(obs_field, 0)
mod_field = np.maximum(mod_field, 0)

# Spatial verification metrics
fss = ms.FSS(obs_field, mod_field, window=5, threshold=10)
sal_s, sal_a, sal_l = ms.SAL(obs_field, mod_field, threshold=5)

print(f"\nSpatial Verification:")
print(f"  Fractions Skill Score: {fss:.3f}")
print(f"  SAL - Structure: {sal_s:.3f}, Amplitude: {sal_a:.3f}, Location: {sal_l:.3f}")
```

## Wind Direction Analysis

### Specialized Wind Metrics

```python
# Wind direction data (degrees)
obs_wind_dir = np.array([45, 90, 180, 270, 355])
mod_wind_dir = np.array([50, 95, 175, 265, 5])

# Wind direction error metrics
wind_rmse = ms.RMSE(obs_wind_dir, mod_wind_dir)
wind_bias = ms.MB(obs_wind_dir, mod_wind_dir)

# Circular bias for wind direction
circular_bias = ms.circlebias(wind_bias)

print(f"\nWind Direction Analysis:")
print(f"  RMSE: {wind_rmse:.1f}°")
print(f"  Bias: {wind_bias:.1f}°")
print(f"  Circular Bias: {circular_bias:.1f}°")
```

## Batch Processing

### Processing Multiple Metrics

```python
def comprehensive_analysis(obs, mod):
    """Perform comprehensive statistical analysis"""
    
    metrics = {
        # Error metrics
        'RMSE': ms.RMSE(obs, mod),
        'MAE': ms.MAE(obs, mod),
        'Mean Bias': ms.MB(obs, mod),
        'Normalized Mean Bias': ms.NMB(obs, mod),
        
        # Skill scores
        'R²': ms.R2(obs, mod),
        'NSE': ms.NSE(obs, mod),
        'Index of Agreement': ms.IOA(obs, mod),
        'Kling-Gupta Efficiency': ms.KGE(obs, mod),
        
        # Relative metrics
        'Mean Percentage Error': ms.MPE(obs, mod),
        'Normalized Mean Error': ms.NME(obs, mod)
    }
    
    return metrics

# Run comprehensive analysis
results = comprehensive_analysis(observed_temps, model_temps)

print("\nComprehensive Analysis Results:")
for metric, value in results.items():
    print(f"  {metric}: {value:.3f}")
```

## Working with Missing Data

### Handling NaN Values

```python
# Data with missing values
obs_with_nan = np.array([15.2, np.nan, 14.5, 13.9, np.nan, 18.5, 19.1])
mod_with_nan = np.array([15.8, 16.2, np.nan, 13.4, 17.8, np.nan, 19.8])

# Monet Stats automatically handles NaN values
rmse_clean = ms.RMSE(obs_with_nan, mod_with_nan)
print(f"\nRMSE with NaN handling: {rmse_clean:.3f}")
```

## Visualization Integration

```python
import matplotlib.pyplot as plt

# Scatter plot with perfect correlation line
plt.figure(figsize=(8, 6))
plt.scatter(observed_temps, model_temps, alpha=0.6, s=50)
plt.plot([observed_temps.min(), observed_temps.max()],
         [observed_temps.min(), observed_temps.max()], 
         'r--', lw=2, label='Perfect correlation')

plt.xlabel('Observed Temperature (°C)')
plt.ylabel('Modeled Temperature (°C)')
plt.title(f'Temperature Forecast Verification (R² = {ms.R2(observed_temps, model_temps):.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Next Steps

Now that you've learned the basics:

1. **Explore Advanced Features**: Check out the [Climate Data Analysis Workflows](workflows/climate-data-analysis.md)
2. **Dive into API**: Browse the complete [API Reference](api/overview.md)
3. **Learn Spatial Methods**: Discover [Spatial Verification](workflows/spatial-verification.md)
4. **Ensemble Analysis**: Understand [Ensemble Verification](workflows/ensemble-verification.md)
5. **View Examples**: See practical [Examples](examples/basic-usage.md)

## Tips for Best Practices

### Data Preparation
- Always check for data quality and missing values
- Ensure temporal/spatial alignment between observations and model data
- Consider data preprocessing (normalization, transformation)

### Metric Selection
- Use multiple complementary metrics for comprehensive evaluation
- Consider the specific application when choosing metrics
- Skill scores are often more interpretable than raw error metrics

### Performance Optimization
- For large datasets, consider chunked processing
- Use NumPy arrays for best performance
- Leverage xarray for multidimensional data

### Documentation
- Always document your analysis methods
- Include units and metadata with your data
- Consider reproducibility when setting up analysis pipelines

## Getting Help

If you need assistance:

- 📖 [Full Documentation](https://noaa-oar-arl.github.io/monet-stats)
- 🐛 [Report Issues](https://github.com/noaa-oar-arl/monet-stats/issues)
- 💬 [Community Discussions](https://github.com/noaa-oar-arl/monet/discussions)
- 📧 Email: arl.webmaster@noaa.gov