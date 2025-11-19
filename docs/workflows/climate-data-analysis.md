# Climate Data Analysis Workflows

This guide provides comprehensive workflows for analyzing climate data using Monet Stats. These workflows cover common scenarios in climate model evaluation, atmospheric data analysis, and environmental monitoring.

## Overview

Climate data analysis requires specialized statistical approaches due to:

- **Multi-scale temporal patterns**: Daily, seasonal, and interannual variability
- **Spatial heterogeneity**: Regional and local climate variations
- **Complex dependencies**: Autocorrelation, teleconnections, and non-stationarity
- **Extreme value analysis**: Rare events and long-term trends
- **Multi-model ensembles**: Uncertainty quantification and model comparison

## Workflow 1: Temperature Trend Analysis

### Objective

Analyze long-term temperature trends and model performance across different time scales.

```python
import numpy as np
import pandas as pd
import monet_stats as ms
import matplotlib.pyplot as plt

# Generate synthetic temperature data (1980-2020)
years = np.arange(1980, 2021)
n_years = len(years)

# Observed temperatures with trend and natural variability
trend = 0.02 * (years - 1980)  # 0.02°C per year warming
observed = 15.0 + trend + np.random.normal(0, 0.5, n_years)

# Model 1: Good performance
model1 = observed + np.random.normal(0, 0.3, n_years)

# Model 2: Biased with poor trend capture
model2 = observed + 1.0 + np.random.normal(0, 0.8, n_years)

# Create DataFrame
temp_data = pd.DataFrame({
    'year': years,
    'observed': observed,
    'model1': model1,
    'model2': model2
})
```

### Step 1: Annual Analysis

```python
# Calculate annual metrics
annual_metrics = temp_data.apply(
    lambda x: pd.Series({
        'RMSE': ms.RMSE(x['observed'], x['model1']),
        'R2': ms.R2(x['observed'], x['model1']),
        'MB': ms.MB(x['observed'], x['model1']),
        'NSE': ms.NSE(x['observed'], x['model1'])
    }),
    axis=1
)

print("Annual Performance Metrics (Model 1):")
print(annual_metrics.describe())
```

### Step 2: Decadal Analysis

```python
# Group by decade
temp_data['decade'] = (temp_data['year'] // 10) * 10
decadal_metrics = temp_data.groupby('decade').apply(
    lambda x: pd.Series({
        'RMSE': ms.RMSE(x['observed'], x['model1']),
        'R2': ms.R2(x['observed'], x['model1']),
        'Trend_RMSE': ms.RMSE(
            np.polyfit(x['year'] - x['year'].iloc[0], x['observed'], 1)[0],
            np.polyfit(x['year'] - x['year'].iloc[0], x['model1'], 1)[0]
        )
    })
).round(4)

print("\nDecadal Performance Trends:")
print(decadal_metrics)
```

### Step 3: Seasonal Analysis

```python
# Add seasonal component
np.random.seed(42)
temp_data['month'] = np.random.choice(range(1, 13), size=len(temp_data))
temp_data['season'] = temp_data['month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

# Seasonal performance analysis
seasonal_performance = temp_data.groupby('season').apply(
    lambda x: pd.Series({
        'RMSE': ms.RMSE(x['observed'], x['model1']),
        'R2': ms.R2(x['observed'], x['model1']),
        'MB': ms.MB(x['observed'], x['model1']),
        'MAE': ms.MAE(x['observed'], x['model1']),
        'NMB': ms.NMB(x['observed'], x['model1'])
    })
)

print("\nSeasonal Performance Analysis:")
print(seasonal_performance)
```

## Workflow 2: Precipitation Extremes Analysis

### Objective

Evaluate model performance for extreme precipitation events and return periods.

```python
# Generate precipitation data with heavy-tailed distribution
np.random.seed(123)
n_years = 50  # 50 years of daily data

# Daily precipitation (mm/day)
precip_daily = np.random.gamma(0.5, 5, n_years * 365)
precip_daily = np.maximum(precip_daily, 0)  # Ensure non-negative

# Add extreme events (rare heavy precipitation)
extreme_events = np.random.choice(len(precip_daily),
                                 size=int(0.001 * len(precip_daily)),
                                 replace=False)
precip_daily[extreme_events] += np.random.exponential(20, len(extreme_events))

# Reshape to yearly
yearly_max = precip_daily.reshape(n_years, 365).max(axis=1)

# Model simulations with different performance
model1_good = yearly_max + np.random.normal(0, 2, n_years)
model2_conservative = yearly_max * 0.8 + np.random.normal(0, 1, n_years)
model3_overpredict = yearly_max * 1.5 + np.random.normal(0, 3, n_years)

# Create DataFrame
precip_data = pd.DataFrame({
    'year': np.arange(1970, 1970 + n_years),
    'observed': yearly_max,
    'model1': model1_good,
    'model2': model2_conservative,
    'model3': model3_overpredict
})
```

### Step 1: Extreme Value Analysis

```python
# Define extreme thresholds
thresholds = [95, 99]  # 95th and 99th percentiles

for threshold in thresholds:
    print(f"\nAnalysis for {threshold}th percentile threshold:")

    # Binary extreme events
    obs_extreme = (precip_data['observed'] >= np.percentile(precip_data['observed'], threshold)).astype(int)

    for model_name in ['model1', 'model2', 'model3']:
        mod_extreme = (precip_data[model_name] >= np.percentile(precip_data['observed'], threshold)).astype(int)

        # Contingency table metrics
        pod = ms.POD(obs_extreme, mod_extreme, threshold=0.5)
        far = ms.FAR(obs_extreme, mod_extreme, threshold=0.5)
        csi = ms.CSI(obs_extreme, mod_extreme, threshold=0.5)
        hss = ms.HSS(obs_extreme, mod_extreme, threshold=0.5)

        print(f"  {model_name}: POD={pod:.3f}, FAR={far:.3f}, CSI={csi:.3f}, HSS={hss:.3f}")
```

### Step 2: Return Period Analysis

```python
def calculate_return_period(values, return_years):
    """Calculate return period values from observed data"""
    sorted_values = np.sort(values)[::-1]
    n = len(values)
    exceedance_probs = np.arange(1, n + 1) / (n + 1)
    return_periods = 1 / exceedance_probs

    # Interpolate to desired return periods
    from scipy.interpolate import interp1d
    interp_func = interp1d(return_periods, sorted_values, bounds_error=False, fill_value='extrapolate')

    return interp_func(return_years)

# Calculate return periods for different models
return_years = [2, 5, 10, 25, 50, 100]
return_period_comparison = pd.DataFrame({
    'Return_Year': return_years
})

for col in ['observed', 'model1', 'model2', 'model3']:
    return_period_comparison[col] = calculate_return_period(precip_data[col], return_years)

print("\nReturn Period Analysis (mm):")
print(return_period_comparison.set_index('Return_Year'))
```

### Step 3: Bias Analysis for Different Intensity Ranges

```python
# Define precipitation intensity categories
def categorize_precipitation(precip):
    """Categorize precipitation by intensity"""
    categories = []
    for p in precip:
        if p == 0:
            categories.append('No rain')
        elif p < 2:
            categories.append('Light')
        elif p < 10:
            categories.append('Moderate')
        elif p < 25:
            categories.append('Heavy')
        else:
            categories.append('Extreme')
    return categories

# Apply categorization
for col in ['observed', 'model1', 'model2', 'model3']:
    precip_data[f'{col}_category'] = categorize_precipitation(precip_data[col])

# Bias analysis by intensity
intensity_bias = []
for category in ['No rain', 'Light', 'Moderate', 'Heavy', 'Extreme']:
    subset = precip_data[precip_data['observed_category'] == category]

    if len(subset) > 0:
        bias = ms.MB(subset['observed'], subset['model1'])
        nmb = ms.NMB(subset['observed'], subset['model1'])

        intensity_bias.append({
            'Intensity': category,
            'Count': len(subset),
            'Mean_Bias': bias,
            'Normalized_Bias': nmb
        })

intensity_bias_df = pd.DataFrame(intensity_bias)
print("\nBias Analysis by Intensity Category:")
print(intensity_bias_df)
```

## Workflow 3: Multi-Model Ensemble Analysis

### Objective

Evaluate and combine multiple climate models to improve prediction accuracy.

```python
# Generate synthetic multi-model data
n_models = 10
n_years = 30
years = np.arange(1990, 1990 + n_years)

# Generate ensemble models with different performance characteristics
ensemble_data = pd.DataFrame({'year': years, 'observed': 15 + 0.01 * (years - 1990)})

for i in range(n_models):
    # Each model has different bias and error characteristics
    bias = np.random.normal(0, 0.5)  # Model bias
    error_std = np.random.uniform(0.2, 1.0)  # Model error

    ensemble_data[f'model_{i+1}'] = (
        ensemble_data['observed'] +
        bias +
        np.random.normal(0, error_std, n_years)
    )
```

### Step 1: Model Evaluation

```python
# Evaluate individual models
model_performance = []
for i in range(1, n_models + 1):
    model_col = f'model_{i}'

    performance = {
        'Model': model_col,
        'RMSE': ms.RMSE(ensemble_data['observed'], ensemble_data[model_col]),
        'R2': ms.R2(ensemble_data['observed'], ensemble_data[model_col]),
        'MB': ms.MB(ensemble_data['observed'], ensemble_data[model_col]),
        'NSE': ms.NSE(ensemble_data['observed'], ensemble_data[model_col]),
        'KGE': ms.KGE(ensemble_data['observed'], ensemble_data[model_col])
    }
    model_performance.append(performance)

performance_df = pd.DataFrame(model_performance).sort_values('RMSE')
print("Individual Model Performance:")
print(performance_df)
```

### Step 2: Ensemble Construction

```python
# Simple mean ensemble
ensemble_mean = ensemble_data.filter(like='model_').mean(axis=1)

# Weighted ensemble based on performance
weights = 1 / performance_df['RMSE'].values
weights = weights / weights.sum()

weighted_ensemble = np.zeros(len(ensemble_data))
for i, (model_col, weight) in enumerate(zip(performance_df['Model'], weights)):
    weighted_ensemble += ensemble_data[model_col] * weight

# Add ensembles to dataframe
ensemble_data['ensemble_mean'] = ensemble_mean
ensemble_data['ensemble_weighted'] = weighted_ensemble

# Evaluate ensemble performance
ensemble_performance = pd.DataFrame({
    'System': ['Best_Model', 'Mean_Ensemble', 'Weighted_Ensemble'],
    'RMSE': [
        performance_df.iloc[0]['RMSE'],
        ms.RMSE(ensemble_data['observed'], ensemble_data['ensemble_mean']),
        ms.RMSE(ensemble_data['observed'], ensemble_data['ensemble_weighted'])
    ],
    'R2': [
        performance_df.iloc[0]['R2'],
        ms.R2(ensemble_data['observed'], ensemble_data['ensemble_mean']),
        ms.R2(ensemble_data['observed'], ensemble_data['ensemble_weighted'])
    ]
})

print("\nEnsemble Performance vs Best Individual Model:")
print(ensemble_performance)
```

### Step 3: Spread-Skill Relationship

```python
def calculate_spread_skill(model_data, observed):
    """Calculate spread and skill for ensemble"""
    # Ensemble spread (standard deviation)
    spread = model_data.std(axis=1)

    # Ensemble skill (RMSE of ensemble mean)
    ensemble_mean = model_data.mean(axis=1)
    skill = ms.RMSE(observed, ensemble_mean)

    return spread, skill

# Calculate spread-skill relationship
spread, skill = calculate_spread_skill(
    ensemble_data.filter(like='model_'),
    ensemble_data['observed']
)

# Analyze correlation
spread_skill_corr = ms.pearsonr(spread, skill)[0]
print(f"\nSpread-Skill Correlation: {spread_skill_corr:.3f}")

# Plot spread-skill relationship
plt.figure(figsize=(8, 6))
plt.scatter(spread, skill, alpha=0.6)
plt.xlabel('Ensemble Spread')
plt.ylabel('Ensemble Skill (RMSE)')
plt.title(f'Spread-Skill Relationship (r = {spread_skill_corr:.3f})')
plt.grid(True, alpha=0.3)
plt.show()
```

## Workflow 4: Spatial Climate Downscaling

### Objective

Evaluate high-resolution downscaled climate model data against observations.

```python
import numpy as np

# Generate spatial data (e.g., temperature field)
grid_size = 20
x, y = np.meshgrid(range(grid_size), range(grid_size))

# Create spatial pattern with some randomness
observed_spatial = (
    20 + 0.1 * x + 0.05 * y +
    2 * np.sin(2 * np.pi * x / grid_size) * np.cos(2 * np.pi * y / grid_size) +
    np.random.normal(0, 0.5, (grid_size, grid_size))
)

# Model 1: Good spatial representation
model1_spatial = observed_spatial + np.random.normal(0, 0.3, (grid_size, grid_size))

# Model 2: Poor spatial representation (over-smoothed)
from scipy.ndimage import gaussian_filter
model2_spatial = gaussian_filter(observed_spatial + np.random.normal(0, 0.5, (grid_size, grid_size)), sigma=2.0)
```

### Step 1: Point-wise Verification

```python
# Flatten arrays for point-wise analysis
observed_flat = observed_spatial.flatten()
model1_flat = model1_spatial.flatten()
model2_flat = model2_spatial.flatten()

# Point-wise metrics
point_metrics = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R2', 'MB', 'NMB'],
    'Model1': [
        ms.RMSE(observed_flat, model1_flat),
        ms.MAE(observed_flat, model1_flat),
        ms.R2(observed_flat, model1_flat),
        ms.MB(observed_flat, model1_flat),
        ms.NMB(observed_flat, model1_flat)
    ],
    'Model2': [
        ms.RMSE(observed_flat, model2_flat),
        ms.MAE(observed_flat, model2_flat),
        ms.R2(observed_flat, model2_flat),
        ms.MB(observed_flat, model2_flat),
        ms.NMB(observed_flat, model2_flat)
    ]
})

print("Point-wise Spatial Verification:")
print(point_metrics.set_index('Metric'))
```

### Step 2: Spatial Structure Analysis

```python
# Calculate spatial gradients
def calculate_spatial_gradients(field):
    """Calculate gradients in x and y directions"""
    grad_x = np.gradient(field, axis=1)
    grad_y = np.gradient(field, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return grad_magnitude

# Spatial structure metrics
obs_grad = calculate_spatial_gradients(observed_spatial)
model1_grad = calculate_spatial_gradients(model1_spatial)
model2_grad = calculate_spatial_gradients(model2_spatial)

# Compare gradient statistics
gradient_stats = pd.DataFrame({
    'Statistic': ['Mean', 'Std', 'Max'],
    'Observed': [obs_grad.mean(), obs_grad.std(), obs_grad.max()],
    'Model1': [model1_grad.mean(), model1_grad.std(), model1_grad.max()],
    'Model2': [model2_grad.mean(), model2_grad.std(), model2_grad.max()]
})

print("\nSpatial Gradient Statistics:")
print(gradient_stats.set_index('Statistic'))

# Spatial verification metrics
fss1 = ms.FSS(observed_spatial, model1_spatial, window=5, threshold=np.percentile(observed_spatial, 75))
fss2 = ms.FSS(observed_spatial, model2_spatial, window=5, threshold=np.percentile(observed_spatial, 75))

sal_s1, sal_a1, sal_l1 = ms.SAL(observed_spatial, model1_spatial)
sal_s2, sal_a2, sal_l2 = ms.SAL(observed_spatial, model2_spatial)

print(f"\nSpatial Verification Metrics:")
print(f"  Model 1 - FSS: {fss1:.3f}, SAL(S={sal_s1:.3f}, A={sal_a1:.3f}, L={sal_l1:.3f})")
print(f"  Model 2 - FSS: {fss2:.3f}, SAL(S={sal_s2:.3f}, A={sal_a2:.3f}, L={sal_l2:.3f})")
```

### Step 3: Spatial Pattern Correlation

```python
# Calculate spatial pattern correlations
from scipy.stats import pearsonr

def spatial_pattern_correlation(obs, mod, window_size=5):
    """Calculate local spatial correlations"""
    correlations = np.zeros_like(obs, dtype=float)

    for i in range(window_size//2, obs.shape[0] - window_size//2):
        for j in range(window_size//2, obs.shape[1] - window_size//2):
            # Extract local window
            obs_window = obs[i-window_size//2:i+window_size//2+1,
                           j-window_size//2:j+window_size//2+1]
            mod_window = mod[i-window_size//2:i+window_size//2+1,
                           j-window_size//2:j+window_size//2+1]

            # Calculate correlation
            if not (np.isnan(obs_window).any() or np.isnan(mod_window).any()):
                corr, _ = pearsonr(obs_window.flatten(), mod_window.flatten())
                correlations[i, j] = corr

    return correlations

# Calculate local correlations
local_corr1 = spatial_pattern_correlation(observed_spatial, model1_spatial)
local_corr2 = spatial_pattern_correlation(observed_spatial, model2_spatial)

print(f"\nLocal Spatial Pattern Correlations:")
print(f"  Model 1 - Mean: {np.nanmean(local_corr1):.3f}, Std: {np.nanstd(local_corr1):.3f}")
print(f"  Model 2 - Mean: {np.nanmean(local_corr2):.3f}, Std: {np.nanstd(local_corr2):.3f}")
```

## Best Practices for Climate Data Analysis

### Data Quality Control

```python
def quality_control_analysis(observed, modeled):
    """Perform comprehensive quality control"""

    # Check for data consistency
    qc_results = {}

    # Temporal consistency
    obs_diff = np.diff(observed)
    mod_diff = np.diff(modeled)
    qc_results['temporal_correlation'] = ms.pearsonr(obs_diff, mod_diff)[0]

    # Outlier detection (using IQR method)
    obs_q75, obs_q25 = np.percentile(observed, [75, 25])
    obs_iqr = obs_q75 - obs_q25
    obs_outliers = np.sum((observed < obs_q25 - 1.5 * obs_iqr) |
                          (observed > obs_q75 + 1.5 * obs_iqr))

    mod_q75, mod_q25 = np.percentile(modeled, [75, 25])
    mod_iqr = mod_q75 - mod_q25
    mod_outliers = np.sum((modeled < mod_q25 - 1.5 * mod_iqr) |
                          (modeled > mod_q75 + 1.5 * mod_iqr))

    qc_results['outlier_ratio'] = obs_outliers / mod_outliers if mod_outliers > 0 else np.nan

    return qc_results

# Example usage
qc = quality_control_analysis(observed_temps, model_temps)
print("\nQuality Control Analysis:")
for key, value in qc.items():
    print(f"  {key}: {value:.3f}")
```

### Trend Analysis

```python
def climate_trend_analysis(time_series, time):
    """Analyze climate trends with appropriate statistical methods"""

    # Linear trend
    trend_slope, trend_intercept = np.polyfit(time - time[0], time_series, 1)

    # Mann-Kendall trend test
    def mann_kendall_test(data):
        """Mann-Kendall trend test"""
        n = len(data)
        s = 0
        for i in range(n):
            for j in range(i + 1, n):
                if data[j] > data[i]:
                    s += 1
                elif data[j] < data[i]:
                    s -= 1

        # Calculate variance
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # Calculate z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # Calculate p-value (two-tailed)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z)))

        return z, p_value

    mk_z, mk_p = mann_kendall_test(time_series)

    # Theil-Sen slope estimator
    def theil_sen_slope(data, time):
        """Theil-Sen slope estimator"""
        slopes = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if time[j] != time[i]:
                    slope = (data[j] - data[i]) / (time[j] - time[i])
                    slopes.append(slope)

        if len(slopes) > 0:
            return np.median(slopes)
        else:
            return np.nan

    theil_slope = theil_sen_slope(time_series, time)

    return {
        'linear_trend': trend_slope,
        'theil_slope': theil_slope,
        'mann_kendall_z': mk_z,
        'mann_kendall_p': mk_p
    }

# Example usage
trend_analysis = climate_trend_analysis(observed_temps, years)
print("\nClimate Trend Analysis:")
for key, value in trend_analysis.items():
    print(f"  {key}: {value:.4f}")
```

## Summary

These climate data analysis workflows demonstrate how to use Monet Stats effectively for:

1. **Temperature Trend Analysis**: Multi-scale evaluation of long-term climate trends
2. **Precipitation Extremes**: Extreme value analysis and return period calculations
3. **Multi-Model Ensembles**: Ensemble construction and spread-skill relationships
4. **Spatial Downscaling**: High-resolution model evaluation and spatial verification

Each workflow can be adapted to specific climate data types and research questions. The key is to combine multiple complementary metrics and consider the specific characteristics of climate data.

For more specialized workflows, see:

- [Spatial Verification](workflows/spatial-verification.md)
- [Ensemble Verification](workflows/ensemble-verification.md)
- [Wind Direction Metrics](workflows/wind-direction-metrics.md)
- [Air Quality Model Evaluation](workflows/air-quality-evaluation.md)
