# Monet Stats

A comprehensive statistics and utility library designed for atmospheric sciences applications, providing a wide range of metrics for model evaluation, verification, and analysis.

## Overview

Monet Stats is a Python library focused on statistical evaluation methods commonly used in atmospheric sciences, meteorology, and environmental modeling. It provides a comprehensive suite of metrics for:

- **Model Verification**: Evaluate the performance of numerical weather prediction and air quality models
- **Contingency Table Analysis**: Assess categorical forecast skill for events like precipitation and air quality exceedances
- **Error Metrics**: Quantify the magnitude and characteristics of model errors
- **Skill Scores**: Measure forecast skill relative to reference forecasts
- **Spatial Verification**: Evaluate the spatial structure and location of modeled fields
- **Ensemble Verification**: Assess probabilistic forecast performance from ensemble systems

## Key Features

- 📊 **Comprehensive Metric Coverage**: 50+ statistical metrics for atmospheric sciences
- 🔧 **Multiple Data Formats**: Support for NumPy arrays, xarray DataArrays, and pandas DataFrames
- 🌪️ **Specialized Metrics**: Wind direction handling, circular statistics, and spatial verification
- 📈 **Skill Score Framework**: Built-in support for Brier, Heidke, and other skill scores
- 🧮 **Mathematical Rigor**: Well-documented mathematical formulations and use cases
- ⚡ **Performance Optimized**: Vectorized operations and efficient algorithms

## Quick Start

```python
import numpy as np
from monet_stats import R2, RMSE, POD, FAR

# Sample data
obs = np.array([1.2, 2.5, 3.7, 4.1, 5.0])
mod = np.array([1.1, 2.6, 3.5, 4.3, 4.8])

# Calculate basic metrics
r_squared = R2(obs, mod)
rmse_value = RMSE(obs, mod)
print(f"R²: {r_squared:.3f}")
print(f"RMSE: {rmse_value:.3f}")
```

## Installation

```bash
pip install monet-stats
```

## Supported Metrics

### By Category

#### Contingency Table Metrics

- Heidke Skill Score (HSS)
- Equitable Threat Score (ETS)
- Critical Success Index (CSI)
- Probability of Detection (POD)
- False Alarm Rate (FAR)
- True Skill Statistic (TSS)

#### Error Metrics

- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Mean Bias (MB)
- Normalized Mean Error (NME)
- Wind Direction RMSE

#### Correlation Metrics

- Coefficient of Determination (R²)
- Pearson Correlation
- Taylor Skill Score
- Kling-Gupta Efficiency (KGE)

#### Skill Scores

- Brier Skill Score (BSS)
- Nash-Sutcliffe Efficiency (NSE)
- Index of Agreement (IOA)
- Mean Absolute Percentage Error (MAPE)

#### Spatial & Ensemble Metrics

- Fractions Skill Score (FSS)
- Continuous Ranked Probability Score (CRPS)
- Structure-Amplitude-Location (SAL)
- Ensemble mean and spread
- Rank histograms

## Documentation Structure

- [Installation Guide](installation.md) - Setup and configuration
- [Getting Started](getting-started.md) - Basic usage and examples
- [User Guides](workflows/climate-data-analysis.md) - Domain-specific workflows
- [API Reference](api/overview.md) - Complete function documentation
- [Mathematical Formulations](math/overview.md) - Theory and equations
- [Examples](examples/basic-usage.md) - Practical use cases
- [Performance Guide](performance.md) - Optimization tips
- [Integration Guide](integration/xarray.md) - Framework integration

## Use Cases

### Climate Model Evaluation

- Compare model outputs against observations
- Analyze seasonal and temporal variations
- Assess extreme event performance

### Weather Forecast Verification

- Evaluate deterministic and probabilistic forecasts
- Analyze categorical event predictions
- Optimize forecast thresholds

### Air Quality Assessment

- Monitor pollutant concentration forecasts
- Evaluate exceedance predictions
- Assess spatial distribution accuracy

### Ensemble Analysis

- Evaluate ensemble spread-skill relationships
- Assess probabilistic forecast performance
- Analyze ensemble member contributions

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details on:

- Setting up the development environment
- Submitting bug reports and feature requests
- Contributing new metrics and improvements

## License

Monet Stats is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: arl.webmaster@noaa.gov
- 🐛 Issues: [GitHub Issues](https://github.com/noaa-oar-arl/monet-stats/issues)
- 📖 Documentation: [Full Documentation](https://noaa-oar-arl.github.io/monet-stats)

---

_Monet Stats is developed and maintained by the NOAA Air Resources Laboratory_
