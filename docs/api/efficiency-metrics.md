# Efficiency Metrics

Model efficiency and performance measures for evaluating forecast skill relative to a reference (usually the mean of observations).

## Overview

Efficiency metrics are used to assess the relative performance of a model compared to a baseline, such as the observed mean or a naive forecast. These metrics are particularly common in hydrology and climatology.

## Core Metrics

### Nash-Sutcliffe Efficiency (NSE)

NSE is a normalized statistic that determines the relative magnitude of the residual variance ("noise") compared to the measured data variance ("information"). NSE indicates how well the plot of observed versus simulated data fits the 1:1 line.

```python
from monet_stats.efficiency_metrics import NSE
nse_val = NSE(obs, mod)
```

- **1.0**: Perfect match.
- **0.0**: Model predictions are as accurate as the mean of the observed data.
- **< 0.0**: Observed mean is a better predictor than the model.

### Kling-Gupta Efficiency (KGE)

KGE is a relatively new metric (Gupta et al., 2009) developed to decompose NSE into three components: correlation, bias, and variability. This allows for a more balanced evaluation of model performance.

```python
from monet_stats.efficiency_metrics import KGE
kge_val = KGE(obs, mod)
```

### Relative NSE (rNSE)

rNSE is a modification of the Nash-Sutcliffe efficiency that normalizes the differences by the observed values, making it less sensitive to high-flow or large-magnitude events in time series.

```python
from monet_stats.efficiency_metrics import rNSE
rnse_val = rNSE(obs, mod)
```

### Percent Correct (PC)

Calculates the percentage of model predictions that are within a specified tolerance of the observations. This is useful for providing a simple, intuitive measure of accuracy.

```python
from monet_stats.efficiency_metrics import PC
pc_val = PC(obs, mod, tolerance=0.1) # 10% tolerance
```

## API Reference

::: monet_stats.efficiency_metrics