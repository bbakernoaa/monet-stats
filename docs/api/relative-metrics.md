# Relative Metrics

Normalized and relative error measures for model evaluation.

## Core Functions

### MG(obs, mod)

Geometric Mean Bias. Standard metric for log-normally distributed variables in atmospheric science.

```python
from monet_stats import MG
mg_score = MG(obs, mod)
```

### VG(obs, mod)

Geometric Variance. Measures the spread of the ratios of model to observations for log-normally distributed variables.

```python
from monet_stats import VG
vg_score = VG(obs, mod)
```

## API Reference

::: monet_stats.relative_metrics
