# Error Metrics

Error analysis and bias quantification for model evaluation.

## Core Functions

### FAC2(obs, mod)

Fraction of predictions within a factor of two. This is a robust metric common in air quality modeling.

```python
from monet_stats import FAC2
fac2_score = FAC2(obs, mod)
```

### RMSLE(obs, mod)

Root Mean Square Logarithmic Error. Useful for variables spanning several orders of magnitude.

```python
from monet_stats import RMSLE
rmsle_score = RMSLE(obs, mod)
```

## API Reference

::: monet_stats.error_metrics