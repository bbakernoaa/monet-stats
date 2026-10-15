---
type: reference
category: reference
tags: [relative-metrics, normalized-bias]
---

# Relative Metrics

Normalized and relative error measures for model evaluation.

## Core Functions

### NMB(obs, mod)

Normalized Mean Bias (%). The bias scaled by the sum of observations, which
makes it comparable across variables and datasets with different units or
scales. Common in regulatory and operational air-quality model performance
reports.

```python
from monet_stats import NMB
nmb = NMB(obs, mod)  # e.g. 10.0 for a 10% high bias
```

### NMB_ABS(obs, mod)

Like `NMB`, but the denominator uses absolute values, so it stays meaningful
when observations have mixed sign.

### NMdnB(obs, mod)

Normalized Median Bias (%). The median-based counterpart of `NMB`, which is
less sensitive to outliers in either series.

### FB(obs, mod)

Fractional Bias (%). Symmetric ratio-form bias measure, common in photochemical
and dispersion model evaluation.

### FE(obs, mod)

Fractional Error (%). The non-directional companion to `FB`, quantifying the
magnitude of fractional disagreement.

### MPE(obs, mod)

Mean Peak Error (%) — note this is **not** mean percentage error. It compares
the peaks of the model and observation series and is used for extreme-event
evaluation such as air-quality exceedances.

```python
from monet_stats import MPE
mpe = MPE(obs, mod)
```

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
