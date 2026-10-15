---
type: reference
category: reference
tags: [spatial, fss, vets, verification]
---

# Spatial Skill Metrics

Spatial skill scores that evaluate the accuracy of multi-dimensional fields.
These metrics live in `monet_stats.spatial_skill_metrics` and are also
re-exported at the top level (`monet_stats.FSS`, `monet_stats.VETS`).

## Core Metrics

### Fractions Skill Score (FSS)

The FSS compares the fractional coverage of an event within a moving square
window, making it a scale-dependent verification metric for fields where exact
point placement matters less than the spatial pattern (e.g., convection,
precipitation). Skill is assessed by sweeping `window_size` to find the scale
at which a forecast becomes useful.

```python
import numpy as np
from monet_stats import FSS

obs_field = np.random.exponential(5, (50, 50))
mod_field = obs_field + np.random.normal(0, 1, (50, 50))

# Skill at a 5x5 window for values above the observed mean
fss = FSS(obs_field, mod_field, window_size=5)
print(f"Fractions Skill Score: {fss:.3f}")

# Explicit event threshold
fss_10 = FSS(obs_field, mod_field, window_size=5, threshold=10.0)
```

**Signature:**

```text
FSS(obs, mod, window_size=3, threshold=None, dim=None, axis=None)
```

**Parameters:**

- `obs`: Observed field (`np.ndarray` or `xr.DataArray`)
- `mod`: Modeled field
- `window_size`: Side length of the square averaging window (default `3`)
- `threshold`: Value defining an event. If `None`, the **mean of the
  observations** is used (lazy/Dask-friendly).
- `dim`: Dimension(s) along which to compute the fractions (xarray only;
  defaults to all dimensions)
- `axis`: Axis or axes along which to compute the fractions

**Returns:** `xr.DataArray`, `np.ndarray`, or `float` — the Fractions Skill
Score, ranging from **0 (no skill) to 1 (perfect skill)**.

### Volumetric Equitable Threat Score (VETS)

VETS extends the threat-score family to volumetric totals (e.g., accumulated
precipitation over a period), and is *equitable* — it discounts the hits
expected from random chance alone.

```python
from monet_stats import VETS

vets = VETS(obs_field, mod_field)
print(f"Volumetric Equitable Threat Score: {vets:.3f}")
```

**Signature:**

```text
VETS(obs, mod, dim=None, axis=None)
```

**Parameters:**

- `obs`: Observed volumetric field (`np.ndarray` or `xr.DataArray`)
- `mod`: Modeled volumetric field
- `dim`: Dimension(s) along which to compute the score (xarray only)
- `axis`: Axis or axes along which to compute the score

**Returns:** `xr.DataArray`, `np.ndarray`, or `float` — the Volumetric
Equitable Threat Score.

## Choosing Between FSS and VETS

| Question | Metric |
| --- | --- |
| "At what spatial scale does my forecast become useful?" | `FSS` (sweep `window_size`) |
| "How skillful is my accumulated/volumetric total, accounting for chance?" | `VETS` |

## API Reference

::: monet_stats.spatial_skill_metrics
