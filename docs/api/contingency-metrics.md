---
type: reference
category: reference
tags: [contingency, categorical, verification]
---

# Contingency Metrics

The contingency metrics module provides comprehensive metrics for evaluating categorical forecasts, particularly useful for binary event prediction in atmospheric sciences. These metrics are based on contingency table analysis and are commonly used for precipitation verification, air quality exceedance predictions, and other categorical forecast evaluations.

## Overview

Contingency metrics analyze the relationship between observed and predicted categorical events by organizing outcomes into a 2×2 contingency table:

|                       | Predicted Event  | Predicted No Event     |
| --------------------- | ---------------- | ---------------------- |
| **Observed Event**    | Hits (A)         | Misses (B)             |
| **Observed No Event** | False Alarms (C) | Correct Rejections (D) |

## Core Functions

### HSS(obs, mod, minval)

Calculate the Heidke Skill Score for categorical forecast verification.

```python
import numpy as np
from monet_stats import HSS

# Example: Precipitation verification
obs_precip = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0])  # 1 = rain, 0 = no rain
mod_precip = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 1])  # Model predictions

hss_score = HSS(obs_precip, mod_precip, 0.1)
print(f"Heidke Skill Score: {hss_score:.3f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional, for range-based events)

**Returns:**

- `float`: Heidke Skill Score (-∞ to 1, where 1 = perfect, 0 = no skill)

**Typical Use Cases:**

- Evaluating categorical forecast skill (e.g., precipitation, air quality events)
- Used in meteorology and environmental modeling for binary event prediction accuracy

**Typical Values and Range:**

- Range: -∞ to 1
- 1: Perfect forecast
- 0: No skill (random forecast)
- Negative values: Worse than random

---

### ETS(obs, mod, minval, maxval=None)

Calculate the Equitable Threat Score (ETS), also known as the Gilbert Skill Score.

```python
ets_score = ETS(obs_precip, mod_precip, 0.1)
print(f"Equitable Threat Score: {ets_score:.3f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional)

**Returns:**

- `float`: Equitable Threat Score (-∞ to 1)

**Use Cases:**

- Fair evaluation of rare events
- Accounting for random hits in skill calculation
- Commonly used for precipitation verification

---

### CSI(obs, mod, minval, maxval=None)

Calculate the Critical Success Index (CSI), also known as the Threat Score.

```python
csi_score = CSI(obs_precip, mod_precip, 0.1)
print(f"Critical Success Index: {csi_score:.3f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional)

**Returns:**

- `float`: Critical Success Index (0 to 1)

**Use Cases:**

- High-impact event verification
- Areas where false alarms and misses are equally costly
- Common in severe weather forecasting

---

### POD(obs, mod, minval, maxval=None)

Calculate the Probability of Detection (POD), also known as the Hit Rate.

```python
pod_score = POD(obs_precip, mod_precip, 0.1)
print(f"Probability of Detection: {pod_score:.3f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional)

**Returns:**

- `float`: Probability of Detection (0 to 1)

**Use Cases:**

- Assessing forecast sensitivity to detect events
- Important for public safety applications
- Used in severe weather warnings

---

### FAR(obs, mod, minval, maxval=None)

Calculate the False Alarm Rate (FAR).

```python
far_score = FAR(obs_precip, mod_precip, 0.1)
print(f"False Alarm Rate: {far_score:.3f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional)

**Returns:**

- `float`: False Alarm Rate (0 to 1)

**Use Cases:**

- Evaluating false positive rates
- Important for resource allocation decisions
- Used in operational forecasting

---

### FBI(obs, mod, minval, maxval=None)

Calculate the False Bias Index.

```python
fbi_score = FBI(obs_precip, mod_precip, 0.1)
print(f"False Bias Index: {fbi_score:.3f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional)

**Returns:**

- `float`: False Bias Index

**Use Cases:**

- Assessing forecast bias toward false alarms
- Comparing model tendency to overpredict events

---

### TSS(obs, mod, minval, maxval=None)

Calculate the True Skill Statistic (TSS), also known as the Hanssen-Kuipers Discriminant.

```python
tss_score = TSS(obs_precip, mod_precip, 0.1)
print(f"True Skill Statistic: {tss_score:.3f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional)

**Returns:**

- `float`: True Skill Statistic (-1 to 1)

**Use Cases:**

- Balanced evaluation of hits and misses
- Comparing forecast systems
- Assessing discrimination skill

---

### scores(obs, mod, minval, maxval=None, axis=None)

Compute the raw 2x2 contingency table counts.

```python
# Get the contingency table counts
hits, misses, false_alarms, correct_negatives = scores(obs_precip, mod_precip, 0.1)
print(f"Hits: {hits}, Misses: {misses}, "
      f"False alarms: {false_alarms}, Correct negatives: {correct_negatives}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional)
- `axis`: Axis along which to compute the counts (optional)

**Returns:**

- `tuple`: The four contingency counts `(hits, misses, false_alarms,
  correct_negatives)` — each a scalar, `np.ndarray`, or `xr.DataArray`
  depending on the input. Use these counts to build custom scores; the
  individual metrics (`POD`, `FAR`, `CSI`, `HSS`, `ETS`, `TSS`) are computed
  by their own functions, not returned here.

---

### BSS_binary(obs, mod, threshold, axis=None)

Calculate the Binary Brier Skill Score for deterministic forecasts. The
`threshold` is **required** and is used internally to convert the inputs to
binary events.

```python
bss_score = BSS_binary(obs_precip, mod_precip, threshold=0.5)
print(f"Binary Brier Skill Score: {bss_score:.3f}")
```

**Parameters:**

- `obs`: Observed binary outcomes (0 or 1) or continuous values
- `mod`: Forecast binary predictions (0 or 1) or continuous values
- `threshold`: Decision threshold used to binarize the inputs
- `axis`: Axis along which to compute the score (optional)

**Returns:**

- `float`: Binary Brier Skill Score (-∞ to 1; 1 = perfect, 0 = reference
  skill, negative = worse than the reference)

**Use Cases:**

- Evaluating binary forecast skill
- Comparing against reference forecasts
- Skill score calculations for categorical predictions

---

### BS(obs, mod)

Calculate the Brier Score (BS).

```python
bs_score = BS(obs_precip, mod_probabilities)
print(f"Brier Score: {bs_score:.3f}")
```

**Parameters:**

- `obs`: Observed binary outcomes (0 or 1)
- `mod`: Forecast probabilities (0 to 1)

**Returns:**

- `float`: Brier Score (0 to 1, where 0 is perfect)

**Use Cases:**

- Evaluating the accuracy of probabilistic binary forecasts
- Standard measure of the mean squared error of probability forecasts

## Advanced Functions

The threshold optimizers scan a **range of `minval` thresholds** and return
the best threshold together with its score. They live in
`monet_stats.contingency_metrics` (not re-exported at the top level), so
import them from the module:

```python
from monet_stats.contingency_metrics import HSS_max_threshold

# Scan minval from 0.2 to 0.8 in steps of 0.1
optimal_threshold, best_score = HSS_max_threshold(obs_precip, mod_precip,
                                                  0.2, 0.8, step_size=0.1)
print(f"Optimal HSS threshold: {optimal_threshold:.2f} (HSS = {best_score:.3f})")
```

### HSS_max_threshold(obs, mod, minval_range, maxval_range, step_size=1.0)

Find the threshold that maximizes the Heidke Skill Score.

**Parameters:**

- `obs`, `mod`: Observed and modeled values (array-like)
- `minval_range`: Lowest threshold to test
- `maxval_range`: Highest threshold to test
- `step_size`: Spacing between tested thresholds (default `1.0`)

**Returns:** `tuple[float, float]` — `(optimal_threshold, max_score)`

---

### ETS_max_threshold(obs, mod, minval_range, maxval_range, step_size=1.0)

As above, maximizing the Equitable Threat Score. Returns
`(optimal_threshold, max_score)`.

---

### POD_max_threshold(obs, mod, minval_range, maxval_range, step_size=1.0)

As above, maximizing the Probability of Detection. Returns
`(optimal_threshold, max_score)`.

---

### FAR_min_threshold(obs, mod, minval_range, maxval_range, step_size=1.0)

As above, minimizing the False Alarm Rate. Returns
`(optimal_threshold, min_score)`.

---

### CSI_max_threshold / TSS_max_threshold

The same pattern for the Critical Success Index and True Skill Statistic.

**Use Cases:**

- Optimizing forecast thresholds for a specific application
- Comparing model performance across different event definitions
- Calibration analysis

## Examples

### Basic Precipitation Verification

```python
import numpy as np
from monet_stats import CSI, FAR, HSS, POD, scores

# Sample precipitation data (0 = no rain, 1 = rain)
observed = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
modeled = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])

# Multiple threshold analysis
thresholds = [0.1, 1.0, 5.0, 10.0]

print("Precipitation Verification Results:")
print("-" * 50)
for threshold in thresholds:
    print(f"\nThreshold: {threshold} mm")

    # scores() returns the raw contingency counts
    hits, misses, false_alarms, correct_negatives = scores(observed, modeled, threshold)
    print(f"  Events: {hits + misses} observed, {hits + false_alarms} predicted")

    # Individual scores are computed by their own functions
    print(f"  POD: {POD(observed, modeled, threshold):.3f}")
    print(f"  FAR: {FAR(observed, modeled, threshold):.3f}")
    print(f"  CSI: {CSI(observed, modeled, threshold):.3f}")
    print(f"  HSS: {HSS(observed, modeled, threshold):.3f}")
```

### Air Quality Exceedance Analysis

```python
import numpy as np
from monet_stats import HSS, ETS, POD

# Air quality concentration data (µg/m³)
observed_conc = np.array([35, 45, 55, 65, 75, 85, 95, 105, 115, 125])
modeled_conc = np.array([40, 48, 52, 68, 72, 88, 92, 108, 112, 130])

# Exceedance threshold for PM2.5 (75 µg/m³)
threshold = 75

# Convert to binary exceedances
obs_binary = (observed_conc >= threshold).astype(int)
mod_binary = (modeled_conc >= threshold).astype(int)

# Calculate metrics
hss = HSS(obs_binary, mod_binary, threshold)
ets = ETS(obs_binary, mod_binary, threshold)
pod = POD(obs_binary, mod_binary, threshold)

print(f"Air Quality Exceedance Analysis (threshold = {threshold} µg/m³):")
print(f"  Heidke Skill Score: {hss:.3f}")
print(f"  Equitable Threat Score: {ets:.3f}")
print(f"  Probability of Detection: {pod:.3f}")

# Find optimal threshold (scalar min/max range, not an array)
optimal_hss_threshold, best_hss = HSS_max_threshold(observed_conc, modeled_conc,
                                                    50, 120, step_size=5)
print(f"  Optimal threshold for HSS: {optimal_hss_threshold} µg/m³ (HSS = {best_hss:.3f})")
```

### Multi-Event Contingency Analysis

`scores()` returns the four raw counts, so build a summary DataFrame from the
individual metric functions:

```python
import pandas as pd
from monet_stats import CSI, FAR, HSS, POD

# Multiple event types
events_data = pd.DataFrame({
    'observed': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],  # Binary events
    'modeled': [0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    'event_type': ['light', 'moderate', 'heavy', 'extreme'] * 2 + ['light', 'moderate']
})

# Analyze by event type
event_analysis = events_data.groupby('event_type').apply(
    lambda x: pd.Series({
        'POD': POD(x['observed'], x['modeled'], 0.5),
        'FAR': FAR(x['observed'], x['modeled'], 0.5),
        'CSI': CSI(x['observed'], x['modeled'], 0.5),
        'HSS': HSS(x['observed'], x['modeled'], 0.5),
    })
)

print("Multi-Event Contingency Analysis:")
print(event_analysis.round(3))
```

## Threshold Optimization

The `*_max_threshold` / `*_min_threshold` helpers scan a scalar threshold range
and return the best threshold with its score. Import them from the module:

```python
import numpy as np
import matplotlib.pyplot as plt
from monet_stats import FAR, HSS, POD
from monet_stats.contingency_metrics import (
    FAR_min_threshold,
    HSS_max_threshold,
    POD_max_threshold,
)

# Generate synthetic verification data
np.random.seed(42)
n_samples = 1000

# Create observed events (rare)
observed = np.random.binomial(1, 0.1, n_samples)  # 10% event rate

# Model with skill but different bias
modeled = observed.copy()
misses = (observed == 1) & (np.random.rand(n_samples) < 0.3)
false_alarms = (observed == 0) & (np.random.rand(n_samples) < 0.2)
modeled[misses] = 0
modeled[false_alarms] = 1

# Sweep thresholds manually to plot the full curve
threshold_range = np.linspace(0, 1, 21)
hss_values, pod_values, far_values = [], [], []
for t in threshold_range:
    obs_binary = (observed >= t).astype(int)
    mod_binary = (modeled >= t).astype(int)
    hss_values.append(HSS(obs_binary, mod_binary, t))
    pod_values.append(POD(obs_binary, mod_binary, t))
    far_values.append(FAR(obs_binary, mod_binary, t))

# Find optimal thresholds (returns (threshold, score) tuples)
optimal_hss, best_hss = HSS_max_threshold(observed, modeled, 0.0, 1.0, step_size=0.05)
optimal_pod, best_pod = POD_max_threshold(observed, modeled, 0.0, 1.0, step_size=0.05)
optimal_far, best_far = FAR_min_threshold(observed, modeled, 0.0, 1.0, step_size=0.05)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(threshold_range, hss_values, "b-", label="HSS")
plt.axvline(optimal_hss, color="r", linestyle="--", label=f"Optimal HSS: {optimal_hss:.2f}")
plt.xlabel("Threshold")
plt.ylabel("HSS")
plt.title("Heidke Skill Score")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(threshold_range, pod_values, "g-", label="POD")
plt.axvline(optimal_pod, color="r", linestyle="--", label=f"Optimal POD: {optimal_pod:.2f}")
plt.xlabel("Threshold")
plt.ylabel("POD")
plt.title("Probability of Detection")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(threshold_range, far_values, "orange", label="FAR")
plt.axvline(optimal_far, color="r", linestyle="--", label=f"Optimal FAR: {optimal_far:.2f}")
plt.xlabel("Threshold")
plt.ylabel("FAR")
plt.title("False Alarm Rate")
plt.legend()

plt.tight_layout()
plt.show()
```

## Best Practices

### 1. Rare Event Handling

For rare events (<5% occurrence rate):

- Use Equitable Threat Score (ETS) instead of CSI
- Consider alternative formulations for skill scores
- Be cautious with small sample sizes

### 2. Threshold Selection

- Use physical thresholds when available
- Optimize thresholds for specific applications
- Consider the costs of false alarms vs. misses

### 3. Sample Size Considerations

- Minimum recommended: 50-100 events for stable statistics
- Use bootstrapping for uncertainty estimation with small samples
- Report confidence intervals for skill scores

### 4. Multiple Threshold Analysis

- Analyze performance across a range of thresholds
- Use contingency tables for detailed breakdown
- Consider ROC curve analysis for comprehensive evaluation

## Common Issues and Solutions

### Issue: NaN Values in Contingency Metrics

```python
# Solution: Handle missing data properly
obs_clean = obs[~np.isnan(obs) & ~np.isnan(mod)]
mod_clean = mod[~np.isnan(obs) & ~np.isnan(mod)]

hits, misses, false_alarms, correct_negatives = scores(obs_clean, mod_clean, threshold)
```

### Issue: Zero Division in FAR Calculation

```python
# Solution: Guard the denominator using the contingency counts
from monet_stats import scores

def safe_far(obs, mod, threshold):
    hits, misses, false_alarms, correct_negatives = scores(obs, mod, threshold)
    if hits + false_alarms == 0:
        return 0.0
    return false_alarms / (hits + false_alarms)

far = safe_far(obs, mod, threshold)
```

### Issue: Imbalanced Classes

```python
# Solution: Use skill scores that account for random chance
# Prefer ETS over CSI for rare events
ets_score = ETS(obs, mod, threshold)
hss_score = HSS(obs, mod, threshold)
```

## API Reference

::: monet_stats.contingency_metrics
