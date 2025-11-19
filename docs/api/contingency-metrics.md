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

hss_score = HSS(obs_precip, mod_precip, threshold=0.1)
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
ets_score = ETS(obs_precip, mod_precip, threshold=0.1)
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
csi_score = CSI(obs_precip, mod_precip, threshold=0.1)
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
pod_score = POD(obs_precip, mod_precip, threshold=0.1)
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
far_score = FAR(obs_precip, mod_precip, threshold=0.1)
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
fbi_score = FBI(obs_precip, mod_precip, threshold=0.1)
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
tss_score = TSS(obs_precip, mod_precip, threshold=0.1)
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

### scores(obs, mod, minval, maxval=None)

Calculate a comprehensive set of contingency table metrics.

```python
# Get all contingency metrics at once
contingency_results = scores(obs_precip, mod_precip, threshold=0.1)
print("Contingency Metrics:")
for metric, value in contingency_results.items():
    print(f"  {metric}: {value:.3f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `minval`: Threshold value for event definition
- `maxval`: Maximum threshold (optional)

**Returns:**

- `dict`: Dictionary containing all contingency metrics:
  - `hits`: Number of hits (A)
  - `misses`: Number of misses (B)
  - `false_alarms`: Number of false alarms (C)
  - `correct_negatives`: Number of correct negatives (D)
  - `POD`: Probability of Detection
  - `FAR`: False Alarm Rate
  - `CSI`: Critical Success Index
  - `HSS`: Heidke Skill Score
  - `ETS`: Equitable Threat Score
  - `TSS`: True Skill Statistic

---

### BSS_binary(obs, mod, threshold)

Calculate the Binary Brier Skill Score for categorical forecasts.

```python
# Convert probabilities to binary predictions
obs_binary = (obs_precip >= threshold).astype(int)
mod_binary = (mod_precip >= threshold).astype(int)

bss_score = BSS_binary(obs_binary, mod_binary)
print(f"Binary Brier Skill Score: {bss_score:.3f}")
```

**Parameters:**

- `obs`: Observed binary outcomes (0 or 1)
- `mod`: Forecast binary predictions (0 or 1)
- `threshold`: Decision threshold

**Returns:**

- `float`: Binary Brier Skill Score

**Use Cases:**

- Evaluating binary forecast skill
- Comparing against reference forecasts
- Skill score calculations for categorical predictions

## Advanced Functions

### HSS_max_threshold(obs, mod, threshold_range)

Find the optimal threshold that maximizes the Heidke Skill Score.

```python
# Test multiple thresholds to find optimal performance
thresholds = np.arange(0.1, 10.0, 0.5)
optimal_threshold = HSS_max_threshold(obs_precip, mod_precip, thresholds)
print(f"Optimal threshold for HSS: {optimal_threshold:.1f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `threshold_range`: Array of threshold values to test

**Returns:**

- `float`: Threshold that maximizes HSS

**Use Cases:**

- Optimizing forecast thresholds
- Comparing model performance across different event definitions
- Calibration analysis

---

### POD_max_threshold(obs, mod, threshold_range)

Find the optimal threshold that maximizes the Probability of Detection.

```python
optimal_pod_threshold = POD_max_threshold(obs_precip, mod_precip, thresholds)
print(f"Optimal threshold for POD: {optimal_pod_threshold:.1f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `threshold_range`: Array of threshold values to test

**Returns:**

- `float`: Threshold that maximizes POD

**Use Cases:**

- Maximizing event detection sensitivity
- Optimizing warning systems
- High-impact weather verification

---

### FAR_min_threshold(obs, mod, threshold_range)

Find the optimal threshold that minimizes the False Alarm Rate.

```python
optimal_far_threshold = FAR_min_threshold(obs_precip, mod_precip, thresholds)
print(f"Optimal threshold for FAR minimization: {optimal_far_threshold:.1f}")
```

**Parameters:**

- `obs`: Observed values (array-like)
- `mod`: Modeled values (array-like)
- `threshold_range`: Array of threshold values to test

**Returns:**

- `float`: Threshold that minimizes FAR

**Use Cases:**

- Minimizing false positive rates
- Optimizing resource allocation
- Cost-sensitive forecasting

## Examples

### Basic Precipitation Verification

```python
import numpy as np
from monet_stats import scores, POD, FAR, CSI

# Sample precipitation data (0 = no rain, 1 = rain)
observed = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0])
modeled = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])

# Multiple threshold analysis
thresholds = [0.1, 1.0, 5.0, 10.0]

print("Precipitation Verification Results:")
print("-" * 50)
for threshold in thresholds:
    print(f"\nThreshold: {threshold} mm")
    results = scores(observed, modeled, threshold)

    print(f"  Events: {results['hits'] + results['misses']} observed, "
          f"{results['hits'] + results['false_alarms']} predicted")
    print(f"  POD: {results['POD']:.3f}")
    print(f"  FAR: {results['FAR']:.3f}")
    print(f"  CSI: {results['CSI']:.3f}")
    print(f"  HSS: {results['HSS']:.3f}")
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

# Find optimal threshold
thresholds = np.arange(50, 120, 5)
optimal_hss_threshold = HSS_max_threshold(obs_binary, mod_binary, thresholds)
print(f"  Optimal threshold for HSS: {optimal_hss_threshold} µg/m³")
```

### Multi-Event Contingency Analysis

```python
import pandas as pd
from monet_stats import scores

# Multiple event types
events_data = pd.DataFrame({
    'observed': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],  # Binary events
    'modeled': [0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    'event_type': ['light', 'moderate', 'heavy', 'extreme'] * 2 + ['light', 'moderate']
})

# Analyze by event type
event_analysis = events_data.groupby('event_type').apply(
    lambda x: scores(x['observed'], x['modeled'], threshold=0.5)
).unstack()

print("Multi-Event Contingency Analysis:")
print(event_analysis[['POD', 'FAR', 'CSI', 'HSS']].round(3))
```

## Threshold Optimization

```python
import numpy as np
import matplotlib.pyplot as plt
from monet_stats import HSS_max_threshold, POD_max_threshold, FAR_min_threshold

# Generate synthetic verification data
np.random.seed(42)
n_samples = 1000

# Create observed events (rare)
observed = np.random.binomial(1, 0.1, n_samples)  # 10% event rate

# Model with skill but different bias
modeled = observed.copy()
# Add some misses and false alarms
misses = (observed == 1) & (np.random.rand(n_samples) < 0.3)
false_alarms = (observed == 0) & (np.random.rand(n_samples) < 0.2)
modeled[misses] = 0
modeled[false_alarms] = 1

# Test threshold range
threshold_range = np.linspace(0, 1, 21)

# Calculate metrics for different thresholds
hss_values = []
pod_values = []
far_values = []

for threshold in threshold_range:
    obs_binary = (observed >= threshold).astype(int)
    mod_binary = (modeled >= threshold).astype(int)

    hss_values.append(HSS(obs_binary, mod_binary, threshold))
    pod_values.append(POD(obs_binary, mod_binary, threshold))
    far_values.append(FAR(obs_binary, mod_binary, threshold))

# Find optimal thresholds
optimal_hss = HSS_max_threshold(observed, modeled, threshold_range)
optimal_pod = POD_max_threshold(observed, modeled, threshold_range)
optimal_far = FAR_min_threshold(observed, modeled, threshold_range)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(threshold_range, hss_values, 'b-', label='HSS')
plt.axvline(optimal_hss, color='r', linestyle='--', label=f'Optimal HSS: {optimal_hss:.2f}')
plt.xlabel('Threshold')
plt.ylabel('HSS')
plt.title('Heidke Skill Score')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(threshold_range, pod_values, 'g-', label='POD')
plt.axvline(optimal_pod, color='r', linestyle='--', label=f'Optimal POD: {optimal_pod:.2f}')
plt.xlabel('Threshold')
plt.ylabel('POD')
plt.title('Probability of Detection')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(threshold_range, far_values, 'orange', label='FAR')
plt.axvline(optimal_far, color='r', linestyle='--', label=f'Optimal FAR: {optimal_far:.2f}')
plt.xlabel('Threshold')
plt.ylabel('FAR')
plt.title('False Alarm Rate')
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

metrics = scores(obs_clean, mod_clean, threshold)
```

### Issue: Zero Division in FAR Calculation

```python
# Solution: Add small epsilon to avoid division by zero
def safe_far(obs, mod, threshold):
    hits, misses, false_alarms, correct_negatives = contingency_table(obs, mod, threshold)
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
