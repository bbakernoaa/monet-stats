# Error Metrics

Error analysis and bias quantification for model evaluation.

## Overview

Error metrics quantify the difference between model predictions and observations. They are fundamental to model evaluation, providing insights into systematic biases, average error magnitudes, and overall predictive accuracy.

## Core Metrics

### Mean Absolute Error (MAE)

MAE measures the average magnitude of errors in a set of predictions, without considering their direction. It is the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

```python
from monet_stats.error_metrics import MAE
mae_val = MAE(obs, mod)
```

### Root Mean Square Error (RMSE)

RMSE is a quadratic scoring rule that also measures the average magnitude of the error. It’s the square root of the average of squared differences between prediction and actual observation. RMSE gives a relatively high weight to large errors.

```python
from monet_stats.error_metrics import RMSE
rmse_val = RMSE(obs, mod)
```

### Mean Bias (MB)

MB calculates the average difference between the model and observations. It indicates whether the model is systematically over-predicting (positive bias) or under-predicting (negative bias).

```python
from monet_stats.error_metrics import MB
mb_val = MB(obs, mod)
```

### FAC2

Fraction of predictions within a factor of two. This is a robust metric common in air quality modeling that is less sensitive to extreme outliers.

```python
from monet_stats.error_metrics import FAC2
fac2_score = FAC2(obs, mod)
```

### RMSLE

Root Mean Square Logarithmic Error. Useful for variables spanning several orders of magnitude (e.g., pollutant concentrations) because it penalizes under-predictions and over-predictions equally in relative terms.

```python
from monet_stats.error_metrics import RMSLE
rmsle_score = RMSLE(obs, mod)
```

## API Reference

::: monet_stats.error_metrics
