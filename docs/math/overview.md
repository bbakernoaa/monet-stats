# Mathematical Formulations

This section provides the mathematical foundations and theoretical background for the statistical metrics implemented in Monet Stats. Understanding these formulations helps in proper interpretation and application of the metrics in atmospheric sciences research.

## Mathematical Notation

- $O$: Observed values
- $M$: Modeled/predicted values
- $N$: Number of observations
- $\bar{O}$: Mean of observed values
- $\bar{M}$: Mean of modeled values
- $\sigma_O$: Standard deviation of observed values
- $\sigma_M$: Standard deviation of modeled values

## Error Metrics

### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |O_i - M_i|
$$

The MAE measures the average magnitude of errors without considering their direction, providing a linear penalty for errors.

### Root Mean Square Error (RMSE)

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (O_i - M_i)^2}
$$

RMSE gives a higher weight to larger errors due to the squaring operation, making it sensitive to outliers.

### Mean Bias (MB)

$$
\text{MB} = \bar{M} - \bar{O} = \frac{1}{N} \sum_{i=1}^{N} M_i - \frac{1}{N} \sum_{i=1}^{N} O_i
$$

MB quantifies the systematic overestimation or underestimation bias in the model.

### Normalized Mean Bias (NMB)

$$
\text{NMB} = \frac{\bar{M} - \bar{O}}{\bar{O}} \times 100\%
$$

NMB expresses bias as a percentage of the observed mean, allowing comparison across different scales.

## Correlation Metrics

### Coefficient of Determination (R²)

$$
R^2 = 1 - \frac{\sum_{i=1}^{N} (O_i - M_i)^2}{\sum_{i=1}^{N} (O_i - \bar{O})^2}
$$

R² represents the proportion of variance in the observed data that is explained by the model.

### Pearson Correlation Coefficient

$$
r = \frac{\sum_{i=1}^{N} (O_i - \bar{O})(M_i - \bar{M})}{\sqrt{\sum_{i=1}^{N} (O_i - \bar{O})^2 \sum_{i=1}^{N} (M_i - \bar{M})^2}}
$$

Pearson correlation measures the linear relationship strength between observed and modeled values.

### Index of Agreement (IOA)

$$
\text{IOA} = 1 - \frac{\sum_{i=1}^{N} (O_i - M_i)^2}{\sum_{i=1}^{N} (|M_i - \bar{O}| + |O_i - \bar{O}|)^2}
$$

IOA ranges from 0 to 1, with values closer to 1 indicating better agreement.

## Skill Scores

### Nash-Sutcliffe Efficiency (NSE)

$$
\text{NSE} = 1 - \frac{\sum_{i=1}^{N} (O_i - M_i)^2}{\sum_{i=1}^{N} (O_i - \bar{O})^2}
$$

NSE compares the model performance to a simple mean forecast, with values > 0 indicating better performance than climatology.

### Kling-Gupta Efficiency (KGE)

$$
\text{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}
$$

Where:

- $r$: Pearson correlation coefficient
- $\alpha = \sigma_M / \sigma_O$: Ratio of standard deviations
- $\beta = \bar{M} / \bar{O}$: Ratio of means

KGE provides a comprehensive evaluation of performance across correlation, variability, and bias dimensions.

## Contingency Table Metrics

### Contingency Table Structure

|                  | Forecast Yes     | Forecast No           | Total |
| ---------------- | ---------------- | --------------------- | ----- |
| **Observed Yes** | A (Hits)         | B (Misses)            | A+B   |
| **Observed No**  | C (False Alarms) | D (Correct Negatives) | C+D   |
| **Total**        | A+C              | B+D                   | N     |

### Probability of Detection (POD)

$$
\text{POD} = \frac{A}{A+B} = \frac{\text{Hits}}{\text{Hits + Misses}}
$$

POD measures the ability to correctly detect the occurrence of an event.

### False Alarm Ratio (FAR)

$$
\text{FAR} = \frac{C}{A+C} = \frac{\text{False Alarms}}{\text{Hits + False Alarms}}
$$

FAR indicates the proportion of predicted events that did not actually occur.

### Critical Success Index (CSI)

$$
\text{CSI} = \frac{A}{A+B+C} = \frac{\text{Hits}}{\text{Hits + Misses + False Alarms}}
$$

CSI measures the accuracy of event forecasts, excluding correct negatives.

### Heidke Skill Score (HSS)

$$
\text{HSS} = \frac{2(AD - BC)}{(A+C)(C+D) + (A+B)(B+D)}
$$

HSS measures the improvement of the forecast over random chance.

## Spatial Verification Metrics

### Fractions Skill Score (FSS)

$$
\text{FSS} = 1 - \frac{\text{MSE}(f_o, f_m)}{\text{MSE}(f_o, \bar{f_o}) + \text{MSE}(f_m, \bar{f_m})}
$$

Where $f_o$ and $f_m$ are observed and modeled fractions in neighborhoods, and $\bar{f_o}$, $\bar{f_m}$ are their means.

FSS evaluates the ability to predict spatial patterns of categorical events.

### Structure-Amplitude-Location (SAL)

$$
S = \log\left(\frac{\sigma_M}{\sigma_O}\right), \quad A = \frac{2}{3} \frac{|\bar{M} - \bar{O}|}{\bar{O} + \bar{M}}, \quad L = \frac{1}{N} \sum_{i=1}^{N} \frac{|x_{M,i} - x_{O,i}|}{\text{distance}_i}
$$

SAL decomposes verification errors into structure, amplitude, and location components.

## Ensemble Metrics

### Continuous Ranked Probability Score (CRPS)

$$
\text{CRPS} = \int_{-\infty}^{\infty} [F_o(x) - F_m(x)]^2 dx
$$

Where $F_o(x)$ is the cumulative distribution function of observations and $F_m(x)$ is the CDF of the ensemble forecast.

CRPS measures the overall quality of probabilistic forecasts, considering both reliability and sharpness.

### Brier Score

$$
\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (p_i - o_i)^2
$$

Where $p_i$ is the predicted probability and $o_i$ is the observed outcome (0 or 1).

The Brier score measures the accuracy of probabilistic binary forecasts.

## Circular Statistics

For wind direction and other circular variables:

### Circular Mean

$$
\bar{\theta}_c = \text{atan2}\left(\frac{1}{N}\sum_{i=1}^{N} \sin\theta_i, \frac{1}{N}\sum_{i=1}^{N} \cos\theta_i\right)
$$

### Circular Variance

$$
V_c = 1 - \frac{1}{N}\left|\sum_{i=1}^{N} e^{i\theta_i}\right|
$$

Circular statistics properly handle the periodic nature of angular measurements like wind direction.

## References

- Willmott, C.J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE). Climate Research, 30(1), 79-82.
- Nash, J.E., & Sutcliffe, J.V. (1970). River flow forecasting through conceptual models part I — A discussion of principles. Journal of Hydrology, 10(3), 282-290.
- Gupta, H.V., et al. (2009). Decomposition of the mean squared error and NSE criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1-2), 80-91.
- Potts, J.M., et al. (1996). A simple, objective method for partitioning variance in model performance evaluation. American Meteorological Society, 29(2), 202-215.
- Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. Weather and Forecasting, 15(5), 559-570.
