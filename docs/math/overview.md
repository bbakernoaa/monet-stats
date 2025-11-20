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

### Mean Absolute Percentage Error (MAPE)

$$
\text{MAPE} = \frac{100}{N} \sum_{i=1}^{N} \left|\frac{O_i - M_i}{O_i}\right|
$$

MAPE measures the average absolute percentage error, useful for scale-independent comparisons.

### Symmetric Mean Absolute Percentage Error (sMAPE)

$$
\text{sMAPE} = \frac{200}{N} \sum_{i=1}^{N} \frac{|O_i - M_i|}{|O_i| + |M_i|}
$$

sMAPE provides a symmetric version of MAPE that is bounded between 0% and 200%.

### Mean Absolute Scaled Error (MASE)

$$
\text{MASE} = \frac{\frac{1}{N}\sum_{i=1}^{N}|O_i - M_i|}{\frac{1}{N-1}\sum_{i=2}^{N}|O_i - O_{i-1}|}
$$

MASE compares forecast errors to the average error of a naive forecast, making it scale-independent.

### Root Mean Square Percentage Error (RMSPE)

$$
\text{RMSPE} = 100 \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\frac{O_i - M_i}{O_i}\right)^2}
$$

RMSPE is the percentage version of RMSE, emphasizing larger errors.

### Normalized Root Mean Square Error (NRMSE)

$$
\text{NRMSE} = \frac{\text{RMSE}}{O_{max} - O_{min}}
$$

NRMSE normalizes RMSE by the range of observed values, allowing comparison across different scales.

### Median Absolute Error (MedAE)

$$
\text{MedAE} = \text{median}(|O_i - M_i|)
$$

MedAE is the median of absolute errors, providing a robust measure less sensitive to outliers than MAE.

### Mean Normalized Bias (MNB)

$$
\text{MNB} = \frac{\sum_{i=1}^{N}(M_i - O_i)}{\sum_{i=1}^{N}O_i} \times 100\%
$$

MNB measures the normalized bias by comparing the sum of differences to the sum of observations.

### Normalized Mean Error (NME)

$$
\text{NME} = \frac{\sum_{i=1}^{N}|M_i - O_i|}{\sum_{i=1}^{N}O_i} \times 100\%
$$

NME normalizes the total absolute error by the sum of observations.

### Fractional Bias (FB)

$$
\text{FB} = 2 \times \frac{\bar{M} - \bar{O}}{\bar{M} + \bar{O}} \times 100\%
$$

FB measures the average bias as a fraction of the sum of model and observed means.

### Fractional Error (FE)

$$
\text{FE} = 2 \times \frac{|M - O|}{M + O} \times 100\%
$$

FE measures the average error as a fraction of the sum of model and observed values.

### Index of Agreement (IOA)

$$
\text{IOA} = 1 - \frac{\sum_{i=1}^{N}(O_i - M_i)^2}{\sum_{i=1}^{N}(|M_i - \bar{O}| + |O_i - \bar{O}|)^2}
$$

IOA ranges from 0 to 1, with values closer to 1 indicating better agreement.

### Modified Index of Agreement (d1)

$$
\text{d1} = 1 - \frac{\sum_{i=1}^{N}|O_i - M_i|}{\sum_{i=1}^{N}(|M_i - \bar{O}| + |O_i - \bar{O}|)}
$$

d1 is a modified version of IOA using absolute differences instead of squared differences.

### Modified Coefficient of Efficiency (E1)

$$
\text{E1} = 1 - \frac{\sum_{i=1}^{N}|O_i - M_i|}{\sum_{i=1}^{N}|O_i - \bar{O}|}
$$

E1 is a robust version of the coefficient of efficiency using absolute differences.

### Center of Mass Error (COE)

$$
\text{COE} = \sqrt{(\bar{x}_o - \bar{x}_m)^2 + (\bar{y}_o - \bar{y}_m)^2}
$$

Where $\bar{x}_o, \bar{y}_o$ and $\bar{x}_m, \bar{y}_m$ are the centers of mass of observed and modeled fields respectively.

### Volumetric Error

$$
\text{Volumetric Error} = \frac{|\sum M_i - \sum O_i|}{|\sum O_i|}
$$

Measures the relative difference in total volume between modeled and observed fields.

### Normalized Mean Square Error (NMSE)

$$
\text{NMSE} = \frac{\frac{1}{N}\sum_{i=1}^{N}(O_i - M_i)^2}{\sigma_O^2}
$$

NMSE normalizes the mean square error by the variance of observations.

### Logarithmic Error

$$
\text{Log Error} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\ln(O_i + \epsilon) - \ln(M_i + \epsilon))^2}
$$

Where $\epsilon$ is a small constant to avoid $\ln(0)$.

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

### Spearman Rank Correlation

$$
\rho = 1 - \frac{6\sum d_i^2}{N(N^2 - 1)}
$$

Where $d_i$ is the difference between ranks of corresponding values, and $N$ is the number of observations.

### Kendall Rank Correlation

$$
\tau = \frac{(\text{number of concordant pairs}) - (\text{number of discordant pairs})}{\frac{1}{2}N(N-1)}
$$

Kendall's tau measures the ordinal association between two measured quantities.

### Anomaly Correlation (AC)

$$
\text{AC} = \frac{\sum_{i=1}^{N} (O_i - \bar{O})(M_i - \bar{M})}{\sqrt{\sum_{i=1}^{N} (O_i - \bar{O})^2 \sum_{i=1}^{N} (M_i - \bar{M})^2}}
$$

AC measures the correlation between anomalies (deviations from mean) of observations and model values.

### Concordance Correlation Coefficient (CCC)

$$
\text{CCC} = \frac{2\rho\sigma_O\sigma_M}{\sigma_O^2 + \sigma_M^2 + (\bar{O} - \bar{M})^2}
$$

CCC measures how far the data deviates from the line of perfect concordance (slope=1, intercept=0).

### Taylor Skill Score (TSS)

$$
\text{TSS} = \frac{2(1 + r)}{(1 + r_0^2)(1 + r^2)}
$$

Where $r$ is the correlation coefficient and $r_0$ is a reference correlation.

### Kling-Gupta Efficiency (KGE)

$$
\text{KGE} = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}
$$

Where:

- $r$: Pearson correlation coefficient
- $\alpha = \sigma_M / \sigma_O$: Ratio of standard deviations
- $\beta = \bar{M} / \bar{O}$: Ratio of means

KGE provides a comprehensive evaluation of performance across correlation, variability, and bias dimensions.

## Efficiency Metrics

### Nash-Sutcliffe Efficiency (NSE)

$$
\text{NSE} = 1 - \frac{\sum_{i=1}^{N} (O_i - M_i)^2}{\sum_{i=1}^{N} (O_i - \bar{O})^2}
$$

NSE compares the model performance to a simple mean forecast, with values > 0 indicating better performance than climatology.

### Log Nash-Sutcliffe Efficiency (NSElog)

$$
\text{NSElog} = 1 - \frac{\sum_{i=1}^{N} (\ln(O_i + \epsilon) - \ln(M_i + \epsilon))^2}{\sum_{i=1}^{N} (\ln(O_i + \epsilon) - \overline{\ln(O + \epsilon)})^2}
$$

NSElog applies the NSE calculation to log-transformed values, useful for variables with skewed distributions.

### Modified Nash-Sutcliffe Efficiency (mNSE)

$$
\text{mNSE} = 1 - \frac{\sum_{i=1}^{N} |O_i - M_i|}{\sum_{i=1}^{N} |O_i - \bar{O}|}
$$

mNSE uses absolute differences instead of squared differences, making it more robust to outliers.

### Relative Nash-Sutcliffe Efficiency (rNSE)

$$
\text{rNSE} = 1 - \frac{\sum_{i=1}^{N} (O_i - M_i)^2}{\sum_{i=1}^{N} (O_i - \bar{O})^2}
$$

rNSE is similar to NSE but normalized by the range of observations.

### Percent of Correct (PC)

$$
\text{PC} = \frac{\text{Number of predictions within tolerance}}{\text{Total number of predictions}} \times 100\%
$$

PC measures the percentage of predictions that fall within a specified tolerance of observations.

### Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (O_i - M_i)^2
$$

MSE measures the average of the squares of the errors, giving more weight to larger errors.

## Relative/Percentage Metrics

### Normalized Median Bias (NMdnB)

$$
\text{NMdnB} = \frac{\text{median}(M_i - O_i)}{\text{median}(O_i)} \times 100\%
$$

NMdnB measures the normalized median bias, robust to outliers.

### Normalized Median Error (NMdnE)

$$
\text{NMdnE} = \frac{\text{median}(|M_i - O_i|)}{\text{median}(O_i)} \times 100\%
$$

NMdnE measures the normalized median error, robust to outliers.

### Unpaired Space/Unpaired Time Peak Bias (USUTPB)

$$
\text{USUTPB} = \frac{M_{max} - O_{max}}{O_{max}} \times 100\%
$$

USUTPB measures the bias in peak values regardless of spatial or temporal pairing.

### Unpaired Space/Unpaired Time Peak Error (USUTPE)

$$
\text{USUTPE} = \frac{|M_{max} - O_{max}|}{O_{max}} \times 10\%
$$

USUTPE measures the error in peak values regardless of spatial or temporal pairing.

### Mean Normalized Peak Bias (MNPB)

$$
\text{MNPB} = \frac{1}{N} \sum_{i=1}^{N} \frac{M_{max,i} - O_{max,i}}{O_{max,i}} \times 100\%
$$

MNPB measures the mean normalized bias in peak values across multiple series.

### Mean Normalized Peak Error (MNPE)

$$
\text{MNPE} = \frac{1}{N} \sum_{i=1}^{N} \frac{|M_{max,i} - O_{max,i}|}{O_{max,i}} \times 100\%
$$

MNPE measures the mean normalized error in peak values across multiple series.

### Normalized Mean Peak Bias (NMPB)

$$
\text{NMPB} = \frac{\overline{M_{max}} - \overline{O_{max}}}{\overline{O_{max}}} \times 100\%
$$

NMPB measures the normalized mean of peak biases across multiple series.

### Normalized Mean Peak Error (NMPE)

$$
\text{NMPE} = \frac{\overline{|M_{max} - O_{max}|}}{\overline{O_{max}}} \times 10\%
$$

NMPE measures the normalized mean of peak errors across multiple series.

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

### Equitable Threat Score (ETS)

$$
\text{ETS} = \frac{A - A_r}{A + B + C - A_r}
$$

Where $A_r = \frac{(A+B)(A+C)}{N}$ is the number of hits expected by random chance.

ETS measures the threat score adjusted for random hits, useful for rare events.

### Frequency Bias Index (FBI)

$$
\text{FBI} = \frac{A + C}{A + B}
$$

FBI measures the ratio of forecast events to observed events.

### True Skill Statistic (TSS)

$$
\text{TSS} = \text{POD} - \text{POFD}
$$

Where POFD (Probability of False Detection) = $\frac{C}{C+D}$

TSS measures the ability to discriminate between events and non-events.

### Binary Brier Skill Score

$$
\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{ref}}
$$

Where $\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2$ is the Brier Score, and $\text{BS}_{ref}$ is the reference Brier Score.

## Spatial Verification Metrics

### Fractions Skill Score (FSS)

$$
\text{FSS} = 1 - \frac{\text{MSE}_{frac}}{\text{MSE}_{ref}}
$$

Where $\text{MSE}_{frac}$ is the mean squared error of fractional coverage, and $\text{MSE}_{ref}$ is the reference MSE.

FSS evaluates the ability to predict spatial patterns of categorical events.

### Structure-Amplitude-Location (SAL)

$$
S = \frac{2(\frac{\text{max}(M)}{\sum M} - \frac{\text{max}(O)}{\sum O})}{\frac{\text{max}(M)}{\sum M} + \frac{\text{max}(O)}{\sum O}}, \quad A = \frac{2(\bar{M} - \bar{O})}{\bar{M} + \bar{O}}, \quad L = L_1 + L_2
$$

Where:
- $S$: Structure component (-2 to 2, 0 is best)
- $A$: Amplitude component (-2 to 2, 0 is best)
- $L$: Location component (0 to 2, 0 is best)

SAL decomposes verification errors into structure, amplitude, and location components.

### Extreme Dependency Score (EDS)

$$
\text{EDS} = \frac{\log(\frac{\text{hits}}{N})}{\log(p \cdot q)}
$$

Where $p = \frac{N_{obs}}{N}$ and $q = \frac{N_{mod}}{N}$ are the observed and forecasted event frequencies.

EDS measures the skill in forecasting rare events.

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

### Brier Skill Score (BSS)

$$
\text{BSS} = 1 - \frac{\text{BS}}{\text{BS}_{ref}}
$$

Where BS is the Brier Score and $\text{BS}_{ref}$ is the reference Brier Score.

### Spread-Error Relationship

$$
\text{Spread} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (M_i - \bar{M})^2}
$$

$$
\text{Error} = |\bar{M} - O|
$$

Where $\bar{M}$ is the ensemble mean. This measures the relationship between ensemble spread (uncertainty) and forecast error.

### Rank Histogram

The rank histogram (Talagrand diagram) assesses ensemble reliability by plotting the frequency of the observation rank among ensemble members. A flat histogram indicates reliable ensemble forecasts.

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

### Wind Direction Mean Bias (WDMB)

$$
\text{WDMB} = \text{mean}(\text{circlebias}(M_i - O_i))
$$

Where $\text{circlebias}$ handles the circular nature of wind direction differences.

### Wind Direction Root Mean Square Error (WDRMSE)

$$
\text{WDRMSE} = \sqrt{\text{mean}((\text{circlebias}(M_i - O_i))^2)}
$$

### Wind Direction Index of Agreement (WDIOA)

$$
\text{WDIOA} = 1 - \frac{\sum_{i=1}^{N} |\text{circlebias}(O_i - M_i)|^2}{\sum_{i=1}^{N} (|\text{circlebias}(M_i - \bar{O}_c)| + |\text{circlebias}(O_i - \bar{O}_c)|)^2}
$$

Where $\bar{O}_c$ is the circular mean of observations.

## References

- Willmott, C.J., & Matsuura, K. (2005). Advantages of the mean absolute error (MAE) over the root mean square error (RMSE). Climate Research, 30(1), 79-82.
- Nash, J.E., & Sutcliffe, J.V. (1970). River flow forecasting through conceptual models part I — A discussion of principles. Journal of Hydrology, 10(3), 282-290.
- Gupta, H.V., et al. (2009). Decomposition of the mean squared error and NSE criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1-2), 80-91.
- Potts, J.M., et al. (1996). A simple, objective method for partitioning variance in model performance evaluation. American Meteorological Society, 29(2), 202-215.
- Hersbach, H. (2000). Decomposition of the continuous ranked probability score for ensemble prediction systems. Weather and Forecasting, 15(5), 559-570.
