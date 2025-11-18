"""
Test aliases for backwards compatibility and testing framework.
This module provides the functions that the test files expect but that may have different names in the actual implementation.
"""

# Error Metrics - Common aliases expected by tests
def mean_bias_error(obs, mod):
    """Alias for MB (Mean Bias)"""
    from .error_metrics import MB
    return MB(obs, mod)

def mean_absolute_error(obs, mod):
    """Alias for MAE (Mean Absolute Error)"""
    from .error_metrics import MAE
    return MAE(obs, mod)

def root_mean_squared_error(obs, mod):
    """Alias for RMSE (Root Mean Square Error)"""
    from .error_metrics import RMSE
    return RMSE(obs, mod)

def normalized_mean_bias_error(obs, mod):
    """Alias for NMB (Normalized Mean Bias)"""
    from .error_metrics import MNB
    return MNB(obs, mod)

def normalized_mean_absolute_error(obs, mod):
    """Alias for NMAE calculation"""
    from .error_metrics import MAE, MO
    return MAE(obs, mod) / abs(MO(obs, mod))

# Correlation Metrics - Common aliases expected by tests  
def pearson_correlation(obs, mod):
    """Alias for pearsonr (Pearson correlation)"""
    from .correlation_metrics import pearsonr
    return pearsonr(obs, mod)

def spearman_correlation(obs, mod):
    """Alias for spearmanr (Spearman correlation)"""
    from .correlation_metrics import spearmanr
    return spearmanr(obs, mod)

def coefficient_of_determination(obs, mod):
    """Alias for R2 (Coefficient of Determination)"""
    from .correlation_metrics import R2
    return R2(obs, mod)

# Contingency Metrics - Common aliases expected by tests
def hit_rate(obs, mod):
    """Alias for POD (Probability of Detection)"""
    from .contingency_metrics import POD
    return POD(obs, mod, 0.5)

def false_alarm_rate(obs, mod):
    """Alias for FAR (False Alarm Rate)"""
    from .contingency_metrics import FAR
    return FAR(obs, mod, 0.5)

def critical_success_index(obs, mod):
    """Alias for CSI (Critical Success Index)"""
    from .contingency_metrics import CSI
    return CSI(obs, mod, 0.5)

def equitable_threat_score(obs, mod):
    """Alias for ETS (Equitable Threat Score)"""
    from .contingency_metrics import ETS
    return ETS(obs, mod, 0.5)

def peirce_skill_score(obs, mod):
    """Alias for TSS (True Skill Statistic)"""
    from .contingency_metrics import TSS
    return TSS(obs, mod, 0.5)

def heidke_skill_score(obs, mod):
    """Alias for HSS (Heidke Skill Score)"""
    from .contingency_metrics import HSS
    return HSS(obs, mod, 0.5)

# Efficiency Metrics - Common aliases expected by tests
def index_of_agreement(obs, mod):
    """Alias for IOA (Index of Agreement)"""
    from .correlation_metrics import IOA
    return IOA(obs, mod)

def modified_index_of_agreement(obs, mod):
    """Alias for d1 (Modified Index of Agreement)"""
    from .correlation_metrics import d1
    return d1(obs, mod)

def akaike_information_criterion(obs, mod):
    """Placeholder for AIC - not implemented yet"""
    import numpy as np
    n = len(obs)
    mse = np.mean((obs - mod) ** 2)
    return n * np.log(mse) + 2 * 2  # k=2 parameters

def bayesian_information_criterion(obs, mod):
    """Placeholder for BIC - not implemented yet"""
    import numpy as np
    n = len(obs)
    mse = np.mean((obs - mod) ** 2)
    return n * np.log(mse) + 2 * np.log(n)  # k=2 parameters

# Add to __all__ for proper exports
__all__ = [
    'mean_bias_error',
    'mean_absolute_error', 
    'root_mean_squared_error',
    'normalized_mean_bias_error',
    'normalized_mean_absolute_error',
    'pearson_correlation',
    'spearman_correlation',
    'coefficient_of_determination',
    'hit_rate',
    'false_alarm_rate',
    'critical_success_index',
    'equitable_threat_score',
    'peirce_skill_score',
    'heidke_skill_score',
    'index_of_agreement',
    'modified_index_of_agreement',
    'akaike_information_criterion',
    'bayesian_information_criterion'
]