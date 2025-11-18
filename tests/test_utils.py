"""
Test utilities and helper functions for monet-stats testing framework.
"""
import numpy as np
import pytest


class TestDataGenerator:
    """Utility class for generating synthetic test data."""
    
    @staticmethod
    def generate_correlated_data(n_samples=100, correlation=0.8, noise_level=0.1, seed=42):
        """
        Generate two correlated datasets.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        correlation : float
            Desired correlation coefficient (-1 to 1)
        noise_level : float
            Level of noise to add
        seed : int
            Random seed for reproducibility
            
        Returns
        -------
        tuple
            (obs, mod) arrays with specified correlation
        """
        np.random.seed(seed)
        
        # Generate base signal
        signal = np.random.normal(0, 1, n_samples)
        
        # Create correlated data
        obs = signal + np.random.normal(0, noise_level, n_samples)
        mod = correlation * signal + np.random.normal(0, np.sqrt(1 - correlation**2) * noise_level, n_samples)
        
        return obs, mod
    
    @staticmethod
    def generate_perfect_relationship(n_samples=50, relationship='linear'):
        """
        Generate data with perfect mathematical relationship.
        
        Parameters
        ----------
        n_samples : int
            Number of samples
        relationship : str
            Type of relationship ('linear', 'quadratic', 'exponential')
            
        Returns
        -------
        tuple
            (x, y) arrays with perfect relationship
        """
        x = np.linspace(0, 10, n_samples)
        
        if relationship == 'linear':
            y = 2 * x + 1
        elif relationship == 'quadratic':
            y = x**2
        elif relationship == 'exponential':
            y = np.exp(x / 5)
        else:
            raise ValueError(f"Unknown relationship: {relationship}")
            
        return x, y
    
    @staticmethod
    def generate_edge_cases():
        """Generate various edge case datasets."""
        return {
            'zeros': np.zeros(50),
            'constants': np.ones(50) * 5,
            'nans': np.full(50, np.nan),
            'infs': np.array([np.inf, -np.inf] * 25),
            'mixed': np.array([1, 2, np.nan, 4, 5, np.inf, 7, 8, -np.inf, 10]),
            'empty': np.array([]),
            'single': np.array([42]),
            'two_values': np.array([1, 2]),
            'alternating': np.array([0, 1] * 25),
        }
    
    @staticmethod
    def generate_contingency_data(n_categories=3, n_samples=100, seed=42):
        """
        Generate categorical data for contingency table analysis.
        
        Parameters
        ----------
        n_categories : int
            Number of categories
        n_samples : int
            Number of samples
        seed : int
            Random seed
            
        Returns
        -------
        tuple
            (obs_categories, mod_categories) arrays
        """
        np.random.seed(seed)
        
        obs_categories = np.random.choice(n_categories, n_samples)
        mod_categories = np.random.choice(n_categories, n_samples)
        
        return obs_categories, mod_categories
    
    @staticmethod
    def generate_spatial_data(shape=(20, 30), spatial_correlation=True, seed=42):
        """
        Generate spatially correlated data.
        
        Parameters
        ----------
        shape : tuple
            Shape of spatial grid (ny, nx)
        spatial_correlation : bool
            Whether to add spatial correlation
        seed : int
            Random seed
            
        Returns
        -------
        tuple
            (obs_grid, mod_grid) 2D arrays
        """
        np.random.seed(seed)
        
        ny, nx = shape
        obs_grid = np.random.normal(0, 1, shape)
        
        if spatial_correlation:
            # Add spatial smoothing to create correlation
            from scipy.ndimage import gaussian_filter
            obs_grid = gaussian_filter(obs_grid, sigma=2)
        
        # Add noise for model data
        mod_grid = obs_grid + np.random.normal(0, 0.1, shape)
        
        return obs_grid, mod_grid


def assert_statistical_property(value, expected_value, tolerance=1e-10, name="statistic"):
    """
    Assert that a statistical property is within tolerance of expected value.
    
    Parameters
    ----------
    value : float or array
        Computed statistic
    expected_value : float or array
        Expected value
    tolerance : float
        Absolute tolerance
    name : str
        Name of statistic for error message
        
    Raises
    ------
    AssertionError
        If statistic is not within tolerance
    """
    if np.isnan(value).any() or np.isnan(expected_value).any():
        raise AssertionError(f"{name} contains NaN values")
    
    if np.isinf(value).any() or np.isinf(expected_value).any():
        raise AssertionError(f"{name} contains infinite values")
    
    if not np.allclose(value, expected_value, atol=tolerance, rtol=1e-12):
        raise AssertionError(
            f"{name} = {value}, expected {expected_value}, "
            f"difference = {abs(value - expected_value)} > {tolerance}"
        )


def assert_correlation_bounds(correlation, name="correlation"):
    """Assert that correlation coefficient is within valid bounds [-1, 1]."""
    if not (-1 <= correlation <= 1):
        raise AssertionError(f"{name} = {correlation} outside valid range [-1, 1]")


def assert_percentage_bounds(percentage, name="percentage"):
    """Assert that percentage value is within valid bounds [0, 100] or reasonable range."""
    if not (0 <= percentage <= 200):  # Allow some flexibility for normalized metrics
        raise AssertionError(f"{name} = {percentage} outside reasonable range [0, 200]")


def assert_positive_value(value, name="value"):
    """Assert that value is positive (for metrics like RMSE, MAE, etc.)."""
    if value < 0:
        raise AssertionError(f"{name} = {value} is negative")


def check_array_shapes(*arrays, name="arrays"):
    """Check that all arrays have compatible shapes for element-wise operations."""
    shapes = [np.asarray(arr).shape for arr in arrays]
    if len(set(shapes)) > 1:
        raise AssertionError(f"{name} have incompatible shapes: {shapes}")


def validate_metric_output(metric_func, obs, mod, expected_type=None, **kwargs):
    """
    Validate output of a metric function.
    
    Parameters
    ----------
    metric_func : callable
        Metric function to test
    obs : array-like
        Observation data
    mod : array-like
        Model data
    expected_type : type, optional
        Expected return type
    **kwargs
        Additional arguments for metric function
        
    Returns
    -------
    result
        Result of metric function
        
    Raises
    ------
    AssertionError
        If validation fails
    """
    obs = np.asarray(obs)
    mod = np.asarray(mod)
    
    # Test with valid data
    try:
        result = metric_func(obs, mod, **kwargs)
    except Exception as e:
        raise AssertionError(f"Metric function failed with valid data: {e}")
    
    # Check return type
    if expected_type is not None and not isinstance(result, expected_type):
        raise AssertionError(f"Expected return type {expected_type}, got {type(result)}")
    
    # Check for NaN/inf values
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        raise AssertionError(f"Metric result contains NaN or inf values: {result}")
    
    return result


class MetricTester:
    """Helper class for testing statistical metrics."""
    
    def __init__(self, metric_func, metric_name):
        self.metric_func = metric_func
        self.metric_name = metric_name
        self.data_gen = TestDataGenerator()
    
    def test_perfect_agreement(self):
        """Test metric with perfectly agreeing data."""
        obs, mod = self.data_gen.generate_perfect_relationship(relationship='linear')
        result = validate_metric_output(self.metric_func, obs, mod)
        return result
    
    def test_perfect_correlation(self):
        """Test metric with perfectly correlated data."""
        obs, mod = self.data_gen.generate_correlated_data(correlation=1.0)
        result = validate_metric_output(self.metric_func, obs, mod)
        return result
    
    def test_no_correlation(self):
        """Test metric with uncorrelated data."""
        obs, mod = self.data_gen.generate_correlated_data(correlation=0.0)
        result = validate_metric_output(self.metric_func, obs, mod)
        return result
    
    def test_edge_cases(self):
        """Test metric with various edge cases."""
        edge_cases = self.data_gen.generate_edge_cases()
        
        results = {}
        for case_name, case_data in edge_cases.items():
            if len(case_data) == 0:  # Skip empty arrays
                continue
                
            try:
                # Test with obs=case_data, mod=normal_data
                normal_data = np.random.normal(0, 1, len(case_data))
                result = self.metric_func(case_data, normal_data)
                results[case_name] = result
            except Exception as e:
                results[case_name] = f"Error: {e}"
        
        return results