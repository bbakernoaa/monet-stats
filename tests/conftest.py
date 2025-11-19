"""
Global pytest configuration and fixtures for monet-stats testing framework.
"""
import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, settings

# Configure hypothesis settings for better test performance
settings.register_profile(
    "default",
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.filter_too_much,
        HealthCheck.data_too_large,
    ],
    deadline=1000,  # 1 second deadline per test
    max_examples=50,  # Reduced from default for faster testing
)
settings.load_profile("default")


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session")
def small_test_data(random_seed):
    """Generate small test datasets for unit tests."""
    np.random.seed(random_seed)
    n_obs = 100
    n_mod = 100

    # Generate correlated observation and model data
    obs = np.random.normal(10, 2, n_obs)
    mod = obs + np.random.normal(0, 0.5, n_mod)  # Model slightly biased

    return {
        'obs': obs,
        'mod': mod,
        'n_obs': n_obs,
        'n_mod': n_mod
    }


@pytest.fixture(scope="session")
def medium_test_data(random_seed):
    """Generate medium test datasets for integration tests."""
    np.random.seed(random_seed)
    n_points = 1000

    # Generate spatial data with some correlation
    x = np.linspace(0, 10, int(np.sqrt(n_points)))
    y = np.linspace(0, 10, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)

    obs_2d = np.sin(X) * np.cos(Y) + np.random.normal(0, 0.1, X.shape)
    mod_2d = obs_2d + np.random.normal(0, 0.05, X.shape)

    # Generate time series data
    time = np.arange(100)
    obs_ts = np.sin(time * 0.1) + np.random.normal(0, 0.2, 100)
    mod_ts = obs_ts + np.random.normal(0, 0.1, 100)

    return {
        'obs_2d': obs_2d,
        'mod_2d': mod_2d,
        'obs_ts': obs_ts,
        'mod_ts': mod_ts,
        'X': X,
        'Y': Y,
        'time': time
    }


@pytest.fixture(scope="session")
def xarray_test_data(random_seed):
    """Generate xarray DataArray test datasets."""
    np.random.seed(random_seed)

    # Create spatial coordinates
    lat = np.linspace(-90, 90, 20)
    lon = np.linspace(-180, 180, 30)
    time = np.arange(10)

    # Create test data arrays
    obs_data = np.random.normal(15, 5, (len(time), len(lat), len(lon)))
    mod_data = obs_data + np.random.normal(0, 1, obs_data.shape)

    obs_da = xr.DataArray(
        obs_data,
        coords={'time': time, 'lat': lat, 'lon': lon},
        dims=['time', 'lat', 'lon'],
        name='observation'
    )

    mod_da = xr.DataArray(
        mod_data,
        coords={'time': time, 'lat': lat, 'lon': lon},
        dims=['time', 'lat', 'lon'],
        name='model'
    )

    return {
        'obs': obs_da,
        'mod': mod_da,
        'lat': lat,
        'lon': lon,
        'time': time
    }


@pytest.fixture
def perfect_correlation_data():
    """Generate data with perfect correlation."""
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1  # Perfect linear relationship
    return x, y


@pytest.fixture
def no_correlation_data():
    """Generate data with no correlation."""
    np.random.seed(123)
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)
    return x, y


@pytest.fixture
def edge_case_data():
    """Generate edge case datasets."""
    return {
        'zeros': np.zeros(50),
        'constants': np.ones(50) * 5,
        'nans': np.full(50, np.nan),
        'mixed': np.array([1, 2, np.nan, 4, 5, np.inf, 7, 8, -np.inf, 10]),
        'empty': np.array([]),
        'single': np.array([42]),
        'two_values': np.array([1, 2])
    }


@pytest.fixture
def contingency_table_data():
    """Generate contingency table test data."""
    # 2x2 contingency table
    table_2x2 = np.array([[30, 10], [20, 40]])

    # 3x3 contingency table
    table_3x3 = np.array([[20, 15, 5], [10, 25, 15], [5, 10, 35]])

    # Binary classification data
    obs_binary = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    mod_binary = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1])

    return {
        'table_2x2': table_2x2,
        'table_3x3': table_3x3,
        'obs_binary': obs_binary,
        'mod_binary': mod_binary
    }


# Markers for test categorization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests as edge case tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "mathematical: marks tests as mathematical correctness tests"
    )
    config.addinivalue_line(
        "markers", "xarray: marks tests that require xarray functionality"
    )


# Custom exception for test validation
class TestValidationError(Exception):
    """Custom exception for test validation errors."""
