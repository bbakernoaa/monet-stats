"""
Tests for Spatial Skill Metrics (Aero Protocol).
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.spatial_skill_metrics import FSS, VETS

try:
    import dask.array as da

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@pytest.fixture
def sample_spatial_data():
    """Create sample spatial data for testing."""
    obs = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    mod = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    return obs, mod


def test_fss_numpy(sample_spatial_data):
    """Test FSS with NumPy inputs."""
    obs, mod = sample_spatial_data
    fss_score = FSS(obs, mod, threshold=0.5, window_size=3)
    assert isinstance(fss_score, float)
    assert 0 <= fss_score <= 1


def test_fss_xarray(sample_spatial_data):
    """Test FSS with xarray inputs."""
    obs, mod = sample_spatial_data
    obs_xr = xr.DataArray(obs, dims=["y", "x"])
    mod_xr = xr.DataArray(mod, dims=["y", "x"])
    fss_score = FSS(obs_xr, mod_xr, threshold=0.5, window_size=3)
    assert isinstance(fss_score, xr.DataArray)
    assert 0 <= fss_score <= 1
    assert "Fractions Skill Score (FSS)" in fss_score.attrs["history"]


def test_fss_lazy(sample_spatial_data):
    """Test FSS with Dask-backed xarray to ensure laziness."""
    if not DASK_AVAILABLE:
        pytest.skip("Dask not available")

    obs, mod = sample_spatial_data
    obs_xr = xr.DataArray(da.from_array(obs, chunks=(5, 5)), dims=["y", "x"])
    mod_xr = xr.DataArray(da.from_array(mod, chunks=(5, 5)), dims=["y", "x"])

    # FSS with default threshold (should be lazy)
    fss_score = FSS(obs_xr, mod_xr, window_size=3)

    assert hasattr(fss_score.data, "chunks")
    # Verify result
    assert 0 <= fss_score.compute() <= 1


def test_fss_multi_dim():
    """Test FSS with multi-dimensional (time, lat, lon) input."""
    data_obs = np.random.rand(5, 10, 10)
    data_mod = np.random.rand(5, 10, 10)
    obs = xr.DataArray(data_obs, dims=["time", "lat", "lon"])
    mod = xr.DataArray(data_mod, dims=["time", "lat", "lon"])

    # Spatial FSS (rolling only over lat/lon)
    score = FSS(obs, mod, window_size=3, dim=["lat", "lon"])
    assert score.dims == ("time",)
    assert score.shape == (5,)


def test_vets_numpy(sample_spatial_data):
    """Test VETS with NumPy inputs."""
    obs, mod = sample_spatial_data
    vets_score = VETS(obs, mod)
    assert isinstance(vets_score, float)


def test_vets_xarray(sample_spatial_data):
    """Test VETS with xarray inputs."""
    obs, mod = sample_spatial_data
    obs_xr = xr.DataArray(obs, dims=["y", "x"])
    mod_xr = xr.DataArray(mod, dims=["y", "x"])
    vets_score = VETS(obs_xr, mod_xr)
    assert isinstance(vets_score, xr.DataArray)
    assert "Volumetric Equitable Threat Score (VETS)" in vets_score.attrs["history"]


def test_vets_dim_reduction():
    """Test VETS with explicit dimension reduction."""
    data_obs = np.random.rand(5, 10, 10)
    data_mod = np.random.rand(5, 10, 10)
    obs = xr.DataArray(data_obs, dims=["time", "lat", "lon"])
    mod = xr.DataArray(data_mod, dims=["time", "lat", "lon"])

    # Reduce over space, keep time
    score = VETS(obs, mod, dim=["lat", "lon"])
    assert score.dims == ("time",)

    # Reduce over all using axis
    score_all = VETS(obs, mod, axis=[0, 1, 2])
    assert score_all.ndim == 0
