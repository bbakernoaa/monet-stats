"""
Tests for Spatial Skill Metrics
"""

import numpy as np
import pytest
import xarray as xr

from monet_stats.spatial_skill_metrics import FSS, VETS


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


def test_fss_perfect_score(sample_spatial_data):
    """Test FSS with identical inputs."""
    obs, _ = sample_spatial_data
    fss_score = FSS(obs, obs, threshold=0.5, window_size=3)
    assert fss_score == 1.0


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


def test_vets_perfect_score(sample_spatial_data):
    """Test VETS with identical inputs."""
    obs, _ = sample_spatial_data
    vets_score = VETS(obs, obs)
    assert vets_score == 1.0
