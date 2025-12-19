"""
Tests for Spatial Skill Metrics
"""

import numpy as np
import pytest
import xarray as xr
import dask.array as da

from monet_stats.spatial_skill_metrics import FSS, VETS


@pytest.fixture
def sample_spatial_data():
    """Create sample spatial xarray DataArrays for testing."""
    obs_np = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    mod_np = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    obs_xr = xr.DataArray(obs_np, dims=["y", "x"])
    mod_xr = xr.DataArray(mod_np, dims=["y", "x"])
    return obs_xr, mod_xr


@pytest.fixture
def dask_spatial_data(sample_spatial_data):
    """Create dask-backed spatial xarray DataArrays for testing."""
    obs_xr, mod_xr = sample_spatial_data
    obs_dask = obs_xr.chunk({"y": 2, "x": 2})
    mod_dask = mod_xr.chunk({"y": 2, "x": 2})
    return obs_dask, mod_dask


class TestSpatialSkillMetrics:
    """Test suite for spatial skill metrics."""

    def test_fss_xarray(self, sample_spatial_data):
        """Test FSS with xarray inputs."""
        obs_xr, mod_xr = sample_spatial_data
        fss_score = FSS(obs_xr, mod_xr, threshold=0.5, window_size=3)
        assert isinstance(fss_score, xr.DataArray)
        assert 0 <= fss_score.item() <= 1

    def test_fss_perfect_score(self, sample_spatial_data):
        """Test FSS with identical inputs."""
        obs_xr, _ = sample_spatial_data
        fss_score = FSS(obs_xr, obs_xr, threshold=0.5, window_size=3)
        assert fss_score.item() == 1.0

    def test_vets_xarray(self, sample_spatial_data):
        """Test VETS with xarray inputs."""
        obs_xr, mod_xr = sample_spatial_data
        vets_score = VETS(obs_xr, mod_xr)
        assert isinstance(vets_score, xr.DataArray)
        assert 0 <= vets_score.item() <= 1

    def test_vets_perfect_score(self, sample_spatial_data):
        """Test VETS with identical inputs."""
        obs_xr, _ = sample_spatial_data
        vets_score = VETS(obs_xr, obs_xr)
        assert vets_score.item() == 1.0


class TestSpatialSkillMetricsDask:
    """Test suite for spatial skill metrics with Dask-backed arrays."""

    def test_fss_dask(self, dask_spatial_data):
        """Test FSS with Dask-backed xarray inputs."""
        obs_dask, mod_dask = dask_spatial_data
        fss_score = FSS(obs_dask, mod_dask, threshold=0.5, window_size=3)
        assert isinstance(fss_score.data, da.Array)
        fss_val = fss_score.compute().item()
        assert 0 <= fss_val <= 1

    def test_fss_perfect_score_dask(self, dask_spatial_data):
        """Test FSS with identical Dask inputs."""
        obs_dask, _ = dask_spatial_data
        fss_score = FSS(obs_dask, obs_dask, threshold=0.5, window_size=3)
        assert isinstance(fss_score.data, da.Array)
        assert fss_score.compute().item() == 1.0

    def test_vets_dask(self, dask_spatial_data):
        """Test VETS with Dask-backed xarray inputs."""
        obs_dask, mod_dask = dask_spatial_data
        vets_score = VETS(obs_dask, mod_dask)
        assert isinstance(vets_score.data, da.Array)
        vets_val = vets_score.compute().item()
        assert 0 <= vets_val <= 1

    def test_vets_perfect_score_dask(self, dask_spatial_data):
        """Test VETS with identical Dask inputs."""
        obs_dask, _ = dask_spatial_data
        vets_score = VETS(obs_dask, obs_dask)
        assert isinstance(vets_score.data, da.Array)
        assert vets_score.compute().item() == 1.0
