"""
Tests for Spatial Skill Metrics
"""

import numpy as np
import pytest
import xarray as xr
import dask.array as da

from monet_stats.spatial_skill_metrics import FSS, VETS


@pytest.fixture(params=[np, xr])
def factory(request):
    """Provide a factory for creating numpy or xarray arrays."""
    return request.param


@pytest.fixture
def sample_spatial_data(factory):
    """Create sample spatial data for testing."""
    obs_data = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    mod_data = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    if factory == xr:
        obs = xr.DataArray(obs_data, dims=["y", "x"])
        mod = xr.DataArray(mod_data, dims=["y", "x"])
    else:
        obs = obs_data
        mod = mod_data
    return obs, mod


class TestSpatialSkillMetrics:
    """Test suite for spatial skill metrics."""

    def test_fss(self, sample_spatial_data):
        """Test FSS with NumPy and xarray inputs."""
        obs, mod = sample_spatial_data
        fss_score = FSS(obs, mod, threshold=0.5, window_size=3)
        assert isinstance(fss_score, (float, np.ndarray, xr.DataArray))
        assert 0 <= np.mean(fss_score) <= 1

    def test_fss_perfect_score(self, sample_spatial_data):
        """Test FSS with identical inputs."""
        obs, _ = sample_spatial_data
        fss_score = FSS(obs, obs, threshold=0.5, window_size=3)
        assert np.isclose(np.mean(fss_score), 1.0)

    def test_vets(self, sample_spatial_data):
        """Test VETS with NumPy and xarray inputs."""
        obs, mod = sample_spatial_data
        vets_score = VETS(obs, mod)
        assert isinstance(vets_score, (float, np.ndarray, xr.DataArray))

    def test_vets_perfect_score(self, sample_spatial_data):
        """Test VETS with identical inputs."""
        obs, _ = sample_spatial_data
        vets_score = VETS(obs, obs)
        assert np.isclose(np.mean(vets_score), 1.0)


class TestSpatialSkillMetricsDask:
    """Test suite for spatial skill metrics with Dask-backed arrays."""

    def test_fss_dask(self, sample_spatial_data):
        """Test FSS with Dask-backed xarray inputs."""
        obs_xr, mod_xr = sample_spatial_data
        if isinstance(obs_xr, xr.DataArray):
            obs_dask = obs_xr.chunk({"y": 2, "x": 2})
            mod_dask = mod_xr.chunk({"y": 2, "x": 2})
            fss_score = FSS(obs_dask, mod_dask, threshold=0.5, window_size=3)
            assert isinstance(fss_score.data, da.Array)
            fss_val = fss_score.compute()
            assert 0 <= fss_val <= 1

    def test_vets_dask(self, sample_spatial_data):
        """Test VETS with Dask-backed xarray inputs."""
        obs_xr, mod_xr = sample_spatial_data
        if isinstance(obs_xr, xr.DataArray):
            obs_dask = obs_xr.chunk({"y": 2, "x": 2})
            mod_dask = mod_xr.chunk({"y": 2, "x": 2})
            vets_score = VETS(obs_dask, mod_dask)
            assert isinstance(vets_score.data, da.Array)
            vets_val = vets_score.compute()
            assert 0 <= vets_val <= 1
