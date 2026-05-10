import numpy as np
import pytest
import xarray as xr
import pandas as pd
from monet_stats.track_metrics import (
    haversine_distance,
    track_error,
    bearing,
    along_track_error,
    cross_track_error,
    translation_speed,
)


def test_haversine_distance():
    # NYC (40.7128, -74.0060) to London (51.5074, -0.1278)
    # Expected: ~5570 km
    nyc = (40.7128, -74.0060)
    lon = (51.5074, -0.1278)
    dist = haversine_distance(nyc[0], nyc[1], lon[0], lon[1])
    assert np.isclose(dist, 5570, atol=20)

    # Identical points should have zero distance
    assert haversine_distance(10, 10, 10, 10) == 0


def test_bearing():
    # North: (0,0) to (10, 0)
    assert np.isclose(bearing(0, 0, 10, 0), 0)
    # East: (0,0) to (0, 10)
    assert np.isclose(bearing(0, 0, 0, 10), 90)
    # South: (10,0) to (0, 0)
    assert np.isclose(bearing(10, 0, 0, 0), 180)
    # West: (0,10) to (0, 0)
    assert np.isclose(bearing(0, 10, 0, 0), 270)


def test_track_errors_basic():
    # Setup a simple case: Moving North
    prev_obs = (0, 0)
    obs = (1, 0)
    # Model is ahead by 1 degree latitude (~111 km)
    mod = (2, 0)

    # All should be ~111.2 km (approx degree to km)
    err = track_error(obs[0], obs[1], mod[0], mod[1])
    at_err = along_track_error(obs[0], obs[1], mod[0], mod[1], prev_obs[0], prev_obs[1])
    ct_err = cross_track_error(obs[0], obs[1], mod[0], mod[1], prev_obs[0], prev_obs[1])

    assert np.isclose(err, 111.19, atol=1.0)
    assert np.isclose(at_err, 111.19, atol=1.0)
    assert np.isclose(ct_err, 0.0, atol=1.0)

    # Model is to the right (East) by 1 degree
    mod_right = (1, 1)
    # Cos(lat) factor for distance at 1 deg N is approx 1, but let's be precise
    err_right = track_error(obs[0], obs[1], mod_right[0], mod_right[1])
    at_err_right = along_track_error(obs[0], obs[1], mod_right[0], mod_right[1], prev_obs[0], prev_obs[1])
    ct_err_right = cross_track_error(obs[0], obs[1], mod_right[0], mod_right[1], prev_obs[0], prev_obs[1])

    assert np.isclose(err_right, ct_err_right)
    assert np.isclose(at_err_right, 0.0, atol=1.0)
    assert ct_err_right > 0  # To the right is positive


def test_translation_speed():
    lats = np.array([0, 1, 2])
    lons = np.array([0, 0, 0])
    times = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"])

    speed = translation_speed(lats, lons, time=times)
    # 1 degree lat per hour is ~111 km/h
    # speed[0] is NaN
    assert np.isnan(speed[0])
    assert np.allclose(speed[1:], 111.19, atol=1.0)


def test_track_metrics_xarray_dask():
    try:
        import dask.array as da
    except ImportError:
        pytest.skip("Dask not installed")

    obs_lat = xr.DataArray(da.from_array([0, 1, 2], chunks=2), dims="time")
    obs_lon = xr.DataArray(da.from_array([0, 0, 0], chunks=2), dims="time")
    mod_lat = xr.DataArray(da.from_array([0.1, 1.1, 2.1], chunks=2), dims="time")
    mod_lon = xr.DataArray(da.from_array([0, 0, 0], chunks=2), dims="time")

    # Verify track_error preserves laziness
    err = track_error(obs_lat, obs_lon, mod_lat, mod_lon)
    assert hasattr(err.data, "chunks")

    res = err.compute()
    assert np.all(res > 0)
    assert "track_error" in res.attrs.get("history", "")


def test_accessor_integration():
    times = pd.date_range("2020-01-01", periods=3, freq="h")
    obs_lat = xr.DataArray([0, 1, 2], coords={"time": times}, dims="time", name="lat")
    obs_lon = xr.DataArray([0, 0, 0], coords={"time": times}, dims="time", name="lon")
    mod_lat = xr.DataArray([0.5, 1.5, 2.5], coords={"time": times}, dims="time", name="lat")
    mod_lon = xr.DataArray([0, 0, 0], coords={"time": times}, dims="time", name="lon")

    # Use accessor on mod_lat
    err = mod_lat.monet_stats.track_error(obs_lat, obs_lon, mod_lon)
    assert isinstance(err, xr.DataArray)
    assert np.all(err > 0)

    speed = mod_lat.monet_stats.translation_speed(mod_lon)
    assert speed.sizes["time"] == 3
    assert np.isnan(speed[0])  # First point has no speed
    assert np.all(speed[1:] > 0)
