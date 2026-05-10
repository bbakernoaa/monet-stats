"""
Hurricane and Tropical Cyclone Track Statistics (Aero Protocol Compliant).
"""

from typing import Optional, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history


def haversine_distance(
    lat1: Union[xr.DataArray, np.ndarray],
    lon1: Union[xr.DataArray, np.ndarray],
    lat2: Union[xr.DataArray, np.ndarray],
    lon2: Union[xr.DataArray, np.ndarray],
    radius: float = 6371.0,
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the great-circle distance between two points on the Earth (Haversine formula).

    Parameters
    ----------
    lat1, lon1 : Union[xr.DataArray, np.ndarray]
        Latitude and longitude of the first point(s) in decimal degrees.
    lat2, lon2 : Union[xr.DataArray, np.ndarray]
        Latitude and longitude of the second point(s) in decimal degrees.
    radius : float, optional
        Radius of the Earth in km. Default is 6371.0 km.

    Returns
    -------
    Union[xr.DataArray, np.ndarray]
        Great-circle distance in km (or same units as radius).
    """

    def _haversine_numpy(l1, ln1, l2, ln2, r):
        p1, lm1 = np.radians(l1), np.radians(ln1)
        p2, lm2 = np.radians(l2), np.radians(ln2)
        dp = p2 - p1
        dlm = lm2 - lm1
        a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlm / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        return r * c

    if any(isinstance(x, xr.DataArray) for x in [lat1, lon1, lat2, lon2]):
        return xr.apply_ufunc(
            _haversine_numpy,
            lat1,
            lon1,
            lat2,
            lon2,
            kwargs={"r": radius},
            dask="parallelized",
            output_dtypes=[float],
        )

    return _haversine_numpy(lat1, lon1, lat2, lon2, radius)


def track_error(
    obs_lat: Union[xr.DataArray, np.ndarray],
    obs_lon: Union[xr.DataArray, np.ndarray],
    mod_lat: Union[xr.DataArray, np.ndarray],
    mod_lon: Union[xr.DataArray, np.ndarray],
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the track error (distance) between observed and modeled positions.

    Parameters
    ----------
    obs_lat, obs_lon : Union[xr.DataArray, np.ndarray]
        Observed latitude and longitude.
    mod_lat, mod_lon : Union[xr.DataArray, np.ndarray]
        Model/predicted latitude and longitude.

    Returns
    -------
    Union[xr.DataArray, np.ndarray]
        Track error in km.
    """
    res = haversine_distance(obs_lat, obs_lon, mod_lat, mod_lon)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, "track_error")
    return res


def bearing(
    lat1: Union[xr.DataArray, np.ndarray],
    lon1: Union[xr.DataArray, np.ndarray],
    lat2: Union[xr.DataArray, np.ndarray],
    lon2: Union[xr.DataArray, np.ndarray],
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the initial bearing (forward azimuth) from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1 : Union[xr.DataArray, np.ndarray]
        Latitude and longitude of the starting point(s).
    lat2, lon2 : Union[xr.DataArray, np.ndarray]
        Latitude and longitude of the destination point(s).

    Returns
    -------
    Union[xr.DataArray, np.ndarray]
        Initial bearing in degrees (0 to 360).
    """

    def _bearing_numpy(l1, ln1, l2, ln2):
        p1, lm1 = np.radians(l1), np.radians(ln1)
        p2, lm2 = np.radians(l2), np.radians(ln2)
        dlm = lm2 - lm1
        y = np.sin(dlm) * np.cos(p2)
        x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlm)
        theta = np.degrees(np.arctan2(y, x))
        return (theta + 360) % 360

    if any(isinstance(x, xr.DataArray) for x in [lat1, lon1, lat2, lon2]):
        return xr.apply_ufunc(_bearing_numpy, lat1, lon1, lat2, lon2, dask="parallelized", output_dtypes=[float])

    return _bearing_numpy(lat1, lon1, lat2, lon2)


def along_track_error(
    obs_lat: Union[xr.DataArray, np.ndarray],
    obs_lon: Union[xr.DataArray, np.ndarray],
    mod_lat: Union[xr.DataArray, np.ndarray],
    mod_lon: Union[xr.DataArray, np.ndarray],
    prev_obs_lat: Union[xr.DataArray, np.ndarray],
    prev_obs_lon: Union[xr.DataArray, np.ndarray],
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the along-track error component.

    Parameters
    ----------
    obs_lat, obs_lon : Union[xr.DataArray, np.ndarray]
        Observed positions at time t.
    mod_lat, mod_lon : Union[xr.DataArray, np.ndarray]
        Model positions at time t.
    prev_obs_lat, prev_obs_lon : Union[xr.DataArray, np.ndarray]
        Observed positions at time t-1 (defines motion vector).

    Returns
    -------
    Union[xr.DataArray, np.ndarray]
        Along-track error in km. Positive means model is ahead of observations.
    """
    # Distance from obs to mod
    d_om = haversine_distance(obs_lat, obs_lon, mod_lat, mod_lon)

    # Bearing from obs to mod
    theta_om = np.deg2rad(bearing(obs_lat, obs_lon, mod_lat, mod_lon))

    # Bearing of motion (from prev_obs to obs)
    theta_motion = np.deg2rad(bearing(prev_obs_lat, prev_obs_lon, obs_lat, obs_lon))

    # Relative angle
    delta_theta = theta_om - theta_motion

    res = d_om * np.cos(delta_theta)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, "along_track_error")
    return res


def cross_track_error(
    obs_lat: Union[xr.DataArray, np.ndarray],
    obs_lon: Union[xr.DataArray, np.ndarray],
    mod_lat: Union[xr.DataArray, np.ndarray],
    mod_lon: Union[xr.DataArray, np.ndarray],
    prev_obs_lat: Union[xr.DataArray, np.ndarray],
    prev_obs_lon: Union[xr.DataArray, np.ndarray],
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the cross-track error component.

    Parameters
    ----------
    obs_lat, obs_lon : Union[xr.DataArray, np.ndarray]
        Observed positions at time t.
    mod_lat, mod_lon : Union[xr.DataArray, np.ndarray]
        Model positions at time t.
    prev_obs_lat, prev_obs_lon : Union[xr.DataArray, np.ndarray]
        Observed positions at time t-1 (defines motion vector).

    Returns
    -------
    Union[xr.DataArray, np.ndarray]
        Cross-track error in km. Positive means model is to the right of observations.
    """
    # Distance from obs to mod
    d_om = haversine_distance(obs_lat, obs_lon, mod_lat, mod_lon)

    # Bearing from obs to mod
    theta_om = np.deg2rad(bearing(obs_lat, obs_lon, mod_lat, mod_lon))

    # Bearing of motion (from prev_obs to obs)
    theta_motion = np.deg2rad(bearing(prev_obs_lat, prev_obs_lon, obs_lat, obs_lon))

    # Relative angle
    delta_theta = theta_om - theta_motion

    res = d_om * np.sin(delta_theta)
    if isinstance(res, (xr.DataArray, xr.Dataset)):
        return _update_history(res, "cross_track_error")
    return res


def translation_speed(
    lat: Union[xr.DataArray, np.ndarray],
    lon: Union[xr.DataArray, np.ndarray],
    time: Optional[Union[xr.DataArray, np.ndarray]] = None,
    dim: str = "time",
) -> Union[xr.DataArray, np.ndarray]:
    """
    Calculate the translation speed of a track.

    Parameters
    ----------
    lat, lon : Union[xr.DataArray, np.ndarray]
        Latitudes and longitudes of the track.
    time : Union[xr.DataArray, np.ndarray], optional
        Timestamps. If None, assumes lat/lon are Xarray and uses 'time' coordinate.
    dim : str, optional
        Dimension along which to compute speed. Default is 'time'.

    Returns
    -------
    Union[xr.DataArray, np.ndarray]
        Translation speed in km/h. Returns same length as input, with NaN at the start.
    """
    if isinstance(lat, xr.DataArray):
        if time is None:
            time = lat[dim]

        # Use shift to get previous positions
        lat_prev = lat.shift({dim: 1})
        lon_prev = lon.shift({dim: 1})

        dist = haversine_distance(lat_prev, lon_prev, lat, lon)

        # Calculate time difference in hours
        dt = time.diff(dim)
        # Ensure dt is in hours
        if np.issubdtype(dt.dtype, np.timedelta64):
            dt_hours = dt / np.timedelta64(1, "h")
        else:
            # Assume it's already numeric and in some unit, but better to be safe
            dt_hours = dt

        # Align to original time index to preserve size
        speed = (dist / dt_hours).reindex_like(lat)
        return _update_history(speed, "translation_speed")

    # NumPy path
    lat_arr = np.asanyarray(lat)
    lon_arr = np.asanyarray(lon)

    dist = haversine_distance(lat_arr[:-1], lon_arr[:-1], lat_arr[1:], lon_arr[1:])

    if time is not None:
        time_arr = np.asanyarray(time)
        dt = np.diff(time_arr)
        if np.issubdtype(dt.dtype, np.timedelta64):
            dt_hours = dt / np.timedelta64(1, "h")
        else:
            dt_hours = dt
    else:
        raise ValueError("time must be provided for translation_speed when using NumPy arrays.")

    # Prepend NaN to match input length
    speed = np.zeros_like(lat_arr, dtype=float)
    speed[0] = np.nan
    speed[1:] = dist / dt_hours
    return speed


__all__ = [
    "haversine_distance",
    "track_error",
    "bearing",
    "along_track_error",
    "cross_track_error",
    "translation_speed",
]
