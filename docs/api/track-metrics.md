---
type: reference
category: reference
tags: [track, tropical-cyclone, haversine, storm]
---

# Track Metrics

Great-circle and storm-track statistics for verifying the position and motion
of moving features such as tropical cyclones (Aero Protocol compliant). All
functions accept NumPy arrays or xarray `DataArray` objects and are re-exported
at the top level (`monet_stats.track_error`, `monet_stats.find_storm_center`,
etc.).

## Distance and Bearing

### haversine_distance

Great-circle distance between two points using the Haversine formula.

```python
import numpy as np
from monet_stats import haversine_distance

dist_km = haversine_distance(25.0, -80.0, 27.5, -77.0)
print(f"Distance: {dist_km:.1f} km")
```

**Signature:** `haversine_distance(lat1, lon1, lat2, lon2, radius=6371.0)`

- `lat1, lon1, lat2, lon2`: Coordinates in decimal degrees.
- `radius`: Earth radius in km (default `6371.0`); the return shares these units.

### bearing

Initial bearing (forward azimuth) from point 1 to point 2.

```python
from monet_stats import bearing

deg = bearing(25.0, -80.0, 27.5, -77.0)  # degrees, 0-360
```

**Signature:** `bearing(lat1, lon1, lat2, lon2)`

## Track Verification

### track_error

Distance between observed and modeled positions — the standard track-forecast
error metric.

```python
from monet_stats import track_error

obs_lat = np.array([20.1, 21.5, 23.0])
obs_lon = np.array([-90.2, -88.0, -85.1])
mod_lat = np.array([20.4, 21.9, 23.6])
mod_lon = np.array([-89.8, -87.4, -84.2])

err_km = track_error(obs_lat, obs_lon, mod_lat, mod_lon)
print(f"Mean track error: {err_km.mean():.1f} km")
```

**Signature:** `track_error(obs_lat, obs_lon, mod_lat, mod_lon)`

### along_track_error / cross_track_error

Decompose the position error into the component parallel to the storm's
motion (speed error) and the component perpendicular to it (steering error).
Both take the current and previous observed/ modeled positions.

```python
from monet_stats import along_track_error, cross_track_error

along = along_track_error(obs_lat, obs_lon, mod_lat, mod_lon,
                          prev_obs_lat, prev_obs_lon)
cross = cross_track_error(obs_lat, obs_lon, mod_lat, mod_lon,
                          prev_obs_lat, prev_obs_lon)
```

**Signatures:**

```text
along_track_error(obs_lat, obs_lon, mod_lat, mod_lon, prev_obs_lat, prev_obs_lon)
cross_track_error(obs_lat, obs_lon, mod_lat, mod_lon, prev_obs_lat, prev_obs_lon)
```

### translation_speed

Speed of movement between consecutive positions.

```python
from monet_stats import translation_speed

speed = translation_speed(obs_lat, obs_lon, time=obs_time, dim="time")
```

**Signature:** `translation_speed(lat, lon, time=None, dim='time')`

## Storm-Center Detection

### find_storm_center

Locate the single storm center (e.g., minimum MSLP or maximum wind speed) in a
field, returning the center coordinates.

```python
from monet_stats import find_storm_center

center = find_storm_center(mslp_da, lat_dim="lat", lon_dim="lon", method="min")
```

**Signature:**
`find_storm_center(data, lat_dim='lat', lon_dim='lon', method='min', dim=None)`

- `method`: `"min"` (e.g., lowest pressure) or `"max"` (e.g., strongest wind).

### find_storm_centers

Detect multiple local-extrema centers in a field, returning a boolean mask
where `True` marks a detected center.

```python
from monet_stats import find_storm_centers

mask = find_storm_centers(field, method="min", window_size=5, threshold=None)
```

**Signature:**
`find_storm_centers(data, lat_dim='lat', lon_dim='lon', method='min', window_size=5, threshold=None)`

## Accessor Methods

On an xarray `DataArray`, the same operations are available through the
`.monet_stats` accessor: `track_error`, `along_track_error`,
`cross_track_error`, `translation_speed`, `find_storm_center`, and
`find_storm_centers`.

## API Reference

::: monet_stats.track_metrics
