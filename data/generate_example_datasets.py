"""
Generate example datasets for Monet Stats demonstration notebooks.

This script creates realistic synthetic climate datasets for various atmospheric variables
and model-observation pairs that can be used in the example notebooks.
"""

import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import xarray as xr


def generate_temperature_data(n_years: int = 10, n_stations: int = 10, n_ensemble_members: int = 10) -> Dict[str, Any]:
    """Generate synthetic temperature data for model-observation comparison."""

    # Time dimension
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010 + n_years, 1, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    # Station coordinates
    station_ids = [f"STN{i:03d}" for i in range(1, n_stations + 1)]
    latitudes = np.random.uniform(30, 50, n_stations)  # Mid-latitude region
    longitudes = np.random.uniform(-120, -70, n_stations)  # North America

    # Generate synthetic temperature data with realistic patterns
    base_temp = 15.0  # Base temperature in Celsius
    seasonal_amplitude = 10.0  # Seasonal variation

    # Create seasonal cycle
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal_cycle = seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)

    # Initialize arrays
    observed_temps = np.zeros((n_days, n_stations))
    modeled_temps = np.zeros((n_days, n_stations))

    for i in range(n_stations):
        # Add spatial and temporal variability
        station_bias = np.random.normal(0, 2)  # Station-specific bias
        noise = np.random.normal(0, 1.5, n_days)  # Random noise

        # True observed values with seasonal cycle
        observed_temps[:, i] = base_temp + seasonal_cycle + station_bias + noise

        # Modeled values with systematic bias and skill
        model_bias = np.random.normal(-0.5, 0.5)  # Model bias
        model_noise = np.random.normal(0, 1.2, n_days)  # Model-specific noise
        correlation = np.random.uniform(0.7, 0.9)  # Model-obs correlation

        # Create correlated model values
        modeled_temps[:, i] = (
            base_temp
            + seasonal_cycle
            + station_bias
            + model_bias
            + correlation * noise
            + (1 - correlation) * model_noise
        )

    # Create ensemble forecasts
    ensemble_forecasts = np.zeros((n_days, n_stations, n_ensemble_members))
    for i in range(n_ensemble_members):
        ensemble_bias = np.random.normal(-0.2, 0.3)  # Ensemble member bias
        ensemble_noise = np.random.normal(0, np.random.uniform(0.8, 1.8), (n_days, n_stations))

        ensemble_forecasts[:, :, i] = modeled_temps + ensemble_bias + ensemble_noise

    # Create DataFrames
    temp_df = pd.DataFrame(
        {
            "date": np.repeat(dates, n_stations),
            "station_id": station_ids * n_days,
            "latitude": np.tile(latitudes, n_days),
            "longitude": np.tile(longitudes, n_days),
            "observed_temp": observed_temps.flatten(),
            "modeled_temp": modeled_temps.flatten(),
        }
    )

    return {
        "temp_df": temp_df,
        "observed_temps": observed_temps,
        "modeled_temps": modeled_temps,
        "ensemble_forecasts": ensemble_forecasts,
        "dates": dates,
        "stations": station_ids,
        "latitudes": latitudes,
        "longitudes": longitudes,
    }


def generate_precipitation_data(n_years: int = 10, n_stations: int = 10) -> Dict[str, Any]:
    """Generate synthetic precipitation data for contingency analysis."""

    start_date = datetime(2010, 1, 1)
    end_date = datetime(2010 + n_years, 1, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_days = len(dates)

    station_ids = [f"STN{i:03d}" for i in range(1, n_stations + 1)]
    latitudes = np.random.uniform(30, 50, n_stations)
    longitudes = np.random.uniform(-120, -70, n_stations)

    # Generate precipitation data (binary and continuous)
    # Use gamma distribution for realistic precipitation patterns
    shape = 0.5  # Shape parameter for gamma distribution
    scale = 2.0  # Scale parameter for gamma distribution

    observed_precip = np.zeros((n_days, n_stations))
    modeled_precip = np.zeros((n_days, n_stations))

    for i in range(n_stations):
        # Base precipitation rate
        base_rate = np.random.uniform(0.5, 2.0)

        # Seasonal variation
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * np.array([d.timetuple().tm_yday for d in dates]) / 365.25)

        # Generate daily precipitation
        daily_precip = np.random.gamma(shape, scale, n_days) * seasonal_factor * base_rate
        # Some days will have no precipitation
        no_precip_prob = 0.7  # 70% of days have no rain
        daily_precip[np.random.random(n_days) < no_precip_prob] = 0

        observed_precip[:, i] = daily_precip

        # Modeled precipitation with systematic differences
        model_scale = np.random.uniform(0.8, 1.2)  # Model scaling factor
        model_noise = np.random.exponential(0.1, n_days)  # Additional model uncertainty

        modeled_precip[:, i] = daily_precip * model_scale + model_noise
        # Ensure no negative precipitation
        modeled_precip[:, i] = np.maximum(modeled_precip[:, i], 0)

    # Create binary precipitation data (0 = no rain, 1 = rain)
    threshold = 0.1  # Precipitation threshold for "rain" event
    obs_binary = (observed_precip > threshold).astype(int)
    mod_binary = (modeled_precip > threshold).astype(int)

    precip_df = pd.DataFrame(
        {
            "date": np.repeat(dates, n_stations),
            "station_id": station_ids * n_days,
            "latitude": np.tile(latitudes, n_days),
            "longitude": np.tile(longitudes, n_days),
            "observed_precip": observed_precip.flatten(),
            "modeled_precip": modeled_precip.flatten(),
            "obs_binary_precip": obs_binary.flatten(),
            "mod_binary_precip": mod_binary.flatten(),
        }
    )

    return {
        "precip_df": precip_df,
        "observed_precip": observed_precip,
        "modeled_precip": modeled_precip,
        "obs_binary": obs_binary,
        "mod_binary": mod_binary,
        "dates": dates,
        "stations": station_ids,
        "latitudes": latitudes,
        "longitudes": longitudes,
    }


def generate_wind_data(n_years: int = 5, n_stations: int = 5) -> Dict[str, Any]:
    """Generate synthetic wind data with direction and speed."""

    start_date = datetime(2015, 1, 1)
    end_date = datetime(2015 + n_years, 1, 1)
    dates = pd.date_range(start=start_date, end=end_date, freq="H")  # Hourly data
    n_hours = len(dates)

    station_ids = [f"STN{i:03d}" for i in range(1, n_stations + 1)]
    latitudes = np.random.uniform(30, 50, n_stations)
    longitudes = np.random.uniform(-120, -70, n_stations)

    # Generate wind speed and direction
    observed_wind_speed = np.zeros((n_hours, n_stations))
    modeled_wind_speed = np.zeros((n_hours, n_stations))
    observed_wind_dir = np.zeros((n_hours, n_stations))
    modeled_wind_dir = np.zeros((n_hours, n_stations))

    for i in range(n_stations):
        # Base wind conditions
        base_speed = np.random.uniform(5, 15)  # m/s
        base_dir = np.random.uniform(0, 360)  # degrees

        # Diurnal and seasonal variations
        hour_of_day = np.array([d.hour for d in dates])
        diurnal_cycle = 2 * np.sin(2 * np.pi * hour_of_day / 24)  # Diurnal variation

        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        seasonal_cycle = 3 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)  # Seasonal variation

        # Generate wind speed with realistic patterns
        speed_noise = np.random.lognormal(0, 0.3, n_hours)  # Lognormal for positive values
        observed_wind_speed[:, i] = (
            base_speed + diurnal_cycle + seasonal_cycle + speed_noise * np.random.uniform(0.5, 1.5)
        )
        observed_wind_speed[:, i] = np.maximum(observed_wind_speed[:, i], 0)  # Ensure positive

        # Generate wind direction with circular statistics
        dir_noise = np.random.vonmises(0, 1, n_hours)  # Von Mises for circular data
        observed_wind_dir[:, i] = (base_dir + dir_noise * 30) % 360  # Apply noise in degrees

        # Modeled wind with systematic differences
        speed_bias = np.random.normal(0, 1)  # Speed bias
        dir_bias = np.random.normal(0, 10)  # Direction bias
        correlation = np.random.uniform(0.6, 0.8)  # Speed correlation

        # Correlated speed
        modeled_wind_speed[:, i] = (
            base_speed
            + diurnal_cycle
            + seasonal_cycle
            + speed_bias
            + correlation * speed_noise * np.random.uniform(0.5, 1.5)
            + (1 - correlation) * np.random.lognormal(0, 0.2, n_hours)
        )
        modeled_wind_speed[:, i] = np.maximum(modeled_wind_speed[:, i], 0)

        # Correlated direction
        dir_corr_noise = correlation * dir_noise * 30
        dir_uncorr_noise = (1 - correlation) * np.random.vonmises(0, 0.7, n_hours) * 20
        modeled_wind_dir[:, i] = (base_dir + dir_bias + dir_corr_noise + dir_uncorr_noise) % 360

    wind_df = pd.DataFrame(
        {
            "date": np.repeat(dates, n_stations),
            "station_id": station_ids * n_hours,
            "latitude": np.tile(latitudes, n_hours),
            "longitude": np.tile(longitudes, n_hours),
            "observed_wind_speed": observed_wind_speed.flatten(),
            "modeled_wind_speed": modeled_wind_speed.flatten(),
            "observed_wind_dir": observed_wind_dir.flatten(),
            "modeled_wind_dir": modeled_wind_dir.flatten(),
        }
    )

    return {
        "wind_df": wind_df,
        "observed_wind_speed": observed_wind_speed,
        "modeled_wind_speed": modeled_wind_speed,
        "observed_wind_dir": observed_wind_dir,
        "modeled_wind_dir": modeled_wind_dir,
        "dates": dates,
        "stations": station_ids,
        "latitudes": latitudes,
        "longitudes": longitudes,
    }


def generate_spatial_data() -> Dict[str, Any]:
    """Generate synthetic spatial data for spatial verification metrics."""

    # Define spatial grid
    lat_range = (30, 50)
    lon_range = (-120, -70)
    n_lat = 50
    n_lon = 80

    lats = np.linspace(lat_range[0], lat_range[1], n_lat)
    lons = np.linspace(lon_range[0], lon_range[1], n_lon)

    # Time dimension
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    n_days = len(dates)

    # Generate spatial fields with realistic patterns
    observed_field = np.zeros((n_days, n_lat, n_lon))
    modeled_field = np.zeros((n_days, n_lat, n_lon))

    for t in range(n_days):
        # Create spatially correlated field using Gaussian random fields
        # Add a large-scale pattern
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

        # Large-scale pattern (e.g., temperature gradient)
        large_scale = 15 + 0.5 * (lat_grid - lat_range[0]) - 0.1 * (lon_grid - lon_range[0])

        # Add spatially correlated noise
        # Use a simple approach with spatial correlation
        noise_field = np.random.normal(0, 2, (n_lat, n_lon))
        # Apply Gaussian smoothing for spatial correlation
        from scipy import ndimage

        correlated_noise = ndimage.gaussian_filter(noise_field, sigma=2.0)

        observed_field[t] = large_scale + correlated_noise

        # Modeled field with systematic differences
        model_bias = np.random.normal(-0.5, 1.0)  # Spatially uniform bias
        model_noise = ndimage.gaussian_filter(np.random.normal(0, 1.5, (n_lat, n_lon)), sigma=1.5)
        correlation = np.random.uniform(0.7, 0.9)

        modeled_field[t] = large_scale + model_bias + correlation * correlated_noise + (1 - correlation) * model_noise

    # Create xarray datasets
    obs_da = xr.DataArray(
        observed_field,
        dims=["time", "lat", "lon"],
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={
            "units": "°C",
            "long_name": "Observed Temperature",
            "description": "Synthetic observed temperature field",
        },
    )

    mod_da = xr.DataArray(
        modeled_field,
        dims=["time", "lat", "lon"],
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={
            "units": "°C",
            "long_name": "Modeled Temperature",
            "description": "Synthetic modeled temperature field",
        },
    )

    return {
        "observed_da": obs_da,
        "modeled_da": mod_da,
        "lats": lats,
        "lons": lons,
        "dates": dates,
    }


def save_datasets() -> None:
    """Generate and save all example datasets."""

    print("Generating temperature dataset...")
    temp_data = generate_temperature_data()
    temp_data["temp_df"].to_csv("data/temperature_obs_mod.csv", index=False)

    print("Generating precipitation dataset...")
    precip_data = generate_precipitation_data()
    precip_data["precip_df"].to_csv("data/precipitation_obs_mod.csv", index=False)

    print("Generating wind dataset...")
    wind_data = generate_wind_data()
    wind_data["wind_df"].to_csv("data/wind_obs_mod.csv", index=False)

    print("Generating spatial dataset...")
    spatial_data = generate_spatial_data()
    spatial_data["observed_da"].to_netcdf("data/spatial_obs.nc")
    spatial_data["modeled_da"].to_netcdf("data/spatial_mod.nc")

    print("All datasets generated and saved successfully!")

    # Create summary file
    summary = f"""
Dataset Summary:
- Temperature: {len(temp_data['dates'])} days, {len(temp_data['stations'])} stations
- Precipitation: {len(precip_data['dates'])} days, {len(precip_data['stations'])} stations
- Wind: {len(wind_data['dates'])} hours, {len(wind_data['stations'])} stations
- Spatial: {len(spatial_data['dates'])} days, {len(spatial_data['lats'])}x{len(spatial_data['lons'])} grid

Files created:
- data/temperature_obs_mod.csv
- data/precipitation_obs_mod.csv
- data/wind_obs_mod.csv
- data/spatial_obs.nc
- data/spatial_mod.nc
"""

    with open("data/dataset_summary.txt", "w") as f:
        f.write(summary)

    print(summary)


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    save_datasets()
