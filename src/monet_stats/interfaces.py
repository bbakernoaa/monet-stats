"""
Core interfaces and base classes for the Monet Stats package.

This module defines the core interfaces, base classes, and validation framework
for the statistical functions in the Monet Stats package.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import xarray as xr


class StatisticalMetric(ABC):
    """
    Abstract base class for all statistical metrics.

    This class defines the common interface for all statistical metrics
    in the Monet Stats package, ensuring consistency across different
    types of metrics (error, correlation, efficiency, etc.).
    """

    @abstractmethod
    def compute(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs,
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Compute the statistical metric.

        Parameters
        ----------
        obs : array-like or xarray.DataArray
            Observed values.
        mod : array-like or xarray.DataArray
            Model/predicted values.
        **kwargs : dict
            Additional parameters specific to the metric.

        Returns
        -------
        float or array-like or xarray.DataArray
            The computed metric value(s).
        """

    @abstractmethod
    def validate_inputs(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs,
    ) -> bool:
        """
        Validate input parameters for the metric.

        Parameters
        ----------
        obs : array-like or xarray.DataArray
            Observed values.
        mod : array-like or xarray.DataArray
            Model/predicted values.
        **kwargs : dict
            Additional parameters specific to the metric.

        Returns
        -------
        bool
            True if inputs are valid, False otherwise.
        """


class BaseStatisticalMetric(StatisticalMetric):
    """
    Base implementation for statistical metrics with common functionality.
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.description = self.__doc__ or "Statistical metric"

    def validate_inputs(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs,
    ) -> bool:
        """
        Validate input parameters for the metric.

        Parameters
        ----------
        obs : array-like or xarray.DataArray
            Observed values.
        mod : array-like or xarray.DataArray
            Model/predicted values.
        **kwargs : dict
            Additional parameters specific to the metric.

        Returns
        -------
        bool
            True if inputs are valid, False otherwise.
        """
        # Check if inputs are arrays or xarray DataArrays
        if not (
            isinstance(obs, (np.ndarray, xr.DataArray))
            and isinstance(mod, (np.ndarray, xr.DataArray))
        ):
            raise TypeError("obs and mod must be numpy arrays or xarray DataArrays")

        # Check if shapes match
        if hasattr(obs, "shape") and hasattr(mod, "shape"):
            if obs.shape != mod.shape:
                raise ValueError(
                    f"obs and mod must have the same shape, got {obs.shape} and {mod.shape}"
                )

        # Check for finite values
        obs_finite = (
            np.isfinite(obs) if isinstance(obs, np.ndarray) else np.isfinite(obs.values)
        )
        mod_finite = (
            np.isfinite(mod) if isinstance(mod, np.ndarray) else np.isfinite(mod.values)
        )

        if not np.any(obs_finite) or not np.any(mod_finite):
            raise ValueError("No finite values in obs or mod")

        return True

    def _handle_xarray(
        self, obs: xr.DataArray, mod: xr.DataArray, func, axis=None, **kwargs
    ):
        """
        Handle xarray DataArray inputs by aligning and applying function.

        Parameters
        ----------
        obs : xarray.DataArray
            Observed values.
        mod : xarray.DataArray
            Model/predicted values.
        func : callable
            Function to apply to the data.
        axis : int or str, optional
            Axis along which to compute.
        **kwargs : dict
            Additional parameters for the function.

        Returns
        -------
        xarray.DataArray
            Result of applying the function.
        """
        obs, mod = xr.align(obs, mod, join="inner")

        if axis is not None:
            # Handle axis parameter for xarray
            if isinstance(axis, int):
                dim = obs.dims[axis]
            else:
                dim = axis
            return func(obs, mod, dim=dim, **kwargs)
        else:
            return func(obs, mod, **kwargs)

    def _handle_numpy(
        self, obs: np.ndarray, mod: np.ndarray, func, axis=None, **kwargs
    ):
        """
        Handle numpy array inputs by applying function.

        Parameters
        ----------
        obs : numpy.ndarray
            Observed values.
        mod : numpy.ndarray
            Model/predicted values.
        func : callable
            Function to apply to the data.
        axis : int, optional
            Axis along which to compute.
        **kwargs : dict
            Additional parameters for the function.

        Returns
        -------
        numpy.ndarray or float
            Result of applying the function.
        """
        return func(obs, mod, axis=axis, **kwargs)

    def _handle_masked_arrays(
        self, obs: np.ndarray, mod: np.ndarray, func, axis=None, **kwargs
    ):
        """
        Handle masked array inputs by applying function.

        Parameters
        ----------
        obs : numpy.ndarray
            Observed values.
        mod : numpy.ndarray
            Model/predicted values.
        func : callable
            Function to apply to the data.
        axis : int, optional
            Axis along which to compute.
        **kwargs : dict
            Additional parameters for the function.

        Returns
        -------
        numpy.ndarray or float
            Result of applying the function.
        """
        # Use numpy masked array operations
        obs_ma = np.ma.masked_invalid(obs)
        mod_ma = np.ma.masked_invalid(mod)
        return func(obs_ma, mod_ma, axis=axis, **kwargs)


class DataProcessor:
    """
    Data processing utilities for handling different data formats.
    """

    @staticmethod
    def to_numpy(data: Union[np.ndarray, xr.DataArray, list]) -> np.ndarray:
        """
        Convert data to numpy array.

        Parameters
        ----------
        data : array-like or xarray.DataArray or list
            Input data.

        Returns
        -------
        numpy.ndarray
            Converted numpy array.
        """
        if isinstance(data, xr.DataArray):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        else:
            return np.asarray(data)

    @staticmethod
    def align_arrays(
        obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray]
    ) -> tuple:
        """
        Align two arrays for comparison.

        Parameters
        ----------
        obs : array-like or xarray.DataArray
            Observed values.
        mod : array-like or xarray.DataArray
            Model/predicted values.

        Returns
        -------
        tuple
            Aligned obs and mod arrays.
        """
        if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
            obs, mod = xr.align(obs, mod, join="inner")
            return obs, mod
        else:
            # For numpy arrays, ensure they have the same shape
            obs = DataProcessor.to_numpy(obs)
            mod = DataProcessor.to_numpy(mod)

            if obs.shape != mod.shape:
                raise ValueError(
                    f"Arrays must have the same shape, got {obs.shape} and {mod.shape}"
                )

            return obs, mod

    @staticmethod
    def handle_missing_values(
        obs: np.ndarray, mod: np.ndarray, strategy: str = "pairwise"
    ) -> tuple:
        """
        Handle missing values in arrays.

        Parameters
        ----------
        obs : numpy.ndarray
            Observed values.
        mod : numpy.ndarray
            Model/predicted values.
        strategy : str, optional
            Strategy for handling missing values ('pairwise', 'listwise').

        Returns
        -------
        tuple
            Arrays with missing values handled according to strategy.
        """
        if strategy == "pairwise":
            # Remove pairs where either value is NaN
            mask = ~(np.isnan(obs) | np.isnan(mod))
            return obs[mask], mod[mask]
        elif strategy == "listwise":
            # Remove all pairs if any value is NaN
            mask = ~(np.isnan(obs) | np.isnan(mod))
            return obs[mask], mod[mask]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class PerformanceOptimizer:
    """
    Performance optimization utilities for statistical computations.
    """

    @staticmethod
    def chunk_array(arr: np.ndarray, chunk_size: int = 1000000) -> list:
        """
        Split array into chunks for memory-efficient processing.

        Parameters
        ----------
        arr : numpy.ndarray
            Input array to chunk.
        chunk_size : int, optional
            Size of each chunk (number of elements).

        Returns
        -------
        list
            List of array chunks.
        """
        if arr.size <= chunk_size:
            return [arr]

        num_chunks = int(np.ceil(arr.size / chunk_size))
        chunks = np.array_split(arr, num_chunks)
        return chunks

    @staticmethod
    def vectorize_function(func, *args, **kwargs):
        """
        Apply function in a vectorized manner.

        Parameters
        ----------
        func : callable
            Function to vectorize.
        *args : tuple
            Arguments to pass to function.
        **kwargs : dict
            Keyword arguments to pass to function.

        Returns
        -------
        result
            Result of vectorized function application.
        """
        return np.vectorize(func)(*args, **kwargs)


class PluginInterface(ABC):
    """
    Interface for creating custom statistical metrics as plugins.
    """

    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""

    @abstractmethod
    def compute(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs,
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Compute the custom metric.

        Parameters
        ----------
        obs : array-like or xarray.DataArray
            Observed values.
        mod : array-like or xarray.DataArray
            Model/predicted values.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        float or array-like or xarray.DataArray
            Computed metric value(s).
        """

    @abstractmethod
    def description(self) -> str:
        """Return the description of the metric."""

    @abstractmethod
    def validate_inputs(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs,
    ) -> bool:
        """
        Validate inputs for the metric.

        Parameters
        ----------
        obs : array-like or xarray.DataArray
            Observed values.
        mod : array-like or xarray.DataArray
            Model/predicted values.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        bool
            True if inputs are valid, False otherwise.
        """
