"""
Core interfaces and base classes for the Monet Stats package (Aero Protocol Compliant).

This module defines the core interfaces, base classes, and validation framework
for the statistical functions in the Monet Stats package.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Tuple, Union

import numpy as np
import xarray as xr

from .utils_stats import _update_history


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
        **kwargs: Any,
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Compute the statistical metric.

        Parameters
        ----------
        obs : numpy.ndarray or xarray.DataArray
            Observed values.
        mod : numpy.ndarray or xarray.DataArray
            Model/predicted values.
        **kwargs : Any
            Additional parameters specific to the metric.

        Returns
        -------
        float or numpy.ndarray or xarray.DataArray
            The computed metric value(s).
        """

    @abstractmethod
    def validate_inputs(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs: Any,
    ) -> bool:
        """
        Validate input parameters for the metric.

        Parameters
        ----------
        obs : numpy.ndarray or xarray.DataArray
            Observed values.
        mod : numpy.ndarray or xarray.DataArray
            Model/predicted values.
        **kwargs : Any
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

    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.description = self.__doc__ or "Statistical metric"

    def validate_inputs(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs: Any,
    ) -> bool:
        """
        Validate input parameters for the metric (Aero Protocol: Lazy-friendly).

        Parameters
        ----------
        obs : numpy.ndarray or xarray.DataArray
            Observed values.
        mod : numpy.ndarray or xarray.DataArray
            Model/predicted values.
        **kwargs : Any
            Additional parameters specific to the metric.

        Returns
        -------
        bool
            True if inputs are valid.

        Raises
        ------
        TypeError
            If inputs are not numpy arrays or xarray DataArrays.
        ValueError
            If shapes mismatch or no finite values are present.
        """
        # Check if inputs are arrays or xarray DataArrays
        if not (isinstance(obs, (np.ndarray, xr.DataArray)) and isinstance(mod, (np.ndarray, xr.DataArray))):
            raise TypeError("obs and mod must be numpy arrays or xarray DataArrays")

        # Check if shapes match (lazy-safe as shape is metadata)
        if hasattr(obs, "shape") and hasattr(mod, "shape"):
            if obs.shape != mod.shape:
                try:
                    np.broadcast_shapes(obs.shape, mod.shape)
                except ValueError:
                    raise ValueError(f"obs and mod must have compatible shapes, got {obs.shape} and {mod.shape}")

        # Note: We avoid .values here to maintain laziness.
        # Deep data validation (like checking for NaNs) is deferred to the computation phase
        # to avoid premature Dask triggers.
        return True

    def _handle_xarray(
        self,
        obs: xr.DataArray,
        mod: xr.DataArray,
        func: Callable,
        axis: Optional[Union[int, str, Iterable[Union[int, str]]]] = None,
        **kwargs: Any,
    ) -> xr.DataArray:
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
        axis : int, str, or iterable, optional
            Axis or dimension along which to compute.
        **kwargs : Any
            Additional parameters for the function.

        Returns
        -------
        xarray.DataArray
            Result of applying the function with history updated.
        """
        from .data_processing import align_arrays

        obs, mod = align_arrays(obs, mod)

        if axis is not None:
            # Handle axis parameter for xarray
            if isinstance(axis, int):
                dim = obs.dims[axis]
            else:
                dim = axis
            res = func(obs, mod, dim=dim, **kwargs)
        else:
            res = func(obs, mod, **kwargs)

        return _update_history(res, self.name)

    def _handle_numpy(
        self,
        obs: np.ndarray,
        mod: np.ndarray,
        func: Callable,
        axis: Optional[Union[int, Iterable[int]]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, float]:
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
        axis : int or iterable of int, optional
            Axis along which to compute.
        **kwargs : Any
            Additional parameters for the function.

        Returns
        -------
        numpy.ndarray or float
            Result of applying the function.
        """
        return func(obs, mod, axis=axis, **kwargs)

    def _handle_masked_arrays(
        self,
        obs: np.ndarray,
        mod: np.ndarray,
        func: Callable,
        axis: Optional[Union[int, Iterable[int]]] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, float]:
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
        axis : int or iterable of int, optional
            Axis along which to compute.
        **kwargs : Any
            Additional parameters for the function.

        Returns
        -------
        numpy.ndarray or float
            Result of applying the function.
        """
        obs_ma = np.ma.masked_invalid(obs)
        mod_ma = np.ma.masked_invalid(mod)
        return func(obs_ma, mod_ma, axis=axis, **kwargs)


class DataProcessor:
    """
    Data processing utilities (Legacy wrapper for data_processing module).
    """

    @staticmethod
    def to_numpy(data: Any) -> np.ndarray:
        """
        Convert data to numpy array (Triggers compute).

        Parameters
        ----------
        data : Any
            Input data.

        Returns
        -------
        numpy.ndarray
            Converted numpy array.
        """
        from .data_processing import to_numpy

        return to_numpy(data)

    @staticmethod
    def align_arrays(
        obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray]
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        Align two arrays for comparison.

        Parameters
        ----------
        obs : numpy.ndarray or xarray.DataArray
            Observed values.
        mod : numpy.ndarray or xarray.DataArray
            Model/predicted values.

        Returns
        -------
        tuple
            Aligned obs and mod arrays.
        """
        from .data_processing import align_arrays

        return align_arrays(obs, mod)

    @staticmethod
    def handle_missing_values(
        obs: Union[np.ndarray, xr.DataArray], mod: Union[np.ndarray, xr.DataArray], strategy: str = "pairwise"
    ) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
        """
        Handle missing values in arrays.

        Parameters
        ----------
        obs : numpy.ndarray or xarray.DataArray
            Observed values.
        mod : numpy.ndarray or xarray.DataArray
            Model/predicted values.
        strategy : str, optional
            Strategy for handling missing values ('pairwise', 'listwise').

        Returns
        -------
        tuple
            Arrays with missing values handled according to strategy.
        """
        from .data_processing import handle_missing_values

        return handle_missing_values(obs, mod, strategy=strategy)


class PerformanceOptimizer:
    """
    Performance optimization utilities (Legacy wrapper for performance module).
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
        from .performance import chunk_array

        return chunk_array(arr, chunk_size=chunk_size)

    @staticmethod
    def vectorize_function(func: Callable, *args: Any, **kwargs: Any) -> Any:
        """
        Apply function in a vectorized manner.

        Parameters
        ----------
        func : callable
            Function to vectorize.
        *args : Any
            Arguments to pass to function.
        **kwargs : Any
            Keyword arguments to pass to function.

        Returns
        -------
        Any
            Result of vectorized function application.
        """
        from .performance import vectorize_function

        return vectorize_function(func, *args, **kwargs)


class PluginInterface(ABC):
    """
    Interface for creating custom statistical metrics as plugins.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the metric.

        Returns
        -------
        str
            Metric name.
        """

    @abstractmethod
    def compute(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs: Any,
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Compute the custom metric.

        Parameters
        ----------
        obs : numpy.ndarray or xarray.DataArray
            Observed values.
        mod : numpy.ndarray or xarray.DataArray
            Model/predicted values.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        float or numpy.ndarray or xarray.DataArray
            Computed metric value(s).
        """

    @abstractmethod
    def description(self) -> str:
        """
        Return the description of the metric.

        Returns
        -------
        str
            Metric description.
        """

    @abstractmethod
    def validate_inputs(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs: Any,
    ) -> bool:
        """
        Validate inputs for the metric.

        Parameters
        ----------
        obs : numpy.ndarray or xarray.DataArray
            Observed values.
        mod : numpy.ndarray or xarray.DataArray
            Model/predicted values.
        **kwargs : Any
            Additional parameters.

        Returns
        -------
        bool
            True if inputs are valid, False otherwise.
        """
