"""
Plugin system architecture for extending statistical metrics (Aero Protocol Compliant).

This module provides the infrastructure to register and execute custom statistical
metrics as plugins, maintaining consistency with the core Monet Stats API.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from .interfaces import PluginInterface
from .utils_stats import _update_history


class PluginManager:
    """
    Manager for registering and executing statistical metric plugins.

    The PluginManager allows users to extend the library with custom metrics
    that follow the same interface as built-in metrics.
    """

    def __init__(self) -> None:
        """Initialize the PluginManager with an empty plugin registry."""
        self._plugins: Dict[str, PluginInterface] = {}

    def register_plugin(self, plugin: PluginInterface) -> None:
        """
        Register a new statistical metric plugin.

        Parameters
        ----------
        plugin : PluginInterface
            The plugin instance to register.
        """
        self._plugins[plugin.name()] = plugin

    def register(self, name: str, description: str) -> Callable[[Callable], Callable]:
        """
        Decorator to register a function as a custom metric plugin.

        Parameters
        ----------
        name : str
            Name of the metric.
        description : str
            Description of the metric.

        Returns
        -------
        Callable[[Callable], Callable]
            The decorator function.
        """

        def decorator(func: Callable) -> Callable:
            plugin = CustomMetric(name, description, func)
            self.register_plugin(plugin)
            return func

        return decorator

    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin by name.

        Parameters
        ----------
        name : str
            Name of the plugin to remove from the registry.
        """
        if name in self._plugins:
            del self._plugins[name]

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """
        Retrieve a registered plugin by its name.

        Parameters
        ----------
        name : str
            The name of the plugin to retrieve.

        Returns
        -------
        Optional[PluginInterface]
            The plugin instance if found, otherwise None.
        """
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """
        List the names of all currently registered plugins.

        Returns
        -------
        List[str]
            A list of registered plugin names.
        """
        return list(self._plugins.keys())

    def compute_metric(
        self,
        name: str,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs: Any,
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Execute a registered plugin to compute a metric.

        Parameters
        ----------
        name : str
            Name of the registered plugin to execute.
        obs : Union[np.ndarray, xr.DataArray]
            Observed values.
        mod : Union[np.ndarray, xr.DataArray]
            Model/predicted values.
        **kwargs : Any
            Additional keyword arguments for the metric computation.

        Returns
        -------
        Union[float, np.ndarray, xr.DataArray]
            The computed metric result.

        Raises
        ------
        ValueError
            If the plugin name is not registered or if input validation fails.
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            raise ValueError(f"Plugin '{name}' not found in the registry.")

        if not plugin.validate_inputs(obs, mod, **kwargs):
            raise ValueError(f"Input validation failed for plugin '{name}'.")

        return plugin.compute(obs, mod, **kwargs)


class CustomMetric(PluginInterface):
    """
    Wrapper for user-defined functions to act as statistical plugins.

    This class implements the PluginInterface, allowing arbitrary functions
    to be integrated into the Monet Stats ecosystem with proper
    backend handling and provenance tracking.
    """

    def __init__(self, name: str, description: str, func: Callable) -> None:
        """
        Initialize the CustomMetric.

        Parameters
        ----------
        name : str
            Human-readable name of the metric.
        description : str
            Detailed description of what the metric calculates.
        func : Callable
            The underlying computation function.
        """
        self._name = name
        self._description = description
        self._func = func

    def name(self) -> str:
        """
        Return the metric name.

        Returns
        -------
        str
            The metric name.
        """
        return self._name

    def description(self) -> str:
        """
        Return the metric description.

        Returns
        -------
        str
            The metric description.
        """
        return self._description

    def compute(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs: Any,
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Compute the custom metric with Aero Protocol enhancements.

        This implementation ensures that if Xarray DataArrays are provided,
        they are aligned, and the scientific history is updated.

        Parameters
        ----------
        obs : Union[np.ndarray, xr.DataArray]
            Observed values.
        mod : Union[np.ndarray, xr.DataArray]
            Model values.
        **kwargs : Any
            Additional arguments passed to the underlying function.

        Returns
        -------
        Union[float, np.ndarray, xr.DataArray]
            The computed metric.
        """
        if isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
            # Alignment is already handled by stats() usually, but kept for direct calls
            from .data_processing import align_arrays

            obs, mod = align_arrays(obs, mod)
            res = self._func(obs, mod, **kwargs)
            return _update_history(res, self._name)

        return self._func(obs, mod, **kwargs)

    def validate_inputs(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs: Any,
    ) -> bool:
        """
        Validate inputs for compatibility (metadata-only check for laziness).

        Parameters
        ----------
        obs : Union[np.ndarray, xr.DataArray]
            Observed values.
        mod : Union[np.ndarray, xr.DataArray]
            Model values.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        bool
            True if inputs are compatible.
        """
        if not (isinstance(obs, (np.ndarray, xr.DataArray)) and isinstance(mod, (np.ndarray, xr.DataArray))):
            return False

        if hasattr(obs, "shape") and hasattr(mod, "shape"):
            try:
                np.broadcast_shapes(obs.shape, mod.shape)
            except ValueError:
                return False

        return True


class ExampleMetrics:
    """
    Standard implementations of extended metrics as plugins.
    """

    @staticmethod
    def wmape_plugin() -> CustomMetric:
        """
        Create a Weighted Mean Absolute Percentage Error (WMAPE) plugin.

        WMAPE = (sum(|mod - obs|) / sum(|obs|)) * 100

        Returns
        -------
        CustomMetric
            A configured WMAPE plugin.
        """

        def wmape_func(
            obs: Union[np.ndarray, xr.DataArray],
            mod: Union[np.ndarray, xr.DataArray],
            axis: Optional[Union[int, str]] = None,
        ) -> Union[float, np.ndarray, xr.DataArray]:
            # Xarray/Dask friendly implementation with zero handling
            if isinstance(obs, xr.DataArray):
                numerator = (np.abs(mod - obs)).sum(dim=axis)
                denominator = (np.abs(obs)).sum(dim=axis)
                res = (numerator / denominator.where(denominator != 0, np.nan)) * 100.0
            else:
                numerator = np.sum(np.abs(mod - obs), axis=axis)
                denominator = np.sum(np.abs(obs), axis=axis)
                with np.errstate(divide="ignore", invalid="ignore"):
                    res = (numerator / np.where(denominator != 0, denominator, np.nan)) * 100.0
            return res

        return CustomMetric(
            name="WMAPE",
            description="Weighted Mean Absolute Percentage Error",
            func=wmape_func,
        )

    @staticmethod
    def mape_bias_plugin() -> CustomMetric:
        """
        Create a MAPE Bias plugin.

        Calculates the difference between average positive and average negative percentage errors.

        Returns
        -------
        CustomMetric
            A configured MAPE Bias plugin.
        """

        def mape_bias_func(
            obs: Union[np.ndarray, xr.DataArray],
            mod: Union[np.ndarray, xr.DataArray],
            axis: Optional[Union[int, str]] = None,
        ) -> Union[float, np.ndarray, xr.DataArray]:
            # Avoid zero division in pe
            if isinstance(obs, xr.DataArray):
                abs_obs = np.abs(obs)
                # Ensure we have NaNs where obs is zero to exclude them from mean correctly
                pe = (mod - obs) / abs_obs.where(abs_obs != 0)
                # Preserve NaNs during positive/negative splitting
                pos_pe = xr.where(pe >= 0, pe, 0).where(~np.isnan(pe))
                neg_pe = xr.where(pe < 0, np.abs(pe), 0).where(~np.isnan(pe))
                pos_errors = pos_pe.mean(dim=axis)
                neg_errors = neg_pe.mean(dim=axis)
            else:
                abs_obs = np.abs(obs)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pe = (mod - obs) / np.where(abs_obs != 0, abs_obs, np.nan)
                # Use nanmean to ignore points where pe is NaN (i.e. obs was 0)
                pos_errors = np.nanmean(np.where(np.isnan(pe), np.nan, np.where(pe >= 0, pe, 0)), axis=axis)
                neg_errors = np.nanmean(np.where(np.isnan(pe), np.nan, np.where(pe < 0, np.abs(pe), 0)), axis=axis)

            return pos_errors - neg_errors

        return CustomMetric(
            name="MAPE_Bias",
            description="MAPE Bias - difference between positive and negative percentage errors",
            func=mape_bias_func,
        )


# Global plugin manager instance
plugin_manager = PluginManager()


def register_builtin_plugins() -> None:
    """Register built-in example plugins to the global manager."""
    plugin_manager.register_plugin(ExampleMetrics.wmape_plugin())
    plugin_manager.register_plugin(ExampleMetrics.mape_bias_plugin())


# Initialize with built-in plugins
register_builtin_plugins()
