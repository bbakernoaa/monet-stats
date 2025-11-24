"""
Plugin system architecture for extending statistical metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import xarray as xr


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


class PluginManager:
    """
    Manager for registering and executing plugins.
    """

    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}

    def register_plugin(self, plugin: PluginInterface) -> None:
        """
        Register a plugin.

        Parameters
        ----------
        plugin : PluginInterface
            Plugin to register.
        """
        self._plugins[plugin.name()] = plugin

    def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin by name.

        Parameters
        ----------
        name : str
            Name of the plugin to unregister.
        """
        if name in self._plugins:
            del self._plugins[name]

    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """
        Get a registered plugin by name.

        Parameters
        ----------
        name : str
            Name of the plugin to retrieve.

        Returns
        -------
        PluginInterface or None
            The requested plugin or None if not found.
        """
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """
        List all registered plugin names.

        Returns
        -------
        list of str
            Names of all registered plugins.
        """
        return list(self._plugins.keys())

    def compute_metric(
        self,
        name: str,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs,
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Compute a metric using a registered plugin.

        Parameters
        ----------
        name : str
            Name of the plugin to use.
        obs : array-like or xarray.DataArray
            Observed values.
        mod : array-like or xarray.DataArray
            Model/predicted values.
        **kwargs : dict
            Additional parameters for the metric.

        Returns
        -------
        float or array-like or xarray.DataArray
            Computed metric value(s).
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            raise ValueError(f"Plugin '{name}' not found")

        if not plugin.validate_inputs(obs, mod, **kwargs):
            raise ValueError(f"Invalid inputs for plugin '{name}'")

        return plugin.compute(obs, mod, **kwargs)


class CustomMetric(PluginInterface):
    """
    Example implementation of a custom metric plugin.
    """

    def __init__(self, name: str, description: str, func):
        self._name = name
        self._description = description
        self._func = func

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def compute(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs,
    ) -> Union[float, np.ndarray, xr.DataArray]:
        """
        Compute the custom metric using the provided function.
        """
        try:
            import xarray as xr
        except ImportError:
            xr = None

        if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
            obs, mod = xr.align(obs, mod, join="inner")
            return self._func(obs, mod, **kwargs)
        else:
            return self._func(obs, mod, **kwargs)

    def validate_inputs(
        self,
        obs: Union[np.ndarray, xr.DataArray],
        mod: Union[np.ndarray, xr.DataArray],
        **kwargs,
    ) -> bool:
        """
        Validate inputs for the custom metric.
        """
        # Check if inputs are arrays or xarray DataArrays
        if not (isinstance(obs, (np.ndarray, xr.DataArray)) and isinstance(mod, (np.ndarray, xr.DataArray))):
            return False

        # Check if shapes match
        if hasattr(obs, "shape") and hasattr(mod, "shape"):
            if obs.shape != mod.shape:
                return False

        return True


class ExampleMetrics:
    """
    Example implementations of statistical metrics as plugins.
    """

    @staticmethod
    def wmape_plugin():
        """Weighted Mean Absolute Percentage Error (WMAPE) as a plugin."""

        def wmape_func(obs, mod, axis=None):
            try:
                import xarray as xr
            except ImportError:
                xr = None

            if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
                obs, mod = xr.align(obs, mod, join="inner")
                numerator = (abs(mod - obs)).sum(dim=axis)
                denominator = (abs(obs)).sum(dim=axis)
                return (numerator / denominator) * 100.0
            else:
                numerator = np.sum(np.abs(mod - obs), axis=axis)
                denominator = np.sum(np.abs(obs), axis=axis)
                return (numerator / denominator) * 100.0

        return CustomMetric(
            name="WMAPE",
            description="Weighted Mean Absolute Percentage Error",
            func=wmape_func,
        )

    @staticmethod
    def mape_bias_plugin():
        """MAPE Bias as a plugin."""

        def mape_bias_func(obs, mod, axis=None):
            try:
                import xarray as xr
            except ImportError:
                xr = None

            if xr is not None and isinstance(obs, xr.DataArray) and isinstance(mod, xr.DataArray):
                obs, mod = xr.align(obs, mod, join="inner")
                positive_errors = ((mod >= obs) * abs(mod - obs) / abs(obs)).mean(dim=axis)
                negative_errors = ((mod < obs) * abs(mod - obs) / abs(obs)).mean(dim=axis)
                return positive_errors - negative_errors
            else:
                positive_mask = mod >= obs
                positive_errors = np.mean(
                    np.where(positive_mask, np.abs(mod - obs) / np.abs(obs), 0),
                    axis=axis,
                )
                negative_errors = np.mean(
                    np.where(~positive_mask, np.abs(mod - obs) / np.abs(obs), 0),
                    axis=axis,
                )
                return positive_errors - negative_errors

        return CustomMetric(
            name="MAPE_Bias",
            description="MAPE Bias - difference between positive and negative percentage errors",
            func=mape_bias_func,
        )


# Global plugin manager instance
plugin_manager = PluginManager()


def register_builtin_plugins():
    """Register built-in example plugins."""
    wmape_plugin = ExampleMetrics.wmape_plugin()
    mape_bias_plugin = ExampleMetrics.mape_bias_plugin()

    plugin_manager.register_plugin(wmape_plugin)
    plugin_manager.register_plugin(mape_bias_plugin)


# Initialize with built-in plugins
register_builtin_plugins()
