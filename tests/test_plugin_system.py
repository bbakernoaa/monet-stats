"""Tests for the `plugin_system` module."""

import numpy as np

from monet_stats import plugin_system


def test_list_plugins():
    """Test the list_plugins function."""
    manager = plugin_system.plugin_manager
    plugin_system.register_builtin_plugins()
    plugins = manager.list_plugins()
    assert isinstance(plugins, list)


def test_get_plugin():
    """Test the get_plugin function."""
    manager = plugin_system.plugin_manager
    plugin_system.register_builtin_plugins()
    plugin = manager.get_plugin("WMAPE")
    assert plugin is not None


def test_register_plugin():
    """Test the register_plugin function."""
    manager = plugin_system.PluginManager()
    plugin = plugin_system.ExampleMetrics.wmape_plugin()
    manager.register_plugin(plugin)
    assert "WMAPE" in manager.list_plugins()


def test_unregister_plugin():
    """Test the unregister_plugin function."""
    manager = plugin_system.PluginManager()
    plugin = plugin_system.ExampleMetrics.wmape_plugin()
    manager.register_plugin(plugin)
    manager.unregister_plugin("WMAPE")
    assert "WMAPE" not in manager.list_plugins()


def test_compute_metric():
    """Test the compute_metric function."""
    manager = plugin_system.plugin_manager
    plugin_system.register_builtin_plugins()
    obs = np.array([1, 2, 3, 4])
    mod = np.array([1.1, 2.2, 3.3, 4.4])
    result = manager.compute_metric("WMAPE", obs, mod)
    assert np.isclose(result, 10.0)


def test_custom_metric():
    """Test the CustomMetric class."""

    def my_metric(obs, mod):
        return np.mean(obs - mod)

    plugin = plugin_system.CustomMetric("my_metric", "My custom metric", my_metric)
    manager = plugin_system.PluginManager()
    manager.register_plugin(plugin)
    obs = np.array([1, 2, 3, 4])
    mod = np.array([2, 2, 2, 2])
    result = manager.compute_metric("my_metric", obs, mod)
    assert np.isclose(result, 0.5)
