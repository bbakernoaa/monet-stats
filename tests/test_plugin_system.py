"""
Tests for plugin_system.py module.
"""

import numpy as np

from src.monet_stats.plugin_system import CustomMetric, PluginManager


class TestPluginSystem:
    """Test suite for the plugin system."""

    def test_plugin_manager(self):
        """Test the PluginManager class."""
        manager = PluginManager()

        def dummy_metric(obs, mod):
            return np.mean(obs - mod)

        plugin = CustomMetric(
            name="dummy", description="A dummy metric", func=dummy_metric
        )

        # Test registration
        manager.register_plugin(plugin)
        assert "dummy" in manager.list_plugins()

        # Test retrieval
        retrieved_plugin = manager.get_plugin("dummy")
        assert retrieved_plugin is not None
        assert retrieved_plugin.name() == "dummy"

        # Test unregistration
        manager.unregister_plugin("dummy")
        assert "dummy" not in manager.list_plugins()

    def test_custom_metric(self):
        """Test the CustomMetric class."""

        def dummy_metric(obs, mod):
            return np.mean(obs - mod)

        plugin = CustomMetric(
            name="dummy", description="A dummy metric", func=dummy_metric
        )

        obs = np.array([1, 2, 3])
        mod = np.array([2, 3, 4])

        # Test validation
        assert plugin.validate_inputs(obs, mod)
        assert not plugin.validate_inputs(obs, np.array([1, 2]))

        # Test computation
        result = plugin.compute(obs, mod)
        assert np.isclose(result, -1.0)
