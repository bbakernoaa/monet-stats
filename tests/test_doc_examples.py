import doctest

import pytest

from monet_stats import (
    contingency_metrics,
    correlation_metrics,
    efficiency_metrics,
    error_metrics,
    relative_metrics,
    spatial_ensemble_metrics,
    utils_stats,
)


@pytest.mark.parametrize(
    "module",
    [
        contingency_metrics,
        correlation_metrics,
        efficiency_metrics,
        error_metrics,
        relative_metrics,
        spatial_ensemble_metrics,
        utils_stats,
    ],
)
def test_doctests(module):
    """Test all doctests in the given module."""
    # The `stats` module in the doctests is an alias for the module being tested.
    # We need to inject this alias into the global namespace for the doctests.
    globs = {"np": __import__("numpy")}

    # Run the doctests
    result = doctest.testmod(module, globs=globs, raise_on_error=True, verbose=True)

    # Check if any tests failed
    assert result.failed == 0, f"Doctest failed in {module.__name__} with {result.failed} failures."
