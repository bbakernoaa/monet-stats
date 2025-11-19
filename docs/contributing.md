# Contributing to Monet Stats

We welcome contributions to the Monet Stats library! This guide provides instructions for developers who want to contribute new metrics, improvements, or bug fixes.

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment (recommended)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/noaa-oar-arl/monet-stats.git
cd monet-stats

# Create virtual environment
python -m venv monet-stats-dev
source monet-stats-dev/bin/activate  # Windows: monet-stats-dev\Scripts\activate

# Install development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

### Development Dependencies

The development setup includes:

- **Code Quality**: `black`, `flake8`, `isort`, `mypy`
- **Testing**: `pytest`, `pytest-cov`, `hypothesis`
- **Documentation**: `mkdocs`, `mkdocstrings`
- **Build**: `setuptools`, `wheel`

## Development Workflow

### 1. Making Changes

1. Create a new branch for your changes:

   ```bash
   git checkout -b feature/new-metric-name
   ```

2. Make your changes following the coding standards

3. Write tests for new functionality

4. Run tests to ensure everything works:

   ```bash
   pytest tests/ -v
   ```

5. Run code quality checks:
   ```bash
   black src/
   isort src/
   flake8 src/
   mypy src/
   ```

### 6. Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality standards. The hooks run automatically before each commit to check code formatting, linting, and other quality checks.

#### 6.1 Installing Pre-commit

To install the pre-commit hooks, run:

```bash
pip install pre-commit
pre-commit install
```

This will install the hooks in your local repository and they will run automatically on each commit.

#### 6.2 Running Pre-commit Manually

You can run the pre-commit hooks manually on all files:

```bash
pre-commit run --all-files
```

Or run on specific files:

```bash
pre-commit run --files file1.py file2.py
```

#### 6.3 Pre-commit Configuration

The pre-commit configuration is defined in `.pre-commit-config.yaml` and includes:

- **Black**: Code formatting following PEP 8 standards
- **Isort**: Import organization and sorting
- **Flake8**: Code linting
- **Pycodestyle**: Additional style checks
- **MyPy**: Static type checking
- **Gitlint**: Commit message formatting
- **Trailing whitespace removal**: Removes trailing whitespace
- **End-of-file fixer**: Ensures files end with a newline
- **Large file checker**: Prevents accidentally committing large files
- **Merge conflict checker**: Detects merge conflict markers
- **JSON/YAML validators**: Ensures configuration files are valid
- **Debug statement checker**: Prevents committing debug statements

### 2. Adding New Metrics

When adding a new statistical metric, follow these steps:

#### Create the Metric Function

```python
# src/monet_stats/new_metrics.py
import numpy as np
from typing import Union

def new_metric(obs: Union[np.ndarray, list],
               mod: Union[np.ndarray, list],
               **kwargs) -> float:
    """
    Calculate the new metric.

    Parameters
    ----------
    obs : array-like
        Observed values
    mod : array-like
        Modeled values
    **kwargs : dict
        Additional parameters

    Returns
    -------
    float
        The calculated metric value

    Raises
    ------
    ValueError
        If input arrays have different lengths or invalid parameters
    """
    obs = np.asarray(obs)
    mod = np.asarray(mod)

    if len(obs) != len(mod):
        raise ValueError("Observed and modeled arrays must have same length")

    # Implementation goes here
    result = np.mean((obs - mod) ** 2)  # Example implementation

    return float(result)
```

#### Add to `__init__.py`

Add the new metric to the imports in `src/monet_stats/__init__.py`:

```python
from .new_metrics import new_metric

__all__ = [
    # ... existing metrics ...
    "new_metric",
]
```

#### Write Tests

Create corresponding test file in `tests/`:

```python
# tests/test_new_metrics.py
import numpy as np
import pytest
from monet_stats import new_metric

def test_new_metric_basic():
    """Test basic functionality of new_metric"""
    obs = [1, 2, 3, 4, 5]
    mod = [1.1, 2.1, 2.9, 4.1, 4.8]

    result = new_metric(obs, mod)
    assert isinstance(result, float)
    assert result >= 0

def test_new_metric_nan_handling():
    """Test that new_metric handles NaN values correctly"""
    obs = [1, 2, np.nan, 4, 5]
    mod = [1.1, 2.1, 3.0, 4.1, 4.8]

    result = new_metric(obs, mod)
    assert not np.isnan(result)

def test_new_metric_edge_cases():
    """Test edge cases for new_metric"""
    # Empty arrays
    with pytest.raises(ValueError):
        new_metric([], [])

    # Different lengths
    with pytest.raises(ValueError):
        new_metric([1, 2], [1, 2, 3])
```

### 3. Testing Guidelines

#### Test Structure

- Use descriptive test function names
- Include tests for:
  - Basic functionality
  - Edge cases (empty arrays, NaN values)
  - Error conditions
  - Mathematical correctness

#### Test Coverage

- Aim for >95% test coverage
- Use `pytest-cov` to measure coverage:
  ```bash
  pytest --cov=src/monet_stats/new_metrics tests/test_new_metrics.py
  ```

#### Property-Based Testing

For complex metrics, consider property-based testing:

```python
# tests/test_new_metrics_property.py
import numpy as np
from hypothesis import given, strategies as st
from monet_stats import new_metric

@given(obs=st.lists(st.floats(min_value=0, max_value=100), min_size=10),
       mod=st.lists(st.floats(min_value=0, max_value=100), min_size=10))
def test_new_metric_properties(obs, mod):
    """Test mathematical properties of new_metric"""
    result = new_metric(obs, mod)

    # Test non-negative result
    assert result >= 0

    # Test symmetry (if applicable)
    if len(obs) == len(mod):
        result_swapped = new_metric(mod, obs)
        # Add specific properties based on metric characteristics
```

### 4. Code Standards

#### Formatting

- Use `black` for code formatting (line length: 88)
- Use `isort` for import sorting
- Follow PEP 8 style guidelines

#### Type Hints

- Add type hints to all functions
- Use `Union` types for flexible input acceptance
- Return appropriate types (typically `float` for scalar metrics)

#### Docstrings

Follow NumPy-style docstrings:

```python
def example_function(param1, param2=0):
    """
    Calculate example metric.

    Parameters
    ----------
    param1 : array-like
        Description of param1
    param2 : float, optional
        Description of param2 (default: 0)

    Returns
    -------
    float
        Description of return value

    Raises
    ------
    ValueError
        Description of error conditions
    """
    # Implementation
```

### 5. Documentation

#### API Documentation

- Add the new metric to the appropriate API documentation file
- Include mathematical formulation if applicable
- Provide usage examples

#### Mathematical Formulations

For metrics with mathematical foundations:

1. Add the formulation to `docs/math/overview.md`
2. Include definitions of all symbols
3. Provide interpretation guidelines

### 6. Performance Considerations

#### Vectorization

- Use NumPy vectorized operations instead of loops
- Avoid creating unnecessary temporary arrays

#### Memory Efficiency

- Process large data in chunks when possible
- Use `np.float32` instead of `np.float64` for large datasets
- Consider memory usage for ensemble operations

#### Benchmarking

Add performance benchmarks for new metrics:

```python
# benchmarks/benchmark_new_metric.py
import numpy as np
import timeit
from monet_stats import new_metric

def benchmark_new_metric():
    """Performance benchmark for new_metric"""
    sizes = [1000, 10000, 100000]

    for size in sizes:
        obs = np.random.normal(20, 5, size)
        mod = obs + np.random.normal(0, 2, size)

        time_taken = timeit.timeit(
            lambda: new_metric(obs, mod),
            number=100
        )

        print(f"Size: {size}, Time: {time_taken:.4f}s per call")

if __name__ == "__main__":
    benchmark_new_metric()
```

## Submitting Changes

### Pull Request Process

1. Ensure all tests pass
2. Run code quality checks
3. Update documentation
4. Create pull request with clear description

### Pull Request Template

```markdown
## Description

Brief description of changes made

## Changes Made

- [ ] Added new metric `new_metric`
- [ ] Updated documentation
- [ ] Added tests
- [ ] Fixed bug in existing function

## Testing

- [ ] All tests pass
- [ ] New functionality tested
- [ ] Edge cases covered

## Checklist

- [ ] Code follows style guidelines
- [ ] Type hints included
- [ ] Documentation updated
- [ ] Performance considered
- [ ] Breaking changes considered
```

### Code Review Guidelines

- Reviewers will check:
  - Code quality and style
  - Test coverage
  - Documentation quality
  - Performance implications
  - Mathematical correctness

## Release Process

### Version Bumping

Follow semantic versioning:

- **Patch**: Bug fixes (0.0.x)
- **Minor**: New features (0.x.0)
- **Major**: Breaking changes (x.0.0)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build and test documentation
5. Create release tag
6. Push to PyPI

## Community Guidelines

### Code of Conduct

- Be respectful and constructive
- Focus on technical merit
- Welcome diverse perspectives
- Follow NOAA's Code of Conduct

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: arl.webmaster@noaa.gov for private matters

### Getting Help

If you need help with development:

1. Check existing documentation and issues
2. Search GitHub discussions
3. Create a new issue with detailed description
4. Join community discussions

## Troubleshooting

### Common Development Issues

#### Test Failures

```bash
# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=src/monet_stats

# Run specific test
pytest tests/test_specific_module.py::test_function_name
```

#### Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall in development mode
pip install -e .
```

#### Documentation Building

```bash
# Install documentation dependencies
pip install mkdocs mkdocstrings-material

# Build documentation locally
mkdocs serve

# Build for production
mkdocs build
```

Thank you for contributing to Monet Stats! Your contributions help make this library a valuable resource for the atmospheric sciences community.
