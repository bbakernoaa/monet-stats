#!/usr/bin/env python3
"""
CI/CD Test Script

This script runs all the quality checks that would be performed in the CI/CD pipeline
to ensure code quality standards are met before pushing to the repository.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return True if successful."""
    print(f"\n{description}")
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    try:
        result = subprocess.run(
            cmd,
            check=False, shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if result.returncode == 0:
            print(f"✓ {description} PASSED")
            return True
        else:
            print(f"✗ {description} FAILED")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {description} FAILED with exception: {e}")
        return False


def main():
    """Run all CI/CD quality checks."""
    print("Running CI/CD Quality Checks...")

    checks = [
        (["ruff", "check", "src/", "tests/"], "Ruff Linting"),
        (["ruff", "format", "--check", "src/", "tests/"], "Ruff Formatting"),
        (["black", "--check", "src/", "tests/"], "Black Formatting"),
        (["isort", "--check-only", "src/", "tests/"], "Import Sorting"),
        (["mypy", "src/", "tests/"], "MyPy Type Checking"),
        (["pytest", "--cov=src/monet_stats", "--cov-report=term-missing", "--cov-fail-under=95"], "Pytest Coverage"),
    ]

    all_passed = True

    for cmd, description in checks:
        if not run_command(cmd, description):
            all_passed = False

    print(f"\n{'='*50}")
    if all_passed:
        print("✓ All CI/CD checks PASSED!")
        return 0
    else:
        print("✗ Some CI/CD checks FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
