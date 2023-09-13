"""Tests for the minimum dependencies in README.rst and pyproject.toml"""


import os
import platform
import re
from pathlib import Path

import pytest

import sklearn
from sklearn._min_dependencies import dependent_packages
from sklearn.utils.fixes import parse_version


def test_min_dependencies_readme():
    # Test that the minimum dependencies in the README.rst file are
    # consistent with the minimum dependencies defined at the file:
    # sklearn/_min_dependencies.py

    if platform.python_implementation() == "PyPy":
        pytest.skip("PyPy does not always share the same minimum deps")

    pattern = re.compile(
        r"(\.\. \|)"
        + r"(([A-Za-z]+\-?)+)"
        + r"(MinVersion\| replace::)"
        + r"( [0-9]+\.[0-9]+(\.[0-9]+)?)"
    )

    readme_path = Path(sklearn.__path__[0]).parents[0]
    readme_file = readme_path / "README.rst"

    if not os.path.exists(readme_file):
        # Skip the test if the README.rst file is not available.
        # For instance, when installing scikit-learn from wheels
        pytest.skip("The README.rst file is not available.")

    with readme_file.open("r") as f:
        for line in f:
            matched = pattern.match(line)

            if not matched:
                continue

            package, version = matched.group(2), matched.group(5)
            package = package.lower()

            if package in dependent_packages:
                version = parse_version(version)
                min_version = parse_version(dependent_packages[package][0])

                assert version == min_version, f"{package} has a mismatched version"


def test_min_dependencies_pyproject_toml():
    """Check versions in pyproject.toml is consistent with _min_dependencies."""
    # tomllib is available in Python 3.11
    tomllib = pytest.importorskip("tomllib")

    root_directory = Path(sklearn.__path__[0]).parent
    pyproject_toml_path = root_directory / "pyproject.toml"

    if not pyproject_toml_path.exists():
        # Skip the test if the pyproject.toml file is not available.
        # For instance, when installing scikit-learn from wheels
        pytest.skip("pyproject.toml is not available.")

    with pyproject_toml_path.open("rb") as f:
        pyproject_toml = tomllib.load(f)

    build_requirements = pyproject_toml["build-system"]["requires"]

    pyproject_build_min_versions = {}
    for requirement in build_requirements:
        if ">=" in requirement:
            package, version = requirement.split(">=")
            package = package.lower()
            pyproject_build_min_versions[package] = version

    # Only scipy and cython are listed in pyproject.toml
    # NumPy is more complex using oldest-supported-numpy.
    assert set(["scipy", "cython"]) == set(pyproject_build_min_versions)

    for package, version in pyproject_build_min_versions.items():
        version = parse_version(version)
        expected_min_version = parse_version(dependent_packages[package][0])

        assert version == expected_min_version, f"{package} has a mismatched version"
