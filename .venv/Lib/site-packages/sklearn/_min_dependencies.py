"""All minimum dependencies for scikit-learn."""
import argparse
from collections import defaultdict

# scipy and cython should by in sync with pyproject.toml
NUMPY_MIN_VERSION = "1.19.5"
SCIPY_MIN_VERSION = "1.6.0"
JOBLIB_MIN_VERSION = "1.2.0"
THREADPOOLCTL_MIN_VERSION = "2.0.0"
PYTEST_MIN_VERSION = "7.1.2"
CYTHON_MIN_VERSION = "0.29.33"


# 'build' and 'install' is included to have structured metadata for CI.
# It will NOT be included in setup's extras_require
# The values are (version_spec, comma separated tags)
dependent_packages = {
    "numpy": (NUMPY_MIN_VERSION, "build, install"),
    "scipy": (SCIPY_MIN_VERSION, "build, install"),
    "joblib": (JOBLIB_MIN_VERSION, "install"),
    "threadpoolctl": (THREADPOOLCTL_MIN_VERSION, "install"),
    "cython": (CYTHON_MIN_VERSION, "build"),
    "matplotlib": ("3.3.4", "benchmark, docs, examples, tests"),
    "scikit-image": ("0.17.2", "docs, examples, tests"),
    "pandas": ("1.1.5", "benchmark, docs, examples, tests"),
    "seaborn": ("0.9.0", "docs, examples"),
    "memory_profiler": ("0.57.0", "benchmark, docs"),
    "pytest": (PYTEST_MIN_VERSION, "tests"),
    "pytest-cov": ("2.9.0", "tests"),
    "ruff": ("0.0.272", "tests"),
    "black": ("23.3.0", "tests"),
    "mypy": ("1.3", "tests"),
    "pyamg": ("4.0.0", "tests"),
    "polars": ("0.19.12", "tests"),
    "pyarrow": ("12.0.0", "tests"),
    "sphinx": ("6.0.0", "docs"),
    "sphinx-copybutton": ("0.5.2", "docs"),
    "sphinx-gallery": ("0.15.0", "docs"),
    "numpydoc": ("1.2.0", "docs, tests"),
    "Pillow": ("7.1.2", "docs"),
    "pooch": ("1.6.0", "docs, examples, tests"),
    "sphinx-prompt": ("1.3.0", "docs"),
    "sphinxext-opengraph": ("0.4.2", "docs"),
    "plotly": ("5.14.0", "docs, examples"),
    # XXX: Pin conda-lock to the latest released version (needs manual update
    # from time to time)
    "conda-lock": ("2.4.2", "maintenance"),
}


# create inverse mapping for setuptools
tag_to_packages: dict = defaultdict(list)
for package, (min_version, extras) in dependent_packages.items():
    for extra in extras.split(", "):
        tag_to_packages[extra].append("{}>={}".format(package, min_version))


# Used by CI to get the min dependencies
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies for a package")

    parser.add_argument("package", choices=dependent_packages)
    args = parser.parse_args()
    min_version = dependent_packages[args.package][0]
    print(min_version)
