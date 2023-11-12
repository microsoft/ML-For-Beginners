"""Test for the show_versions helper. Based on the sklearn tests."""
# Author: Alexander L. Hayes <hayesall@iu.edu>
# License: MIT

from imblearn.utils._show_versions import _get_deps_info, show_versions


def test_get_deps_info():
    _deps_info = _get_deps_info()
    assert "pip" in _deps_info
    assert "setuptools" in _deps_info
    assert "imbalanced-learn" in _deps_info
    assert "scikit-learn" in _deps_info
    assert "numpy" in _deps_info
    assert "scipy" in _deps_info
    assert "Cython" in _deps_info
    assert "pandas" in _deps_info
    assert "joblib" in _deps_info


def test_show_versions_default(capsys):
    show_versions()
    out, err = capsys.readouterr()
    assert "python" in out
    assert "executable" in out
    assert "machine" in out
    assert "pip" in out
    assert "setuptools" in out
    assert "imbalanced-learn" in out
    assert "scikit-learn" in out
    assert "numpy" in out
    assert "scipy" in out
    assert "Cython" in out
    assert "pandas" in out
    assert "keras" in out
    assert "tensorflow" in out
    assert "joblib" in out


def test_show_versions_github(capsys):
    show_versions(github=True)
    out, err = capsys.readouterr()
    assert "<details><summary>System, Dependency Information</summary>" in out
    assert "**System Information**" in out
    assert "* python" in out
    assert "* executable" in out
    assert "* machine" in out
    assert "**Python Dependencies**" in out
    assert "* pip" in out
    assert "* setuptools" in out
    assert "* imbalanced-learn" in out
    assert "* scikit-learn" in out
    assert "* numpy" in out
    assert "* scipy" in out
    assert "* Cython" in out
    assert "* pandas" in out
    assert "* keras" in out
    assert "* tensorflow" in out
    assert "* joblib" in out
    assert "</details>" in out
