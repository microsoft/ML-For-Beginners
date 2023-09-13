"""Tests for paths"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import re
import shutil
import site
import stat
import subprocess
import sys
import tempfile
import warnings
from unittest.mock import patch

import pytest
from platformdirs import __version_info__

from jupyter_core import paths
from jupyter_core.paths import (
    UF_HIDDEN,
    _win32_restrict_file_to_user_ctypes,
    exists,
    is_file_hidden,
    is_hidden,
    issue_insecure_write_warning,
    jupyter_config_dir,
    jupyter_config_path,
    jupyter_data_dir,
    jupyter_path,
    jupyter_runtime_dir,
    prefer_environment_over_user,
    secure_write,
)

pjoin = os.path.join

macos = pytest.mark.skipif(sys.platform != "darwin", reason="only run on macos")
windows = pytest.mark.skipif(sys.platform != "win32", reason="only run on windows")
linux = pytest.mark.skipif(sys.platform != "linux", reason="only run on linux")

xdg_env = {
    "XDG_CONFIG_HOME": "/tmp/xdg/config",
    "XDG_DATA_HOME": "/tmp/xdg/data",
    "XDG_RUNTIME_DIR": "/tmp/xdg/runtime",
}
xdg = patch.dict("os.environ", xdg_env)
no_xdg = patch.dict(
    "os.environ",
    {},
)

environment = patch.dict("os.environ")

use_platformdirs = patch.dict("os.environ", {"JUPYTER_PLATFORM_DIRS": "1"})

jupyter_config_env = "/jupyter-cfg"
config_env = patch.dict("os.environ", {"JUPYTER_CONFIG_DIR": jupyter_config_env})
prefer_env = patch.dict("os.environ", {"JUPYTER_PREFER_ENV_PATH": "True"})
prefer_user = patch.dict("os.environ", {"JUPYTER_PREFER_ENV_PATH": "False"})


def setup_function():
    environment.start()
    for var in [
        "JUPYTER_CONFIG_DIR",
        "JUPYTER_CONFIG_PATH",
        "JUPYTER_DATA_DIR",
        "JUPYTER_NO_CONFIG",
        "JUPYTER_PATH",
        "JUPYTER_PLATFORM_DIRS",
        "JUPYTER_RUNTIME_DIR",
    ]:
        os.environ.pop(var, None)
    # For these tests, default to preferring the user-level over environment-level paths
    # Tests can override this preference using the prefer_env decorator/context manager
    os.environ["JUPYTER_PREFER_ENV_PATH"] = "no"


def teardown_function():
    environment.stop()


def realpath(path):
    return os.path.abspath(os.path.realpath(os.path.expanduser(path)))


home_jupyter = realpath("~/.jupyter")


def test_envset():
    true_values = ["", "True", "on", "yes", "Y", "1", "anything"]
    false_values = ["n", "No", "N", "fAlSE", "0", "0.0", "Off"]
    with patch.dict("os.environ", ((f"FOO_{v}", v) for v in true_values + false_values)):
        for v in true_values:
            assert paths.envset(f"FOO_{v}")
        for v in false_values:
            assert not paths.envset(f"FOO_{v}")
        # Test default value is False
        assert paths.envset("THIS_VARIABLE_SHOULD_NOT_BE_SET") is False
        # Test envset returns the given default if supplied
        assert paths.envset("THIS_VARIABLE_SHOULD_NOT_BE_SET", None) is None


def test_config_dir():
    config = jupyter_config_dir()
    assert config == home_jupyter


@macos
@use_platformdirs
def test_config_dir_darwin():
    config = jupyter_config_dir()

    assert config == realpath("~/Library/Application Support/Jupyter")


@windows
@use_platformdirs
def test_config_dir_windows():
    config = jupyter_config_dir()
    assert config == realpath(pjoin(os.environ.get("LOCALAPPDATA", ""), "Jupyter"))


@linux
@use_platformdirs
def test_config_dir_linux():
    config = jupyter_config_dir()
    assert config == realpath("~/.config/jupyter")


def test_config_env_legacy():
    with config_env:
        config = jupyter_config_dir()
        assert config == jupyter_config_env


@use_platformdirs
def test_config_env():
    with config_env:
        config = jupyter_config_dir()
        assert config == jupyter_config_env


def test_data_dir_env_legacy():
    data_env = "runtime-dir"
    with patch.dict("os.environ", {"JUPYTER_DATA_DIR": data_env}):
        data = jupyter_data_dir()
        assert data == data_env


@use_platformdirs
def test_data_dir_env():
    data_env = "runtime-dir"
    with patch.dict("os.environ", {"JUPYTER_DATA_DIR": data_env}):
        data = jupyter_data_dir()
        assert data == data_env


@macos
def test_data_dir_darwin_legacy():
    data = jupyter_data_dir()
    assert data == realpath("~/Library/Jupyter")


@macos
@use_platformdirs
def test_data_dir_darwin():
    data = jupyter_data_dir()
    assert data == realpath("~/Library/Application Support/Jupyter")


@windows
def test_data_dir_windows_legacy():
    data = jupyter_data_dir()
    assert data == realpath(pjoin(os.environ.get("APPDATA", ""), "jupyter"))


@windows
@use_platformdirs
def test_data_dir_windows():
    data = jupyter_data_dir()
    assert data == realpath(pjoin(os.environ.get("LOCALAPPDATA", ""), "Jupyter"))


@linux
def test_data_dir_linux_legacy():
    with no_xdg:
        data = jupyter_data_dir()
        assert data == realpath("~/.local/share/jupyter")

    with xdg:
        data = jupyter_data_dir()
        assert data == pjoin(xdg_env["XDG_DATA_HOME"], "jupyter")


@linux
@use_platformdirs
def test_data_dir_linux():
    with no_xdg:
        data = jupyter_data_dir()
        assert data == realpath("~/.local/share/jupyter")

    with xdg:
        data = jupyter_data_dir()
        assert data == pjoin(xdg_env["XDG_DATA_HOME"], "jupyter")


def test_runtime_dir_env_legacy():
    rtd_env = "runtime-dir"
    with patch.dict("os.environ", {"JUPYTER_RUNTIME_DIR": rtd_env}):
        runtime = jupyter_runtime_dir()
        assert runtime == rtd_env


@use_platformdirs
def test_runtime_dir_env():
    rtd_env = "runtime-dir"
    with patch.dict("os.environ", {"JUPYTER_RUNTIME_DIR": rtd_env}):
        runtime = jupyter_runtime_dir()
        assert runtime == rtd_env


@macos
def test_runtime_dir_darwin_legacy():
    runtime = jupyter_runtime_dir()
    assert runtime == realpath("~/Library/Jupyter/runtime")


@macos
@use_platformdirs
def test_runtime_dir_darwin():
    runtime = jupyter_runtime_dir()
    if __version_info__[0] < 3:
        assert runtime == realpath("~/Library/Preferences/Jupyter/runtime")
        return
    assert runtime == realpath("~/Library/Application Support/Jupyter/runtime")


@windows
def test_runtime_dir_windows_legacy():
    runtime = jupyter_runtime_dir()
    assert runtime == realpath(pjoin(os.environ.get("APPDATA", ""), "jupyter", "runtime"))


@windows
@use_platformdirs
def test_runtime_dir_windows():
    runtime = jupyter_runtime_dir()
    assert runtime == realpath(pjoin(os.environ.get("LOCALAPPDATA", ""), "Jupyter", "runtime"))


@linux
def test_runtime_dir_linux_legacy():
    with no_xdg:
        runtime = jupyter_runtime_dir()
        assert runtime == realpath("~/.local/share/jupyter/runtime")

    with xdg:
        runtime = jupyter_runtime_dir()
        assert runtime == pjoin(xdg_env["XDG_DATA_HOME"], "jupyter", "runtime")


@linux
@use_platformdirs
def test_runtime_dir_linux():
    with no_xdg:
        runtime = jupyter_runtime_dir()
        assert runtime == realpath("~/.local/share/jupyter/runtime")

    with xdg:
        runtime = jupyter_runtime_dir()
        assert runtime == pjoin(xdg_env["XDG_DATA_HOME"], "jupyter", "runtime")


def test_jupyter_path():
    system_path = ["system", "path"]
    with patch.object(paths, "SYSTEM_JUPYTER_PATH", system_path):
        path = jupyter_path()
        assert path[0] == jupyter_data_dir()
        assert path[-2:] == system_path


def test_jupyter_path_user_site():
    with patch.object(site, "ENABLE_USER_SITE", True):
        path = jupyter_path()

        # deduplicated expected values
        values = list(
            dict.fromkeys(
                [
                    jupyter_data_dir(),
                    os.path.join(site.getuserbase(), "share", "jupyter"),
                    paths.ENV_JUPYTER_PATH[0],
                ]
            )
        )
        for p, v in zip(path, values):
            assert p == v


def test_jupyter_path_no_user_site():
    with patch.object(site, "ENABLE_USER_SITE", False):
        path = jupyter_path()
        assert path[0] == jupyter_data_dir()
        assert path[1] == paths.ENV_JUPYTER_PATH[0]


def test_jupyter_path_prefer_env():
    with prefer_env:
        path = jupyter_path()
        assert path[0] == paths.ENV_JUPYTER_PATH[0]
        assert path[1] == jupyter_data_dir()


def test_jupyter_path_env():
    path_env = os.pathsep.join(
        [
            pjoin("foo", "bar"),
            pjoin("bar", "baz", ""),  # trailing /
        ]
    )

    with patch.dict("os.environ", {"JUPYTER_PATH": path_env}):
        path = jupyter_path()
    assert path[:2] == [pjoin("foo", "bar"), pjoin("bar", "baz")]


def test_jupyter_path_sys_prefix():
    with patch.object(paths, "ENV_JUPYTER_PATH", ["sys_prefix"]):
        path = jupyter_path()
    assert "sys_prefix" in path


def test_jupyter_path_subdir():
    path = jupyter_path("sub1", "sub2")
    for p in path:
        assert p.endswith(pjoin("", "sub1", "sub2"))


def test_jupyter_config_path():
    with patch.object(site, "ENABLE_USER_SITE", True):
        path = jupyter_config_path()

    # deduplicated expected values
    values = list(
        dict.fromkeys(
            [
                jupyter_config_dir(),
                os.path.join(site.getuserbase(), "etc", "jupyter"),
                paths.ENV_CONFIG_PATH[0],
            ]
        )
    )
    for p, v in zip(path, values):
        assert p == v


def test_jupyter_config_path_no_user_site():
    with patch.object(site, "ENABLE_USER_SITE", False):
        path = jupyter_config_path()
    assert path[0] == jupyter_config_dir()
    assert path[1] == paths.ENV_CONFIG_PATH[0]


def test_jupyter_config_path_prefer_env():
    with prefer_env, patch.object(site, "ENABLE_USER_SITE", True):
        path = jupyter_config_path()

    # deduplicated expected values
    values = list(
        dict.fromkeys(
            [
                paths.ENV_CONFIG_PATH[0],
                jupyter_config_dir(),
                os.path.join(site.getuserbase(), "etc", "jupyter"),
            ]
        )
    )
    for p, v in zip(path, values):
        assert p == v


def test_jupyter_config_path_env():
    path_env = os.pathsep.join(
        [
            pjoin("foo", "bar"),
            pjoin("bar", "baz", ""),  # trailing /
        ]
    )

    with patch.dict("os.environ", {"JUPYTER_CONFIG_PATH": path_env}):
        path = jupyter_config_path()
    assert path[:2] == [pjoin("foo", "bar"), pjoin("bar", "baz")]


def test_prefer_environment_over_user():
    with prefer_env:
        assert prefer_environment_over_user() is True

    with prefer_user:
        assert prefer_environment_over_user() is False

    # Test default if environment variable is not set, and try to determine if we are in a virtual environment
    os.environ.pop("JUPYTER_PREFER_ENV_PATH", None)

    # base prefix differs, venv
    with patch.object(sys, "base_prefix", "notthesame"):
        assert prefer_environment_over_user() == paths._do_i_own(sys.prefix)

    # conda
    with patch.object(sys, "base_prefix", sys.prefix):
        # in base env, don't prefer it
        with patch.dict(os.environ, {"CONDA_PREFIX": sys.prefix, "CONDA_DEFAULT_ENV": "base"}):
            assert not prefer_environment_over_user()
        # in non-base env, prefer it
        with patch.dict(os.environ, {"CONDA_PREFIX": sys.prefix, "CONDA_DEFAULT_ENV": "/tmp"}):
            assert prefer_environment_over_user() == paths._do_i_own(sys.prefix)

        # conda env defined, but we aren't using it
        with patch.dict(
            os.environ, {"CONDA_PREFIX": "/somewherelese", "CONDA_DEFAULT_ENV": "/tmp"}
        ):
            assert not prefer_environment_over_user()


def test_is_hidden():
    with tempfile.TemporaryDirectory() as root:
        subdir1 = os.path.join(root, "subdir")
        os.makedirs(subdir1)
        assert not is_hidden(subdir1, root)
        assert not is_file_hidden(subdir1)

        subdir2 = os.path.join(root, ".subdir2")
        os.makedirs(subdir2)
        assert is_hidden(subdir2, root)
        assert is_file_hidden(subdir2)
        # root dir is always visible
        assert not is_hidden(subdir2, subdir2)

        subdir34 = os.path.join(root, "subdir3", ".subdir4")
        os.makedirs(subdir34)
        assert is_hidden(subdir34, root)
        assert is_hidden(subdir34)

        subdir56 = os.path.join(root, ".subdir5", "subdir6")
        os.makedirs(subdir56)
        assert is_hidden(subdir56, root)
        assert is_hidden(subdir56)
        assert not is_file_hidden(subdir56)
        assert not is_file_hidden(subdir56, os.stat(subdir56))

        assert not is_file_hidden(os.path.join(root, "does_not_exist"))
        subdir78 = os.path.join(root, "subdir7", "subdir8")
        os.makedirs(subdir78)
        assert not is_hidden(subdir78, root)
        if hasattr(os, "chflags"):
            os.chflags(subdir78, UF_HIDDEN)
            assert is_hidden(subdir78, root)


@pytest.mark.skipif(
    not (
        sys.platform == "win32"
        and (("__pypy__" not in sys.modules) or (sys.implementation.version >= (7, 3, 6)))
    ),
    reason="only run on windows/cpython or pypy >= 7.3.6: https://foss.heptapod.net/pypy/pypy/-/issues/3469",
)
def test_is_hidden_win32_cpython():
    import ctypes  # noqa

    with tempfile.TemporaryDirectory() as root:
        subdir1 = os.path.join(root, "subdir")
        os.makedirs(subdir1)
        assert not is_hidden(subdir1, root)
        subprocess.check_call(["attrib", "+h", subdir1])  # noqa
        assert is_hidden(subdir1, root)
        assert is_file_hidden(subdir1)


@pytest.mark.skipif(
    not (
        sys.platform == "win32"
        and "__pypy__" in sys.modules
        and sys.implementation.version < (7, 3, 6)
    ),
    reason="only run on windows/pypy < 7.3.6: https://foss.heptapod.net/pypy/pypy/-/issues/3469",
)
def test_is_hidden_win32_pypy():
    import ctypes  # noqa

    with tempfile.TemporaryDirectory() as root:
        subdir1 = os.path.join(root, "subdir")
        os.makedirs(subdir1)
        assert not is_hidden(subdir1, root)
        subprocess.check_call(["attrib", "+h", subdir1])  # noqa

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            assert not is_hidden(subdir1, root)
            # Verify the warning was triggered
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "hidden files are not detectable on this system" in str(w[-1].message)

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            assert not is_file_hidden(subdir1)
            # Verify the warning was triggered
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "hidden files are not detectable on this system" in str(w[-1].message)


@pytest.mark.skipif(sys.platform != "win32", reason="only runs on windows")
def test_win32_restrict_file_to_user_ctypes(tmp_path):
    _win32_restrict_file_to_user_ctypes(str(tmp_path))


@pytest.mark.skipif(sys.platform != "win32", reason="only runs on windows")
def test_secure_write_win32():
    def fetch_win32_permissions(filename):
        """Extracts file permissions on windows using icacls"""
        role_permissions = {}
        proc = os.popen("icacls %s" % filename)  # noqa
        lines = proc.read().splitlines()
        proc.close()
        for index, line in enumerate(lines):
            if index == 0:
                line = line.split(filename)[-1].strip().lower()  # noqa
            match = re.match(r"\s*([^:]+):\(([^\)]*)\)", line)
            if match:
                usergroup, permissions = match.groups()
                usergroup = usergroup.lower().split("\\")[-1]
                permissions = {p.lower() for p in permissions.split(",")}
                role_permissions[usergroup] = permissions
            elif not line.strip():
                break
        return role_permissions

    def check_user_only_permissions(fname):
        # Windows has it's own permissions ACL patterns
        username = os.environ["USERNAME"].lower()
        permissions = fetch_win32_permissions(fname)
        print(permissions)  # for easier debugging
        assert username in permissions
        assert permissions[username] == {"r", "w", "d"}
        assert "administrators" in permissions
        assert permissions["administrators"] == {"f"}
        assert "everyone" not in permissions
        assert len(permissions) == 2

    directory = tempfile.mkdtemp()
    fname = os.path.join(directory, "check_perms")
    try:
        with secure_write(fname) as f:
            f.write("test 1")
        check_user_only_permissions(fname)
        with open(fname, encoding="utf-8") as f:
            assert f.read() == "test 1"
    finally:
        shutil.rmtree(directory)


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_secure_write_unix():
    directory = tempfile.mkdtemp()
    fname = os.path.join(directory, "check_perms")
    try:
        with secure_write(fname) as f:
            f.write("test 1")
        mode = os.stat(fname).st_mode
        assert 0o0600 == (stat.S_IMODE(mode) & 0o7677)  # noqa # tolerate owner-execute bit
        with open(fname, encoding="utf-8") as f:
            assert f.read() == "test 1"

        # Try changing file permissions ahead of time
        os.chmod(fname, 0o755)  # noqa
        with secure_write(fname) as f:
            f.write("test 2")
        mode = os.stat(fname).st_mode
        assert 0o0600 == (stat.S_IMODE(mode) & 0o7677)  # noqa # tolerate owner-execute bit
        with open(fname, encoding="utf-8") as f:
            assert f.read() == "test 2"
    finally:
        shutil.rmtree(directory)


def test_exists(tmpdir):
    assert exists(str(tmpdir))


def test_insecure_write_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        issue_insecure_write_warning()
