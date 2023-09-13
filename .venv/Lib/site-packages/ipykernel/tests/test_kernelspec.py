# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import json
import os
import shutil
import sys
import tempfile
from unittest import mock

import pytest
from jupyter_core.paths import jupyter_data_dir

from ipykernel.kernelspec import (
    KERNEL_NAME,
    RESOURCES,
    InstallIPythonKernelSpecApp,
    get_kernel_dict,
    install,
    make_ipkernel_cmd,
    write_kernel_spec,
)

pjoin = os.path.join


def test_make_ipkernel_cmd():
    cmd = make_ipkernel_cmd()
    assert cmd == [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]


def assert_kernel_dict(d):
    assert d["argv"] == make_ipkernel_cmd()
    assert d["display_name"] == "Python %i (ipykernel)" % sys.version_info[0]
    assert d["language"] == "python"


def test_get_kernel_dict():
    d = get_kernel_dict()
    assert_kernel_dict(d)


def assert_kernel_dict_with_profile(d):
    assert d["argv"] == make_ipkernel_cmd(extra_arguments=["--profile", "test"])
    assert d["display_name"] == "Python %i (ipykernel)" % sys.version_info[0]
    assert d["language"] == "python"


def test_get_kernel_dict_with_profile():
    d = get_kernel_dict(["--profile", "test"])
    assert_kernel_dict_with_profile(d)


def assert_is_spec(path):
    for fname in os.listdir(RESOURCES):
        dst = pjoin(path, fname)
        assert os.path.exists(dst)
    kernel_json = pjoin(path, "kernel.json")
    assert os.path.exists(kernel_json)
    with open(kernel_json, encoding="utf8") as f:
        json.load(f)


def test_write_kernel_spec():
    path = write_kernel_spec()
    assert_is_spec(path)
    shutil.rmtree(path)


def test_write_kernel_spec_path():
    path = os.path.join(tempfile.mkdtemp(), KERNEL_NAME)
    path2 = write_kernel_spec(path)
    assert path == path2
    assert_is_spec(path)
    shutil.rmtree(path)


def test_install_kernelspec():
    path = tempfile.mkdtemp()
    try:
        InstallIPythonKernelSpecApp.launch_instance(argv=["--prefix", path])
        assert_is_spec(os.path.join(path, "share", "jupyter", "kernels", KERNEL_NAME))
    finally:
        shutil.rmtree(path)


def test_install_user():
    tmp = tempfile.mkdtemp()

    with mock.patch.dict(os.environ, {"HOME": tmp}):
        install(user=True)
        data_dir = jupyter_data_dir()

    assert_is_spec(os.path.join(data_dir, "kernels", KERNEL_NAME))


def test_install():
    system_jupyter_dir = tempfile.mkdtemp()

    with mock.patch("jupyter_client.kernelspec.SYSTEM_JUPYTER_PATH", [system_jupyter_dir]):
        install()

    assert_is_spec(os.path.join(system_jupyter_dir, "kernels", KERNEL_NAME))


def test_install_profile():
    system_jupyter_dir = tempfile.mkdtemp()

    with mock.patch("jupyter_client.kernelspec.SYSTEM_JUPYTER_PATH", [system_jupyter_dir]):
        install(profile="Test")

    spec_file = os.path.join(system_jupyter_dir, "kernels", KERNEL_NAME, "kernel.json")
    with open(spec_file) as f:
        spec = json.load(f)
    assert spec["display_name"].endswith(" [profile=Test]")
    assert spec["argv"][-2:] == ["--profile", "Test"]


def test_install_display_name_overrides_profile():
    system_jupyter_dir = tempfile.mkdtemp()

    with mock.patch("jupyter_client.kernelspec.SYSTEM_JUPYTER_PATH", [system_jupyter_dir]):
        install(display_name="Display", profile="Test")

    spec_file = os.path.join(system_jupyter_dir, "kernels", KERNEL_NAME, "kernel.json")
    with open(spec_file) as f:
        spec = json.load(f)
    assert spec["display_name"] == "Display"


@pytest.mark.parametrize("env", [None, dict(spam="spam"), dict(spam="spam", foo="bar")])
def test_install_env(tmp_path, env):
    # python 3.5 // tmp_path must be converted to str
    with mock.patch("jupyter_client.kernelspec.SYSTEM_JUPYTER_PATH", [str(tmp_path)]):
        install(env=env)

    spec = tmp_path / "kernels" / KERNEL_NAME / "kernel.json"
    with spec.open() as f:
        spec = json.load(f)

    if env:
        assert len(env) == len(spec["env"])
        for k, v in env.items():
            assert spec["env"][k] == v
    else:
        assert "env" not in spec
