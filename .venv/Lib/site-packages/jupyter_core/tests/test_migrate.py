# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
"""Test config file migration"""

import os
import re
import shutil
from tempfile import mkdtemp
from unittest.mock import patch

import pytest

from jupyter_core import migrate as migrate_mod
from jupyter_core.application import JupyterApp
from jupyter_core.migrate import (
    migrate,
    migrate_config,
    migrate_dir,
    migrate_file,
    migrate_one,
    migrate_static_custom,
)
from jupyter_core.utils import ensure_dir_exists

pjoin = os.path.join
here = os.path.dirname(__file__)
dotipython = pjoin(here, "dotipython")
dotipython_empty = pjoin(here, "dotipython_empty")


@pytest.fixture
def td(request):
    """Fixture for a temporary directory"""
    td = mkdtemp("μnïcø∂e")
    request.addfinalizer(lambda: shutil.rmtree(td))
    return td


@pytest.fixture
def env(request):
    """Fixture for a full testing environment"""
    td = mkdtemp()
    env = {
        "TESTDIR": td,
        "IPYTHONDIR": pjoin(td, "ipython"),
        "JUPYTER_CONFIG_DIR": pjoin(td, "jupyter"),
        "JUPYTER_DATA_DIR": pjoin(td, "jupyter_data"),
        "JUPYTER_RUNTIME_DIR": pjoin(td, "jupyter_runtime"),
        "JUPYTER_PATH": "",
    }
    env_patch = patch.dict(os.environ, env)
    env_patch.start()

    def fin():
        """Cleanup test env"""
        env_patch.stop()
        shutil.rmtree(td, ignore_errors=os.name == 'nt')

    request.addfinalizer(fin)

    return env


def touch(path, content=""):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def assert_files_equal(a, b):
    """Verify that two files match"""

    assert os.path.exists(b)
    with open(a, encoding="utf-8") as f:
        a_txt = f.read()

    with open(b, encoding="utf-8") as f:
        b_txt = f.read()

    assert a_txt == b_txt


def test_migrate_file(td):
    src = pjoin(td, "src")
    dst = pjoin(td, "dst")
    touch(src, "test file")
    assert migrate_file(src, dst)
    assert_files_equal(src, dst)

    src2 = pjoin(td, "src2")
    touch(src2, "different src")
    assert not migrate_file(src2, dst)
    assert_files_equal(src, dst)


def test_migrate_dir(td):
    src = pjoin(td, "src")
    dst = pjoin(td, "dst")
    os.mkdir(src)
    assert not migrate_dir(src, dst)
    assert not os.path.exists(dst)

    touch(pjoin(src, "f"), "test file")
    assert migrate_dir(src, dst)
    assert_files_equal(pjoin(src, "f"), pjoin(dst, "f"))

    touch(pjoin(src, "g"), "other test file")
    assert not migrate_dir(src, dst)
    assert not os.path.exists(pjoin(dst, "g"))

    shutil.rmtree(dst)
    os.mkdir(dst)
    assert migrate_dir(src, dst)
    assert_files_equal(pjoin(src, "f"), pjoin(dst, "f"))
    assert_files_equal(pjoin(src, "g"), pjoin(dst, "g"))


def test_migrate_one(td):
    src = pjoin(td, "src")
    srcdir = pjoin(td, "srcdir")
    dst = pjoin(td, "dst")
    dstdir = pjoin(td, "dstdir")

    touch(src, "test file")
    touch(pjoin(srcdir, "f"), "test dir file")

    called = {}

    def notice_m_file(src, dst):
        called["migrate_file"] = True
        return migrate_file(src, dst)

    def notice_m_dir(src, dst):
        called["migrate_dir"] = True
        return migrate_dir(src, dst)

    with patch.object(migrate_mod, "migrate_file", notice_m_file), patch.object(
        migrate_mod, "migrate_dir", notice_m_dir
    ):
        assert migrate_one(src, dst)
        assert called == {"migrate_file": True}
        called.clear()
        assert migrate_one(srcdir, dstdir)
        assert called == {"migrate_dir": True}
        called.clear()
        assert not migrate_one(pjoin(td, "dne"), dst)
        assert called == {}


def test_migrate_config(td):
    profile = pjoin(td, "profile")
    jpy = pjoin(td, "jupyter_config")
    ensure_dir_exists(profile)

    env = {
        "profile": profile,
        "jupyter_config": jpy,
    }
    cfg_py = pjoin(profile, "ipython_test_config.py")
    touch(cfg_py, "c.Klass.trait = 5\n")
    empty_cfg_py = pjoin(profile, "ipython_empty_config.py")
    touch(empty_cfg_py, "# c.Klass.trait = 5\n")

    assert not migrate_config("empty", env)
    assert not os.path.exists(jpy)

    with patch.dict(
        migrate_mod.config_substitutions,
        {
            re.compile(r"\bKlass\b"): "Replaced",
        },
    ):
        assert migrate_config("test", env)

    assert os.path.isdir(jpy)
    assert sorted(os.listdir(jpy)) == [
        "jupyter_test_config.py",
    ]

    with open(pjoin(jpy, "jupyter_test_config.py"), encoding="utf-8") as f:
        text = f.read()
    assert text == "c.Replaced.trait = 5\n"


def test_migrate_custom_default(td):
    profile = pjoin(dotipython, "profile_default")
    src = pjoin(profile, "static", "custom")
    assert os.path.exists(src)
    assert not migrate_static_custom(src, td)

    src = pjoin(td, "src")
    dst = pjoin(td, "dst")
    os.mkdir(src)
    src_custom_js = pjoin(src, "custom.js")
    src_custom_css = pjoin(src, "custom.css")
    touch(src_custom_js, "var a=5;")
    touch(src_custom_css, "div { height: 5px; }")

    assert migrate_static_custom(src, dst)


def test_migrate_nothing(env):
    migrate()
    assert os.listdir(env["JUPYTER_CONFIG_DIR"]) == ["migrated"]
    assert not os.path.exists(env["JUPYTER_DATA_DIR"])


def test_migrate_default(env):
    shutil.copytree(dotipython_empty, env["IPYTHONDIR"])
    migrate()
    assert os.listdir(env["JUPYTER_CONFIG_DIR"]) == ["migrated"]
    assert not os.path.exists(env["JUPYTER_DATA_DIR"])


def test_migrate(env):
    shutil.copytree(dotipython, env["IPYTHONDIR"])
    migrate()
    assert os.path.exists(env["JUPYTER_CONFIG_DIR"])
    assert os.path.exists(env["JUPYTER_DATA_DIR"])


def test_app_migrate(env):
    shutil.copytree(dotipython, env["IPYTHONDIR"])
    app = JupyterApp()
    app.initialize([])
    assert os.path.exists(env["JUPYTER_CONFIG_DIR"])
    assert os.path.exists(env["JUPYTER_DATA_DIR"])


def test_app_migrate_skip_if_marker(env):
    shutil.copytree(dotipython, env["IPYTHONDIR"])
    touch(pjoin(env["JUPYTER_CONFIG_DIR"], "migrated"), "done")
    app = JupyterApp()
    app.initialize([])
    assert os.listdir(env["JUPYTER_CONFIG_DIR"]) == ["migrated"]
    assert not os.path.exists(env["JUPYTER_DATA_DIR"])


def test_app_migrate_skip_unwritable_marker(env):
    shutil.copytree(dotipython, env["IPYTHONDIR"])
    migrated_marker = pjoin(env["JUPYTER_CONFIG_DIR"], "migrated")
    touch(migrated_marker, "done")
    os.chmod(migrated_marker, 0)  # make it unworkable
    app = JupyterApp()
    app.initialize([])
    assert os.listdir(env["JUPYTER_CONFIG_DIR"]) == ["migrated"]
    assert not os.path.exists(env["JUPYTER_DATA_DIR"])
