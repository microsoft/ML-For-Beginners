"""Test the Jupyter command-line"""

import json
import os
import sys
import sysconfig
from subprocess import PIPE, CalledProcessError, check_output
from unittest.mock import patch

import pytest

from jupyter_core.command import list_subcommands
from jupyter_core.paths import (
    jupyter_config_dir,
    jupyter_config_path,
    jupyter_data_dir,
    jupyter_path,
    jupyter_runtime_dir,
)

resetenv = patch.dict(os.environ)


def setup_function():
    resetenv.start()
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


def teardown_function():
    resetenv.stop()


def get_jupyter_output(cmd):
    """Get output of a jupyter command"""
    if not isinstance(cmd, list):
        cmd = [cmd]
    return (
        check_output([sys.executable, "-m", "jupyter_core", *cmd], stderr=PIPE)
        .decode("utf8")
        .strip()
    )


def write_executable(path, source):
    if sys.platform == "win32":
        script = path.dirpath() / path.purebasename + "-script.py"
        exe = path.dirpath() / path.purebasename + ".exe"
    else:
        script = path

    script.write(source)
    script.chmod(0o700)

    if sys.platform == "win32":
        try:
            import importlib.resources

            if not hasattr(importlib.resources, 'files'):
                raise ImportError
            wp = importlib.resources.files('setuptools').joinpath('cli-32.exe')
            w = wp.read_bytes()
        except (ImportError, FileNotFoundError, SystemError):
            pytest.skip(
                "Need importlib.resources and setuptools to make scripts executable on Windows"
            )
        exe.write(w, "wb")
        exe.chmod(0o700)


def assert_output(cmd, expected):
    assert get_jupyter_output(cmd) == expected


def test_config_dir():
    assert_output("--config-dir", jupyter_config_dir())


def test_data_dir():
    assert_output("--data-dir", jupyter_data_dir())


def test_runtime_dir():
    assert_output("--runtime-dir", jupyter_runtime_dir())


def test_paths():
    output = get_jupyter_output("--paths")
    for d in (jupyter_config_dir(), jupyter_data_dir(), jupyter_runtime_dir()):
        assert d in output
    for key in ("config", "data", "runtime"):
        assert ("%s:" % key) in output

    for path in (jupyter_config_path(), jupyter_path()):
        for d in path:
            assert d in output


def test_paths_json():
    output = get_jupyter_output(["--paths", "--json"])
    data = json.loads(output)
    assert sorted(data) == ["config", "data", "runtime"]
    for _, path in data.items():
        assert isinstance(path, list)


def test_paths_debug():
    names = [
        "JUPYTER_PREFER_ENV_PATH",
        "JUPYTER_NO_CONFIG",
        "JUPYTER_CONFIG_PATH",
        "JUPYTER_CONFIG_DIR",
        "JUPYTER_PATH",
        "JUPYTER_DATA_DIR",
        "JUPYTER_RUNTIME_DIR",
    ]
    output = get_jupyter_output(["--paths", "--debug"])
    for v in names:
        assert f"{v} is not set" in output

    with patch.dict("os.environ", [(v, "y") for v in names]):
        output = get_jupyter_output(["--paths", "--debug"])
    for v in names:
        assert f"{v} is set" in output


def test_subcommand_not_given():
    with pytest.raises(CalledProcessError):
        get_jupyter_output([])


def test_help():
    output = get_jupyter_output("-h")
    assert "--help" in output


def test_subcommand_not_found():
    with pytest.raises(CalledProcessError) as excinfo:
        get_jupyter_output("nonexistant-subcommand")
    stderr = excinfo.value.stderr.decode("utf8")
    assert "Jupyter command `jupyter-nonexistant-subcommand` not found." in stderr


@patch.object(sys, "argv", [__file__] + sys.argv[1:])
def test_subcommand_list(tmpdir):
    a = tmpdir.mkdir("a")
    for cmd in ("jupyter-foo-bar", "jupyter-xyz", "jupyter-babel-fish"):
        a.join(cmd).write("")
    b = tmpdir.mkdir("b")
    for cmd in ("jupyter-foo", "jupyterstuff", "jupyter-yo-eyropa-ganymyde-callysto"):
        b.join(cmd).write("")
    c = tmpdir.mkdir("c")
    for cmd in ("jupyter-baz", "jupyter-bop"):
        c.join(cmd).write("")

    path = os.pathsep.join(map(str, [a, b]))

    def get_path(dummy):
        return str(c)

    with patch.object(sysconfig, "get_path", get_path), patch.dict("os.environ", {"PATH": path}):
        subcommands = list_subcommands()
        assert subcommands == [
            "babel-fish",
            "baz",
            "bop",
            "foo",
            "xyz",
            "yo-eyropa-ganymyde-callysto",
        ]


skip_darwin = pytest.mark.skipif(sys.platform == "darwin", reason="Fails on macos")


@skip_darwin
def test_not_on_path(tmpdir):
    a = tmpdir.mkdir("a")
    jupyter = a.join("jupyter")
    jupyter.write("from jupyter_core import command; command.main()")
    jupyter.chmod(0o700)
    witness = a.join("jupyter-witness")
    witness_src = "#!{}\n{}\n".format(sys.executable, 'print("WITNESS ME")')
    write_executable(witness, witness_src)

    env = {"PATH": ""}
    if "SYSTEMROOT" in os.environ:  # Windows http://bugs.python.org/issue20614
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    if sys.platform == "win32":
        env["PATHEXT"] = ".EXE"
    # This won't work on windows unless
    out = check_output([sys.executable, str(jupyter), "witness"], env=env)
    assert b"WITNESS" in out


@skip_darwin
def test_path_priority(tmpdir):
    a = tmpdir.mkdir("a")
    jupyter = a.join("jupyter")
    jupyter.write("from jupyter_core import command; command.main()")
    jupyter.chmod(0o700)
    witness_a = a.join("jupyter-witness")
    witness_a_src = "#!{}\n{}\n".format(sys.executable, 'print("WITNESS A")')
    write_executable(witness_a, witness_a_src)

    b = tmpdir.mkdir("b")
    witness_b = b.join("jupyter-witness")
    witness_b_src = "#!{}\n{}\n".format(sys.executable, 'print("WITNESS B")')
    write_executable(witness_b, witness_b_src)

    env = {"PATH": str(b)}
    if "SYSTEMROOT" in os.environ:  # Windows http://bugs.python.org/issue20614
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    if sys.platform == "win32":
        env["PATHEXT"] = ".EXE"
    out = check_output([sys.executable, str(jupyter), "witness"], env=env)
    assert b"WITNESS A" in out


@skip_darwin
def test_argv0(tmpdir):
    a = tmpdir.mkdir("a")
    jupyter = a.join("jupyter")
    jupyter.write("from jupyter_core import command; command.main()")
    jupyter.chmod(0o700)
    witness_a = a.join("jupyter-witness")
    witness_a_src = f"""#!{sys.executable}
import sys
print(sys.argv[0])
"""
    write_executable(witness_a, witness_a_src)

    env = {}
    if "SYSTEMROOT" in os.environ:  # Windows http://bugs.python.org/issue20614
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    if sys.platform == "win32":
        env["PATHEXT"] = ".EXE"
    out = check_output([sys.executable, str(jupyter), "witness"], env=env)

    # Make sure the first argv is the full path to the executing script
    assert f"{jupyter}-witness".encode() in out
