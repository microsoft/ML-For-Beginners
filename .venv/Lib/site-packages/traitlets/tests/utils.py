from __future__ import annotations

import sys
from subprocess import PIPE, Popen
from typing import Any, Sequence


def get_output_error_code(cmd: str | Sequence[str]) -> tuple[str, str, Any]:
    """Get stdout, stderr, and exit code from running a command"""
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)  # noqa: S603
    out, err = p.communicate()
    out_str = out.decode("utf8", "replace")
    err_str = err.decode("utf8", "replace")
    return out_str, err_str, p.returncode


def check_help_output(pkg: str, subcommand: Sequence[str] | None = None) -> tuple[str, str]:
    """test that `python -m PKG [subcommand] -h` works"""
    cmd = [sys.executable, "-m", pkg]
    if subcommand:
        cmd.extend(subcommand)
    cmd.append("-h")
    out, err, rc = get_output_error_code(cmd)
    assert rc == 0, err
    assert "Traceback" not in err
    assert "Options" in out
    assert "--help-all" in out
    return out, err


def check_help_all_output(pkg: str, subcommand: Sequence[str] | None = None) -> tuple[str, str]:
    """test that `python -m PKG --help-all` works"""
    cmd = [sys.executable, "-m", pkg]
    if subcommand:
        cmd.extend(subcommand)
    cmd.append("--help-all")
    out, err, rc = get_output_error_code(cmd)
    assert rc == 0, err
    assert "Traceback" not in err
    assert "Options" in out
    assert "Class options" in out
    return out, err
