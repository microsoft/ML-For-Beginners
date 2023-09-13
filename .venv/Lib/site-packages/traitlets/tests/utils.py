import sys
from subprocess import PIPE, Popen


def get_output_error_code(cmd):
    """Get stdout, stderr, and exit code from running a command"""
    p = Popen(cmd, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out = out.decode("utf8", "replace")  # type:ignore
    err = err.decode("utf8", "replace")  # type:ignore
    return out, err, p.returncode


def check_help_output(pkg, subcommand=None):
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


def check_help_all_output(pkg, subcommand=None):
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
