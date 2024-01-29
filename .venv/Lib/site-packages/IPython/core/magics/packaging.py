"""Implementation of packaging-related magic functions.
"""
#-----------------------------------------------------------------------------
#  Copyright (c) 2018 The IPython Development Team.
#
#  Distributed under the terms of the Modified BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import functools
import re
import shlex
import sys
from pathlib import Path

from IPython.core.magic import Magics, magics_class, line_magic


def is_conda_environment(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Return True if the current Python executable is in a conda env"""
        # TODO: does this need to change on windows?
        if not Path(sys.prefix, "conda-meta", "history").exists():
            raise ValueError(
                "The python kernel does not appear to be a conda environment.  "
                "Please use ``%pip install`` instead."
            )
        return func(*args, **kwargs)

    return wrapper


def _get_conda_like_executable(command):
    """Find the path to the given executable

    Parameters
    ----------

    executable: string
        Value should be: conda, mamba or micromamba
    """
    # Check if there is a conda executable in the same directory as the Python executable.
    # This is the case within conda's root environment.
    executable = Path(sys.executable).parent / command
    if executable.is_file():
        return str(executable)

    # Otherwise, attempt to extract the executable from conda history.
    # This applies in any conda environment.
    history = Path(sys.prefix, "conda-meta", "history").read_text(encoding="utf-8")
    match = re.search(
        rf"^#\s*cmd:\s*(?P<command>.*{executable})\s[create|install]",
        history,
        flags=re.MULTILINE,
    )
    if match:
        return match.groupdict()["command"]

    # Fallback: assume the executable is available on the system path.
    return command


CONDA_COMMANDS_REQUIRING_PREFIX = {
    'install', 'list', 'remove', 'uninstall', 'update', 'upgrade',
}
CONDA_COMMANDS_REQUIRING_YES = {
    'install', 'remove', 'uninstall', 'update', 'upgrade',
}
CONDA_ENV_FLAGS = {'-p', '--prefix', '-n', '--name'}
CONDA_YES_FLAGS = {'-y', '--y'}


@magics_class
class PackagingMagics(Magics):
    """Magics related to packaging & installation"""

    @line_magic
    def pip(self, line):
        """Run the pip package manager within the current kernel.

        Usage:
          %pip install [pkgs]
        """
        python = sys.executable
        if sys.platform == "win32":
            python = '"' + python + '"'
        else:
            python = shlex.quote(python)

        self.shell.system(" ".join([python, "-m", "pip", line]))

        print("Note: you may need to restart the kernel to use updated packages.")

    def _run_command(self, cmd, line):
        args = shlex.split(line)
        command = args[0] if len(args) > 0 else ""
        args = args[1:] if len(args) > 1 else [""]

        extra_args = []

        # When the subprocess does not allow us to respond "yes" during the installation,
        # we need to insert --yes in the argument list for some commands
        stdin_disabled = getattr(self.shell, 'kernel', None) is not None
        needs_yes = command in CONDA_COMMANDS_REQUIRING_YES
        has_yes = set(args).intersection(CONDA_YES_FLAGS)
        if stdin_disabled and needs_yes and not has_yes:
            extra_args.append("--yes")

        # Add --prefix to point conda installation to the current environment
        needs_prefix = command in CONDA_COMMANDS_REQUIRING_PREFIX
        has_prefix = set(args).intersection(CONDA_ENV_FLAGS)
        if needs_prefix and not has_prefix:
            extra_args.extend(["--prefix", sys.prefix])

        self.shell.system(" ".join([cmd, command] + extra_args + args))
        print("\nNote: you may need to restart the kernel to use updated packages.")

    @line_magic
    @is_conda_environment
    def conda(self, line):
        """Run the conda package manager within the current kernel.

        Usage:
          %conda install [pkgs]
        """
        conda = _get_conda_like_executable("conda")
        self._run_command(conda, line)

    @line_magic
    @is_conda_environment
    def mamba(self, line):
        """Run the mamba package manager within the current kernel.

        Usage:
          %mamba install [pkgs]
        """
        mamba = _get_conda_like_executable("mamba")
        self._run_command(mamba, line)

    @line_magic
    @is_conda_environment
    def micromamba(self, line):
        """Run the conda package manager within the current kernel.

        Usage:
          %micromamba install [pkgs]
        """
        micromamba = _get_conda_like_executable("micromamba")
        self._run_command(micromamba, line)
