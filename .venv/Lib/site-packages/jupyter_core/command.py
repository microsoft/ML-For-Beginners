# PYTHON_ARGCOMPLETE_OK
"""The root `jupyter` command.

This does nothing other than dispatch to subcommands or output path info.
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import argparse
import errno
import json
import os
import site
import sys
import sysconfig
from shutil import which
from subprocess import Popen
from typing import List

from . import paths
from .version import __version__


class JupyterParser(argparse.ArgumentParser):
    """A Jupyter argument parser."""

    @property
    def epilog(self):
        """Add subcommands to epilog on request

        Avoids searching PATH for subcommands unless help output is requested.
        """
        return "Available subcommands: %s" % " ".join(list_subcommands())

    @epilog.setter
    def epilog(self, x):
        """Ignore epilog set in Parser.__init__"""
        pass

    def argcomplete(self):
        """Trigger auto-completion, if enabled"""
        try:
            import argcomplete  # type: ignore[import]

            argcomplete.autocomplete(self)
        except ImportError:
            pass


def jupyter_parser() -> JupyterParser:
    """Create a jupyter parser object."""
    parser = JupyterParser(
        description="Jupyter: Interactive Computing",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    # don't use argparse's version action because it prints to stderr on py2
    group.add_argument(
        "--version", action="store_true", help="show the versions of core jupyter packages and exit"
    )
    subcommand_action = group.add_argument(
        "subcommand", type=str, nargs="?", help="the subcommand to launch"
    )
    # For argcomplete, supply all known subcommands
    subcommand_action.completer = lambda *args, **kwargs: list_subcommands()  # type: ignore[attr-defined]

    group.add_argument("--config-dir", action="store_true", help="show Jupyter config dir")
    group.add_argument("--data-dir", action="store_true", help="show Jupyter data dir")
    group.add_argument("--runtime-dir", action="store_true", help="show Jupyter runtime dir")
    group.add_argument(
        "--paths",
        action="store_true",
        help="show all Jupyter paths. Add --json for machine-readable format.",
    )
    parser.add_argument("--json", action="store_true", help="output paths as machine-readable json")
    parser.add_argument("--debug", action="store_true", help="output debug information about paths")

    return parser


def list_subcommands() -> List[str]:
    """List all jupyter subcommands

    searches PATH for `jupyter-name`

    Returns a list of jupyter's subcommand names, without the `jupyter-` prefix.
    Nested children (e.g. jupyter-sub-subsub) are not included.
    """
    subcommand_tuples = set()
    # construct a set of `('foo', 'bar') from `jupyter-foo-bar`
    for d in _path_with_self():
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if name.startswith("jupyter-"):
                if sys.platform.startswith("win"):
                    # remove file-extension on Windows
                    name = os.path.splitext(name)[0]  # noqa
                subcommand_tuples.add(tuple(name.split("-")[1:]))
    # build a set of subcommand strings, excluding subcommands whose parents are defined
    subcommands = set()
    # Only include `jupyter-foo-bar` if `jupyter-foo` is not already present
    for sub_tup in subcommand_tuples:
        if not any(sub_tup[:i] in subcommand_tuples for i in range(1, len(sub_tup))):
            subcommands.add("-".join(sub_tup))
    return sorted(subcommands)


def _execvp(cmd, argv):
    """execvp, except on Windows where it uses Popen

    Python provides execvp on Windows, but its behavior is problematic (Python bug#9148).
    """
    if sys.platform.startswith("win"):
        # PATH is ignored when shell=False,
        # so rely on shutil.which
        cmd_path = which(cmd)
        if cmd_path is None:
            raise OSError("%r not found" % cmd, errno.ENOENT)
        p = Popen([cmd_path] + argv[1:])  # noqa
        # Don't raise KeyboardInterrupt in the parent process.
        # Set this after spawning, to avoid subprocess inheriting handler.
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        p.wait()
        sys.exit(p.returncode)
    else:
        os.execvp(cmd, argv)  # noqa


def _jupyter_abspath(subcommand):
    """This method get the abspath of a specified jupyter-subcommand with no
    changes on ENV.
    """
    # get env PATH with self
    search_path = os.pathsep.join(_path_with_self())
    # get the abs path for the jupyter-<subcommand>
    jupyter_subcommand = f"jupyter-{subcommand}"
    abs_path = which(jupyter_subcommand, path=search_path)
    if abs_path is None:
        msg = f"\nJupyter command `{jupyter_subcommand}` not found."
        raise Exception(msg)

    if not os.access(abs_path, os.X_OK):
        msg = f"\nJupyter command `{jupyter_subcommand}` is not executable."
        raise Exception(msg)

    return abs_path


def _path_with_self():
    """Put `jupyter`'s dir at the front of PATH

    Ensures that /path/to/jupyter subcommand
    will do /path/to/jupyter-subcommand
    even if /other/jupyter-subcommand is ahead of it on PATH
    """
    path_list = (os.environ.get("PATH") or os.defpath).split(os.pathsep)

    # Insert the "scripts" directory for this Python installation
    # This allows the "jupyter" command to be relocated, while still
    # finding subcommands that have been installed in the default
    # location.
    # We put the scripts directory at the *end* of PATH, so that
    # if the user explicitly overrides a subcommand, that override
    # still takes effect.
    try:
        bindir = sysconfig.get_path("scripts")
    except KeyError:
        # The Python environment does not specify a "scripts" location
        pass
    else:
        path_list.append(bindir)

    scripts = [sys.argv[0]]
    if os.path.islink(scripts[0]):
        # include realpath, if `jupyter` is a symlink
        scripts.append(os.path.realpath(scripts[0]))

    for script in scripts:
        bindir = os.path.dirname(script)
        if os.path.isdir(bindir) and os.access(script, os.X_OK):  # only if it's a script
            # ensure executable's dir is on PATH
            # avoids missing subcommands when jupyter is run via absolute path
            path_list.insert(0, bindir)
    return path_list


def _evaluate_argcomplete(parser: JupyterParser) -> List[str]:
    """If argcomplete is enabled, trigger autocomplete or return current words

    If the first word looks like a subcommand, return the current command
    that is attempting to be completed so that the subcommand can evaluate it;
    otherwise auto-complete using the main parser.
    """
    try:
        # traitlets >= 5.8 provides some argcomplete support,
        # use helper methods to jump to argcomplete
        from traitlets.config.argcomplete_config import (
            get_argcomplete_cwords,
            increment_argcomplete_index,
        )

        cwords = get_argcomplete_cwords()
        if cwords and len(cwords) > 1 and not cwords[1].startswith("-"):
            # If first completion word looks like a subcommand,
            # increment word from which to start handling arguments
            increment_argcomplete_index()
            return cwords
        else:
            # Otherwise no subcommand, directly autocomplete and exit
            parser.argcomplete()
    except ImportError:
        # traitlets >= 5.8 not available, just try to complete this without
        # worrying about subcommands
        parser.argcomplete()
    msg = "Control flow should not reach end of autocomplete()"
    raise AssertionError(msg)


def main() -> None:  # noqa
    """The command entry point."""
    parser = jupyter_parser()
    argv = sys.argv
    subcommand = None
    if "_ARGCOMPLETE" in os.environ:
        argv = _evaluate_argcomplete(parser)
        subcommand = argv[1]
    elif len(argv) > 1 and not argv[1].startswith("-"):
        # Don't parse if a subcommand is given
        # Avoids argparse gobbling up args passed to subcommand, such as `-h`.
        subcommand = argv[1]
    else:
        args, opts = parser.parse_known_args()
        subcommand = args.subcommand
        if args.version:
            print("Selected Jupyter core packages...")
            for package in [
                "IPython",
                "ipykernel",
                "ipywidgets",
                "jupyter_client",
                "jupyter_core",
                "jupyter_server",
                "jupyterlab",
                "nbclient",
                "nbconvert",
                "nbformat",
                "notebook",
                "qtconsole",
                "traitlets",
            ]:
                try:
                    if package == "jupyter_core":  # We're already here
                        version = __version__
                    else:
                        mod = __import__(package)
                        version = mod.__version__
                except ImportError:
                    version = "not installed"
                print(f"{package:<17}:", version)
            return
        if args.json and not args.paths:
            sys.exit("--json is only used with --paths")
        if args.debug and not args.paths:
            sys.exit("--debug is only used with --paths")
        if args.debug and args.json:
            sys.exit("--debug cannot be used with --json")
        if args.config_dir:
            print(paths.jupyter_config_dir())
            return
        if args.data_dir:
            print(paths.jupyter_data_dir())
            return
        if args.runtime_dir:
            print(paths.jupyter_runtime_dir())
            return
        if args.paths:
            data = {}
            data["runtime"] = [paths.jupyter_runtime_dir()]
            data["config"] = paths.jupyter_config_path()
            data["data"] = paths.jupyter_path()
            if args.json:
                print(json.dumps(data))
            else:
                if args.debug:
                    env = os.environ

                    if paths.use_platform_dirs():
                        print(
                            "JUPYTER_PLATFORM_DIRS is set to a true value, so we use platformdirs to find platform-specific directories"
                        )
                    else:
                        print(
                            "JUPYTER_PLATFORM_DIRS is set to a false value, or is not set, so we use hardcoded legacy paths for platform-specific directories"
                        )

                    if paths.prefer_environment_over_user():
                        print(
                            "JUPYTER_PREFER_ENV_PATH is set to a true value, or JUPYTER_PREFER_ENV_PATH is not set and we detected a virtual environment, making the environment-level path preferred over the user-level path for data and config"
                        )
                    else:
                        print(
                            "JUPYTER_PREFER_ENV_PATH is set to a false value, or JUPYTER_PREFER_ENV_PATH is not set and we did not detect a virtual environment, making the user-level path preferred over the environment-level path for data and config"
                        )

                    # config path list
                    if env.get("JUPYTER_NO_CONFIG"):
                        print(
                            "JUPYTER_NO_CONFIG is set, making the config path list only a single temporary directory"
                        )
                    else:
                        print(
                            "JUPYTER_NO_CONFIG is not set, so we use the full path list for config"
                        )

                    if env.get("JUPYTER_CONFIG_PATH"):
                        print(
                            f"JUPYTER_CONFIG_PATH is set to '{env.get('JUPYTER_CONFIG_PATH')}', which is prepended to the config path list (unless JUPYTER_NO_CONFIG is set)"
                        )
                    else:
                        print(
                            "JUPYTER_CONFIG_PATH is not set, so we do not prepend anything to the config paths"
                        )

                    if env.get("JUPYTER_CONFIG_DIR"):
                        print(
                            f"JUPYTER_CONFIG_DIR is set to '{env.get('JUPYTER_CONFIG_DIR')}', overriding the default user-level config directory"
                        )
                    else:
                        print(
                            "JUPYTER_CONFIG_DIR is not set, so we use the default user-level config directory"
                        )

                    if site.ENABLE_USER_SITE:
                        print(
                            f"Python's site.ENABLE_USER_SITE is True, so we add the user site directory '{site.getuserbase()}'"
                        )
                    else:
                        print(
                            f"Python's site.ENABLE_USER_SITE is not True, so we do not add the Python site user directory '{site.getuserbase()}'"
                        )

                    # data path list
                    if env.get("JUPYTER_PATH"):
                        print(
                            f"JUPYTER_PATH is set to '{env.get('JUPYTER_PATH')}', which is prepended to the data paths"
                        )
                    else:
                        print(
                            "JUPYTER_PATH is not set, so we do not prepend anything to the data paths"
                        )

                    if env.get("JUPYTER_DATA_DIR"):
                        print(
                            f"JUPYTER_DATA_DIR is set to '{env.get('JUPYTER_DATA_DIR')}', overriding the default user-level data directory"
                        )
                    else:
                        print(
                            "JUPYTER_DATA_DIR is not set, so we use the default user-level data directory"
                        )

                    # runtime directory
                    if env.get("JUPYTER_RUNTIME_DIR"):
                        print(
                            f"JUPYTER_RUNTIME_DIR is set to '{env.get('JUPYTER_RUNTIME_DIR')}', overriding the default runtime directory"
                        )
                    else:
                        print(
                            "JUPYTER_RUNTIME_DIR is not set, so we use the default runtime directory"
                        )

                    print()

                for name in sorted(data):
                    path = data[name]
                    print("%s:" % name)
                    for p in path:
                        print("    " + p)
            return

    if not subcommand:
        parser.print_help(file=sys.stderr)
        sys.exit("\nPlease specify a subcommand or one of the optional arguments.")

    try:
        command = _jupyter_abspath(subcommand)
    except Exception as e:
        parser.print_help(file=sys.stderr)
        # special-case alias of "jupyter help" to "jupyter --help"
        if subcommand == "help":
            return
        sys.exit(str(e))

    try:
        _execvp(command, [command] + argv[2:])
    except OSError as e:
        sys.exit(f"Error executing Jupyter command {subcommand!r}: {e}")


if __name__ == "__main__":
    main()
