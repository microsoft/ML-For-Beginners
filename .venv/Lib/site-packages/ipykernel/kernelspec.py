"""The IPython kernel spec for Jupyter"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import errno
import json
import os
import shutil
import stat
import sys
import tempfile

from jupyter_client.kernelspec import KernelSpecManager
from traitlets import Unicode

try:
    from .debugger import _is_debugpy_available
except ImportError:
    _is_debugpy_available = False

pjoin = os.path.join

KERNEL_NAME = "python%i" % sys.version_info[0]

# path to kernelspec resources
RESOURCES = pjoin(os.path.dirname(__file__), "resources")


def make_ipkernel_cmd(mod="ipykernel_launcher", executable=None, extra_arguments=None):
    """Build Popen command list for launching an IPython kernel.

    Parameters
    ----------
    mod : str, optional (default 'ipykernel')
        A string of an IPython module whose __main__ starts an IPython kernel
    executable : str, optional (default sys.executable)
        The Python executable to use for the kernel process.
    extra_arguments : list, optional
        A list of extra arguments to pass when executing the launch code.

    Returns
    -------
    A Popen command list
    """
    if executable is None:
        executable = sys.executable
    extra_arguments = extra_arguments or []
    arguments = [executable, "-m", mod, "-f", "{connection_file}"]
    arguments.extend(extra_arguments)

    return arguments


def get_kernel_dict(extra_arguments=None):
    """Construct dict for kernel.json"""
    return {
        "argv": make_ipkernel_cmd(extra_arguments=extra_arguments),
        "display_name": "Python %i (ipykernel)" % sys.version_info[0],
        "language": "python",
        "metadata": {"debugger": _is_debugpy_available},
    }


def write_kernel_spec(path=None, overrides=None, extra_arguments=None):
    """Write a kernel spec directory to `path`

    If `path` is not specified, a temporary directory is created.
    If `overrides` is given, the kernelspec JSON is updated before writing.

    The path to the kernelspec is always returned.
    """
    if path is None:
        path = os.path.join(tempfile.mkdtemp(suffix="_kernels"), KERNEL_NAME)

    # stage resources
    shutil.copytree(RESOURCES, path)

    # ensure path is writable
    mask = os.stat(path).st_mode
    if not mask & stat.S_IWUSR:
        os.chmod(path, mask | stat.S_IWUSR)

    # write kernel.json
    kernel_dict = get_kernel_dict(extra_arguments)

    if overrides:
        kernel_dict.update(overrides)
    with open(pjoin(path, "kernel.json"), "w") as f:
        json.dump(kernel_dict, f, indent=1)

    return path


def install(
    kernel_spec_manager=None,
    user=False,
    kernel_name=KERNEL_NAME,
    display_name=None,
    prefix=None,
    profile=None,
    env=None,
):
    """Install the IPython kernelspec for Jupyter

    Parameters
    ----------
    kernel_spec_manager : KernelSpecManager [optional]
        A KernelSpecManager to use for installation.
        If none provided, a default instance will be created.
    user : bool [default: False]
        Whether to do a user-only install, or system-wide.
    kernel_name : str, optional
        Specify a name for the kernelspec.
        This is needed for having multiple IPython kernels for different environments.
    display_name : str, optional
        Specify the display name for the kernelspec
    profile : str, optional
        Specify a custom profile to be loaded by the kernel.
    prefix : str, optional
        Specify an install prefix for the kernelspec.
        This is needed to install into a non-default location, such as a conda/virtual-env.
    env : dict, optional
        A dictionary of extra environment variables for the kernel.
        These will be added to the current environment variables before the
        kernel is started

    Returns
    -------
    The path where the kernelspec was installed.
    """
    if kernel_spec_manager is None:
        kernel_spec_manager = KernelSpecManager()

    if (kernel_name != KERNEL_NAME) and (display_name is None):
        # kernel_name is specified and display_name is not
        # default display_name to kernel_name
        display_name = kernel_name
    overrides = {}
    if display_name:
        overrides["display_name"] = display_name
    if profile:
        extra_arguments = ["--profile", profile]
        if not display_name:
            # add the profile to the default display name
            overrides["display_name"] = "Python %i [profile=%s]" % (sys.version_info[0], profile)
    else:
        extra_arguments = None
    if env:
        overrides["env"] = env
    path = write_kernel_spec(overrides=overrides, extra_arguments=extra_arguments)
    dest = kernel_spec_manager.install_kernel_spec(
        path, kernel_name=kernel_name, user=user, prefix=prefix
    )
    # cleanup afterward
    shutil.rmtree(path)
    return dest


# Entrypoint

from traitlets.config import Application


class InstallIPythonKernelSpecApp(Application):
    """Dummy app wrapping argparse"""

    name = Unicode("ipython-kernel-install")

    def initialize(self, argv=None):
        """Initialize the app."""
        if argv is None:
            argv = sys.argv[1:]
        self.argv = argv

    def start(self):
        """Start the app."""
        import argparse

        parser = argparse.ArgumentParser(
            prog=self.name, description="Install the IPython kernel spec."
        )
        parser.add_argument(
            "--user",
            action="store_true",
            help="Install for the current user instead of system-wide",
        )
        parser.add_argument(
            "--name",
            type=str,
            default=KERNEL_NAME,
            help="Specify a name for the kernelspec."
            " This is needed to have multiple IPython kernels at the same time.",
        )
        parser.add_argument(
            "--display-name",
            type=str,
            help="Specify the display name for the kernelspec."
            " This is helpful when you have multiple IPython kernels.",
        )
        parser.add_argument(
            "--profile",
            type=str,
            help="Specify an IPython profile to load. "
            "This can be used to create custom versions of the kernel.",
        )
        parser.add_argument(
            "--prefix",
            type=str,
            help="Specify an install prefix for the kernelspec."
            " This is needed to install into a non-default location, such as a conda/virtual-env.",
        )
        parser.add_argument(
            "--sys-prefix",
            action="store_const",
            const=sys.prefix,
            dest="prefix",
            help="Install to Python's sys.prefix."
            " Shorthand for --prefix='%s'. For use in conda/virtual-envs." % sys.prefix,
        )
        parser.add_argument(
            "--env",
            action="append",
            nargs=2,
            metavar=("ENV", "VALUE"),
            help="Set environment variables for the kernel.",
        )
        opts = parser.parse_args(self.argv)
        if opts.env:
            opts.env = dict(opts.env)
        try:
            dest = install(
                user=opts.user,
                kernel_name=opts.name,
                profile=opts.profile,
                prefix=opts.prefix,
                display_name=opts.display_name,
                env=opts.env,
            )
        except OSError as e:
            if e.errno == errno.EACCES:
                print(e, file=sys.stderr)
                if opts.user:
                    print("Perhaps you want `sudo` or `--user`?", file=sys.stderr)
                self.exit(1)
            raise
        print(f"Installed kernelspec {opts.name} in {dest}")


if __name__ == "__main__":
    InstallIPythonKernelSpecApp.launch_instance()
