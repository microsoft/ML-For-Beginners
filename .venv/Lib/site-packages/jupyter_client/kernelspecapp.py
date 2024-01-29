"""Apps for managing kernel specs."""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import errno
import json
import os.path
import sys
import typing as t

from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, Dict, Instance, List, Unicode
from traitlets.config.application import Application

from . import __version__
from .kernelspec import KernelSpecManager
from .provisioning.factory import KernelProvisionerFactory


class ListKernelSpecs(JupyterApp):
    """An app to list kernel specs."""

    version = __version__
    description = """List installed kernel specifications."""
    kernel_spec_manager = Instance(KernelSpecManager)
    json_output = Bool(
        False,
        help="output spec name and location as machine-readable json.",
        config=True,
    )

    flags = {
        "json": (
            {"ListKernelSpecs": {"json_output": True}},
            "output spec name and location as machine-readable json.",
        ),
        "debug": base_flags["debug"],
    }

    def _kernel_spec_manager_default(self) -> KernelSpecManager:
        return KernelSpecManager(parent=self, data_dir=self.data_dir)

    def start(self) -> dict[str, t.Any] | None:  # type:ignore[override]
        """Start the application."""
        paths = self.kernel_spec_manager.find_kernel_specs()
        specs = self.kernel_spec_manager.get_all_specs()
        if not self.json_output:
            if not specs:
                print("No kernels available")
                return None
            # pad to width of longest kernel name
            name_len = len(sorted(paths, key=lambda name: len(name))[-1])

            def path_key(item: t.Any) -> t.Any:
                """sort key function for Jupyter path priority"""
                path = item[1]
                for idx, prefix in enumerate(self.jupyter_path):
                    if path.startswith(prefix):
                        return (idx, path)
                # not in jupyter path, artificially added to the front
                return (-1, path)

            print("Available kernels:")
            for kernelname, path in sorted(paths.items(), key=path_key):
                print(f"  {kernelname.ljust(name_len)}    {path}")
        else:
            print(json.dumps({"kernelspecs": specs}, indent=2))
        return specs


class InstallKernelSpec(JupyterApp):
    """An app to install a kernel spec."""

    version = __version__
    description = """Install a kernel specification directory.

    Given a SOURCE DIRECTORY containing a kernel spec,
    jupyter will copy that directory into one of the Jupyter kernel directories.
    The default is to install kernelspecs for all users.
    `--user` can be specified to install a kernel only for the current user.
    """
    examples = """
    jupyter kernelspec install /path/to/my_kernel --user
    """
    usage = "jupyter kernelspec install SOURCE_DIR [--options]"
    kernel_spec_manager = Instance(KernelSpecManager)

    def _kernel_spec_manager_default(self) -> KernelSpecManager:
        return KernelSpecManager(data_dir=self.data_dir)

    sourcedir = Unicode()
    kernel_name = Unicode("", config=True, help="Install the kernel spec with this name")

    def _kernel_name_default(self) -> str:
        return os.path.basename(self.sourcedir)

    user = Bool(
        False,
        config=True,
        help="""
        Try to install the kernel spec to the per-user directory instead of
        the system or environment directory.
        """,
    )
    prefix = Unicode(
        "",
        config=True,
        help="""Specify a prefix to install to, e.g. an env.
        The kernelspec will be installed in PREFIX/share/jupyter/kernels/
        """,
    )
    replace = Bool(False, config=True, help="Replace any existing kernel spec with this name.")

    aliases = {
        "name": "InstallKernelSpec.kernel_name",
        "prefix": "InstallKernelSpec.prefix",
    }
    aliases.update(base_aliases)

    flags = {
        "user": (
            {"InstallKernelSpec": {"user": True}},
            "Install to the per-user kernel registry",
        ),
        "replace": (
            {"InstallKernelSpec": {"replace": True}},
            "Replace any existing kernel spec with this name.",
        ),
        "sys-prefix": (
            {"InstallKernelSpec": {"prefix": sys.prefix}},
            "Install to Python's sys.prefix. Useful in conda/virtual environments.",
        ),
        "debug": base_flags["debug"],
    }

    def parse_command_line(self, argv: None | list[str]) -> None:  # type:ignore[override]
        """Parse the command line args."""
        super().parse_command_line(argv)
        # accept positional arg as profile name
        if self.extra_args:
            self.sourcedir = self.extra_args[0]
        else:
            print("No source directory specified.", file=sys.stderr)
            self.exit(1)

    def start(self) -> None:
        """Start the application."""
        if self.user and self.prefix:
            self.exit("Can't specify both user and prefix. Please choose one or the other.")
        try:
            self.kernel_spec_manager.install_kernel_spec(
                self.sourcedir,
                kernel_name=self.kernel_name,
                user=self.user,
                prefix=self.prefix,
                replace=self.replace,
            )
        except OSError as e:
            if e.errno == errno.EACCES:
                print(e, file=sys.stderr)
                if not self.user:
                    print("Perhaps you want to install with `sudo` or `--user`?", file=sys.stderr)
                self.exit(1)
            elif e.errno == errno.EEXIST:
                print(f"A kernel spec is already present at {e.filename}", file=sys.stderr)
                self.exit(1)
            raise


class RemoveKernelSpec(JupyterApp):
    """An app to remove a kernel spec."""

    version = __version__
    description = """Remove one or more Jupyter kernelspecs by name."""
    examples = """jupyter kernelspec remove python2 [my_kernel ...]"""

    force = Bool(False, config=True, help="""Force removal, don't prompt for confirmation.""")
    spec_names = List(Unicode())

    kernel_spec_manager = Instance(KernelSpecManager)

    def _kernel_spec_manager_default(self) -> KernelSpecManager:
        return KernelSpecManager(data_dir=self.data_dir, parent=self)

    flags = {
        "f": ({"RemoveKernelSpec": {"force": True}}, force.help),
    }
    flags.update(JupyterApp.flags)

    def parse_command_line(self, argv: list[str] | None) -> None:  # type:ignore[override]
        """Parse the command line args."""
        super().parse_command_line(argv)
        # accept positional arg as profile name
        if self.extra_args:
            self.spec_names = sorted(set(self.extra_args))  # remove duplicates
        else:
            self.exit("No kernelspec specified.")

    def start(self) -> None:
        """Start the application."""
        self.kernel_spec_manager.ensure_native_kernel = False
        spec_paths = self.kernel_spec_manager.find_kernel_specs()
        missing = set(self.spec_names).difference(set(spec_paths))
        if missing:
            self.exit("Couldn't find kernel spec(s): %s" % ", ".join(missing))

        if not (self.force or self.answer_yes):
            print("Kernel specs to remove:")
            for name in self.spec_names:
                path = spec_paths.get(name, name)
                print(f"  {name.ljust(20)}\t{path.ljust(20)}")
            answer = input("Remove %i kernel specs [y/N]: " % len(self.spec_names))
            if not answer.lower().startswith("y"):
                return

        for kernel_name in self.spec_names:
            try:
                path = self.kernel_spec_manager.remove_kernel_spec(kernel_name)
            except OSError as e:
                if e.errno == errno.EACCES:
                    print(e, file=sys.stderr)
                    print("Perhaps you want sudo?", file=sys.stderr)
                    self.exit(1)
                else:
                    raise
            print(f"Removed {path}")


class InstallNativeKernelSpec(JupyterApp):
    """An app to install the native kernel spec."""

    version = __version__
    description = """[DEPRECATED] Install the IPython kernel spec directory for this Python."""
    kernel_spec_manager = Instance(KernelSpecManager)

    def _kernel_spec_manager_default(self) -> KernelSpecManager:  # pragma: no cover
        return KernelSpecManager(data_dir=self.data_dir)

    user = Bool(
        False,
        config=True,
        help="""
        Try to install the kernel spec to the per-user directory instead of
        the system or environment directory.
        """,
    )

    flags = {
        "user": (
            {"InstallNativeKernelSpec": {"user": True}},
            "Install to the per-user kernel registry",
        ),
        "debug": base_flags["debug"],
    }

    def start(self) -> None:  # pragma: no cover
        """Start the application."""
        self.log.warning(
            "`jupyter kernelspec install-self` is DEPRECATED as of 4.0."
            " You probably want `ipython kernel install` to install the IPython kernelspec."
        )
        try:
            from ipykernel import kernelspec
        except ModuleNotFoundError:
            print("ipykernel not available, can't install its spec.", file=sys.stderr)
            self.exit(1)
        try:
            kernelspec.install(self.kernel_spec_manager, user=self.user)
        except OSError as e:
            if e.errno == errno.EACCES:
                print(e, file=sys.stderr)
                if not self.user:
                    print(
                        "Perhaps you want to install with `sudo` or `--user`?",
                        file=sys.stderr,
                    )
                self.exit(1)
            self.exit(e)  # type:ignore[arg-type]


class ListProvisioners(JupyterApp):
    """An app to list provisioners."""

    version = __version__
    description = """List available provisioners for use in kernel specifications."""

    def start(self) -> None:
        """Start the application."""
        kfp = KernelProvisionerFactory.instance(parent=self)
        print("Available kernel provisioners:")
        provisioners = kfp.get_provisioner_entries()

        # pad to width of longest kernel name
        name_len = len(sorted(provisioners, key=lambda name: len(name))[-1])

        for name in sorted(provisioners):
            print(f"  {name.ljust(name_len)}    {provisioners[name]}")


class KernelSpecApp(Application):
    """An app to manage kernel specs."""

    version = __version__
    name = "jupyter kernelspec"
    description = """Manage Jupyter kernel specifications."""

    subcommands = Dict(
        {
            "list": (ListKernelSpecs, ListKernelSpecs.description.splitlines()[0]),
            "install": (
                InstallKernelSpec,
                InstallKernelSpec.description.splitlines()[0],
            ),
            "uninstall": (RemoveKernelSpec, "Alias for remove"),
            "remove": (RemoveKernelSpec, RemoveKernelSpec.description.splitlines()[0]),
            "install-self": (
                InstallNativeKernelSpec,
                InstallNativeKernelSpec.description.splitlines()[0],
            ),
            "provisioners": (ListProvisioners, ListProvisioners.description.splitlines()[0]),
        }
    )

    aliases = {}
    flags = {}

    def start(self) -> None:
        """Start the application."""
        if self.subapp is None:
            print("No subcommand specified. Must specify one of: %s" % list(self.subcommands))
            print()
            self.print_description()
            self.print_subcommands()
            self.exit(1)
        else:
            return self.subapp.start()


if __name__ == "__main__":
    KernelSpecApp.launch_instance()
