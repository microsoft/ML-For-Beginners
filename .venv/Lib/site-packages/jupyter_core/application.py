"""
A base Application class for Jupyter applications.

All Jupyter applications should inherit from this.
"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import logging
import os
import sys
import typing as t
from copy import deepcopy
from pathlib import Path
from shutil import which

from traitlets import Bool, List, Unicode, observe
from traitlets.config.application import Application, catch_config_error
from traitlets.config.loader import ConfigFileNotFound

from .paths import (
    allow_insecure_writes,
    issue_insecure_write_warning,
    jupyter_config_dir,
    jupyter_config_path,
    jupyter_data_dir,
    jupyter_path,
    jupyter_runtime_dir,
)
from .utils import ensure_dir_exists, ensure_event_loop

# mypy: disable-error-code="no-untyped-call"

# aliases and flags

base_aliases: dict[str, t.Any] = {}
if isinstance(Application.aliases, dict):
    # traitlets 5
    base_aliases.update(Application.aliases)
_jupyter_aliases = {
    "log-level": "Application.log_level",
    "config": "JupyterApp.config_file",
}
base_aliases.update(_jupyter_aliases)

base_flags: dict[str, t.Any] = {}
if isinstance(Application.flags, dict):
    # traitlets 5
    base_flags.update(Application.flags)
_jupyter_flags: dict[str, t.Any] = {
    "debug": (
        {"Application": {"log_level": logging.DEBUG}},
        "set log level to logging.DEBUG (maximize logging output)",
    ),
    "generate-config": ({"JupyterApp": {"generate_config": True}}, "generate default config file"),
    "y": (
        {"JupyterApp": {"answer_yes": True}},
        "Answer yes to any questions instead of prompting.",
    ),
}
base_flags.update(_jupyter_flags)


class NoStart(Exception):
    """Exception to raise when an application shouldn't start"""


class JupyterApp(Application):
    """Base class for Jupyter applications"""

    name = "jupyter"  # override in subclasses
    description = "A Jupyter Application"

    aliases = base_aliases
    flags = base_flags

    def _log_level_default(self) -> int:
        return logging.INFO

    jupyter_path = List(Unicode())

    def _jupyter_path_default(self) -> list[str]:
        return jupyter_path()

    config_dir = Unicode()

    def _config_dir_default(self) -> str:
        return jupyter_config_dir()

    @property
    def config_file_paths(self) -> list[str]:
        path = jupyter_config_path()
        if self.config_dir not in path:
            # Insert config dir as first item.
            path.insert(0, self.config_dir)
        return path

    data_dir = Unicode()

    def _data_dir_default(self) -> str:
        d = jupyter_data_dir()
        ensure_dir_exists(d, mode=0o700)
        return d

    runtime_dir = Unicode()

    def _runtime_dir_default(self) -> str:
        rd = jupyter_runtime_dir()
        ensure_dir_exists(rd, mode=0o700)
        return rd

    @observe("runtime_dir")
    def _runtime_dir_changed(self, change: t.Any) -> None:
        ensure_dir_exists(change["new"], mode=0o700)

    generate_config = Bool(False, config=True, help="""Generate default config file.""")

    config_file_name = Unicode(config=True, help="Specify a config file to load.")

    def _config_file_name_default(self) -> str:
        if not self.name:
            return ""
        return self.name.replace("-", "_") + "_config"

    config_file = Unicode(
        config=True,
        help="""Full path of a config file.""",
    )

    answer_yes = Bool(False, config=True, help="""Answer yes to any prompts.""")

    def write_default_config(self) -> None:
        """Write our default config to a .py config file"""
        if self.config_file:
            config_file = self.config_file
        else:
            config_file = str(Path(self.config_dir, self.config_file_name + ".py"))

        if Path(config_file).exists() and not self.answer_yes:
            answer = ""

            def ask() -> str:
                prompt = "Overwrite %s with default config? [y/N]" % config_file
                try:
                    return input(prompt).lower() or "n"
                except KeyboardInterrupt:
                    print("")  # empty line
                    return "n"

            answer = ask()
            while not answer.startswith(("y", "n")):
                print("Please answer 'yes' or 'no'")
                answer = ask()
            if answer.startswith("n"):
                return

        config_text = self.generate_config_file()
        print("Writing default config to: %s" % config_file)
        ensure_dir_exists(Path(config_file).parent.resolve(), 0o700)
        with Path.open(Path(config_file), mode="w", encoding="utf-8") as f:
            f.write(config_text)

    def migrate_config(self) -> None:
        """Migrate config/data from IPython 3"""
        try:  # let's see if we can open the marker file
            # for reading and updating (writing)
            f_marker = Path.open(Path(self.config_dir, "migrated"), "r+")
        except FileNotFoundError:  # cannot find the marker file
            pass  # that means we have not migrated yet, so continue
        except OSError:  # not readable and/or writable
            return  # so let's give up migration in such an environment
        else:  # if we got here without raising anything,
            # that means the file exists
            f_marker.close()
            return  # so we must have already migrated -> bail out

        from .migrate import get_ipython_dir, migrate

        # No IPython dir, nothing to migrate
        if not Path(get_ipython_dir()).exists():
            return

        migrate()

    def load_config_file(self, suppress_errors: bool = True) -> None:  # type:ignore[override]
        """Load the config file.

        By default, errors in loading config are handled, and a warning
        printed on screen. For testing, the suppress_errors option is set
        to False, so errors will make tests fail.
        """
        self.log.debug("Searching %s for config files", self.config_file_paths)
        base_config = "jupyter_config"
        try:
            super().load_config_file(
                base_config,
                path=self.config_file_paths,
            )
        except ConfigFileNotFound:
            # ignore errors loading parent
            self.log.debug("Config file %s not found", base_config)

        if self.config_file:
            path, config_file_name = os.path.split(self.config_file)
        else:
            path = self.config_file_paths  # type:ignore[assignment]
            config_file_name = self.config_file_name

            if not config_file_name or (config_file_name == base_config):
                return

        try:
            super().load_config_file(config_file_name, path=path)
        except ConfigFileNotFound:
            self.log.debug("Config file not found, skipping: %s", config_file_name)
        except Exception:
            # Reraise errors for testing purposes, or if set in
            # self.raise_config_file_errors
            if (not suppress_errors) or self.raise_config_file_errors:
                raise
            self.log.warning("Error loading config file: %s", config_file_name, exc_info=True)

    # subcommand-related
    def _find_subcommand(self, name: str) -> str:
        name = f"{self.name}-{name}"
        return which(name) or ""

    @property
    def _dispatching(self) -> bool:
        """Return whether we are dispatching to another command

        or running ourselves.
        """
        return bool(self.generate_config or self.subapp or self.subcommand)

    subcommand = Unicode()

    @catch_config_error
    def initialize(self, argv: t.Any = None) -> None:
        """Initialize the application."""
        # don't hook up crash handler before parsing command-line
        if argv is None:
            argv = sys.argv[1:]
        if argv:
            subc = self._find_subcommand(argv[0])
            if subc:
                self.argv = argv
                self.subcommand = subc
                return
        self.parse_command_line(argv)
        cl_config = deepcopy(self.config)
        if self._dispatching:
            return
        self.migrate_config()
        self.load_config_file()
        # enforce cl-opts override configfile opts:
        self.update_config(cl_config)
        if allow_insecure_writes:
            issue_insecure_write_warning()

    def start(self) -> None:
        """Start the whole thing"""
        if self.subcommand:
            os.execv(self.subcommand, [self.subcommand] + self.argv[1:])  # noqa: S606
            raise NoStart()

        if self.subapp:
            self.subapp.start()
            raise NoStart()

        if self.generate_config:
            self.write_default_config()
            raise NoStart()

    @classmethod
    def launch_instance(cls, argv: t.Any = None, **kwargs: t.Any) -> None:
        """Launch an instance of a Jupyter Application"""
        # Ensure an event loop is set before any other code runs.
        loop = ensure_event_loop()
        try:
            super().launch_instance(argv=argv, **kwargs)
        except NoStart:
            return
        loop.close()


class JupyterAsyncApp(JupyterApp):
    """A Jupyter application that runs on an asyncio loop."""

    name = "jupyter_async"  # override in subclasses
    description = "An Async Jupyter Application"

    # Set to True for tornado-based apps.
    _prefer_selector_loop = False

    async def initialize_async(self, argv: t.Any = None) -> None:
        """Initialize the application asynchronoously."""

    async def start_async(self) -> None:
        """Run the application in an event loop."""

    @classmethod
    async def _launch_instance(cls, argv: t.Any = None, **kwargs: t.Any) -> None:
        app = cls.instance(**kwargs)
        app.initialize(argv)
        await app.initialize_async(argv)
        await app.start_async()

    @classmethod
    def launch_instance(cls, argv: t.Any = None, **kwargs: t.Any) -> None:
        """Launch an instance of an async Jupyter Application"""
        loop = ensure_event_loop(cls._prefer_selector_loop)
        coro = cls._launch_instance(argv, **kwargs)
        loop.run_until_complete(coro)
        loop.close()


if __name__ == "__main__":
    JupyterApp.launch_instance()
