# PYTHON_ARGCOMPLETE_OK
"""Migrating IPython < 4.0 to Jupyter

This *copies* configuration and resources to their new locations in Jupyter

Migrations:

- .ipython/
  - nbextensions -> JUPYTER_DATA_DIR/nbextensions
  - kernels ->  JUPYTER_DATA_DIR/kernels

- .ipython/profile_default/
  - static/custom -> .jupyter/custom
  - nbconfig -> .jupyter/nbconfig
  - security/

    - notebook_secret, notebook_cookie_secret, nbsignatures.db -> JUPYTER_DATA_DIR

  - ipython_{notebook,nbconvert,qtconsole}_config.py -> .jupyter/jupyter_{name}_config.py


"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from traitlets.config.loader import JSONFileConfigLoader, PyFileConfigLoader
from traitlets.log import get_logger

from .application import JupyterApp
from .paths import jupyter_config_dir, jupyter_data_dir
from .utils import ensure_dir_exists

# mypy: disable-error-code="no-untyped-call"


migrations = {
    str(Path("{ipython_dir}", "nbextensions")): str(Path("{jupyter_data}", "nbextensions")),
    str(Path("{ipython_dir}", "kernels")): str(Path("{jupyter_data}", "kernels")),
    str(Path("{profile}", "nbconfig")): str(Path("{jupyter_config}", "nbconfig")),
}

custom_src_t = str(Path("{profile}", "static", "custom"))
custom_dst_t = str(Path("{jupyter_config}", "custom"))

for security_file in ("notebook_secret", "notebook_cookie_secret", "nbsignatures.db"):
    src = str(Path("{profile}", "security", security_file))
    dst = str(Path("{jupyter_data}", security_file))
    migrations[src] = dst

config_migrations = ["notebook", "nbconvert", "qtconsole"]

regex = re.compile

config_substitutions = {
    regex(r"\bIPythonQtConsoleApp\b"): "JupyterQtConsoleApp",
    regex(r"\bIPythonWidget\b"): "JupyterWidget",
    regex(r"\bRichIPythonWidget\b"): "RichJupyterWidget",
    regex(r"\bIPython\.html\b"): "notebook",
    regex(r"\bIPython\.nbconvert\b"): "nbconvert",
}


def get_ipython_dir() -> str:
    """Return the IPython directory location.

    Not imported from IPython because the IPython implementation
    ensures that a writable directory exists,
    creating a temporary directory if not.
    We don't want to trigger that when checking if migration should happen.

    We only need to support the IPython < 4 behavior for migration,
    so importing for forward-compatibility and edge cases is not important.
    """
    return os.environ.get("IPYTHONDIR", str(Path("~/.ipython").expanduser()))


def migrate_dir(src: str, dst: str) -> bool:
    """Migrate a directory from src to dst"""
    log = get_logger()
    if not os.listdir(src):
        log.debug("No files in %s", src)
        return False
    if Path(dst).exists():
        if os.listdir(dst):
            # already exists, non-empty
            log.debug("%s already exists", dst)
            return False
        Path(dst).rmdir()
    log.info("Copying %s -> %s", src, dst)
    ensure_dir_exists(Path(dst).parent)
    shutil.copytree(src, dst, symlinks=True)
    return True


def migrate_file(src: str | Path, dst: str | Path, substitutions: Any = None) -> bool:
    """Migrate a single file from src to dst

    substitutions is an optional dict of {regex: replacement} for performing replacements on the file.
    """
    log = get_logger()
    if Path(dst).exists():
        # already exists
        log.debug("%s already exists", dst)
        return False
    log.info("Copying %s -> %s", src, dst)
    ensure_dir_exists(Path(dst).parent)
    shutil.copy(src, dst)
    if substitutions:
        with Path.open(Path(dst), encoding="utf-8") as f:
            text = f.read()
        for pat, replacement in substitutions.items():
            text = pat.sub(replacement, text)
        with Path.open(Path(dst), "w", encoding="utf-8") as f:
            f.write(text)
    return True


def migrate_one(src: str, dst: str) -> bool:
    """Migrate one item

    dispatches to migrate_dir/_file
    """
    log = get_logger()
    if Path(src).is_file():
        return migrate_file(src, dst)
    if Path(src).is_dir():
        return migrate_dir(src, dst)
    log.debug("Nothing to migrate for %s", src)
    return False


def migrate_static_custom(src: str, dst: str) -> bool:
    """Migrate non-empty custom.js,css from src to dst

    src, dst are 'custom' directories containing custom.{js,css}
    """
    log = get_logger()
    migrated = False

    custom_js = Path(src, "custom.js")
    custom_css = Path(src, "custom.css")
    # check if custom_js is empty:
    custom_js_empty = True
    if Path(custom_js).is_file():
        with Path.open(custom_js, encoding="utf-8") as f:
            js = f.read().strip()
            for line in js.splitlines():
                if not (line.isspace() or line.strip().startswith(("/*", "*", "//"))):
                    custom_js_empty = False
                    break

    # check if custom_css is empty:
    custom_css_empty = True
    if Path(custom_css).is_file():
        with Path.open(custom_css, encoding="utf-8") as f:
            css = f.read().strip()
            custom_css_empty = css.startswith("/*") and css.endswith("*/")

    if custom_js_empty:
        log.debug("Ignoring empty %s", custom_js)
    if custom_css_empty:
        log.debug("Ignoring empty %s", custom_css)

    if custom_js_empty and custom_css_empty:
        # nothing to migrate
        return False
    ensure_dir_exists(dst)

    if not custom_js_empty or not custom_css_empty:
        ensure_dir_exists(dst)

    if not custom_js_empty and migrate_file(custom_js, Path(dst, "custom.js")):
        migrated = True
    if not custom_css_empty and migrate_file(custom_css, Path(dst, "custom.css")):
        migrated = True

    return migrated


def migrate_config(name: str, env: Any) -> list[Any]:
    """Migrate a config file.

    Includes substitutions for updated configurable names.
    """
    log = get_logger()
    src_base = str(Path(f"{env['profile']}", f"ipython_{name}_config"))
    dst_base = str(Path(f"{env['jupyter_config']}", f"jupyter_{name}_config"))
    loaders = {
        ".py": PyFileConfigLoader,
        ".json": JSONFileConfigLoader,
    }
    migrated = []
    for ext in (".py", ".json"):
        src = src_base + ext
        dst = dst_base + ext
        if Path(src).exists():
            cfg = loaders[ext](src).load_config()
            if cfg:
                if migrate_file(src, dst, substitutions=config_substitutions):
                    migrated.append(src)
            else:
                # don't migrate empty config files
                log.debug("Not migrating empty config file: %s", src)
    return migrated


def migrate() -> bool:
    """Migrate IPython configuration to Jupyter"""
    env = {
        "jupyter_data": jupyter_data_dir(),
        "jupyter_config": jupyter_config_dir(),
        "ipython_dir": get_ipython_dir(),
        "profile": str(Path(get_ipython_dir(), "profile_default")),
    }
    migrated = False
    for src_t, dst_t in migrations.items():
        src = src_t.format(**env)
        dst = dst_t.format(**env)
        if Path(src).exists() and migrate_one(src, dst):
            migrated = True

    for name in config_migrations:
        if migrate_config(name, env):
            migrated = True

    custom_src = custom_src_t.format(**env)
    custom_dst = custom_dst_t.format(**env)

    if Path(custom_src).exists() and migrate_static_custom(custom_src, custom_dst):
        migrated = True

    # write a marker to avoid re-running migration checks
    ensure_dir_exists(env["jupyter_config"])
    with Path.open(Path(env["jupyter_config"], "migrated"), "w", encoding="utf-8") as f:
        f.write(datetime.now(tz=timezone.utc).isoformat())

    return migrated


class JupyterMigrate(JupyterApp):
    """A Jupyter Migration App."""

    name = "jupyter-migrate"
    description = """
    Migrate configuration and data from .ipython prior to 4.0 to Jupyter locations.

    This migrates:

    - config files in the default profile
    - kernels in ~/.ipython/kernels
    - notebook javascript extensions in ~/.ipython/extensions
    - custom.js/css to .jupyter/custom

    to their new Jupyter locations.

    All files are copied, not moved.
    If the destinations already exist, nothing will be done.
    """

    def start(self) -> None:
        """Start the application."""
        if not migrate():
            self.log.info("Found nothing to migrate.")


main = JupyterMigrate.launch_instance


if __name__ == "__main__":
    main()
