# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import contextlib
from importlib import import_module
import os
import sys

from . import _util


VENDORED_ROOT = os.path.dirname(os.path.abspath(__file__))
# TODO: Move the "pydevd" git submodule to the debugpy/_vendored directory
# and then drop the following fallback.
if "pydevd" not in os.listdir(VENDORED_ROOT):
    VENDORED_ROOT = os.path.dirname(VENDORED_ROOT)


def list_all(resolve=False):
    """Return the list of vendored projects."""
    # TODO: Derive from os.listdir(VENDORED_ROOT)?
    projects = ["pydevd"]
    if not resolve:
        return projects
    return [project_root(name) for name in projects]


def project_root(project):
    """Return the path the root dir of the vendored project.

    If "project" is an empty string then the path prefix for vendored
    projects (e.g. "debugpy/_vendored/") will be returned.
    """
    if not project:
        project = ""
    return os.path.join(VENDORED_ROOT, project)


def iter_project_files(project, relative=False, **kwargs):
    """Yield (dirname, basename, filename) for all files in the project."""
    if relative:
        with _util.cwd(VENDORED_ROOT):
            for result in _util.iter_all_files(project, **kwargs):
                yield result
    else:
        root = project_root(project)
        for result in _util.iter_all_files(root, **kwargs):
            yield result


def iter_packaging_files(project):
    """Yield the filenames for all files in the project.

    The filenames are relative to "debugpy/_vendored".  This is most
    useful for the "package data" in a setup.py.
    """
    # TODO: Use default filters?  __pycache__ and .pyc?
    prune_dir = None
    exclude_file = None
    try:
        mod = import_module("._{}_packaging".format(project), __name__)
    except ImportError:
        pass
    else:
        prune_dir = getattr(mod, "prune_dir", prune_dir)
        exclude_file = getattr(mod, "exclude_file", exclude_file)
    results = iter_project_files(
        project, relative=True, prune_dir=prune_dir, exclude_file=exclude_file
    )
    for _, _, filename in results:
        yield filename


def prefix_matcher(*prefixes):
    """Return a module match func that matches any of the given prefixes."""
    assert prefixes

    def match(name, module):
        for prefix in prefixes:
            if name.startswith(prefix):
                return True
        else:
            return False

    return match


def check_modules(project, match, root=None):
    """Verify that only vendored modules have been imported."""
    if root is None:
        root = project_root(project)
    extensions = []
    unvendored = {}
    for modname, mod in list(sys.modules.items()):
        if not match(modname, mod):
            continue
        try:
            filename = getattr(mod, "__file__", None)
        except:  # In theory it's possible that any error is raised when accessing __file__
            filename = None
        if not filename:  # extension module
            extensions.append(modname)
        elif not filename.startswith(root):
            unvendored[modname] = filename
    return unvendored, extensions


@contextlib.contextmanager
def vendored(project, root=None):
    """A context manager under which the vendored project will be imported."""
    if root is None:
        root = project_root(project)
    # Add the vendored project directory, so that it gets tried first.
    sys.path.insert(0, root)
    try:
        yield root
    finally:
        sys.path.remove(root)


def preimport(project, modules, **kwargs):
    """Import each of the named modules out of the vendored project."""
    with vendored(project, **kwargs):
        for name in modules:
            import_module(name)
