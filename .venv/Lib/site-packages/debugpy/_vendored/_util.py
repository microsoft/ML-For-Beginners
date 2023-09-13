# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import contextlib
import os


@contextlib.contextmanager
def cwd(dirname):
    """A context manager for operating in a different directory."""
    orig = os.getcwd()
    os.chdir(dirname)
    try:
        yield orig
    finally:
        os.chdir(orig)


def iter_all_files(root, prune_dir=None, exclude_file=None):
    """Yield (dirname, basename, filename) for each file in the tree.

    This is an alternative to os.walk() that flattens out the tree and
    with filtering.
    """
    pending = [root]
    while pending:
        dirname = pending.pop(0)
        for result in _iter_files(dirname, pending, prune_dir, exclude_file):
            yield result


def iter_tree(root, prune_dir=None, exclude_file=None):
    """Yield (dirname, files) for each directory in the tree.

    The list of files is actually a list of (basename, filename).

    This is an alternative to os.walk() with filtering."""
    pending = [root]
    while pending:
        dirname = pending.pop(0)
        files = []
        for _, b, f in _iter_files(dirname, pending, prune_dir, exclude_file):
            files.append((b, f))
        yield dirname, files


def _iter_files(dirname, subdirs, prune_dir, exclude_file):
    for basename in os.listdir(dirname):
        filename = os.path.join(dirname, basename)
        if os.path.isdir(filename):
            if prune_dir is not None and prune_dir(dirname, basename):
                continue
            subdirs.append(filename)
        else:
            # TODO: Use os.path.isfile() to narrow it down?
            if exclude_file is not None and exclude_file(dirname, basename):
                continue
            yield dirname, basename, filename
