# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

from . import VENDORED_ROOT
from ._util import cwd, iter_all_files


INCLUDES = [
    'setup_pydevd_cython.py',
]


def iter_files():
    # From the root of pydevd repo, we want only scripts and
    # subdirectories that constitute the package itself (not helper
    # scripts, tests etc). But when walking down into those
    # subdirectories, we want everything below.

    with cwd(VENDORED_ROOT):
        return iter_all_files('pydevd', prune_dir, exclude_file)


def prune_dir(dirname, basename):
    if basename == '__pycache__':
        return True
    elif dirname != 'pydevd':
        return False
    elif basename.startswith('pydev'):
        return False
    elif basename.startswith('_pydev'):
        return False
    return True


def exclude_file(dirname, basename):
    if dirname == 'pydevd':
        if basename in INCLUDES:
            return False
        elif not basename.endswith('.py'):
            return True
        elif 'pydev' not in basename:
            return True
        return False

    if basename.endswith('.pyc'):
        return True
    return False
