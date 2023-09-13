# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import shutil
import sys
import tempfile
from unittest.mock import patch

from ipykernel.kernelspec import install

pjoin = os.path.join

tmp = None
patchers: list = []


def setup():
    """setup temporary env for tests"""
    global tmp
    tmp = tempfile.mkdtemp()
    patchers[:] = [
        patch.dict(
            os.environ,
            {
                "HOME": tmp,
                # Let tests work with --user install when HOME is changed:
                "PYTHONPATH": os.pathsep.join(sys.path),
            },
        ),
    ]
    for p in patchers:
        p.start()

    # install IPython in the temp home:
    install(user=True)


def teardown():
    for p in patchers:
        p.stop()

    try:
        shutil.rmtree(tmp)  # type:ignore
    except OSError:
        # no such file
        pass
