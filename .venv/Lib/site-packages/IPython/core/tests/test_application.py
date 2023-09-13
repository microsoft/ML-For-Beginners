# coding: utf-8
"""Tests for IPython.core.application"""

import os
import tempfile

from tempfile import TemporaryDirectory
from traitlets import Unicode

from IPython.core.application import BaseIPythonApplication
from IPython.testing import decorators as dec


@dec.onlyif_unicode_paths
def test_unicode_cwd():
    """Check that IPython starts with non-ascii characters in the path."""
    wd = tempfile.mkdtemp(suffix=u"€")
    
    old_wd = os.getcwd()
    os.chdir(wd)
    #raise Exception(repr(os.getcwd()))
    try:
        app = BaseIPythonApplication()
        # The lines below are copied from Application.initialize()
        app.init_profile_dir()
        app.init_config_files()
        app.load_config_file(suppress_errors=False)
    finally:
        os.chdir(old_wd)

@dec.onlyif_unicode_paths
def test_unicode_ipdir():
    """Check that IPython starts with non-ascii characters in the IP dir."""
    ipdir = tempfile.mkdtemp(suffix=u"€")
    
    # Create the config file, so it tries to load it.
    with open(os.path.join(ipdir, "ipython_config.py"), "w", encoding="utf-8") as f:
        pass
    
    old_ipdir1 = os.environ.pop("IPYTHONDIR", None)
    old_ipdir2 = os.environ.pop("IPYTHON_DIR", None)
    os.environ["IPYTHONDIR"] = ipdir
    try:
        app = BaseIPythonApplication()
        # The lines below are copied from Application.initialize()
        app.init_profile_dir()
        app.init_config_files()
        app.load_config_file(suppress_errors=False)
    finally:
        if old_ipdir1:
            os.environ["IPYTHONDIR"] = old_ipdir1
        if old_ipdir2:
            os.environ["IPYTHONDIR"] = old_ipdir2

def test_cli_priority():
    with TemporaryDirectory() as td:

        class TestApp(BaseIPythonApplication):
            test = Unicode().tag(config=True)

        # Create the config file, so it tries to load it.
        with open(os.path.join(td, "ipython_config.py"), "w", encoding="utf-8") as f:
            f.write("c.TestApp.test = 'config file'")

        app = TestApp()
        app.initialize(["--profile-dir", td])
        assert app.test == "config file"
        app = TestApp()
        app.initialize(["--profile-dir", td, "--TestApp.test=cli"])
        assert app.test == "cli"
