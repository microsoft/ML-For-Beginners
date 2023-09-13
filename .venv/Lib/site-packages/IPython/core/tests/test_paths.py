import errno
import os
import shutil
import tempfile
import warnings
from unittest.mock import patch

from tempfile import TemporaryDirectory
from testpath import assert_isdir, assert_isfile, modified_env

from IPython import paths
from IPython.testing.decorators import skip_win32

TMP_TEST_DIR = os.path.realpath(tempfile.mkdtemp())
HOME_TEST_DIR = os.path.join(TMP_TEST_DIR, "home_test_dir")
XDG_TEST_DIR = os.path.join(HOME_TEST_DIR, "xdg_test_dir")
XDG_CACHE_DIR = os.path.join(HOME_TEST_DIR, "xdg_cache_dir")
IP_TEST_DIR = os.path.join(HOME_TEST_DIR,'.ipython')

def setup_module():
    """Setup testenvironment for the module:

            - Adds dummy home dir tree
    """
    # Do not mask exceptions here.  In particular, catching WindowsError is a
    # problem because that exception is only defined on Windows...
    os.makedirs(IP_TEST_DIR)
    os.makedirs(os.path.join(XDG_TEST_DIR, 'ipython'))
    os.makedirs(os.path.join(XDG_CACHE_DIR, 'ipython'))


def teardown_module():
    """Teardown testenvironment for the module:

            - Remove dummy home dir tree
    """
    # Note: we remove the parent test dir, which is the root of all test
    # subdirs we may have created.  Use shutil instead of os.removedirs, so
    # that non-empty directories are all recursively removed.
    shutil.rmtree(TMP_TEST_DIR)

def patch_get_home_dir(dirpath):
    return patch.object(paths, 'get_home_dir', return_value=dirpath)


def test_get_ipython_dir_1():
    """test_get_ipython_dir_1, Testcase to see if we can call get_ipython_dir without Exceptions."""
    env_ipdir = os.path.join("someplace", ".ipython")
    with patch.object(paths, '_writable_dir', return_value=True), \
            modified_env({'IPYTHONDIR': env_ipdir}):
        ipdir = paths.get_ipython_dir()

    assert ipdir == env_ipdir

def test_get_ipython_dir_2():
    """test_get_ipython_dir_2, Testcase to see if we can call get_ipython_dir without Exceptions."""
    with patch_get_home_dir('someplace'), \
            patch.object(paths, 'get_xdg_dir', return_value=None), \
            patch.object(paths, '_writable_dir', return_value=True), \
            patch('os.name', "posix"), \
            modified_env({'IPYTHON_DIR': None,
                          'IPYTHONDIR': None,
                          'XDG_CONFIG_HOME': None
                         }):
        ipdir = paths.get_ipython_dir()

    assert ipdir == os.path.join("someplace", ".ipython")

def test_get_ipython_dir_3():
    """test_get_ipython_dir_3, use XDG if defined and exists, and .ipython doesn't exist."""
    tmphome = TemporaryDirectory()
    try:
        with patch_get_home_dir(tmphome.name), \
                patch('os.name', 'posix'), \
                modified_env({
                    'IPYTHON_DIR': None,
                    'IPYTHONDIR': None,
                    'XDG_CONFIG_HOME': XDG_TEST_DIR,
                }), warnings.catch_warnings(record=True) as w:
            ipdir = paths.get_ipython_dir()

        assert ipdir == os.path.join(tmphome.name, XDG_TEST_DIR, "ipython")
        assert len(w) == 0
    finally:
        tmphome.cleanup()

def test_get_ipython_dir_4():
    """test_get_ipython_dir_4, warn if XDG and home both exist."""
    with patch_get_home_dir(HOME_TEST_DIR), \
            patch('os.name', 'posix'):
        try:
            os.mkdir(os.path.join(XDG_TEST_DIR, 'ipython'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


        with modified_env({
            'IPYTHON_DIR': None,
            'IPYTHONDIR': None,
            'XDG_CONFIG_HOME': XDG_TEST_DIR,
        }), warnings.catch_warnings(record=True) as w:
            ipdir = paths.get_ipython_dir()

        assert len(w) == 1
        assert "Ignoring" in str(w[0])


def test_get_ipython_dir_5():
    """test_get_ipython_dir_5, use .ipython if exists and XDG defined, but doesn't exist."""
    with patch_get_home_dir(HOME_TEST_DIR), \
            patch('os.name', 'posix'):
        try:
            os.rmdir(os.path.join(XDG_TEST_DIR, 'ipython'))
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

        with modified_env({
            'IPYTHON_DIR': None,
            'IPYTHONDIR': None,
            'XDG_CONFIG_HOME': XDG_TEST_DIR,
        }):
            ipdir = paths.get_ipython_dir()

        assert ipdir == IP_TEST_DIR

def test_get_ipython_dir_6():
    """test_get_ipython_dir_6, use home over XDG if defined and neither exist."""
    xdg = os.path.join(HOME_TEST_DIR, 'somexdg')
    os.mkdir(xdg)
    shutil.rmtree(os.path.join(HOME_TEST_DIR, '.ipython'))
    print(paths._writable_dir)
    with patch_get_home_dir(HOME_TEST_DIR), \
            patch.object(paths, 'get_xdg_dir', return_value=xdg), \
            patch('os.name', 'posix'), \
            modified_env({
                'IPYTHON_DIR': None,
                'IPYTHONDIR': None,
                'XDG_CONFIG_HOME': None,
            }), warnings.catch_warnings(record=True) as w:
        ipdir = paths.get_ipython_dir()

    assert ipdir == os.path.join(HOME_TEST_DIR, ".ipython")
    assert len(w) == 0

def test_get_ipython_dir_7():
    """test_get_ipython_dir_7, test home directory expansion on IPYTHONDIR"""
    home_dir = os.path.normpath(os.path.expanduser('~'))
    with modified_env({'IPYTHONDIR': os.path.join('~', 'somewhere')}), \
            patch.object(paths, '_writable_dir', return_value=True):
        ipdir = paths.get_ipython_dir()
    assert ipdir == os.path.join(home_dir, "somewhere")


@skip_win32
def test_get_ipython_dir_8():
    """test_get_ipython_dir_8, test / home directory"""
    if not os.access("/", os.W_OK):
        # test only when HOME directory actually writable
        return

    with patch.object(paths, "_writable_dir", lambda path: bool(path)), patch.object(
        paths, "get_xdg_dir", return_value=None
    ), modified_env(
        {
            "IPYTHON_DIR": None,
            "IPYTHONDIR": None,
            "HOME": "/",
        }
    ):
        assert paths.get_ipython_dir() == "/.ipython"


def test_get_ipython_cache_dir():
    with modified_env({'HOME': HOME_TEST_DIR}):
        if os.name == "posix":
            # test default
            os.makedirs(os.path.join(HOME_TEST_DIR, ".cache"))
            with modified_env({'XDG_CACHE_HOME': None}):
                ipdir = paths.get_ipython_cache_dir()
            assert os.path.join(HOME_TEST_DIR, ".cache", "ipython") == ipdir
            assert_isdir(ipdir)

            # test env override
            with modified_env({"XDG_CACHE_HOME": XDG_CACHE_DIR}):
                ipdir = paths.get_ipython_cache_dir()
            assert_isdir(ipdir)
            assert ipdir == os.path.join(XDG_CACHE_DIR, "ipython")
        else:
            assert paths.get_ipython_cache_dir() == paths.get_ipython_dir()

def test_get_ipython_package_dir():
    ipdir = paths.get_ipython_package_dir()
    assert_isdir(ipdir)


def test_get_ipython_module_path():
    ipapp_path = paths.get_ipython_module_path('IPython.terminal.ipapp')
    assert_isfile(ipapp_path)
