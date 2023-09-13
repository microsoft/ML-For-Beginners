# coding: utf-8
import os.path
from _pydevd_bundle.pydevd_constants import IS_WINDOWS, IS_MAC
import io
from _pydev_bundle.pydev_log import log_context
import pytest
import sys


@pytest.fixture(autouse=True)
def _reset_ide_os():
    yield
    from pydevd_file_utils import set_ide_os
    set_ide_os('WINDOWS' if sys.platform == 'win32' else 'UNIX')


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only test.')
def test_get_path_with_real_case_windows_unc_path(monkeypatch):
    import pydevd_file_utils
    from pydevd_file_utils import get_path_with_real_case

    def temp_listdir(d):
        # When we have a UNC drive in windows the "drive" is something as:
        # \\MACHINE_NAME\MOUNT_POINT\
        if d == '\\\\A\\B\\':
            return ['Cc']
        raise AssertionError('Unexpected: %s' % (d,))

    monkeypatch.setattr(pydevd_file_utils, 'os_path_exists', lambda *args: True)
    monkeypatch.setattr(pydevd_file_utils, 'os_listdir', temp_listdir)
    assert get_path_with_real_case(r'\\a\b\cc') == r'\\A\B\Cc'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only test.')
def test_get_path_with_real_case_windows_slashes_drive(tmpdir):
    from pydevd_file_utils import get_path_with_real_case
    test_dir = str(tmpdir.mkdir("Test_Convert_Utilities")).lower()
    real_case = get_path_with_real_case(test_dir)
    assert real_case.endswith("Test_Convert_Utilities")

    prefix = '\\\\?\\'
    path = prefix + test_dir
    real_case = get_path_with_real_case(path)
    assert real_case.endswith("Test_Convert_Utilities")
    assert path.startswith(prefix)


@pytest.mark.skipif(not IS_MAC, reason='Mac-only test.')
def test_get_path_with_real_case_mac_os(tmpdir):
    from pydevd_file_utils import get_path_with_real_case
    test_dir = str(tmpdir.mkdir("Test_Convert_Utilities")).lower()
    real_case = get_path_with_real_case(test_dir)
    assert real_case.endswith("Test_Convert_Utilities")


@pytest.mark.skipif(not IS_MAC, reason='Mac-only test.')
def test_double_slash_mac(monkeypatch):
    import pydevd_file_utils
    from pydevd_file_utils import get_path_with_real_case

    def temp_listdir(d):
        if d == '//':
            return ['A']
        if d == '//A':
            return ['Bb']
        raise AssertionError('Unexpected: %s' % (d,))

    monkeypatch.setattr(pydevd_file_utils, 'os_path_exists', lambda *args: True)
    monkeypatch.setattr(pydevd_file_utils, 'os_listdir', temp_listdir)
    assert get_path_with_real_case(r'//a/bb') == r'//A/Bb'


def test_convert_utilities(tmpdir):
    import pydevd_file_utils

    test_dir = str(tmpdir.mkdir("Test_Convert_Utilities"))

    if IS_WINDOWS:
        normalized = pydevd_file_utils.normcase(test_dir)
        assert isinstance(normalized, str)
        assert normalized.lower() == normalized

        upper_version = os.path.join(test_dir, 'ÁÉÍÓÚ')
        with open(upper_version, 'w') as stream:
            stream.write('test')

        with open(upper_version, 'r') as stream:
            assert stream.read() == 'test'

        with open(pydevd_file_utils.normcase(upper_version), 'r') as stream:
            assert stream.read() == 'test'

        assert '~' not in normalized

        for i in range(3):  # Check if cache is ok.

            if i == 2:
                pydevd_file_utils._listdir_cache.clear()

            assert pydevd_file_utils.get_path_with_real_case('<does not EXIST>') == '<does not EXIST>'
            real_case = pydevd_file_utils.get_path_with_real_case(normalized)
            assert isinstance(real_case, str)
            # Note test_dir itself cannot be compared with because pytest may
            # have passed the case normalized.
            assert real_case.endswith("Test_Convert_Utilities")

            if i == 2:
                # Check that we have the expected paths in the cache.
                assert pydevd_file_utils._listdir_cache[os.path.dirname(normalized).lower()] == ['Test_Convert_Utilities']
                assert pydevd_file_utils._listdir_cache[(os.path.dirname(normalized).lower(), 'Test_Convert_Utilities'.lower())] == real_case

        # Check that it works with a shortened path.
        shortened = pydevd_file_utils.convert_to_short_pathname(normalized)
        assert '~' in shortened
        with_real_case = pydevd_file_utils.get_path_with_real_case(shortened)
        assert with_real_case.endswith('Test_Convert_Utilities')
        assert '~' not in with_real_case

    elif IS_MAC:
        assert pydevd_file_utils.normcase(test_dir) == test_dir.lower()
        assert pydevd_file_utils.get_path_with_real_case(test_dir) == test_dir

    else:
        # On Linux, nothing should change
        assert pydevd_file_utils.normcase(test_dir) == test_dir
        assert pydevd_file_utils.get_path_with_real_case(test_dir) == test_dir


def test_source_reference(tmpdir):
    import pydevd_file_utils

    pydevd_file_utils.set_ide_os('WINDOWS')
    if IS_WINDOWS:
        # Client and server are on windows.
        pydevd_file_utils.setup_client_server_paths([('c:\\foo', 'c:\\bar')])

        assert pydevd_file_utils.map_file_to_client('c:\\bar\\my') == ('c:\\foo\\my', True)
        assert pydevd_file_utils.get_client_filename_source_reference('c:\\foo\\my') == 0

        assert pydevd_file_utils.map_file_to_client('c:\\another\\my') == ('c:\\another\\my', False)
        source_reference = pydevd_file_utils.get_client_filename_source_reference('c:\\another\\my')
        assert source_reference != 0
        assert pydevd_file_utils.get_server_filename_from_source_reference(source_reference) == 'c:\\another\\my'

    else:
        # Client on windows and server on unix
        pydevd_file_utils.set_ide_os('WINDOWS')

        pydevd_file_utils.setup_client_server_paths([('c:\\foo', '/bar')])

        assert pydevd_file_utils.map_file_to_client('/bar/my') == ('c:\\foo\\my', True)
        assert pydevd_file_utils.get_client_filename_source_reference('c:\\foo\\my') == 0

        assert pydevd_file_utils.map_file_to_client('/another/my') == ('\\another\\my', False)
        source_reference = pydevd_file_utils.get_client_filename_source_reference('\\another\\my')
        assert source_reference != 0
        assert pydevd_file_utils.get_server_filename_from_source_reference(source_reference) == '/another/my'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only test.')
def test_translate_only_drive():
    import pydevd_file_utils
    assert pydevd_file_utils.get_path_with_real_case('c:\\') == 'C:\\'


def test_to_server_and_to_client(tmpdir):
    try:

        def check(obtained, expected):
            assert obtained == expected, '%s (%s) != %s (%s)' % (obtained, type(obtained), expected, type(expected))
            if isinstance(obtained, tuple):
                assert isinstance(obtained[0], str)
            else:
                assert isinstance(obtained, str)

            if isinstance(expected, tuple):
                assert isinstance(expected[0], str)
            else:
                assert isinstance(expected, str)

        import pydevd_file_utils
        if IS_WINDOWS:
            # Check with made-up files

            pydevd_file_utils.setup_client_server_paths([('c:\\foo', 'c:\\bar'), ('c:\\foo2', 'c:\\bar2')])

            stream = io.StringIO()
            with log_context(0, stream=stream):
                pydevd_file_utils.map_file_to_server('y:\\only_exists_in_client_not_in_server')
            assert r'pydev debugger: unable to find translation for: "y:\only_exists_in_client_not_in_server" in ["c:\foo\", "c:\foo2\", "c:\foo", "c:\foo2"] (please revise your path mappings).' in stream.getvalue()

            # Client and server are on windows.
            pydevd_file_utils.set_ide_os('WINDOWS')
            for in_eclipse, in_python  in ([
                    ('c:\\foo', 'c:\\bar'),
                    ('c:/foo', 'c:\\bar'),
                    ('c:\\foo', 'c:/bar'),
                    ('c:\\foo', 'c:\\bar\\'),
                    ('c:/foo', 'c:\\bar\\'),
                    ('c:\\foo', 'c:/bar/'),
                    ('c:\\foo\\', 'c:\\bar'),
                    ('c:/foo/', 'c:\\bar'),
                    ('c:\\foo\\', 'c:/bar'),

                ]):
                PATHS_FROM_ECLIPSE_TO_PYTHON = [
                    (in_eclipse, in_python)
                ]
                pydevd_file_utils.setup_client_server_paths(PATHS_FROM_ECLIPSE_TO_PYTHON)
                check(pydevd_file_utils.map_file_to_server('c:\\foo\\my'), 'c:\\bar\\my')
                check(pydevd_file_utils.map_file_to_server('c:/foo/my'), 'c:\\bar\\my')
                check(pydevd_file_utils.map_file_to_server('c:/foo/my/'), 'c:\\bar\\my')
                check(pydevd_file_utils.map_file_to_server('c:\\foo\\áéíóú'.upper()), 'c:\\bar' + '\\áéíóú'.upper())
                check(pydevd_file_utils.map_file_to_client('c:\\bar\\my'), ('c:\\foo\\my', True))

            # Client on unix and server on windows
            pydevd_file_utils.set_ide_os('UNIX')
            for in_eclipse, in_python  in ([
                    ('/foo', 'c:\\bar'),
                    ('/foo', 'c:/bar'),
                    ('/foo', 'c:\\bar\\'),
                    ('/foo', 'c:/bar/'),
                    ('/foo/', 'c:\\bar'),
                    ('/foo/', 'c:\\bar\\'),
                ]):

                PATHS_FROM_ECLIPSE_TO_PYTHON = [
                    (in_eclipse, in_python)
                ]
                pydevd_file_utils.setup_client_server_paths(PATHS_FROM_ECLIPSE_TO_PYTHON)
                check(pydevd_file_utils.map_file_to_server('/foo/my'), 'c:\\bar\\my')
                check(pydevd_file_utils.map_file_to_client('c:\\bar\\my'), ('/foo/my', True))
                check(pydevd_file_utils.map_file_to_client('c:\\bar\\my\\'), ('/foo/my', True))
                check(pydevd_file_utils.map_file_to_client('c:/bar/my'), ('/foo/my', True))
                check(pydevd_file_utils.map_file_to_client('c:/bar/my/'), ('/foo/my', True))

            # Test with 'real' files
            # Client and server are on windows.
            pydevd_file_utils.set_ide_os('WINDOWS')

            test_dir = pydevd_file_utils.get_path_with_real_case(str(tmpdir.mkdir("Foo")))
            os.makedirs(os.path.join(test_dir, "Another"))

            in_eclipse = os.path.join(os.path.dirname(test_dir), 'Bar')
            in_python = test_dir
            PATHS_FROM_ECLIPSE_TO_PYTHON = [
                (in_eclipse, in_python)
            ]
            pydevd_file_utils.setup_client_server_paths(PATHS_FROM_ECLIPSE_TO_PYTHON)

            if pydevd_file_utils.map_file_to_server(in_eclipse) != in_python.lower():
                raise AssertionError('%s != %s\ntmpdir:%s\nin_eclipse: %s\nin_python: %s\ntest_dir: %s' % (
                    pydevd_file_utils.map_file_to_server(in_eclipse), in_python.lower(), tmpdir, in_eclipse, in_python, test_dir))

            found_in_eclipse = pydevd_file_utils.map_file_to_client(in_python)[0]
            assert found_in_eclipse.endswith('Bar')

            assert pydevd_file_utils.map_file_to_server(
                os.path.join(in_eclipse, 'another')) == os.path.join(in_python, 'another').lower()
            found_in_eclipse = pydevd_file_utils.map_file_to_client(
                os.path.join(in_python, 'another'))[0]
            assert found_in_eclipse.endswith('Bar\\Another')

            # Client on unix and server on windows
            pydevd_file_utils.set_ide_os('UNIX')
            in_eclipse = '/foo'
            in_python = test_dir
            PATHS_FROM_ECLIPSE_TO_PYTHON = [
                (in_eclipse, in_python)
            ]
            pydevd_file_utils.setup_client_server_paths(PATHS_FROM_ECLIPSE_TO_PYTHON)
            assert pydevd_file_utils.map_file_to_server('/foo').lower() == in_python.lower()
            assert pydevd_file_utils.map_file_to_client(in_python) == (in_eclipse, True)

            # Test without translation in place (still needs to fix case and separators)
            pydevd_file_utils.set_ide_os('WINDOWS')
            PATHS_FROM_ECLIPSE_TO_PYTHON = []
            pydevd_file_utils.setup_client_server_paths(PATHS_FROM_ECLIPSE_TO_PYTHON)
            assert pydevd_file_utils.map_file_to_server(test_dir) == test_dir
            assert pydevd_file_utils.map_file_to_client(test_dir.lower())[0].endswith('\\Foo')
        else:
            # Client on windows and server on unix
            pydevd_file_utils.set_ide_os('WINDOWS')

            PATHS_FROM_ECLIPSE_TO_PYTHON = [
                ('c:\\BAR', '/bar')
            ]

            pydevd_file_utils.setup_client_server_paths(PATHS_FROM_ECLIPSE_TO_PYTHON)
            assert pydevd_file_utils.map_file_to_server('c:\\bar\\my') == '/bar/my'
            assert pydevd_file_utils.map_file_to_client('/bar/my') == ('c:\\BAR\\my', True)

            for in_eclipse, in_python  in ([
                    ('c:\\foo', '/báéíóúr'),
                    ('c:/foo', '/báéíóúr'),
                    ('c:/foo/', '/báéíóúr'),
                    ('c:/foo/', '/báéíóúr/'),
                    ('c:\\foo\\', '/báéíóúr/'),
                ]):

                PATHS_FROM_ECLIPSE_TO_PYTHON = [
                    (in_eclipse, in_python)
                ]

                pydevd_file_utils.setup_client_server_paths(PATHS_FROM_ECLIPSE_TO_PYTHON)
                assert pydevd_file_utils.map_file_to_server('c:\\foo\\my') == '/báéíóúr/my'
                assert pydevd_file_utils.map_file_to_server('C:\\foo\\my') == '/báéíóúr/my'
                assert pydevd_file_utils.map_file_to_server('C:\\foo\\MY') == '/báéíóúr/MY'
                assert pydevd_file_utils.map_file_to_server('C:\\foo\\MY\\') == '/báéíóúr/MY'
                assert pydevd_file_utils.map_file_to_server('c:\\foo\\my\\file.py') == '/báéíóúr/my/file.py'
                assert pydevd_file_utils.map_file_to_server('c:\\foo\\my\\other\\file.py') == '/báéíóúr/my/other/file.py'
                assert pydevd_file_utils.map_file_to_server('c:/foo/my') == '/báéíóúr/my'
                assert pydevd_file_utils.map_file_to_server('c:\\foo\\my\\') == '/báéíóúr/my'
                assert pydevd_file_utils.map_file_to_server('c:/foo/my/') == '/báéíóúr/my'

                assert pydevd_file_utils.map_file_to_client('/báéíóúr/my') == ('c:\\foo\\my', True)
                assert pydevd_file_utils.map_file_to_client('/báéíóúr/my/') == ('c:\\foo\\my', True)

                # Files for which there's no translation have only their separators updated.
                assert pydevd_file_utils.map_file_to_client('/usr/bin/x.py') == ('\\usr\\bin\\x.py', False)
                assert pydevd_file_utils.map_file_to_client('/usr/bin') == ('\\usr\\bin', False)
                assert pydevd_file_utils.map_file_to_client('/usr/bin/') == ('\\usr\\bin', False)
                assert pydevd_file_utils.map_file_to_server('\\usr\\bin') == '/usr/bin'
                assert pydevd_file_utils.map_file_to_server('\\usr\\bin\\') == '/usr/bin'

                # When we have a client file and there'd be no translation, and making it absolute would
                # do something as '$cwd/$file_received' (i.e.: $cwd/c:/another in the case below),
                # warn the user that it's not correct and the path that should be translated instead
                # and don't make it absolute.
                assert pydevd_file_utils.map_file_to_server('c:\\Another') == 'c:/Another'

                assert pydevd_file_utils.map_file_to_server('c:/FoO/my/BAR') == '/báéíóúr/my/BAR'
                assert pydevd_file_utils.map_file_to_client('/báéíóúr/my/BAR') == ('c:\\foo\\my\\BAR', True)

            # Client and server on unix
            pydevd_file_utils.set_ide_os('UNIX')
            in_eclipse = '/foo'
            in_python = '/báéíóúr'
            PATHS_FROM_ECLIPSE_TO_PYTHON = [
                (in_eclipse, in_python)
            ]
            pydevd_file_utils.setup_client_server_paths(PATHS_FROM_ECLIPSE_TO_PYTHON)
            assert pydevd_file_utils.map_file_to_server('/foo/my') == '/báéíóúr/my'
            assert pydevd_file_utils.map_file_to_client('/báéíóúr/my') == ('/foo/my', True)
    finally:
        pydevd_file_utils.setup_client_server_paths([])


def test_relative_paths(tmpdir):
    '''
    We need to check that we can deal with relative paths.

    Use cases:
        - Relative path of file that does not exist:
            Use case is a cython-generated module which is generated from a .pyx which
            is not distributed. In this case we need to resolve the file to a library path file.

        - Relative path of a file that exists but not when resolved from the working directory:
            Use case is a cython-generated module which is generated from a .pyx which is
            distributed. In this case we need to resolve to the real file based on the sys.path
            entries.
    '''
    import pydevd_file_utils
    import sys
    sys.path.append(str(tmpdir))
    try:
        pydevd_file_utils.NORM_PATHS_AND_BASE_CONTAINER.clear()
        pydevd_file_utils.NORM_PATHS_CONTAINER.clear()
        abs_path = pydevd_file_utils.get_abs_path_real_path_and_base_from_file('my_dir/my_file.pyx')[0]
        assert 'site-packages' in abs_path
        assert os.path.normcase(str(tmpdir)) not in abs_path
        assert not pydevd_file_utils.exists('my_dir/my_file.pyx')

        # If the relative file exists when joined with some entry in the PYTHONPATH we'll consider
        # that the relative path points to that absolute path.
        target_dir = os.path.join(str(tmpdir), 'my_dir')
        os.makedirs(target_dir)
        with open(os.path.join(target_dir, 'my_file.pyx'), 'w') as stream:
            stream.write('empty')

        pydevd_file_utils.NORM_PATHS_AND_BASE_CONTAINER.clear()
        pydevd_file_utils.NORM_PATHS_CONTAINER.clear()
        abs_path = pydevd_file_utils.get_abs_path_real_path_and_base_from_file('my_dir/my_file.pyx')[0]
        assert 'site-packages' not in abs_path
        assert str(tmpdir) in abs_path
        assert pydevd_file_utils.exists('my_dir/my_file.pyx')
    finally:
        sys.path.remove(str(tmpdir))


def test_zip_paths(tmpdir):
    import pydevd_file_utils
    import sys
    import zipfile

    for i, zip_basename in enumerate(('MY1.zip', 'my2.egg!')):
        zipfile_path = str(tmpdir.join(zip_basename))
        zip_file = zipfile.ZipFile(zipfile_path, 'w')
        zip_file.writestr('zipped%s/__init__.py' % (i,), '')
        zip_file.writestr('zipped%s/zipped_contents.py' % (i,), 'def call_in_zip():\n    return 1')
        zip_file.close()

        sys.path.append(zipfile_path)
        try:
            import importlib
        except ImportError:
            __import__('zipped%s' % (i,))  # Py2.6 does not have importlib
        else:
            importlib.import_module('zipped%s' % (i,))  # Check that it's importable.

        # Check that we can deal with the zip path.
        assert pydevd_file_utils.exists(zipfile_path)
        abspath, realpath, basename = pydevd_file_utils.get_abs_path_real_path_and_base_from_file(zipfile_path)
        if IS_WINDOWS or IS_MAC:
            assert abspath == zipfile_path
            assert basename == zip_basename.lower()
        else:
            assert abspath == zipfile_path
            assert basename == zip_basename

        # Check that we can deal with zip contents.
        for path in [
                zipfile_path + '/zipped%s/__init__.py' % (i,),
                zipfile_path + '/zipped%s/zipped_contents.py' % (i,),
                zipfile_path + '\\zipped%s\\__init__.py' % (i,),
                zipfile_path + '\\zipped%s\\zipped_contents.py' % (i,),
            ]:
            assert pydevd_file_utils.exists(path), 'Expected exists to return True for path:\n%s' % (path,)
            abspath, realpath, basename = pydevd_file_utils.get_abs_path_real_path_and_base_from_file(path)
            assert pydevd_file_utils.exists(abspath), 'Expected exists to return True for path:\n%s' % (abspath,)
            assert pydevd_file_utils.exists(realpath), 'Expected exists to return True for path:\n%s' % (realpath,)

        assert zipfile_path in pydevd_file_utils._ZIP_SEARCH_CACHE, '%s not in %s' % (
            zipfile_path, '\n'.join(sorted(pydevd_file_utils._ZIP_SEARCH_CACHE.keys())))


def test_source_mapping():

    from _pydevd_bundle.pydevd_source_mapping import SourceMapping, SourceMappingEntry
    from _pydevd_bundle import pydevd_api

    class _DummyPyDB(object):
        source_mapping = SourceMapping()
        api_received_breakpoints = {}
        file_to_id_to_line_breakpoint = {}
        file_to_id_to_plugin_breakpoint = {}
        breakpoints = {}

    source_mapping = _DummyPyDB.source_mapping

    mapping = [
        SourceMappingEntry(line=3, end_line=6, runtime_line=5, runtime_source='<cell1>'),
        SourceMappingEntry(line=10, end_line=11, runtime_line=1, runtime_source='<cell2>'),
    ]

    api = pydevd_api.PyDevdAPI()
    py_db = _DummyPyDB()
    filename = 'c:\\temp\\bar.py' if IS_WINDOWS else '/temp/bar.py'
    api.set_source_mapping(py_db, filename, mapping)

    # Map to server
    assert source_mapping.map_to_server(filename, 1) == (filename, 1, False)
    assert source_mapping.map_to_server(filename, 2) == (filename, 2, False)

    assert source_mapping.map_to_server(filename, 3) == ('<cell1>', 5, True)
    assert source_mapping.map_to_server(filename, 4) == ('<cell1>', 6, True)
    assert source_mapping.map_to_server(filename, 5) == ('<cell1>', 7, True)
    assert source_mapping.map_to_server(filename, 6) == ('<cell1>', 8, True)

    assert source_mapping.map_to_server(filename, 7) == (filename, 7, False)

    assert source_mapping.map_to_server(filename, 10) == ('<cell2>', 1, True)
    assert source_mapping.map_to_server(filename, 11) == ('<cell2>', 2, True)

    assert source_mapping.map_to_server(filename, 12) == (filename, 12, False)

    # Map to client
    assert source_mapping.map_to_client(filename, 1) == (filename, 1, False)
    assert source_mapping.map_to_client(filename, 2) == (filename, 2, False)

    assert source_mapping.map_to_client('<cell1>', 5) == (filename, 3, True)
    assert source_mapping.map_to_client('<cell1>', 6) == (filename, 4, True)
    assert source_mapping.map_to_client('<cell1>', 7) == (filename, 5, True)
    assert source_mapping.map_to_client('<cell1>', 8) == (filename, 6, True)

    assert source_mapping.map_to_client(filename, 7) == (filename, 7, False)

    assert source_mapping.map_to_client('<cell2>', 1) == (filename, 10, True)
    assert source_mapping.map_to_client('<cell2>', 2) == (filename, 11, True)

    assert source_mapping.map_to_client(filename, 12) == (filename, 12, False)


@pytest.mark.skipif(IS_WINDOWS, reason='Linux/Mac-only test')
def test_mapping_conflict_to_client():
    import pydevd_file_utils

    path_mappings = []
    for pathMapping in _MAPPING_CONFLICT:
        localRoot = pathMapping.get('localRoot', '')
        remoteRoot = pathMapping.get('remoteRoot', '')
        if (localRoot != '') and (remoteRoot != ''):
            path_mappings.append((localRoot, remoteRoot))

    pydevd_file_utils.setup_client_server_paths(path_mappings)

    assert pydevd_file_utils.map_file_to_client('/opt/pathsomething/foo.py') == \
        ('/var/home/p2/foo.py', True)

    assert pydevd_file_utils.map_file_to_client('/opt/v2/pathsomething/foo.py') == \
        ('/var/home/p4/foo.py', True)

    # This is an odd case, but the user didn't really put a slash in the end,
    # so, it's possible that this is what the user actually wants.
    assert pydevd_file_utils.map_file_to_client('/opt/v2/path_r1/foo.py') == \
        ('/var/home/p3_r1/foo.py', True)

    # The client said both local and remote end with a slash, so, we can only
    # match it with the slash in the end.
    assert pydevd_file_utils.map_file_to_client('/opt/pathsomething_foo.py') == \
        ('/opt/pathsomething_foo.py', False)


_MAPPING_CONFLICT = [
    {
        "localRoot": "/var/home/p1/",
        "remoteRoot": "/opt/path/"
    },
    {
        "localRoot": "/var/home/p2/",
        "remoteRoot": "/opt/pathsomething/"
    },
    {
        "localRoot": "/var/home/p3",
        "remoteRoot": "/opt/v2/path"
    },
    {
        "localRoot": "/var/home/p4",
        "remoteRoot": "/opt/v2/pathsomething"
    },
]


@pytest.mark.skipif(IS_WINDOWS, reason='Linux/Mac-only test')
def test_mapping_conflict_to_server():
    import pydevd_file_utils

    path_mappings = []
    for pathMapping in _MAPPING_CONFLICT_TO_SERVER:
        localRoot = pathMapping.get('localRoot', '')
        remoteRoot = pathMapping.get('remoteRoot', '')
        if (localRoot != '') and (remoteRoot != ''):
            path_mappings.append((localRoot, remoteRoot))

    pydevd_file_utils.setup_client_server_paths(path_mappings)

    assert pydevd_file_utils.map_file_to_server('/opt/pathsomething/foo.py') == '/var/home/p2/foo.py'

    assert pydevd_file_utils.map_file_to_server('/opt/v2/pathsomething/foo.py') == '/var/home/p4/foo.py'

    # This is an odd case, but the user didn't really put a slash in the end,
    # so, it's possible that this is what the user actually wants.
    assert pydevd_file_utils.map_file_to_server('/opt/v2/path_r1/foo.py') == '/var/home/p3_r1/foo.py'

    # The client said both local and remote end with a slash, so, we can only
    # match it with the slash in the end.
    assert pydevd_file_utils.map_file_to_server('/opt/pathsomething_foo.py') == '/opt/pathsomething_foo.py'


_MAPPING_CONFLICT_TO_SERVER = [
    {
        "remoteRoot": "/var/home/p1/",
        "localRoot": "/opt/path/"
    },
    {
        "remoteRoot": "/var/home/p2/",
        "localRoot": "/opt/pathsomething/"
    },
    {
        "remoteRoot": "/var/home/p3",
        "localRoot": "/opt/v2/path"
    },
    {
        "remoteRoot": "/var/home/p4",
        "localRoot": "/opt/v2/pathsomething"
    },
]

