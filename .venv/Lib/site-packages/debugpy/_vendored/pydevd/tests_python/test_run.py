import pytest
pytest_plugins = [
    str('_pytest.pytester'),
]


def _run_and_check(testdir_or_pytester, path, check_for='Worked'):
    result = testdir_or_pytester.runpython(path)
    result.stdout.fnmatch_lines([
        check_for
    ])


if hasattr(pytest, 'version_tuple') and pytest.version_tuple[0] >= 7:

    @pytest.fixture
    def testdir_or_pytester(pytester):
        return pytester

else:

    @pytest.fixture
    def testdir_or_pytester(testdir):
        return testdir


def test_run(testdir_or_pytester):
    from tests_python import debugger_unittest
    import sys
    import os

    foo_dir = debugger_unittest._get_debugger_test_file(os.path.join('resources', 'launch', 'foo'))
    foo_module = 'tests_python.resources.launch.foo'

    pydevd_dir = os.path.dirname(os.path.dirname(__file__))
    assert os.path.exists(os.path.join(pydevd_dir, 'pydevd.py'))

    _run_and_check(testdir_or_pytester, testdir_or_pytester.makepyfile('''
import sys
sys.path.append(%(pydevd_dir)r)
import pydevd
py_db = pydevd.PyDB()
py_db.ready_to_run = True
py_db.run(%(foo_dir)r)
''' % locals()))

    _run_and_check(testdir_or_pytester, testdir_or_pytester.makepyfile('''
import sys
sys.path.append(%(pydevd_dir)r)
import pydevd
py_db = pydevd.PyDB()
py_db.run(%(foo_dir)r, set_trace=False)
''' % locals()))

    if sys.version_info[0:2] == (2, 6):
        # Not valid for Python 2.6
        return

    _run_and_check(testdir_or_pytester, testdir_or_pytester.makepyfile('''
import sys
sys.path.append(%(pydevd_dir)r)
sys.argv.append('--as-module')
import pydevd
py_db = pydevd.PyDB()
py_db.ready_to_run = True
py_db.run(%(foo_module)r, is_module=True)
''' % locals()))

    _run_and_check(testdir_or_pytester, testdir_or_pytester.makepyfile('''
import sys
sys.argv.append('--as-module')
sys.path.append(%(pydevd_dir)r)
import pydevd
py_db = pydevd.PyDB()
py_db.run(%(foo_module)r, is_module=True, set_trace=False)
''' % locals()))


def test_run_on_local_module_without_adding_to_pythonpath(testdir_or_pytester):
    import sys
    import os

    pydevd_dir = os.path.dirname(os.path.dirname(__file__))
    assert os.path.exists(os.path.join(pydevd_dir, 'pydevd.py'))

    foo_module = 'local_foo'
    with open(os.path.join(os.getcwd(), 'local_foo.py'), 'w') as stream:
        stream.write('print("WorkedLocalFoo")')

    _run_and_check(testdir_or_pytester, testdir_or_pytester.makepyfile('''
import sys
import os
sys.path.append(%(pydevd_dir)r)
sys.argv.append('--as-module')
cwd = os.path.abspath(os.getcwd())
while cwd in sys.path:
    sys.path.remove(cwd)
import pydevd
py_db = pydevd.PyDB()
py_db.ready_to_run = True
py_db.run(%(foo_module)r, is_module=True)
''' % locals()), check_for='WorkedLocalFoo')

    _run_and_check(testdir_or_pytester, testdir_or_pytester.makepyfile('''
import sys
import os
sys.argv.append('--as-module')
sys.path.append(%(pydevd_dir)r)
cwd = os.path.abspath(os.getcwd())
while cwd in sys.path:
    sys.path.remove(cwd)
import pydevd
py_db = pydevd.PyDB()
py_db.run(%(foo_module)r, is_module=True, set_trace=False)
''' % locals()), check_for='WorkedLocalFoo')
