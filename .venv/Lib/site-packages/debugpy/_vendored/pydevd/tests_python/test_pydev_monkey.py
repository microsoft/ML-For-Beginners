# coding: utf-8
import os
import sys

import pytest

from _pydev_bundle.pydev_monkey import pydev_src_dir
from _pydevd_bundle.pydevd_constants import sorted_dict_repr
from pydevd import SetupHolder

try:
    from _pydev_bundle import pydev_monkey
except:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from _pydev_bundle import pydev_monkey


@pytest.fixture(autouse=True)
def save_setup_holder():
    original = SetupHolder.setup
    yield
    SetupHolder.setup = original


def test_monkey():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True}
    check = '''C:\\bin\\python.exe -u -c connect(\\"127.0.0.1\\")'''
    debug_command = (
        'import sys; '
        'sys.path.insert(0, r\'%s\'); '
        "import pydevd; pydevd.config('quoted-line', ''); "
        "pydevd.settrace(host='127.0.0.1', port=0, suspend=False, "
        'trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command

    assert 'C:\\bin\\python.exe -u -c %s' % debug_command == pydev_monkey.patch_arg_str_win(check)


def test_str_to_args_windows():
    assert ['a', 'b'] == pydev_monkey.str_to_args_windows('a "b"')


def test_monkey_patch_return_original_args():
    check = ['echo', '"my"', '"args"']
    res = pydev_monkey.patch_args(check[:])
    assert res == check


def test_monkey_patch_pathlib_args():
    try:
        import pathlib
    except ImportError:
        pytest.skip('pathlib not available.')

    check = [pathlib.Path('echo'), '"my"', '"args"']
    res = pydev_monkey.patch_args(check[:])
    assert res == check


def test_monkey_patch_wrong_object_type():
    check = [1, 22, '"my"', '"args"']
    res = pydev_monkey.patch_args(check[:])
    assert res == check


def test_monkey_patch_wrong_object_type_2():
    check = ['C:\\bin\\python.exe', '-u', 1, '-qcconnect("127.0.0.1")']
    res = pydev_monkey.patch_args(check[:])
    assert res == check


def test_monkey_patch_args_module_subprocess_pathlib():
    try:
        import pathlib
    except ImportError:
        pytest.skip('pathlib not available.')

    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'multiprocess': True}
    if sys.platform == 'win32':
        python_path = 'C:\\bin\\python.exe'
    else:
        python_path = '/bin/python'
    check = [pathlib.Path(python_path), '-mtest', pathlib.Path('bar')]
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file
    assert pydev_monkey.patch_args(check) == [
        python_path,
        get_pydevd_file(),
        '--module',
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--multiprocess',
        '--protocol-quoted-line',
        '--file',
        'test',
        'bar',
    ]


def test_monkey_patch_args_indc():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-u', '-c', 'connect("127.0.0.1")']
    debug_command = (
        "import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-u',
        '-c',
        debug_command
    ]


def test_separate_future_imports():
    found = pydev_monkey._separate_future_imports('''from __future__ import print_function\nprint(1)''')
    assert found == ('from __future__ import print_function;', '\nprint(1)')

    found = pydev_monkey._separate_future_imports('''from __future__ import print_function;print(1)''')
    assert found == ('from __future__ import print_function;', 'print(1)')

    found = pydev_monkey._separate_future_imports('''from __future__ import (\nprint_function);print(1)''')
    assert found == ('from __future__ import (\nprint_function);', 'print(1)')

    found = pydev_monkey._separate_future_imports('''"line";from __future__ import (\n\nprint_function, absolute_imports\n);print(1)''')
    assert found == ('"line";from __future__ import (\n\nprint_function, absolute_imports\n);', 'print(1)')

    found = pydev_monkey._separate_future_imports('''from __future__ import bar\nfrom __future__ import (\n\nprint_function, absolute_imports\n);print(1)''')
    assert found == ('from __future__ import bar\nfrom __future__ import (\n\nprint_function, absolute_imports\n);', 'print(1)')


def test_monkey_patch_args_indc_future_import():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-u', '-c', 'from __future__ import print_function;connect("127.0.0.1")']
    debug_command = (
        "from __future__ import print_function;import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-u',
        '-c',
        debug_command
    ]


def test_monkey_patch_args_indc_future_import2():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-u', '-c', 'from __future__ import print_function\nconnect("127.0.0.1")']
    debug_command = (
        "from __future__ import print_function;import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        '\nconnect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-u',
        '-c',
        debug_command
    ]


def test_monkey_patch_args_indc2():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-u', '-qcconnect("127.0.0.1")']
    debug_command = (
        "import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-u',
        '-qc',
        debug_command
    ]


def test_monkey_patch_args_x_flag():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-X', 'faulthandler', '-c', 'connect("127.0.0.1")']
    debug_command = (
        "import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-X',
        'faulthandler',
        '-c',
        debug_command
    ]


def test_monkey_patch_args_flag_in_single_arg_1():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-qX', 'faulthandler', '-c', 'connect("127.0.0.1")']
    debug_command = (
        "import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-qX',
        'faulthandler',
        '-c',
        debug_command
    ]


def test_monkey_patch_args_flag_in_single_arg_2():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-qX', 'faulthandler', '-c', 'connect("127.0.0.1")']
    debug_command = (
        "import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-qX',
        'faulthandler',
        '-c',
        debug_command
    ]


def test_monkey_patch_args_flag_in_single_arg_3():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-qc', 'connect("127.0.0.1")']
    debug_command = (
        "import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-qc',
        debug_command
    ]


def test_monkey_patch_args_x_flag_inline():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-Xfaulthandler', '-c', 'connect("127.0.0.1")', 'arg1']
    debug_command = (
        "import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-Xfaulthandler',
        '-c',
        debug_command,
        'arg1'
    ]


def test_monkey_patch_args_c_flag_inline():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'ppid': os.getpid(), 'protocol-quoted-line': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-X', 'faulthandler', '-cconnect("127.0.0.1")', 'arg1']
    debug_command = (
        "import sys; sys.path.insert(0, r\'%s\'); import pydevd; pydevd.config('quoted-line', ''); "
        'pydevd.settrace(host=\'127.0.0.1\', port=0, suspend=False, trace_only_current_thread=False, patch_multiprocessing=True, access_token=None, client_access_token=None, __setup_holder__=%s); '
        ''
        'connect("127.0.0.1")') % (pydev_src_dir, sorted_dict_repr(SetupHolder.setup))
    if sys.platform == "win32":
        debug_command = debug_command.replace('"', '\\"')
        debug_command = '"%s"' % debug_command
    res = pydev_monkey.patch_args(check)
    assert res == [
        'C:\\bin\\python.exe',
        '-X',
        'faulthandler',
        '-c',
        debug_command,
        'arg1'
    ]


def test_monkey_patch_args_module():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'multiprocess': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-m', 'test']
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file
    assert pydev_monkey.patch_args(check) == [
        'C:\\bin\\python.exe',
        get_pydevd_file(),
        '--module',
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--multiprocess',
        '--skip-notify-stdin',
        '--protocol-quoted-line',
        '--file',
        'test',
    ]


def test_monkey_patch_args_unbuffered_module():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'multiprocess': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-u', '-m', 'test']
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file
    assert pydev_monkey.patch_args(check) == [
        'C:\\bin\\python.exe',
        '-u',
        get_pydevd_file(),
        '--module',
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--multiprocess',
        '--skip-notify-stdin',
        '--protocol-quoted-line',
        '--file',
        'test',
    ]


def test_monkey_patch_args_module_inline():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'multiprocess': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-qOmtest']
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file
    assert pydev_monkey.patch_args(check) == [
        'C:\\bin\\python.exe',
        '-qO',
        get_pydevd_file(),
        '--module',
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--multiprocess',
        '--skip-notify-stdin',
        '--protocol-quoted-line',
        '--file',
        'test',
    ]


def test_monkey_patch_args_module_inline2():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'multiprocess': True, 'skip-notify-stdin': True}
    check = ['C:\\bin\\python.exe', '-qOm', 'test']
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file
    assert pydev_monkey.patch_args(check) == [
        'C:\\bin\\python.exe',
        '-qO',
        get_pydevd_file(),
        '--module',
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--multiprocess',
        '--skip-notify-stdin',
        '--protocol-quoted-line',
        '--file',
        'test',
    ]


def test_monkey_patch_args_no_indc():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0'}
    check = ['C:\\bin\\python.exe', 'connect(\\"127.0.0.1\\")', 'with spaces']
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file
    assert pydev_monkey.patch_args(check) == [
        'C:\\bin\\python.exe',
        get_pydevd_file(),
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--protocol-quoted-line',
        '--file',
        '"connect(\\\\\\"127.0.0.1\\\\\\")"' if sys.platform == 'win32' else 'connect(\\"127.0.0.1\\")',
        '"with spaces"'  if sys.platform == 'win32' else 'with spaces',
    ]


def test_monkey_patch_args_no_indc_with_pydevd():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0'}
    check = ['C:\\bin\\python.exe', 'pydevd.py', 'connect(\\"127.0.0.1\\")', 'bar']

    assert pydev_monkey.patch_args(check) == [
        'C:\\bin\\python.exe', 'pydevd.py', 'connect(\\"127.0.0.1\\")', 'bar']


def test_monkey_patch_args_no_indc_without_pydevd():
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file

    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0'}
    check = ['C:\\bin\\python.exe', 'target.py', 'connect(\\"127.0.0.1\\")', 'bar']
    assert pydev_monkey.patch_args(check) == [
        'C:\\bin\\python.exe',
        get_pydevd_file(),
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--protocol-quoted-line',
        '--file',
        'target.py',
        '"connect(\\\\\\"127.0.0.1\\\\\\")"' if sys.platform == 'win32' else 'connect(\\"127.0.0.1\\")',
        'bar',
    ]


@pytest.mark.parametrize('use_bytes', [True, False])
def test_monkey_patch_c_program_arg(use_bytes):
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file

    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'module': 'ignore.this'}
    check = ['C:\\bin\\python.exe', '-u', 'target.py', '-c', '-áéíóú']

    encode = lambda s:s
    if use_bytes:
        check = [c.encode('utf-8') for c in check]
        encode = lambda s:s.encode('utf-8')

    assert pydev_monkey.patch_args(check) == [
        encode('C:\\bin\\python.exe'),
        encode('-u'),
        get_pydevd_file(),
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--protocol-quoted-line',
        '--file',
        encode('target.py'),
        encode('-c'),
        encode('-áéíóú')
    ]


def test_monkey_patch_args_module_single_arg():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'multiprocess': True, 'module': 'ignore.this'}
    check = ['C:\\bin\\python.exe', '-mtest', 'bar']
    from _pydevd_bundle.pydevd_command_line_handling import get_pydevd_file
    assert pydev_monkey.patch_args(check) == [
        'C:\\bin\\python.exe',
        get_pydevd_file(),
        '--module',
        '--port',
        '0',
        '--ppid',
        str(os.getpid()),
        '--client',
        '127.0.0.1',
        '--multiprocess',
        '--protocol-quoted-line',
        '--file',
        'test',
        'bar',
    ]


def test_monkey_patch_args_stdin():
    SetupHolder.setup = {'client': '127.0.0.1', 'port': '0', 'multiprocess': True, 'module': 'ignore.this'}
    check = ['C:\\bin\\python.exe', '-Xfaulthandler' , '-']
    # i.e.: we don't deal with the stdin.
    assert pydev_monkey.patch_args(check) == check
