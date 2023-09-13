from _pydevd_bundle.pydevd_io import IORedirector
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
import pytest
import sys


def test_io_redirector():

    class MyRedirection1(object):
        encoding = 'foo'

    class MyRedirection2(object):
        pass

    my_redirector = IORedirector(MyRedirection1(), MyRedirection2(), wrap_buffer=True)
    none_redirector = IORedirector(None, None, wrap_buffer=True)

    assert my_redirector.encoding == 'foo'
    with pytest.raises(AttributeError):
        none_redirector.encoding

    # Check that we don't fail creating the IORedirector if the original
    # doesn't have a 'buffer'.
    for redirector in (
            my_redirector,
            none_redirector,
        ):
        redirector.write('test')
        redirector.flush()

    assert not redirector.isatty()


class _DummyWriter(object):

    __slots__ = ['commands', 'command_meanings']

    def __init__(self):
        self.commands = []
        self.command_meanings = []

    def add_command(self, cmd):
        from _pydevd_bundle.pydevd_comm import ID_TO_MEANING
        meaning = ID_TO_MEANING[str(cmd.id)]
        self.command_meanings.append(meaning)
        self.commands.append(cmd)


class _DummyPyDb(object):

    def __init__(self):
        self.cmd_factory = NetCommandFactory()
        self.writer = _DummyWriter()


def test_patch_stdin():
    from pydevd import _internal_patch_stdin

    py_db = _DummyPyDb()

    class _Stub(object):
        pass

    actions = []

    class OriginalStdin(object):

        def readline(self):
            # On a readline we keep the patched version.
            assert sys_mod.stdin is not original_stdin
            actions.append('readline')
            return 'read'

    def getpass_stub(*args, **kwargs):
        # On getpass we need to revert to the original version.
        actions.append('getpass')
        assert sys_mod.stdin is original_stdin
        return 'pass'

    sys_mod = _Stub()
    original_stdin = sys_mod.stdin = OriginalStdin()

    getpass_mod = _Stub()
    getpass_mod.getpass = getpass_stub

    _internal_patch_stdin(py_db, sys_mod, getpass_mod)

    assert sys_mod.stdin.readline() == 'read'

    assert py_db.writer.command_meanings == ['CMD_INPUT_REQUESTED', 'CMD_INPUT_REQUESTED']
    del py_db.writer.command_meanings[:]
    assert actions == ['readline']
    del actions[:]

    assert getpass_mod.getpass() == 'pass'
    assert py_db.writer.command_meanings == ['CMD_INPUT_REQUESTED', 'CMD_INPUT_REQUESTED']
    del py_db.writer.command_meanings[:]


def test_debug_console():
    from _pydev_bundle.pydev_console_utils import DebugConsoleStdIn

    class OriginalStdin(object):

        def readline(self):
            return 'read'

    original_stdin = OriginalStdin()

    py_db = _DummyPyDb()
    debug_console_std_in = DebugConsoleStdIn(py_db, original_stdin)
    assert debug_console_std_in.readline() == 'read'

    assert py_db.writer.command_meanings == ['CMD_INPUT_REQUESTED', 'CMD_INPUT_REQUESTED']
    del py_db.writer.command_meanings[:]

    with debug_console_std_in.notify_input_requested():
        with debug_console_std_in.notify_input_requested():
            pass
    assert py_db.writer.command_meanings == ['CMD_INPUT_REQUESTED', 'CMD_INPUT_REQUESTED']


@pytest.yield_fixture
def _redirect_context():
    from _pydevd_bundle.pydevd_io import RedirectToPyDBIoMessages
    from _pydevd_bundle.pydevd_io import _RedirectionsHolder
    py_db = _DummyPyDb()

    _original_get_pydb = RedirectToPyDBIoMessages.get_pydb
    _original_stack_stdout = _RedirectionsHolder._stack_stdout
    _original_stack_stderr = _RedirectionsHolder._stack_stderr
    _original_stdout_redirect = _RedirectionsHolder._pydevd_stdout_redirect_
    _original_stderr_redirect = _RedirectionsHolder._pydevd_stderr_redirect_

    RedirectToPyDBIoMessages.get_pydb = lambda *args, **kwargs: py_db
    _RedirectionsHolder._stack_stdout = []
    _RedirectionsHolder._stack_stderr = []
    _RedirectionsHolder._pydevd_stdout_redirect_ = None
    _RedirectionsHolder._pydevd_stderr_redirect_ = None

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    yield {'py_db': py_db}

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    RedirectToPyDBIoMessages.get_pydb = _original_get_pydb
    _RedirectionsHolder._stack_stdout = _original_stack_stdout
    _RedirectionsHolder._stack_stderr = _original_stack_stderr
    _RedirectionsHolder._pydevd_stdout_redirect_ = _original_stdout_redirect
    _RedirectionsHolder._pydevd_stderr_redirect_ = _original_stderr_redirect


def test_redirect_to_pyd_io_messages_basic(_redirect_context):
    from _pydevd_bundle.pydevd_io import redirect_stream_to_pydb_io_messages
    from _pydevd_bundle.pydevd_io import redirect_stream_to_pydb_io_messages_context
    from _pydevd_bundle.pydevd_io import stop_redirect_stream_to_pydb_io_messages
    from _pydevd_bundle.pydevd_io import _RedirectionsHolder

    py_db = _redirect_context['py_db']

    redirect_stream_to_pydb_io_messages(std='stdout')
    assert len(_RedirectionsHolder._stack_stdout) == 1
    assert _RedirectionsHolder._pydevd_stdout_redirect_ is not None
    sys.stdout.write('aaa')
    assert py_db.writer.command_meanings == ['CMD_WRITE_TO_CONSOLE']

    with redirect_stream_to_pydb_io_messages_context():
        assert len(_RedirectionsHolder._stack_stdout) == 1
        assert _RedirectionsHolder._pydevd_stdout_redirect_ is not None
        sys.stdout.write('bbb')

        assert py_db.writer.command_meanings == ['CMD_WRITE_TO_CONSOLE', 'CMD_WRITE_TO_CONSOLE']

    assert len(_RedirectionsHolder._stack_stdout) == 1
    assert _RedirectionsHolder._pydevd_stdout_redirect_ is not None
    sys.stdout.write('ccc')
    assert py_db.writer.command_meanings == ['CMD_WRITE_TO_CONSOLE', 'CMD_WRITE_TO_CONSOLE', 'CMD_WRITE_TO_CONSOLE']

    stop_redirect_stream_to_pydb_io_messages(std='stdout')
    assert len(_RedirectionsHolder._stack_stdout) == 0
    assert _RedirectionsHolder._pydevd_stdout_redirect_ is None
    sys.stdout.write('ddd')
    assert py_db.writer.command_meanings == ['CMD_WRITE_TO_CONSOLE', 'CMD_WRITE_TO_CONSOLE', 'CMD_WRITE_TO_CONSOLE']


@pytest.mark.parametrize('std', ['stderr', 'stdout'])
def test_redirect_to_pyd_io_messages_user_change_stdout(_redirect_context, std):
    from _pydevd_bundle.pydevd_io import redirect_stream_to_pydb_io_messages
    from _pydevd_bundle.pydevd_io import stop_redirect_stream_to_pydb_io_messages
    from _pydevd_bundle.pydevd_io import _RedirectionsHolder

    py_db = _redirect_context['py_db']
    stack = getattr(_RedirectionsHolder, '_stack_%s' % (std,))

    def get_redirect():
        return getattr(_RedirectionsHolder, '_pydevd_%s_redirect_' % (std,))

    def write(s):
        getattr(sys, std).write(s)

    redirect_stream_to_pydb_io_messages(std=std)
    assert len(stack) == 1
    assert get_redirect() is not None
    write('aaa')
    assert py_db.writer.command_meanings == ['CMD_WRITE_TO_CONSOLE']

    from io import StringIO
    stream = StringIO()
    setattr(sys, std, stream)

    write(u'bbb')
    assert py_db.writer.command_meanings == ['CMD_WRITE_TO_CONSOLE']
    assert stream.getvalue() == u'bbb'

    # i.e.: because the user changed the sys.stdout, we cannot change it to our previous version.
    stop_redirect_stream_to_pydb_io_messages(std=std)
    assert len(stack) == 0
    assert get_redirect() is None
    write(u'ccc')
    assert py_db.writer.command_meanings == ['CMD_WRITE_TO_CONSOLE']
    assert stream.getvalue() == u'bbbccc'

