from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager


class IORedirector:
    '''
    This class works to wrap a stream (stdout/stderr) with an additional redirect.
    '''

    def __init__(self, original, new_redirect, wrap_buffer=False):
        '''
        :param stream original:
            The stream to be wrapped (usually stdout/stderr, but could be None).

        :param stream new_redirect:
            Usually IOBuf (below).

        :param bool wrap_buffer:
            Whether to create a buffer attribute (needed to mimick python 3 s
            tdout/stderr which has a buffer to write binary data).
        '''
        self._lock = ForkSafeLock(rlock=True)
        self._writing = False
        self._redirect_to = (original, new_redirect)
        if wrap_buffer and hasattr(original, 'buffer'):
            self.buffer = IORedirector(original.buffer, new_redirect.buffer, False)

    def write(self, s):
        # Note that writing to the original stream may fail for some reasons
        # (such as trying to write something that's not a string or having it closed).
        with self._lock:
            if self._writing:
                return
            self._writing = True
            try:
                for r in self._redirect_to:
                    if hasattr(r, 'write'):
                        r.write(s)
            finally:
                self._writing = False

    def isatty(self):
        for r in self._redirect_to:
            if hasattr(r, 'isatty'):
                return r.isatty()
        return False

    def flush(self):
        for r in self._redirect_to:
            if hasattr(r, 'flush'):
                r.flush()

    def __getattr__(self, name):
        for r in self._redirect_to:
            if hasattr(r, name):
                return getattr(r, name)
        raise AttributeError(name)


class RedirectToPyDBIoMessages(object):

    def __init__(self, out_ctx, wrap_stream, wrap_buffer, on_write=None):
        '''
        :param out_ctx:
            1=stdout and 2=stderr

        :param wrap_stream:
            Either sys.stdout or sys.stderr.

        :param bool wrap_buffer:
            If True the buffer attribute (which wraps writing bytes) should be
            wrapped.

        :param callable(str) on_write:
            May be a custom callable to be called when to write something.
            If not passed the default implementation will create an io message
            and send it through the debugger.
        '''
        encoding = getattr(wrap_stream, 'encoding', None)
        if not encoding:
            encoding = os.environ.get('PYTHONIOENCODING', 'utf-8')
        self.encoding = encoding
        self._out_ctx = out_ctx
        if wrap_buffer:
            self.buffer = RedirectToPyDBIoMessages(out_ctx, wrap_stream, wrap_buffer=False, on_write=on_write)
        self._on_write = on_write

    def get_pydb(self):
        # Note: separate method for mocking on tests.
        return get_global_debugger()

    def flush(self):
        pass  # no-op here

    def write(self, s):
        if self._on_write is not None:
            self._on_write(s)
            return

        if s:
            # Need s in str
            if isinstance(s, bytes):
                s = s.decode(self.encoding, errors='replace')

            py_db = self.get_pydb()
            if py_db is not None:
                # Note that the actual message contents will be a xml with utf-8, although
                # the entry is str on py3 and bytes on py2.
                cmd = py_db.cmd_factory.make_io_message(s, self._out_ctx)
                if py_db.writer is not None:
                    py_db.writer.add_command(cmd)


class IOBuf:
    '''This class works as a replacement for stdio and stderr.
    It is a buffer and when its contents are requested, it will erase what
    it has so far so that the next return will not return the same contents again.
    '''

    def __init__(self):
        self.buflist = []
        import os
        self.encoding = os.environ.get('PYTHONIOENCODING', 'utf-8')

    def getvalue(self):
        b = self.buflist
        self.buflist = []  # clear it
        return ''.join(b)  # bytes on py2, str on py3.

    def write(self, s):
        if isinstance(s, bytes):
            s = s.decode(self.encoding, errors='replace')
        self.buflist.append(s)

    def isatty(self):
        return False

    def flush(self):
        pass

    def empty(self):
        return len(self.buflist) == 0


class _RedirectInfo(object):

    def __init__(self, original, redirect_to):
        self.original = original
        self.redirect_to = redirect_to


class _RedirectionsHolder:
    _lock = ForkSafeLock(rlock=True)
    _stack_stdout = []
    _stack_stderr = []

    _pydevd_stdout_redirect_ = None
    _pydevd_stderr_redirect_ = None


def start_redirect(keep_original_redirection=False, std='stdout', redirect_to=None):
    '''
    @param std: 'stdout', 'stderr', or 'both'
    '''
    with _RedirectionsHolder._lock:
        if redirect_to is None:
            redirect_to = IOBuf()

        if std == 'both':
            config_stds = ['stdout', 'stderr']
        else:
            config_stds = [std]

        for std in config_stds:
            original = getattr(sys, std)
            stack = getattr(_RedirectionsHolder, '_stack_%s' % std)

            if keep_original_redirection:
                wrap_buffer = True if hasattr(redirect_to, 'buffer') else False
                new_std_instance = IORedirector(getattr(sys, std), redirect_to, wrap_buffer=wrap_buffer)
                setattr(sys, std, new_std_instance)
            else:
                new_std_instance = redirect_to
                setattr(sys, std, redirect_to)

            stack.append(_RedirectInfo(original, new_std_instance))

        return redirect_to


def end_redirect(std='stdout'):
    with _RedirectionsHolder._lock:
        if std == 'both':
            config_stds = ['stdout', 'stderr']
        else:
            config_stds = [std]
        for std in config_stds:
            stack = getattr(_RedirectionsHolder, '_stack_%s' % std)
            redirect_info = stack.pop()
            setattr(sys, std, redirect_info.original)


def redirect_stream_to_pydb_io_messages(std):
    '''
    :param std:
        'stdout' or 'stderr'
    '''
    with _RedirectionsHolder._lock:
        redirect_to_name = '_pydevd_%s_redirect_' % (std,)
        if getattr(_RedirectionsHolder, redirect_to_name) is None:
            wrap_buffer = True
            original = getattr(sys, std)

            redirect_to = RedirectToPyDBIoMessages(1 if std == 'stdout' else 2, original, wrap_buffer)
            start_redirect(keep_original_redirection=True, std=std, redirect_to=redirect_to)

            stack = getattr(_RedirectionsHolder, '_stack_%s' % std)
            setattr(_RedirectionsHolder, redirect_to_name, stack[-1])
            return True

        return False


def stop_redirect_stream_to_pydb_io_messages(std):
    '''
    :param std:
        'stdout' or 'stderr'
    '''
    with _RedirectionsHolder._lock:
        redirect_to_name = '_pydevd_%s_redirect_' % (std,)
        redirect_info = getattr(_RedirectionsHolder, redirect_to_name)
        if redirect_info is not None:  # :type redirect_info: _RedirectInfo
            setattr(_RedirectionsHolder, redirect_to_name, None)

            stack = getattr(_RedirectionsHolder, '_stack_%s' % std)
            prev_info = stack.pop()

            curr = getattr(sys, std)
            if curr is redirect_info.redirect_to:
                setattr(sys, std, redirect_info.original)


@contextmanager
def redirect_stream_to_pydb_io_messages_context():
    with _RedirectionsHolder._lock:
        redirecting = []
        for std in ('stdout', 'stderr'):
            if redirect_stream_to_pydb_io_messages(std):
                redirecting.append(std)

        try:
            yield
        finally:
            for std in redirecting:
                stop_redirect_stream_to_pydb_io_messages(std)

