"""
Makes it possible to do the compiled analysis in a subprocess. This has two
goals:

1. Making it safer - Segfaults and RuntimeErrors as well as stdout/stderr can
   be ignored and dealt with.
2. Make it possible to handle different Python versions as well as virtualenvs.
"""

import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread

from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, \
    SignatureParam
from jedi.api.exceptions import InternalError


_MAIN_PATH = os.path.join(os.path.dirname(__file__), '__main__.py')
PICKLE_PROTOCOL = 4


def _GeneralizedPopen(*args, **kwargs):
    if os.name == 'nt':
        try:
            # Was introduced in Python 3.7.
            CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW
        except AttributeError:
            CREATE_NO_WINDOW = 0x08000000
        kwargs['creationflags'] = CREATE_NO_WINDOW
    # The child process doesn't need file descriptors except 0, 1, 2.
    # This is unix only.
    kwargs['close_fds'] = 'posix' in sys.builtin_module_names

    return subprocess.Popen(*args, **kwargs)


def _enqueue_output(out, queue_):
    for line in iter(out.readline, b''):
        queue_.put(line)


def _add_stderr_to_debug(stderr_queue):
    while True:
        # Try to do some error reporting from the subprocess and print its
        # stderr contents.
        try:
            line = stderr_queue.get_nowait()
            line = line.decode('utf-8', 'replace')
            debug.warning('stderr output: %s' % line.rstrip('\n'))
        except queue.Empty:
            break


def _get_function(name):
    return getattr(functions, name)


def _cleanup_process(process, thread):
    try:
        process.kill()
        process.wait()
    except OSError:
        # Raised if the process is already killed.
        pass
    thread.join()
    for stream in [process.stdin, process.stdout, process.stderr]:
        try:
            stream.close()
        except OSError:
            # Raised if the stream is broken.
            pass


class _InferenceStateProcess:
    def __init__(self, inference_state):
        self._inference_state_weakref = weakref.ref(inference_state)
        self._inference_state_id = id(inference_state)
        self._handles = {}

    def get_or_create_access_handle(self, obj):
        id_ = id(obj)
        try:
            return self.get_access_handle(id_)
        except KeyError:
            access = DirectObjectAccess(self._inference_state_weakref(), obj)
            handle = AccessHandle(self, access, id_)
            self.set_access_handle(handle)
            return handle

    def get_access_handle(self, id_):
        return self._handles[id_]

    def set_access_handle(self, handle):
        self._handles[handle.id] = handle


class InferenceStateSameProcess(_InferenceStateProcess):
    """
    Basically just an easy access to functions.py. It has the same API
    as InferenceStateSubprocess and does the same thing without using a subprocess.
    This is necessary for the Interpreter process.
    """
    def __getattr__(self, name):
        return partial(_get_function(name), self._inference_state_weakref())


class InferenceStateSubprocess(_InferenceStateProcess):
    def __init__(self, inference_state, compiled_subprocess):
        super().__init__(inference_state)
        self._used = False
        self._compiled_subprocess = compiled_subprocess

    def __getattr__(self, name):
        func = _get_function(name)

        def wrapper(*args, **kwargs):
            self._used = True

            result = self._compiled_subprocess.run(
                self._inference_state_weakref(),
                func,
                args=args,
                kwargs=kwargs,
            )
            # IMO it should be possible to create a hook in pickle.load to
            # mess with the loaded objects. However it's extremely complicated
            # to work around this so just do it with this call. ~ dave
            return self._convert_access_handles(result)

        return wrapper

    def _convert_access_handles(self, obj):
        if isinstance(obj, SignatureParam):
            return SignatureParam(*self._convert_access_handles(tuple(obj)))
        elif isinstance(obj, tuple):
            return tuple(self._convert_access_handles(o) for o in obj)
        elif isinstance(obj, list):
            return [self._convert_access_handles(o) for o in obj]
        elif isinstance(obj, AccessHandle):
            try:
                # Rewrite the access handle to one we're already having.
                obj = self.get_access_handle(obj.id)
            except KeyError:
                obj.add_subprocess(self)
                self.set_access_handle(obj)
        elif isinstance(obj, AccessPath):
            return AccessPath(self._convert_access_handles(obj.accesses))
        return obj

    def __del__(self):
        if self._used and not self._compiled_subprocess.is_crashed:
            self._compiled_subprocess.delete_inference_state(self._inference_state_id)


class CompiledSubprocess:
    is_crashed = False

    def __init__(self, executable, env_vars=None):
        self._executable = executable
        self._env_vars = env_vars
        self._inference_state_deletion_queue = collections.deque()
        self._cleanup_callable = lambda: None

    def __repr__(self):
        pid = os.getpid()
        return '<%s _executable=%r, is_crashed=%r, pid=%r>' % (
            self.__class__.__name__,
            self._executable,
            self.is_crashed,
            pid,
        )

    @memoize_method
    def _get_process(self):
        debug.dbg('Start environment subprocess %s', self._executable)
        parso_path = sys.modules['parso'].__file__
        args = (
            self._executable,
            _MAIN_PATH,
            os.path.dirname(os.path.dirname(parso_path)),
            '.'.join(str(x) for x in sys.version_info[:3]),
        )
        process = _GeneralizedPopen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env_vars
        )
        self._stderr_queue = queue.Queue()
        self._stderr_thread = t = Thread(
            target=_enqueue_output,
            args=(process.stderr, self._stderr_queue)
        )
        t.daemon = True
        t.start()
        # Ensure the subprocess is properly cleaned up when the object
        # is garbage collected.
        self._cleanup_callable = weakref.finalize(self,
                                                  _cleanup_process,
                                                  process,
                                                  t)
        return process

    def run(self, inference_state, function, args=(), kwargs={}):
        # Delete old inference_states.
        while True:
            try:
                inference_state_id = self._inference_state_deletion_queue.pop()
            except IndexError:
                break
            else:
                self._send(inference_state_id, None)

        assert callable(function)
        return self._send(id(inference_state), function, args, kwargs)

    def get_sys_path(self):
        return self._send(None, functions.get_sys_path, (), {})

    def _kill(self):
        self.is_crashed = True
        self._cleanup_callable()

    def _send(self, inference_state_id, function, args=(), kwargs={}):
        if self.is_crashed:
            raise InternalError("The subprocess %s has crashed." % self._executable)

        data = inference_state_id, function, args, kwargs
        try:
            pickle_dump(data, self._get_process().stdin, PICKLE_PROTOCOL)
        except BrokenPipeError:
            self._kill()
            raise InternalError("The subprocess %s was killed. Maybe out of memory?"
                                % self._executable)

        try:
            is_exception, traceback, result = pickle_load(self._get_process().stdout)
        except EOFError as eof_error:
            try:
                stderr = self._get_process().stderr.read().decode('utf-8', 'replace')
            except Exception as exc:
                stderr = '<empty/not available (%r)>' % exc
            self._kill()
            _add_stderr_to_debug(self._stderr_queue)
            raise InternalError(
                "The subprocess %s has crashed (%r, stderr=%s)." % (
                    self._executable,
                    eof_error,
                    stderr,
                ))

        _add_stderr_to_debug(self._stderr_queue)

        if is_exception:
            # Replace the attribute error message with a the traceback. It's
            # way more informative.
            result.args = (traceback,)
            raise result
        return result

    def delete_inference_state(self, inference_state_id):
        """
        Currently we are not deleting inference_state instantly. They only get
        deleted once the subprocess is used again. It would probably a better
        solution to move all of this into a thread. However, the memory usage
        of a single inference_state shouldn't be that high.
        """
        # With an argument - the inference_state gets deleted.
        self._inference_state_deletion_queue.append(inference_state_id)


class Listener:
    def __init__(self):
        self._inference_states = {}
        # TODO refactor so we don't need to process anymore just handle
        # controlling.
        self._process = _InferenceStateProcess(Listener)

    def _get_inference_state(self, function, inference_state_id):
        from jedi.inference import InferenceState

        try:
            inference_state = self._inference_states[inference_state_id]
        except KeyError:
            from jedi import InterpreterEnvironment
            inference_state = InferenceState(
                # The project is not actually needed. Nothing should need to
                # access it.
                project=None,
                environment=InterpreterEnvironment()
            )
            self._inference_states[inference_state_id] = inference_state
        return inference_state

    def _run(self, inference_state_id, function, args, kwargs):
        if inference_state_id is None:
            return function(*args, **kwargs)
        elif function is None:
            del self._inference_states[inference_state_id]
        else:
            inference_state = self._get_inference_state(function, inference_state_id)

            # Exchange all handles
            args = list(args)
            for i, arg in enumerate(args):
                if isinstance(arg, AccessHandle):
                    args[i] = inference_state.compiled_subprocess.get_access_handle(arg.id)
            for key, value in kwargs.items():
                if isinstance(value, AccessHandle):
                    kwargs[key] = inference_state.compiled_subprocess.get_access_handle(value.id)

            return function(inference_state, *args, **kwargs)

    def listen(self):
        stdout = sys.stdout
        # Mute stdout. Nobody should actually be able to write to it,
        # because stdout is used for IPC.
        sys.stdout = open(os.devnull, 'w')
        stdin = sys.stdin
        stdout = stdout.buffer
        stdin = stdin.buffer

        while True:
            try:
                payload = pickle_load(stdin)
            except EOFError:
                # It looks like the parent process closed.
                # Don't make a big fuss here and just exit.
                exit(0)
            try:
                result = False, None, self._run(*payload)
            except Exception as e:
                result = True, traceback.format_exc(), e

            pickle_dump(result, stdout, PICKLE_PROTOCOL)


class AccessHandle:
    def __init__(self, subprocess, access, id_):
        self.access = access
        self._subprocess = subprocess
        self.id = id_

    def add_subprocess(self, subprocess):
        self._subprocess = subprocess

    def __repr__(self):
        try:
            detail = self.access
        except AttributeError:
            detail = '#' + str(self.id)
        return '<%s of %s>' % (self.__class__.__name__, detail)

    def __getstate__(self):
        return self.id

    def __setstate__(self, state):
        self.id = state

    def __getattr__(self, name):
        if name in ('id', 'access') or name.startswith('_'):
            raise AttributeError("Something went wrong with unpickling")

        # print('getattr', name, file=sys.stderr)
        return partial(self._workaround, name)

    def _workaround(self, name, *args, **kwargs):
        """
        TODO Currently we're passing slice objects around. This should not
        happen. They are also the only unhashable objects that we're passing
        around.
        """
        if args and isinstance(args[0], slice):
            return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)
        return self._cached_results(name, *args, **kwargs)

    @memoize_method
    def _cached_results(self, name, *args, **kwargs):
        return self._subprocess.get_compiled_method_return(self.id, name, *args, **kwargs)
