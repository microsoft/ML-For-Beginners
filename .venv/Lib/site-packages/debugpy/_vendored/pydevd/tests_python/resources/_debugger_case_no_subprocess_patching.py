import subprocess
import sys
import os
from contextlib import contextmanager
env = os.environ.copy()
pythonpath = env.get('PYTHONPATH', '')
env['PYTHONPATH'] = os.path.dirname(__file__)


def call():
    print("called")  # break 1 here


@contextmanager
def dummy_context_manager():
    yield


if __name__ == '__main__':
    if 'apply-multiprocessing-patch' in sys.argv:
        context_manager = dummy_context_manager  # It's applied by default.
    else:
        import pydevd
        context_manager = pydevd.skip_subprocess_arg_patch

    with context_manager():
        p = subprocess.Popen(
            [sys.executable, '-c', 'import _debugger_case_no_subprocess_patching;_debugger_case_no_subprocess_patching.call()'],
            stdout=subprocess.PIPE,
            env=env,
        )

    stdout, stderr = p.communicate()
    assert b'called' in stdout, 'Did not find b"called" in: %s' % (stdout,)
    print('TEST SUCEEDED!')  # break 2 here
