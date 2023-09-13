import subprocess
import sys
import os
env = os.environ.copy()
pythonpath = env.get('PYTHONPATH', '')
env['PYTHONPATH'] = os.path.dirname(__file__)


def call():
    print("called")  # break 1 here


if __name__ == '__main__':
    p = subprocess.Popen(
        [sys.executable, '-c', 'import _debugger_case_python_c;_debugger_case_python_c.call()'],
        stdout=subprocess.PIPE,
        env=env,
    )
    stdout, stderr = p.communicate()
    assert b'called' in stdout, 'Did not find b"called" in: %s' % (stdout,)
    print('TEST SUCEEDED!')  # break 2 here
