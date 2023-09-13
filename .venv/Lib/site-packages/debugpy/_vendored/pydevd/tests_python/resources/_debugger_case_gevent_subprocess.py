from gevent import monkey
monkey.patch_all()

import subprocess
import sys
import os

if __name__ == "__main__":
    if '-foo' in sys.argv:
        print('foo called')
    else:
        if os.environ.get('CALL_PYTHON_SUB') == '1':
            assert 'foo called' in subprocess.check_output([sys.executable, __file__, '-foo']).decode('utf-8')
        else:
            subprocess.check_output("tput -T xterm-256color bold".split())
        print('TEST SUCEEDED')
