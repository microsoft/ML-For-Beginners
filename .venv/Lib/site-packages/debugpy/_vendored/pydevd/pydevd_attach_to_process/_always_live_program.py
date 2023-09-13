import sys
import struct
print('Executable: %s' % sys.executable)
import os
def loop_in_thread():
    while True:
        import time
        time.sleep(.5)
        sys.stdout.write('#')
        sys.stdout.flush()

import threading
threading.Thread(target=loop_in_thread).start()


def is_python_64bit():
    return (struct.calcsize('P') == 8)

print('Is 64: %s' % is_python_64bit())

if __name__ == '__main__':
    print('pid:%s' % (os.getpid()))
    i = 0
    while True:
        i += 1
        import time
        time.sleep(.5)
        sys.stdout.write('.')
        sys.stdout.flush()
        if i % 40 == 0:
            sys.stdout.write('\n')
            sys.stdout.flush()
