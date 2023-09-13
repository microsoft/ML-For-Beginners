import subprocess
import sys

args = [
    'connect(\\"127 . 0.0.1\\")',
    '"',
    '',
    '\\',
    '\\"',
    '""',
]


def main():
    retcode = subprocess.call([sys.executable, __file__] + args)
    assert retcode == 0


if __name__ == '__main__':
    sys_args = sys.argv[1:]
    if sys_args:
        assert sys_args == args, 'Expected that %r == %r' % (sys_args, args)
        print('break here')
        print('TEST SUCEEDED!')
    else:
        main()
