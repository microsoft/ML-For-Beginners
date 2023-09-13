"""Error script. DO NOT EDIT FURTHER! It will break exception doctests!!!"""
import sys

def div0():
    "foo"
    x = 1
    y = 0
    x/y

def sysexit(stat, mode):
    raise SystemExit(stat, f"Mode = {mode}")


def bar(mode):
    "bar"
    if mode=='div':
        div0()
    elif mode=='exit':
        try:
            stat = int(sys.argv[2])
        except:
            stat = 1
        sysexit(stat, mode)
    else:
        raise ValueError('Unknown mode')

if __name__ == '__main__':
    try:
        mode = sys.argv[1]
    except IndexError:
        mode = 'div'

    bar(mode)
