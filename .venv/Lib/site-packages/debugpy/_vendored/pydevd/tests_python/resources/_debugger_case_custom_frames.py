import sys

from _pydevd_bundle.pydevd_custom_frames import add_custom_frame
import threading


def call1():
    add_custom_frame(sys._getframe(), 'call1', threading.current_thread().ident)


def call2():
    add_custom_frame(sys._getframe(), 'call2', threading.current_thread().ident)


def call3():
    add_custom_frame(sys._getframe(), 'call3', threading.current_thread().ident)


if __name__ == '__main__':
    call1()  # break here
    call2()
    call3()
    print('TEST SUCEEDED')
