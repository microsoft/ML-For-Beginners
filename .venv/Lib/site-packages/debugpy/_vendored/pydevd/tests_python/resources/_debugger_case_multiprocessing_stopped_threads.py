import time
import multiprocessing
import threading
import sys
import os

event = threading.Event()


class MyThread(threading.Thread):

    def run(self):
        _a = 10
        _b = 20  # break in thread here
        event.set()
        _c = 20


def run_in_multiprocess():
    _a = 30
    _b = 40  # break in process here


if __name__ == '__main__':
    MyThread().start()
    event.wait()
    if sys.version_info[0] >= 3 and sys.platform != 'win32':
        multiprocessing.set_start_method('fork')
    p = multiprocessing.Process(target=run_in_multiprocess, args=())
    p.start()
    print('TEST SUCEEDED!')  # break in main here
    p.join()
