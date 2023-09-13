import multiprocessing
import sys


def run(name):
    print("argument: ", name)  # break 1 here


if __name__ == '__main__':
    if sys.version_info[0] >= 3 and sys.platform != 'win32':
        multiprocessing.set_start_method('fork')
    p = multiprocessing.Process(target=run, args=("argument to run method",))
    p.start()
    print('TEST SUCEEDED!')  # break 2 here
    p.join()
