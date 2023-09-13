from __future__ import print_function
from multiprocessing import Pool
from time import sleep


def call(arg):
    if arg == '2':
        print("called")  # break 1 here


if __name__ == '__main__':
    pool = Pool(1)
    pool.map(call, ['1', '2', '3'])
    pool.close()
    pool.join()
    sleep(1)
    print('TEST SUCEEDED!')  # break 2 here
