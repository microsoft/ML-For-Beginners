import threading


class KeepInLoop(object):
    keep_in_loop = True  # Debugger should change to False to break.


def stop_loop():
    KeepInLoop.keep_in_loop = False
    return 'stopped_loop'


def double_number(number):
    while KeepInLoop.keep_in_loop:
        doubled = number * 2
        print(doubled)
        import time
        time.sleep(.5)
    return doubled


if __name__ == '__main__':
    threads = []
    for num in range(2):
        thread = threading.Thread(target=double_number, args=(num,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print('TEST SUCEEDED!')
