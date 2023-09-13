import time
import threading


class ProceedContainer:
    tid_to_proceed = {
        1: False,
        2: False,
    }


def exit_while_loop(tid):
    ProceedContainer.tid_to_proceed[tid] = True
    return 'ok'


def thread_func(tid):
    while not ProceedContainer.tid_to_proceed[tid]:  # The debugger should change the proceed to True to exit the loop.
        time.sleep(.1)


if __name__ == '__main__':
    threads = [
        threading.Thread(target=thread_func, args=(1,)),
        threading.Thread(target=thread_func, args=(2,)),
    ]
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print('TEST SUCEEDED')
