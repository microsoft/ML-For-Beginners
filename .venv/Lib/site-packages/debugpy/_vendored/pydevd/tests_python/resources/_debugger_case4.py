import time


class ProceedContainer:
    proceed = False


def exit_while_loop():
    ProceedContainer.proceed = True
    return 'ok'


def sleep():
    while not ProceedContainer.proceed:  # The debugger should change the proceed to True to exit the loop.
        time.sleep(.1)


if __name__ == '__main__':
    sleep()

    print('TEST SUCEEDED')

