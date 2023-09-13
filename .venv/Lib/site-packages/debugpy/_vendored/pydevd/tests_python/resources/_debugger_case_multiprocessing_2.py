from multiprocessing import Process, Queue


class Foo:

    def __init__(self, value):
        self.value = value  # break 2 here


def f(q):
    q.put(Foo(1))


if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p.start()
    print(q.get().value)  # break 1 here
    print('TEST SUCEEDED!')
    p.join()
