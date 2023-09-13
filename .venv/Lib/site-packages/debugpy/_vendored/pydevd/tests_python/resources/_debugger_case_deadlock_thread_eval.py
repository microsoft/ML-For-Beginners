'''
The idea here is that a secondary thread does the processing of instructions,
so, when all threads are stopped, doing an evaluation for:

processor.process('xxx')

would be locked until secondary threads start running.
See: https://github.com/microsoft/debugpy/issues/157
'''

import threading
try:
    from queue import Queue
except:
    from Queue import Queue


class EchoThread(threading.Thread):

    def __init__(self, queue):
        threading.Thread.__init__(self)
        self._queue = queue
        self.started = threading.Event()

    def run(self):
        self.started.set()
        while True:
            obj = self._queue.get()
            if obj == 'finish':
                break

            print('processed', obj.value)
            obj.event.set()  # Break here 2


class NotificationObject(object):

    def __init__(self, value):
        self.value = value
        self.event = threading.Event()


class Processor(object):

    def __init__(self, queue):
        self._queue = queue

    def process(self, i):
        obj = NotificationObject(i)
        self._queue.put(obj)
        assert obj.event.wait()

    def finish(self):
        self._queue.put('finish')


def main():
    queue = Queue()
    echo_thread = EchoThread(queue)
    processor = Processor(queue)
    echo_thread.start()
    echo_thread.started.wait()

    processor.process(1)  # Break here 1
    processor.process(2)
    processor.process(3)
    processor.finish()


if __name__ == '__main__':
    main()
    print('TEST SUCEEDED!')
