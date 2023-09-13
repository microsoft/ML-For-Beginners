import threading


class MyThread(threading.Thread):

    def run(self):
        pass


threads = [MyThread() for i in range(3)]

for t in threads:
    t.start()

for t in threads:
    t.join()

print('TEST SUCEEDED!')  # Break here
