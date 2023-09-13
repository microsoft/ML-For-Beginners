import threading

semaphore1 = threading.Semaphore(0)
proceed = False


def thread_target():
    semaphore1.release()
    import time
    
    while True:
        if proceed:
            break
        time.sleep(1 / 30.)


for i in range(2):
    t = threading.Thread(target=thread_target)
    t.start()
    
semaphore1.acquire()  # let first thread run
semaphore1.acquire()  # let second thread run

# At this point we know both other threads are already running.
print('break here')

proceed = True

print('TEST SUCEEDED!')
