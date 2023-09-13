import pydevd
import threading

original = pydevd.PyDB.notify_thread_created

found = set()

def new_notify_thread_created(self, thread_id, thread, *args, **kwargs):
    found.add(thread)
    return original(self, thread_id, thread, *args, **kwargs)

pydevd.PyDB.notify_thread_created = new_notify_thread_created 

ok = []
class MyThread(threading.Thread):
    
    def run(self):
        if self not in found:
            ok.append(False)
        else:
            ok.append(True)
        
if __name__ == '__main__':
    threads = []
    for i in range(15):
        t = MyThread()
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    assert len(ok) == len(threads)
    assert all(ok), 'Expected all threads to be notified of their creation before starting to run. Found: %s' % (ok,)
    
    found.clear()
    print('TEST SUCEEDED')
        
