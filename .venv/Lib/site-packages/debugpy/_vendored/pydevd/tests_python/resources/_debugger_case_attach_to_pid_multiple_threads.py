
import time
import sys
try:
    import _thread
except:
    import thread as _thread

if __name__ == '__main__':

    lock = _thread.allocate_lock()
    initialized = [False]
    print('Main thread ident should be: %s' % (_thread.get_ident()))

    def new_thread_function():
        sys.secondary_id = _thread.get_ident()
        print('Secondary thread ident should be: %s' % (_thread.get_ident()))
        wait = True

        with lock:
            initialized[0] = True
            while wait:
                time.sleep(.1)  # break thread here

    _thread.start_new_thread(new_thread_function, ())

    wait = True

    while not initialized[0]:
        time.sleep(.1)

    with lock:  # It'll be here until the secondary thread finishes (i.e.: releases the lock).
        pass

    import threading  # Note: only import after the attach.
    curr_thread_ident = threading.current_thread().ident
    if hasattr(threading, 'main_thread'):
        main_thread_ident = threading.main_thread().ident
    else:
        # Python 2 does not have main_thread, but we can still get the reference.
        main_thread_ident = threading._shutdown.im_self.ident

    if curr_thread_ident != main_thread_ident:
        raise AssertionError('Expected current thread ident (%s) to be the main thread ident (%s)' % (
            curr_thread_ident, main_thread_ident))

    print('TEST SUCEEDED')
