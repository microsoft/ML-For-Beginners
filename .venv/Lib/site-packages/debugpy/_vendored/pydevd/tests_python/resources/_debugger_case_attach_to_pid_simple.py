import time

if __name__ == '__main__':
    wait = True

    while wait:
        time.sleep(1)  # break here

    # Ok, if it got here things are looking good, let's just make
    # sure that the threading module main thread has the correct ident.
    import threading  # Note: only import after the attach.
    if hasattr(threading, 'main_thread'):
        assert threading.current_thread().ident == threading.main_thread().ident
    else:
        # Python 2 does not have main_thread, but we can still get the reference.
        assert threading.current_thread().ident == threading._shutdown.im_self.ident
    print('TEST SUCEEDED')
