# Note: this module should be self-contained to run tests (as it relies on `threading` not being
# imported and having no other threads running).


def wait_for_condition(condition, msg=None, timeout=5, sleep=.05):
    import time
    curtime = time.time()
    while True:
        if condition():
            break
        if time.time() - curtime > timeout:
            error_msg = 'Condition not reached in %s seconds' % (timeout,)
            if msg is not None:
                error_msg += '\n'
                if callable(msg):
                    error_msg += msg()
                else:
                    error_msg += str(msg)

            raise AssertionError('Timeout: %s' % (error_msg,))
        time.sleep(sleep)


def check_main_thread_id_simple():
    import attach_script
    import sys
    assert 'threading' not in sys.modules
    try:
        import thread
    except ImportError:
        import _thread as thread

    main_thread_id, log_msg = attach_script.get_main_thread_id(None)
    assert main_thread_id == thread.get_ident(), 'Found: %s, Expected: %s' % (main_thread_id, thread.get_ident())
    assert not log_msg
    assert 'threading' not in sys.modules
    wait_for_condition(lambda: len(sys._current_frames()) == 1)


def check_main_thread_id_multiple_threads():
    import attach_script
    import sys
    import time
    assert 'threading' not in sys.modules
    try:
        import thread
    except ImportError:
        import _thread as thread

    lock = thread.allocate_lock()
    lock2 = thread.allocate_lock()

    def method():
        lock2.acquire()
        with lock:
            pass  # Will only finish when lock is released.

    with lock:
        thread.start_new_thread(method, ())
        while not lock2.locked():
            time.sleep(.1)

        wait_for_condition(lambda: len(sys._current_frames()) == 2)

        main_thread_id, log_msg = attach_script.get_main_thread_id(None)
        assert main_thread_id == thread.get_ident(), 'Found: %s, Expected: %s' % (main_thread_id, thread.get_ident())
        assert not log_msg
        assert 'threading' not in sys.modules
    wait_for_condition(lambda: len(sys._current_frames()) == 1)


def check_fix_main_thread_id_multiple_threads():
    import attach_script
    import sys
    import time
    assert 'threading' not in sys.modules
    try:
        import thread
    except ImportError:
        import _thread as thread

    lock = thread.allocate_lock()
    lock2 = thread.allocate_lock()

    def method():
        lock2.acquire()
        import threading  # Note: imported on wrong thread
        assert threading.current_thread().ident == thread.get_ident()
        assert threading.current_thread() is attach_script.get_main_thread_instance(threading)

        attach_script.fix_main_thread_id()

        assert threading.current_thread().ident == thread.get_ident()
        assert threading.current_thread() is not attach_script.get_main_thread_instance(threading)

        with lock:
            pass  # Will only finish when lock is released.

    with lock:
        thread.start_new_thread(method, ())
        while not lock2.locked():
            time.sleep(.1)

        wait_for_condition(lambda: len(sys._current_frames()) == 2)

        main_thread_id, log_msg = attach_script.get_main_thread_id(None)
        assert main_thread_id == thread.get_ident(), 'Found: %s, Expected: %s' % (main_thread_id, thread.get_ident())
        assert not log_msg
        assert 'threading' in sys.modules
        import threading
        assert threading.current_thread().ident == main_thread_id
    wait_for_condition(lambda: len(sys._current_frames()) == 1)


def check_win_threads():
    import sys
    if sys.platform != 'win32':
        return

    import attach_script
    import time
    assert 'threading' not in sys.modules
    try:
        import thread
    except ImportError:
        import _thread as thread
    from ctypes import windll, WINFUNCTYPE, c_uint32, c_void_p, c_size_t

    ThreadProc = WINFUNCTYPE(c_uint32, c_void_p)

    lock = thread.allocate_lock()
    lock2 = thread.allocate_lock()

    @ThreadProc
    def method(_):
        lock2.acquire()
        with lock:
            pass  # Will only finish when lock is released.
        return 0

    with lock:
        windll.kernel32.CreateThread(None, c_size_t(0), method, None, c_uint32(0), None)
        while not lock2.locked():
            time.sleep(.1)

        wait_for_condition(lambda: len(sys._current_frames()) == 2)

        main_thread_id, log_msg = attach_script.get_main_thread_id(None)
        assert main_thread_id == thread.get_ident(), 'Found: %s, Expected: %s' % (main_thread_id, thread.get_ident())
        assert not log_msg
        assert 'threading' not in sys.modules
    wait_for_condition(lambda: len(sys._current_frames()) == 1)


if __name__ == '__main__':
    check_main_thread_id_simple()
    check_main_thread_id_multiple_threads()
    check_win_threads()
    check_fix_main_thread_id_multiple_threads()  # Important: must be the last test checked!
