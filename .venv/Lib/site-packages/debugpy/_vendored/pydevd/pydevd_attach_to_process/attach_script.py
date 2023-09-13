

def get_main_thread_instance(threading):
    if hasattr(threading, 'main_thread'):
        return threading.main_thread()
    else:
        # On Python 2 we don't really have an API to get the main thread,
        # so, we just get it from the 'shutdown' bound method.
        return threading._shutdown.im_self


def get_main_thread_id(unlikely_thread_id=None):
    '''
    :param unlikely_thread_id:
        Pass to mark some thread id as not likely the main thread.

    :return tuple(thread_id, critical_warning)
    '''
    import sys
    import os

    current_frames = sys._current_frames()
    possible_thread_ids = []
    for thread_ident, frame in current_frames.items():
        while frame.f_back is not None:
            frame = frame.f_back

        basename = os.path.basename(frame.f_code.co_filename)
        if basename.endswith(('.pyc', '.pyo')):
            basename = basename[:-1]

        if (frame.f_code.co_name, basename) in [
                ('_run_module_as_main', 'runpy.py'),
                ('_run_module_as_main', '<frozen runpy>'),
                ('run_module_as_main', 'runpy.py'),
                ('run_module', 'runpy.py'),
                ('run_path', 'runpy.py'),
            ]:
            # This is the case for python -m <module name> (this is an ideal match, so,
            # let's return it).
            return thread_ident, ''

        if frame.f_code.co_name == '<module>':
            if frame.f_globals.get('__name__') == '__main__':
                possible_thread_ids.insert(0, thread_ident)  # Add with higher priority
                continue

            # Usually the main thread will be started in the <module>, whereas others would
            # be started in another place (but when Python is embedded, this may not be
            # correct, so, just add to the available possibilities as we'll have to choose
            # one if there are multiple).
            possible_thread_ids.append(thread_ident)

    if len(possible_thread_ids) > 0:
        if len(possible_thread_ids) == 1:
            return possible_thread_ids[0], ''  # Ideal: only one match

        while unlikely_thread_id in possible_thread_ids:
            possible_thread_ids.remove(unlikely_thread_id)

        if len(possible_thread_ids) == 1:
            return possible_thread_ids[0], ''  # Ideal: only one match

        elif len(possible_thread_ids) > 1:
            # Bad: we can't really be certain of anything at this point.
            return possible_thread_ids[0], \
                'Multiple thread ids found (%s). Choosing main thread id randomly (%s).' % (
                    possible_thread_ids, possible_thread_ids[0])

    # If we got here we couldn't discover the main thread id.
    return None, 'Unable to discover main thread id.'


def fix_main_thread_id(on_warn=lambda msg:None, on_exception=lambda msg:None, on_critical=lambda msg:None):
    # This means that we weren't able to import threading in the main thread (which most
    # likely means that the main thread is paused or in some very long operation).
    # In this case we'll import threading here and hotfix what may be wrong in the threading
    # module (if we're on Windows where we create a thread to do the attach and on Linux
    # we are not certain on which thread we're executing this code).
    #
    # The code below is a workaround for https://bugs.python.org/issue37416
    import sys
    import threading

    try:
        with threading._active_limbo_lock:
            main_thread_instance = get_main_thread_instance(threading)

            if sys.platform == 'win32':
                # On windows this code would be called in a secondary thread, so,
                # the current thread is unlikely to be the main thread.
                if hasattr(threading, '_get_ident'):
                    unlikely_thread_id = threading._get_ident()  # py2
                else:
                    unlikely_thread_id = threading.get_ident()  # py3
            else:
                unlikely_thread_id = None

            main_thread_id, critical_warning = get_main_thread_id(unlikely_thread_id)

            if main_thread_id is not None:
                main_thread_id_attr = '_ident'
                if not hasattr(main_thread_instance, main_thread_id_attr):
                    main_thread_id_attr = '_Thread__ident'
                    assert hasattr(main_thread_instance, main_thread_id_attr)

                if main_thread_id != getattr(main_thread_instance, main_thread_id_attr):
                    # Note that we also have to reset the '_tstack_lock' for a regular lock.
                    # This is needed to avoid an error on shutdown because this lock is bound
                    # to the thread state and will be released when the secondary thread
                    # that initialized the lock is finished -- making an assert fail during
                    # process shutdown.
                    main_thread_instance._tstate_lock = threading._allocate_lock()
                    main_thread_instance._tstate_lock.acquire()

                    # Actually patch the thread ident as well as the threading._active dict
                    # (we should have the _active_limbo_lock to do that).
                    threading._active.pop(getattr(main_thread_instance, main_thread_id_attr), None)
                    setattr(main_thread_instance, main_thread_id_attr, main_thread_id)
                    threading._active[getattr(main_thread_instance, main_thread_id_attr)] = main_thread_instance

        # Note: only import from pydevd after the patching is done (we want to do the minimum
        # possible when doing that patching).
        on_warn('The threading module was not imported by user code in the main thread. The debugger will attempt to work around https://bugs.python.org/issue37416.')

        if critical_warning:
            on_critical('Issue found when debugger was trying to work around https://bugs.python.org/issue37416:\n%s' % (critical_warning,))
    except:
        on_exception('Error patching main thread id.')


def attach(port, host, protocol='', debug_mode=''):
    try:
        import sys
        fix_main_thread = 'threading' not in sys.modules

        if fix_main_thread:

            def on_warn(msg):
                from _pydev_bundle import pydev_log
                pydev_log.warn(msg)

            def on_exception(msg):
                from _pydev_bundle import pydev_log
                pydev_log.exception(msg)

            def on_critical(msg):
                from _pydev_bundle import pydev_log
                pydev_log.critical(msg)

            fix_main_thread_id(on_warn=on_warn, on_exception=on_exception, on_critical=on_critical)

        else:
            from _pydev_bundle import pydev_log  # @Reimport
            pydev_log.debug('The threading module is already imported by user code.')

        if protocol:
            from _pydevd_bundle import pydevd_defaults
            pydevd_defaults.PydevdCustomization.DEFAULT_PROTOCOL = protocol

        if debug_mode:
            from _pydevd_bundle import pydevd_defaults
            pydevd_defaults.PydevdCustomization.DEBUG_MODE = debug_mode

        import pydevd

        # I.e.: disconnect/reset if already connected.

        pydevd.SetupHolder.setup = None

        py_db = pydevd.get_global_debugger()
        if py_db is not None:
            py_db.dispose_and_kill_all_pydevd_threads(wait=False)

        # pydevd.DebugInfoHolder.DEBUG_TRACE_LEVEL = 3
        pydevd.settrace(
            port=port,
            host=host,
            stdoutToServer=True,
            stderrToServer=True,
            overwrite_prev_trace=True,
            suspend=False,
            trace_only_current_thread=False,
            patch_multiprocessing=False,
        )
    except:
        import traceback
        traceback.print_exc()
