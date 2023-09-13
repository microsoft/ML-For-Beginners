from _pydev_bundle._pydev_saved_modules import threading


def wrapper(fun):

    def pydev_after_run_call():
        pass

    def inner(*args, **kwargs):
        fun(*args, **kwargs)
        pydev_after_run_call()

    return inner


def wrap_attr(obj, attr):
    t_save_start = getattr(obj, attr)
    setattr(obj, attr, wrapper(t_save_start))
    obj._pydev_run_patched = True


class ObjectWrapper(object):

    def __init__(self, obj):
        self.wrapped_object = obj
        try:
            import functools
            functools.update_wrapper(self, obj)
        except:
            pass

    def __getattr__(self, attr):
        orig_attr = getattr(self.wrapped_object, attr)  # .__getattribute__(attr)
        if callable(orig_attr):

            def patched_attr(*args, **kwargs):
                self.call_begin(attr)
                result = orig_attr(*args, **kwargs)
                self.call_end(attr)
                if result == self.wrapped_object:
                    return self
                return result

            return patched_attr
        else:
            return orig_attr

    def call_begin(self, attr):
        pass

    def call_end(self, attr):
        pass

    def __enter__(self):
        self.call_begin("__enter__")
        self.wrapped_object.__enter__()
        self.call_end("__enter__")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.call_begin("__exit__")
        self.wrapped_object.__exit__(exc_type, exc_val, exc_tb)


def factory_wrapper(fun):

    def inner(*args, **kwargs):
        obj = fun(*args, **kwargs)
        return ObjectWrapper(obj)

    return inner


def wrap_threads():
    # TODO: add wrappers for thread and _thread
    # import _thread as mod
    # print("Thread imported")
    # mod.start_new_thread = wrapper(mod.start_new_thread)
    threading.Lock = factory_wrapper(threading.Lock)
    threading.RLock = factory_wrapper(threading.RLock)

    # queue patching
    import queue  # @UnresolvedImport
    queue.Queue = factory_wrapper(queue.Queue)
