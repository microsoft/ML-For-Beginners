"""
Utility for saving locals.
"""
import sys

try:
    import types

    frame_type = types.FrameType
except:
    frame_type = type(sys._getframe())


def is_save_locals_available():
    return save_locals_impl is not None


def save_locals(frame):
    """
    Copy values from locals_dict into the fast stack slots in the given frame.

    Note: the 'save_locals' branch had a different approach wrapping the frame (much more code, but it gives ideas
    on how to save things partially, not the 'whole' locals).
    """
    if not isinstance(frame, frame_type):
        # Fix exception when changing Django variable (receiving DjangoTemplateFrame)
        return

    if save_locals_impl is not None:
        try:
            save_locals_impl(frame)
        except:
            pass


def make_save_locals_impl():
    """
    Factory for the 'save_locals_impl' method. This may seem like a complicated pattern but it is essential that the method is created at
    module load time. Inner imports after module load time would cause an occasional debugger deadlock due to the importer lock and debugger
    lock being taken in different order in  different threads.
    """
    try:
        if '__pypy__' in sys.builtin_module_names:
            import __pypy__  # @UnresolvedImport
            save_locals = __pypy__.locals_to_fast
    except:
        pass
    else:
        if '__pypy__' in sys.builtin_module_names:

            def save_locals_pypy_impl(frame):
                save_locals(frame)

            return save_locals_pypy_impl

    try:
        import ctypes
        locals_to_fast = ctypes.pythonapi.PyFrame_LocalsToFast
    except:
        pass
    else:

        def save_locals_ctypes_impl(frame):
            locals_to_fast(ctypes.py_object(frame), ctypes.c_int(0))

        return save_locals_ctypes_impl

    return None


save_locals_impl = make_save_locals_impl()

_SENTINEL = []  # Any mutable will do.


def update_globals_and_locals(updated_globals, initial_globals, frame):
    # We don't have the locals and passed all in globals, so, we have to
    # manually choose how to update the variables.
    #
    # Note that the current implementation is a bit tricky: it does work in general
    # but if we do something as 'some_var = 10' and 'some_var' is already defined to have
    # the value '10' in the globals, we won't actually put that value in the locals
    # (which means that the frame locals won't be updated).
    # Still, the approach to have a single namespace was chosen because it was the only
    # one that enabled creating and using variables during the same evaluation.
    assert updated_globals is not None
    f_locals = None

    removed = set(initial_globals).difference(updated_globals)

    for key, val in updated_globals.items():
        if val is not initial_globals.get(key, _SENTINEL):
            if f_locals is None:
                # Note: we call f_locals only once because each time
                # we call it the values may be reset.
                f_locals = frame.f_locals

            f_locals[key] = val

    if removed:
        if f_locals is None:
            # Note: we call f_locals only once because each time
            # we call it the values may be reset.
            f_locals = frame.f_locals

        for key in removed:
            try:
                del f_locals[key]
            except KeyError:
                pass

    if f_locals is not None:
        save_locals(frame)
