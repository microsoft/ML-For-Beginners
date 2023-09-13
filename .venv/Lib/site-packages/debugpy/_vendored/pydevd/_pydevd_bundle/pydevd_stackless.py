from __future__ import nested_scopes

import weakref
import sys

from _pydevd_bundle.pydevd_comm import get_global_debugger
from _pydevd_bundle.pydevd_constants import call_only_once
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import update_custom_frame, remove_custom_frame, add_custom_frame
import stackless  # @UnresolvedImport
from _pydev_bundle import pydev_log


# Used so that we don't loose the id (because we'll remove when it's not alive and would generate a new id for the
# same tasklet).
class TaskletToLastId:
    '''
    So, why not a WeakKeyDictionary?
    The problem is that removals from the WeakKeyDictionary will create a new tasklet (as it adds a callback to
    remove the key when it's garbage-collected), so, we can get into a recursion.
    '''

    def __init__(self):
        self.tasklet_ref_to_last_id = {}
        self._i = 0

    def get(self, tasklet):
        return self.tasklet_ref_to_last_id.get(weakref.ref(tasklet))

    def __setitem__(self, tasklet, last_id):
        self.tasklet_ref_to_last_id[weakref.ref(tasklet)] = last_id
        self._i += 1
        if self._i % 100 == 0:  # Collect at each 100 additions to the dict (no need to rush).
            for tasklet_ref in list(self.tasklet_ref_to_last_id.keys()):
                if tasklet_ref() is None:
                    del self.tasklet_ref_to_last_id[tasklet_ref]


_tasklet_to_last_id = TaskletToLastId()


#=======================================================================================================================
# _TaskletInfo
#=======================================================================================================================
class _TaskletInfo:

    _last_id = 0

    def __init__(self, tasklet_weakref, tasklet):
        self.frame_id = None
        self.tasklet_weakref = tasklet_weakref

        last_id = _tasklet_to_last_id.get(tasklet)
        if last_id is None:
            _TaskletInfo._last_id += 1
            last_id = _TaskletInfo._last_id
            _tasklet_to_last_id[tasklet] = last_id

        self._tasklet_id = last_id

        self.update_name()

    def update_name(self):
        tasklet = self.tasklet_weakref()
        if tasklet:
            if tasklet.blocked:
                state = 'blocked'
            elif tasklet.paused:
                state = 'paused'
            elif tasklet.scheduled:
                state = 'scheduled'
            else:
                state = '<UNEXPECTED>'

            try:
                name = tasklet.name
            except AttributeError:
                if tasklet.is_main:
                    name = 'MainTasklet'
                else:
                    name = 'Tasklet-%s' % (self._tasklet_id,)

            thread_id = tasklet.thread_id
            if thread_id != -1:
                for thread in threading.enumerate():
                    if thread.ident == thread_id:
                        if thread.name:
                            thread_name = "of %s" % (thread.name,)
                        else:
                            thread_name = "of Thread-%s" % (thread.name or str(thread_id),)
                        break
                else:
                    # should not happen.
                    thread_name = "of Thread-%s" % (str(thread_id),)
                thread = None
            else:
                # tasklet is no longer bound to a thread, because its thread ended
                thread_name = "without thread"

            tid = id(tasklet)
            tasklet = None
        else:
            state = 'dead'
            name = 'Tasklet-%s' % (self._tasklet_id,)
            thread_name = ""
            tid = '-'
        self.tasklet_name = '%s %s %s (%s)' % (state, name, thread_name, tid)

    if not hasattr(stackless.tasklet, "trace_function"):

        # bug https://bitbucket.org/stackless-dev/stackless/issue/42
        # is not fixed. Stackless releases before 2014
        def update_name(self):
            tasklet = self.tasklet_weakref()
            if tasklet:
                try:
                    name = tasklet.name
                except AttributeError:
                    if tasklet.is_main:
                        name = 'MainTasklet'
                    else:
                        name = 'Tasklet-%s' % (self._tasklet_id,)

                thread_id = tasklet.thread_id
                for thread in threading.enumerate():
                    if thread.ident == thread_id:
                        if thread.name:
                            thread_name = "of %s" % (thread.name,)
                        else:
                            thread_name = "of Thread-%s" % (thread.name or str(thread_id),)
                        break
                else:
                    # should not happen.
                    thread_name = "of Thread-%s" % (str(thread_id),)
                thread = None

                tid = id(tasklet)
                tasklet = None
            else:
                name = 'Tasklet-%s' % (self._tasklet_id,)
                thread_name = ""
                tid = '-'
            self.tasklet_name = '%s %s (%s)' % (name, thread_name, tid)


_weak_tasklet_registered_to_info = {}


#=======================================================================================================================
# get_tasklet_info
#=======================================================================================================================
def get_tasklet_info(tasklet):
    return register_tasklet_info(tasklet)


#=======================================================================================================================
# register_tasklet_info
#=======================================================================================================================
def register_tasklet_info(tasklet):
    r = weakref.ref(tasklet)
    info = _weak_tasklet_registered_to_info.get(r)
    if info is None:
        info = _weak_tasklet_registered_to_info[r] = _TaskletInfo(r, tasklet)

    return info


_application_set_schedule_callback = None


#=======================================================================================================================
# _schedule_callback
#=======================================================================================================================
def _schedule_callback(prev, next):
    '''
    Called when a context is stopped or a new context is made runnable.
    '''
    try:
        if not prev and not next:
            return

        current_frame = sys._getframe()

        if next:
            register_tasklet_info(next)

            # Ok, making next runnable: set the tracing facility in it.
            debugger = get_global_debugger()
            if debugger is not None:
                next.trace_function = debugger.get_thread_local_trace_func()
                frame = next.frame
                if frame is current_frame:
                    frame = frame.f_back
                if hasattr(frame, 'f_trace'):  # Note: can be None (but hasattr should cover for that too).
                    frame.f_trace = debugger.get_thread_local_trace_func()

            debugger = None

        if prev:
            register_tasklet_info(prev)

        try:
            for tasklet_ref, tasklet_info in list(_weak_tasklet_registered_to_info.items()):  # Make sure it's a copy!
                tasklet = tasklet_ref()
                if tasklet is None or not tasklet.alive:
                    # Garbage-collected already!
                    try:
                        del _weak_tasklet_registered_to_info[tasklet_ref]
                    except KeyError:
                        pass
                    if tasklet_info.frame_id is not None:
                        remove_custom_frame(tasklet_info.frame_id)
                else:
                    is_running = stackless.get_thread_info(tasklet.thread_id)[1] is tasklet
                    if tasklet is prev or (tasklet is not next and not is_running):
                        # the tasklet won't run after this scheduler action:
                        # - the tasklet is the previous tasklet
                        # - it is not the next tasklet and it is not an already running tasklet
                        frame = tasklet.frame
                        if frame is current_frame:
                            frame = frame.f_back
                        if frame is not None:
                            # print >>sys.stderr, "SchedCB: %r, %d, '%s', '%s'" % (tasklet, frame.f_lineno, _filename, base)
                            debugger = get_global_debugger()
                            if debugger is not None and debugger.get_file_type(frame) is None:
                                tasklet_info.update_name()
                                if tasklet_info.frame_id is None:
                                    tasklet_info.frame_id = add_custom_frame(frame, tasklet_info.tasklet_name, tasklet.thread_id)
                                else:
                                    update_custom_frame(tasklet_info.frame_id, frame, tasklet.thread_id, name=tasklet_info.tasklet_name)
                            debugger = None

                    elif tasklet is next or is_running:
                        if tasklet_info.frame_id is not None:
                            # Remove info about stackless suspended when it starts to run.
                            remove_custom_frame(tasklet_info.frame_id)
                            tasklet_info.frame_id = None

        finally:
            tasklet = None
            tasklet_info = None
            frame = None

    except:
        pydev_log.exception()

    if _application_set_schedule_callback is not None:
        return _application_set_schedule_callback(prev, next)


if not hasattr(stackless.tasklet, "trace_function"):

    # Older versions of Stackless, released before 2014
    # This code does not work reliable! It is affected by several
    # stackless bugs: Stackless issues #44, #42, #40
    def _schedule_callback(prev, next):
        '''
        Called when a context is stopped or a new context is made runnable.
        '''
        try:
            if not prev and not next:
                return

            if next:
                register_tasklet_info(next)

                # Ok, making next runnable: set the tracing facility in it.
                debugger = get_global_debugger()
                if debugger is not None and next.frame:
                    if hasattr(next.frame, 'f_trace'):
                        next.frame.f_trace = debugger.get_thread_local_trace_func()
                debugger = None

            if prev:
                register_tasklet_info(prev)

            try:
                for tasklet_ref, tasklet_info in list(_weak_tasklet_registered_to_info.items()):  # Make sure it's a copy!
                    tasklet = tasklet_ref()
                    if tasklet is None or not tasklet.alive:
                        # Garbage-collected already!
                        try:
                            del _weak_tasklet_registered_to_info[tasklet_ref]
                        except KeyError:
                            pass
                        if tasklet_info.frame_id is not None:
                            remove_custom_frame(tasklet_info.frame_id)
                    else:
                        if tasklet.paused or tasklet.blocked or tasklet.scheduled:
                            if tasklet.frame and tasklet.frame.f_back:
                                f_back = tasklet.frame.f_back
                                debugger = get_global_debugger()
                                if debugger is not None and debugger.get_file_type(f_back) is None:
                                    if tasklet_info.frame_id is None:
                                        tasklet_info.frame_id = add_custom_frame(f_back, tasklet_info.tasklet_name, tasklet.thread_id)
                                    else:
                                        update_custom_frame(tasklet_info.frame_id, f_back, tasklet.thread_id)
                                debugger = None

                        elif tasklet.is_current:
                            if tasklet_info.frame_id is not None:
                                # Remove info about stackless suspended when it starts to run.
                                remove_custom_frame(tasklet_info.frame_id)
                                tasklet_info.frame_id = None

            finally:
                tasklet = None
                tasklet_info = None
                f_back = None

        except:
            pydev_log.exception()

        if _application_set_schedule_callback is not None:
            return _application_set_schedule_callback(prev, next)

    _original_setup = stackless.tasklet.setup

    #=======================================================================================================================
    # setup
    #=======================================================================================================================
    def setup(self, *args, **kwargs):
        '''
        Called to run a new tasklet: rebind the creation so that we can trace it.
        '''

        f = self.tempval

        def new_f(old_f, args, kwargs):

            debugger = get_global_debugger()
            if debugger is not None:
                debugger.enable_tracing()

            debugger = None

            # Remove our own traces :)
            self.tempval = old_f
            register_tasklet_info(self)

            # Hover old_f to see the stackless being created and *args and **kwargs to see its parameters.
            return old_f(*args, **kwargs)

        # This is the way to tell stackless that the function it should execute is our function, not the original one. Note:
        # setting tempval is the same as calling bind(new_f), but it seems that there's no other way to get the currently
        # bound function, so, keeping on using tempval instead of calling bind (which is actually the same thing in a better
        # API).

        self.tempval = new_f

        return _original_setup(self, f, args, kwargs)

    #=======================================================================================================================
    # __call__
    #=======================================================================================================================
    def __call__(self, *args, **kwargs):
        '''
        Called to run a new tasklet: rebind the creation so that we can trace it.
        '''

        return setup(self, *args, **kwargs)

    _original_run = stackless.run

    #=======================================================================================================================
    # run
    #=======================================================================================================================
    def run(*args, **kwargs):
        debugger = get_global_debugger()
        if debugger is not None:
            debugger.enable_tracing()
        debugger = None

        return _original_run(*args, **kwargs)


#=======================================================================================================================
# patch_stackless
#=======================================================================================================================
def patch_stackless():
    '''
    This function should be called to patch the stackless module so that new tasklets are properly tracked in the
    debugger.
    '''
    global _application_set_schedule_callback
    _application_set_schedule_callback = stackless.set_schedule_callback(_schedule_callback)

    def set_schedule_callback(callable):
        global _application_set_schedule_callback
        old = _application_set_schedule_callback
        _application_set_schedule_callback = callable
        return old

    def get_schedule_callback():
        global _application_set_schedule_callback
        return _application_set_schedule_callback

    set_schedule_callback.__doc__ = stackless.set_schedule_callback.__doc__
    if hasattr(stackless, "get_schedule_callback"):
        get_schedule_callback.__doc__ = stackless.get_schedule_callback.__doc__
    stackless.set_schedule_callback = set_schedule_callback
    stackless.get_schedule_callback = get_schedule_callback

    if not hasattr(stackless.tasklet, "trace_function"):
        # Older versions of Stackless, released before 2014
        __call__.__doc__ = stackless.tasklet.__call__.__doc__
        stackless.tasklet.__call__ = __call__

        setup.__doc__ = stackless.tasklet.setup.__doc__
        stackless.tasklet.setup = setup

        run.__doc__ = stackless.run.__doc__
        stackless.run = run


patch_stackless = call_only_once(patch_stackless)
