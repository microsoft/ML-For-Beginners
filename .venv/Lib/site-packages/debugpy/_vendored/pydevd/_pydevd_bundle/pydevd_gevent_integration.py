import pydevd_tracing
import greenlet
import gevent
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import add_custom_frame, update_custom_frame, remove_custom_frame
from _pydevd_bundle.pydevd_constants import GEVENT_SHOW_PAUSED_GREENLETS, get_global_debugger, \
    thread_get_ident
from _pydev_bundle import pydev_log
from pydevd_file_utils import basename

_saved_greenlets_to_custom_frame_thread_id = {}

if GEVENT_SHOW_PAUSED_GREENLETS:

    def _get_paused_name(py_db, g):
        frame = g.gr_frame
        use_frame = frame

        # i.e.: Show in the description of the greenlet the last user-code found.
        while use_frame is not None:
            if py_db.apply_files_filter(use_frame, use_frame.f_code.co_filename, True):
                frame = use_frame
                use_frame = use_frame.f_back
            else:
                break

        if use_frame is None:
            use_frame = frame

        return '%s: %s - %s' % (type(g).__name__, use_frame.f_code.co_name, basename(use_frame.f_code.co_filename))

    def greenlet_events(event, args):
        if event in ('switch', 'throw'):
            py_db = get_global_debugger()
            origin, target = args

            if not origin.dead and origin.gr_frame is not None:
                frame_custom_thread_id = _saved_greenlets_to_custom_frame_thread_id.get(origin)
                if frame_custom_thread_id is None:
                    _saved_greenlets_to_custom_frame_thread_id[origin] = add_custom_frame(
                        origin.gr_frame, _get_paused_name(py_db, origin), thread_get_ident())
                else:
                    update_custom_frame(
                        frame_custom_thread_id, origin.gr_frame, _get_paused_name(py_db, origin), thread_get_ident())
            else:
                frame_custom_thread_id = _saved_greenlets_to_custom_frame_thread_id.pop(origin, None)
                if frame_custom_thread_id is not None:
                    remove_custom_frame(frame_custom_thread_id)

            # This one will be resumed, so, remove custom frame from it.
            frame_custom_thread_id = _saved_greenlets_to_custom_frame_thread_id.pop(target, None)
            if frame_custom_thread_id is not None:
                remove_custom_frame(frame_custom_thread_id)

        # The tracing needs to be reapplied for each greenlet as gevent
        # clears the tracing set through sys.settrace for each greenlet.
        pydevd_tracing.reapply_settrace()

else:

    # i.e.: no logic related to showing paused greenlets is needed.
    def greenlet_events(event, args):
        pydevd_tracing.reapply_settrace()


def enable_gevent_integration():
    # References:
    # https://greenlet.readthedocs.io/en/latest/api.html#greenlet.settrace
    # https://greenlet.readthedocs.io/en/latest/tracing.html

    # Note: gevent.version_info is WRONG (gevent.__version__ must be used).
    try:
        if tuple(int(x) for x in gevent.__version__.split('.')[:2]) <= (20, 0):
            if not GEVENT_SHOW_PAUSED_GREENLETS:
                return

            if not hasattr(greenlet, 'settrace'):
                # In older versions it was optional.
                # We still try to use if available though.
                pydev_log.debug('greenlet.settrace not available. GEVENT_SHOW_PAUSED_GREENLETS will have no effect.')
                return
        try:
            greenlet.settrace(greenlet_events)
        except:
            pydev_log.exception('Error with greenlet.settrace.')
    except:
        pydev_log.exception('Error setting up gevent %s.', gevent.__version__)


def log_gevent_debug_info():
    pydev_log.debug('Greenlet version: %s', greenlet.__version__)
    pydev_log.debug('Gevent version: %s', gevent.__version__)
    pydev_log.debug('Gevent install location: %s', gevent.__file__)
