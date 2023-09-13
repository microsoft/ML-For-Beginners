
import sys
from _pydev_bundle import pydev_log

backends = {'tk': 'TkAgg',
            'gtk': 'GTKAgg',
            'wx': 'WXAgg',
            'qt': 'QtAgg', # Auto-choose qt4/5
            'qt4': 'Qt4Agg',
            'qt5': 'Qt5Agg',
            'osx': 'MacOSX'}

# We also need a reverse backends2guis mapping that will properly choose which
# GUI support to activate based on the desired matplotlib backend.  For the
# most part it's just a reverse of the above dict, but we also need to add a
# few others that map to the same GUI manually:
backend2gui = dict(zip(backends.values(), backends.keys()))
# In the reverse mapping, there are a few extra valid matplotlib backends that
# map to the same GUI support
backend2gui['GTK'] = backend2gui['GTKCairo'] = 'gtk'
backend2gui['WX'] = 'wx'
backend2gui['CocoaAgg'] = 'osx'


def do_enable_gui(guiname):
    from _pydev_bundle.pydev_versioncheck import versionok_for_gui
    if versionok_for_gui():
        try:
            from pydev_ipython.inputhook import enable_gui
            enable_gui(guiname)
        except:
            sys.stderr.write("Failed to enable GUI event loop integration for '%s'\n" % guiname)
            pydev_log.exception()
    elif guiname not in ['none', '', None]:
        # Only print a warning if the guiname was going to do something
        sys.stderr.write("Debug console: Python version does not support GUI event loop integration for '%s'\n" % guiname)
    # Return value does not matter, so return back what was sent
    return guiname


def find_gui_and_backend():
    """Return the gui and mpl backend."""
    matplotlib = sys.modules['matplotlib']
    # WARNING: this assumes matplotlib 1.1 or newer!!
    backend = matplotlib.rcParams['backend']
    # In this case, we need to find what the appropriate gui selection call
    # should be for IPython, so we can activate inputhook accordingly
    gui = backend2gui.get(backend, None)
    return gui, backend


def is_interactive_backend(backend):
    """ Check if backend is interactive """
    matplotlib = sys.modules['matplotlib']
    from matplotlib.rcsetup import interactive_bk, non_interactive_bk  # @UnresolvedImport
    if backend in interactive_bk:
        return True
    elif backend in non_interactive_bk:
        return False
    else:
        return matplotlib.is_interactive()


def patch_use(enable_gui_function):
    """ Patch matplotlib function 'use' """
    matplotlib = sys.modules['matplotlib']

    def patched_use(*args, **kwargs):
        matplotlib.real_use(*args, **kwargs)
        gui, backend = find_gui_and_backend()
        enable_gui_function(gui)

    matplotlib.real_use = matplotlib.use
    matplotlib.use = patched_use


def patch_is_interactive():
    """ Patch matplotlib function 'use' """
    matplotlib = sys.modules['matplotlib']

    def patched_is_interactive():
        return matplotlib.rcParams['interactive']

    matplotlib.real_is_interactive = matplotlib.is_interactive
    matplotlib.is_interactive = patched_is_interactive


def activate_matplotlib(enable_gui_function):
    """Set interactive to True for interactive backends.
    enable_gui_function - Function which enables gui, should be run in the main thread.
    """
    matplotlib = sys.modules['matplotlib']
    gui, backend = find_gui_and_backend()
    is_interactive = is_interactive_backend(backend)
    if is_interactive:
        enable_gui_function(gui)
        if not matplotlib.is_interactive():
            sys.stdout.write("Backend %s is interactive backend. Turning interactive mode on.\n" % backend)
        matplotlib.interactive(True)
    else:
        if matplotlib.is_interactive():
            sys.stdout.write("Backend %s is non-interactive backend. Turning interactive mode off.\n" % backend)
        matplotlib.interactive(False)
    patch_use(enable_gui_function)
    patch_is_interactive()


def flag_calls(func):
    """Wrap a function to detect and flag when it gets called.

    This is a decorator which takes a function and wraps it in a function with
    a 'called' attribute. wrapper.called is initialized to False.

    The wrapper.called attribute is set to False right before each call to the
    wrapped function, so if the call fails it remains False.  After the call
    completes, wrapper.called is set to True and the output is returned.

    Testing for truth in wrapper.called allows you to determine if a call to
    func() was attempted and succeeded."""

    # don't wrap twice
    if hasattr(func, 'called'):
        return func

    def wrapper(*args, **kw):
        wrapper.called = False
        out = func(*args, **kw)
        wrapper.called = True
        return out

    wrapper.called = False
    wrapper.__doc__ = func.__doc__
    return wrapper


def activate_pylab():
    pylab = sys.modules['pylab']
    pylab.show._needmain = False
    # We need to detect at runtime whether show() is called by the user.
    # For this, we wrap it into a decorator which adds a 'called' flag.
    pylab.draw_if_interactive = flag_calls(pylab.draw_if_interactive)


def activate_pyplot():
    pyplot = sys.modules['matplotlib.pyplot']
    pyplot.show._needmain = False
    # We need to detect at runtime whether show() is called by the user.
    # For this, we wrap it into a decorator which adds a 'called' flag.
    pyplot.draw_if_interactive = flag_calls(pyplot.draw_if_interactive)
