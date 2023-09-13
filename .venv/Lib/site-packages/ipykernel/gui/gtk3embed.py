"""GUI support for the IPython ZeroMQ kernel - GTK toolkit support.
"""
# -----------------------------------------------------------------------------
#  Copyright (C) 2010-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file LICENSE, distributed as part of this software.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
# stdlib
import sys
import warnings

warnings.warn(
    "The Gtk3 event loop for ipykernel is deprecated", category=DeprecationWarning, stacklevel=2
)

# Third-party
import gi

gi.require_version("Gdk", "3.0")
gi.require_version("Gtk", "3.0")
from gi.repository import GObject, Gtk

# -----------------------------------------------------------------------------
# Classes and functions
# -----------------------------------------------------------------------------


class GTKEmbed:
    """A class to embed a kernel into the GTK main event loop."""

    def __init__(self, kernel):
        """Initialize the embed."""
        self.kernel = kernel
        # These two will later store the real gtk functions when we hijack them
        self.gtk_main = None
        self.gtk_main_quit = None

    def start(self):
        """Starts the GTK main event loop and sets our kernel startup routine."""
        # Register our function to initiate the kernel and start gtk
        GObject.idle_add(self._wire_kernel)
        Gtk.main()

    def _wire_kernel(self):
        """Initializes the kernel inside GTK.

        This is meant to run only once at startup, so it does its job and
        returns False to ensure it doesn't get run again by GTK.
        """
        self.gtk_main, self.gtk_main_quit = self._hijack_gtk()
        GObject.timeout_add(int(1000 * self.kernel._poll_interval), self.iterate_kernel)
        return False

    def iterate_kernel(self):
        """Run one iteration of the kernel and return True.

        GTK timer functions must return True to be called again, so we make the
        call to :meth:`do_one_iteration` and then return True for GTK.
        """
        self.kernel.do_one_iteration()
        return True

    def stop(self):
        """Stop the embed."""
        # FIXME: this one isn't getting called because we have no reliable
        # kernel shutdown.  We need to fix that: once the kernel has a
        # shutdown mechanism, it can call this.
        if self.gtk_main_quit:
            self.gtk_main_quit()
        sys.exit()

    def _hijack_gtk(self):
        """Hijack a few key functions in GTK for IPython integration.

        Modifies pyGTK's main and main_quit with a dummy so user code does not
        block IPython.  This allows us to use %run to run arbitrary pygtk
        scripts from a long-lived IPython session, and when they attempt to
        start or stop

        Returns
        -------
        The original functions that have been hijacked:
        - Gtk.main
        - Gtk.main_quit
        """

        def dummy(*args, **kw):
            """No-op."""
            pass

        # save and trap main and main_quit from gtk
        orig_main, Gtk.main = Gtk.main, dummy
        orig_main_quit, Gtk.main_quit = Gtk.main_quit, dummy
        return orig_main, orig_main_quit
