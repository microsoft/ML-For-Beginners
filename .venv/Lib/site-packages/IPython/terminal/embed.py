# encoding: utf-8
"""
An embedded IPython shell.
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import sys
import warnings

from IPython.core import ultratb, compilerop
from IPython.core import magic_arguments
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.interactiveshell import DummyMod, InteractiveShell
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.ipapp import load_default_config

from traitlets import Bool, CBool, Unicode
from IPython.utils.io import ask_yes_no

from typing import Set

class KillEmbedded(Exception):pass

# kept for backward compatibility as IPython 6 was released with
# the typo. See https://github.com/ipython/ipython/pull/10706
KillEmbeded = KillEmbedded

# This is an additional magic that is exposed in embedded shells.
@magics_class
class EmbeddedMagics(Magics):

    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('-i', '--instance', action='store_true',
                              help='Kill instance instead of call location')
    @magic_arguments.argument('-x', '--exit', action='store_true',
                              help='Also exit the current session')
    @magic_arguments.argument('-y', '--yes', action='store_true',
                              help='Do not ask confirmation')
    def kill_embedded(self, parameter_s=''):
        """%kill_embedded : deactivate for good the current embedded IPython

        This function (after asking for confirmation) sets an internal flag so
        that an embedded IPython will never activate again for the given call
        location. This is useful to permanently disable a shell that is being
        called inside a loop: once you've figured out what you needed from it,
        you may then kill it and the program will then continue to run without
        the interactive shell interfering again.

        Kill Instance Option:

            If for some reasons you need to kill the location where the instance
            is created and not called, for example if you create a single
            instance in one place and debug in many locations, you can use the
            ``--instance`` option to kill this specific instance. Like for the
            ``call location`` killing an "instance" should work even if it is
            recreated within a loop.

        .. note::

            This was the default behavior before IPython 5.2

        """

        args = magic_arguments.parse_argstring(self.kill_embedded, parameter_s)
        print(args)
        if args.instance:
            # let no ask
            if not args.yes:
                kill = ask_yes_no(
                    "Are you sure you want to kill this embedded instance? [y/N] ", 'n')
            else:
                kill = True
            if kill:
                self.shell._disable_init_location()
                print("This embedded IPython instance will not reactivate anymore "
                      "once you exit.")
        else:
            if not args.yes:
                kill = ask_yes_no(
                    "Are you sure you want to kill this embedded call_location? [y/N] ", 'n')
            else:
                kill = True
            if kill:
                self.shell.embedded_active = False
                print("This embedded IPython  call location will not reactivate anymore "
                      "once you exit.")

        if args.exit:
            # Ask-exit does not really ask, it just set internals flags to exit
            # on next loop.
            self.shell.ask_exit()


    @line_magic
    def exit_raise(self, parameter_s=''):
        """%exit_raise Make the current embedded kernel exit and raise and exception.

        This function sets an internal flag so that an embedded IPython will
        raise a `IPython.terminal.embed.KillEmbedded` Exception on exit, and then exit the current I. This is
        useful to permanently exit a loop that create IPython embed instance.
        """

        self.shell.should_raise = True
        self.shell.ask_exit()


class _Sentinel:
    def __init__(self, repr):
        assert isinstance(repr, str)
        self.repr = repr

    def __repr__(self):
        return repr


class InteractiveShellEmbed(TerminalInteractiveShell):

    dummy_mode = Bool(False)
    exit_msg = Unicode('')
    embedded = CBool(True)
    should_raise = CBool(False)
    # Like the base class display_banner is not configurable, but here it
    # is True by default.
    display_banner = CBool(True)
    exit_msg = Unicode()

    # When embedding, by default we don't change the terminal title
    term_title = Bool(False,
        help="Automatically set the terminal title"
    ).tag(config=True)

    _inactive_locations: Set[str] = set()

    def _disable_init_location(self):
        """Disable the current Instance creation location"""
        InteractiveShellEmbed._inactive_locations.add(self._init_location_id)

    @property
    def embedded_active(self):
        return (self._call_location_id not in InteractiveShellEmbed._inactive_locations)\
            and (self._init_location_id not in InteractiveShellEmbed._inactive_locations)

    @embedded_active.setter
    def embedded_active(self, value):
        if value:
            InteractiveShellEmbed._inactive_locations.discard(
                self._call_location_id)
            InteractiveShellEmbed._inactive_locations.discard(
                self._init_location_id)
        else:
            InteractiveShellEmbed._inactive_locations.add(
                self._call_location_id)

    def __init__(self, **kw):
        assert (
            "user_global_ns" not in kw
        ), "Key word argument `user_global_ns` has been replaced by `user_module` since IPython 4.0."
        # temporary fix for https://github.com/ipython/ipython/issues/14164
        cls = type(self)
        if cls._instance is None:
            for subclass in cls._walk_mro():
                subclass._instance = self
            cls._instance = self

        clid = kw.pop('_init_location_id', None)
        if not clid:
            frame = sys._getframe(1)
            clid = '%s:%s' % (frame.f_code.co_filename, frame.f_lineno)
        self._init_location_id = clid

        super(InteractiveShellEmbed,self).__init__(**kw)

        # don't use the ipython crash handler so that user exceptions aren't
        # trapped
        sys.excepthook = ultratb.FormattedTB(color_scheme=self.colors,
                                             mode=self.xmode,
                                             call_pdb=self.pdb)

    def init_sys_modules(self):
        """
        Explicitly overwrite :mod:`IPython.core.interactiveshell` to do nothing.
        """
        pass

    def init_magics(self):
        super(InteractiveShellEmbed, self).init_magics()
        self.register_magics(EmbeddedMagics)

    def __call__(
        self,
        header="",
        local_ns=None,
        module=None,
        dummy=None,
        stack_depth=1,
        compile_flags=None,
        **kw
    ):
        """Activate the interactive interpreter.

        __call__(self,header='',local_ns=None,module=None,dummy=None) -> Start
        the interpreter shell with the given local and global namespaces, and
        optionally print a header string at startup.

        The shell can be globally activated/deactivated using the
        dummy_mode attribute. This allows you to turn off a shell used
        for debugging globally.

        However, *each* time you call the shell you can override the current
        state of dummy_mode with the optional keyword parameter 'dummy'. For
        example, if you set dummy mode on with IPShell.dummy_mode = True, you
        can still have a specific call work by making it as IPShell(dummy=False).
        """

        # we are called, set the underlying interactiveshell not to exit.
        self.keep_running = True

        # If the user has turned it off, go away
        clid = kw.pop('_call_location_id', None)
        if not clid:
            frame = sys._getframe(1)
            clid = '%s:%s' % (frame.f_code.co_filename, frame.f_lineno)
        self._call_location_id = clid

        if not self.embedded_active:
            return

        # Normal exits from interactive mode set this flag, so the shell can't
        # re-enter (it checks this variable at the start of interactive mode).
        self.exit_now = False

        # Allow the dummy parameter to override the global __dummy_mode
        if dummy or (dummy != 0 and self.dummy_mode):
            return

        # self.banner is auto computed
        if header:
            self.old_banner2 = self.banner2
            self.banner2 = self.banner2 + '\n' + header + '\n'
        else:
            self.old_banner2 = ''

        if self.display_banner:
            self.show_banner()

        # Call the embedding code with a stack depth of 1 so it can skip over
        # our call and get the original caller's namespaces.
        self.mainloop(
            local_ns, module, stack_depth=stack_depth, compile_flags=compile_flags
        )

        self.banner2 = self.old_banner2

        if self.exit_msg is not None:
            print(self.exit_msg)

        if self.should_raise:
            raise KillEmbedded('Embedded IPython raising error, as user requested.')

    def mainloop(
        self,
        local_ns=None,
        module=None,
        stack_depth=0,
        compile_flags=None,
    ):
        """Embeds IPython into a running python program.

        Parameters
        ----------
        local_ns, module
            Working local namespace (a dict) and module (a module or similar
            object). If given as None, they are automatically taken from the scope
            where the shell was called, so that program variables become visible.
        stack_depth : int
            How many levels in the stack to go to looking for namespaces (when
            local_ns or module is None). This allows an intermediate caller to
            make sure that this function gets the namespace from the intended
            level in the stack. By default (0) it will get its locals and globals
            from the immediate caller.
        compile_flags
            A bit field identifying the __future__ features
            that are enabled, as passed to the builtin :func:`compile` function.
            If given as None, they are automatically taken from the scope where
            the shell was called.

        """
        
        # Get locals and globals from caller
        if ((local_ns is None or module is None or compile_flags is None)
            and self.default_user_namespaces):
            call_frame = sys._getframe(stack_depth).f_back

            if local_ns is None:
                local_ns = call_frame.f_locals
            if module is None:
                global_ns = call_frame.f_globals
                try:
                    module = sys.modules[global_ns['__name__']]
                except KeyError:
                    warnings.warn("Failed to get module %s" % \
                        global_ns.get('__name__', 'unknown module')
                    )
                    module = DummyMod()
                    module.__dict__ = global_ns
            if compile_flags is None:
                compile_flags = (call_frame.f_code.co_flags &
                                 compilerop.PyCF_MASK)
        
        # Save original namespace and module so we can restore them after 
        # embedding; otherwise the shell doesn't shut down correctly.
        orig_user_module = self.user_module
        orig_user_ns = self.user_ns
        orig_compile_flags = self.compile.flags
        
        # Update namespaces and fire up interpreter
        
        # The global one is easy, we can just throw it in
        if module is not None:
            self.user_module = module

        # But the user/local one is tricky: ipython needs it to store internal
        # data, but we also need the locals. We'll throw our hidden variables
        # like _ih and get_ipython() into the local namespace, but delete them
        # later.
        if local_ns is not None:
            reentrant_local_ns = {k: v for (k, v) in local_ns.items() if k not in self.user_ns_hidden.keys()}
            self.user_ns = reentrant_local_ns
            self.init_user_ns()

        # Compiler flags
        if compile_flags is not None:
            self.compile.flags = compile_flags

        # make sure the tab-completer has the correct frame information, so it
        # actually completes using the frame's locals/globals
        self.set_completer_frame()

        with self.builtin_trap, self.display_trap:
            self.interact()
        
        # now, purge out the local namespace of IPython's hidden variables.
        if local_ns is not None:
            local_ns.update({k: v for (k, v) in self.user_ns.items() if k not in self.user_ns_hidden.keys()})

        
        # Restore original namespace so shell can shut down when we exit.
        self.user_module = orig_user_module
        self.user_ns = orig_user_ns
        self.compile.flags = orig_compile_flags


def embed(*, header="", compile_flags=None, **kwargs):
    """Call this to embed IPython at the current point in your program.

    The first invocation of this will create a :class:`terminal.embed.InteractiveShellEmbed`
    instance and then call it.  Consecutive calls just call the already
    created instance.

    If you don't want the kernel to initialize the namespace
    from the scope of the surrounding function,
    and/or you want to load full IPython configuration,
    you probably want `IPython.start_ipython()` instead.

    Here is a simple example::

        from IPython import embed
        a = 10
        b = 20
        embed(header='First time')
        c = 30
        d = 40
        embed()

    Parameters
    ----------

    header : str
        Optional header string to print at startup.
    compile_flags
        Passed to the `compile_flags` parameter of :py:meth:`terminal.embed.InteractiveShellEmbed.mainloop()`,
        which is called when the :class:`terminal.embed.InteractiveShellEmbed` instance is called.
    **kwargs : various, optional
        Any other kwargs will be passed to the :class:`terminal.embed.InteractiveShellEmbed` constructor.
        Full customization can be done by passing a traitlets :class:`Config` in as the
        `config` argument (see :ref:`configure_start_ipython` and :ref:`terminal_options`).
    """
    config = kwargs.get('config')
    if config is None:
        config = load_default_config()
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        kwargs['config'] = config
    using = kwargs.get('using', 'sync')
    if using :
        kwargs['config'].update({'TerminalInteractiveShell':{'loop_runner':using, 'colors':'NoColor', 'autoawait': using!='sync'}})
    #save ps1/ps2 if defined
    ps1 = None
    ps2 = None
    try:
        ps1 = sys.ps1
        ps2 = sys.ps2
    except AttributeError:
        pass
    #save previous instance
    saved_shell_instance = InteractiveShell._instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
    frame = sys._getframe(1)
    shell = InteractiveShellEmbed.instance(_init_location_id='%s:%s' % (
        frame.f_code.co_filename, frame.f_lineno), **kwargs)
    shell(header=header, stack_depth=2, compile_flags=compile_flags,
        _call_location_id='%s:%s' % (frame.f_code.co_filename, frame.f_lineno))
    InteractiveShellEmbed.clear_instance()
    #restore previous instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
        for subclass in cls._walk_mro():
            subclass._instance = saved_shell_instance
    if ps1 is not None:
        sys.ps1 = ps1
        sys.ps2 = ps2
