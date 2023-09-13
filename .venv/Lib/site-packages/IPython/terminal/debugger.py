import asyncio
import os
import sys

from IPython.core.debugger import Pdb
from IPython.core.completer import IPCompleter
from .ptutils import IPythonPTCompleter
from .shortcuts import create_ipython_shortcuts
from . import embed

from pathlib import Path
from pygments.token import Token
from prompt_toolkit.application import create_app_session
from prompt_toolkit.shortcuts.prompt import PromptSession
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.formatted_text import PygmentsTokens
from prompt_toolkit.history import InMemoryHistory, FileHistory
from concurrent.futures import ThreadPoolExecutor

from prompt_toolkit import __version__ as ptk_version
PTK3 = ptk_version.startswith('3.')


# we want to avoid ptk as much as possible when using subprocesses
# as it uses cursor positioning requests, deletes color ....
_use_simple_prompt = "IPY_TEST_SIMPLE_PROMPT" in os.environ


class TerminalPdb(Pdb):
    """Standalone IPython debugger."""

    def __init__(self, *args, pt_session_options=None, **kwargs):
        Pdb.__init__(self, *args, **kwargs)
        self._ptcomp = None
        self.pt_init(pt_session_options)
        self.thread_executor = ThreadPoolExecutor(1)

    def pt_init(self, pt_session_options=None):
        """Initialize the prompt session and the prompt loop
        and store them in self.pt_app and self.pt_loop.

        Additional keyword arguments for the PromptSession class
        can be specified in pt_session_options.
        """
        if pt_session_options is None:
            pt_session_options = {}

        def get_prompt_tokens():
            return [(Token.Prompt, self.prompt)]

        if self._ptcomp is None:
            compl = IPCompleter(
                shell=self.shell, namespace={}, global_namespace={}, parent=self.shell
            )
            # add a completer for all the do_ methods
            methods_names = [m[3:] for m in dir(self) if m.startswith("do_")]

            def gen_comp(self, text):
                return [m for m in methods_names if m.startswith(text)]
            import types
            newcomp = types.MethodType(gen_comp, compl)
            compl.custom_matchers.insert(0, newcomp)
            # end add completer.

            self._ptcomp = IPythonPTCompleter(compl)

        # setup history only when we start pdb
        if self.shell.debugger_history is None:
            if self.shell.debugger_history_file is not None:
                p = Path(self.shell.debugger_history_file).expanduser()
                if not p.exists():
                    p.touch()
                self.debugger_history = FileHistory(os.path.expanduser(str(p)))
            else:
                self.debugger_history = InMemoryHistory()
        else:
            self.debugger_history = self.shell.debugger_history

        options = dict(
            message=(lambda: PygmentsTokens(get_prompt_tokens())),
            editing_mode=getattr(EditingMode, self.shell.editing_mode.upper()),
            key_bindings=create_ipython_shortcuts(self.shell),
            history=self.debugger_history,
            completer=self._ptcomp,
            enable_history_search=True,
            mouse_support=self.shell.mouse_support,
            complete_style=self.shell.pt_complete_style,
            style=getattr(self.shell, "style", None),
            color_depth=self.shell.color_depth,
        )

        if not PTK3:
            options['inputhook'] = self.shell.inputhook
        options.update(pt_session_options)
        if not _use_simple_prompt:
            self.pt_loop = asyncio.new_event_loop()
            self.pt_app = PromptSession(**options)

    def _prompt(self):
        """
        In case other prompt_toolkit apps have to run in parallel to this one (e.g. in madbg),
        create_app_session must be used to prevent mixing up between them. According to the prompt_toolkit docs:

        > If you need multiple applications running at the same time, you have to create a separate
        > `AppSession` using a `with create_app_session():` block.
        """
        with create_app_session():
            return self.pt_app.prompt()

    def cmdloop(self, intro=None):
        """Repeatedly issue a prompt, accept input, parse an initial prefix
        off the received input, and dispatch to action methods, passing them
        the remainder of the line as argument.

        override the same methods from cmd.Cmd to provide prompt toolkit replacement.
        """
        if not self.use_rawinput:
            raise ValueError('Sorry ipdb does not support use_rawinput=False')

        # In order to make sure that prompt, which uses asyncio doesn't
        # interfere with applications in which it's used, we always run the
        # prompt itself in a different thread (we can't start an event loop
        # within an event loop). This new thread won't have any event loop
        # running, and here we run our prompt-loop.
        self.preloop()

        try:
            if intro is not None:
                self.intro = intro
            if self.intro:
                print(self.intro, file=self.stdout)
            stop = None
            while not stop:
                if self.cmdqueue:
                    line = self.cmdqueue.pop(0)
                else:
                    self._ptcomp.ipy_completer.namespace = self.curframe_locals
                    self._ptcomp.ipy_completer.global_namespace = self.curframe.f_globals

                    # Run the prompt in a different thread.
                    if not _use_simple_prompt:
                        try:
                            line = self.thread_executor.submit(self._prompt).result()
                        except EOFError:
                            line = "EOF"
                    else:
                        line = input("ipdb> ")

                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
            self.postloop()
        except Exception:
            raise

    def do_interact(self, arg):
        ipshell = embed.InteractiveShellEmbed(
            config=self.shell.config,
            banner1="*interactive*",
            exit_msg="*exiting interactive console...*",
        )
        global_ns = self.curframe.f_globals
        ipshell(
            module=sys.modules.get(global_ns["__name__"], None),
            local_ns=self.curframe_locals,
        )


def set_trace(frame=None):
    """
    Start debugging from `frame`.

    If frame is not specified, debugging starts from caller's frame.
    """
    TerminalPdb().set_trace(frame or sys._getframe().f_back)


if __name__ == '__main__':
    import pdb
    # IPython.core.debugger.Pdb.trace_dispatch shall not catch
    # bdb.BdbQuit. When started through __main__ and an exception
    # happened after hitting "c", this is needed in order to
    # be able to quit the debugging session (see #9950).
    old_trace_dispatch = pdb.Pdb.trace_dispatch
    pdb.Pdb = TerminalPdb  # type: ignore
    pdb.Pdb.trace_dispatch = old_trace_dispatch  # type: ignore
    pdb.main()
