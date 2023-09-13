"""A Jupyter console app to run files."""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import queue
import signal
import sys
import time

from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Any, Dict, Float
from traitlets.config import catch_config_error

from . import __version__
from .consoleapp import JupyterConsoleApp, app_aliases, app_flags

OUTPUT_TIMEOUT = 10

# copy flags from mixin:
flags = dict(base_flags)
# start with mixin frontend flags:
frontend_flags_dict = dict(app_flags)
# update full dict with frontend flags:
flags.update(frontend_flags_dict)

# copy flags from mixin
aliases = dict(base_aliases)
# start with mixin frontend flags
frontend_aliases_dict = dict(app_aliases)
# load updated frontend flags into full dict
aliases.update(frontend_aliases_dict)

# get flags&aliases into sets, and remove a couple that
# shouldn't be scrubbed from backend flags:
frontend_aliases = set(frontend_aliases_dict.keys())
frontend_flags = set(frontend_flags_dict.keys())


class RunApp(JupyterApp, JupyterConsoleApp):
    """An Jupyter Console app to run files."""

    version = __version__
    name = "jupyter run"
    description = """Run Jupyter kernel code."""
    flags = Dict(flags)  # type:ignore[assignment]
    aliases = Dict(aliases)  # type:ignore[assignment]
    frontend_aliases = Any(frontend_aliases)
    frontend_flags = Any(frontend_flags)
    kernel_timeout = Float(
        60,
        config=True,
        help="""Timeout for giving up on a kernel (in seconds).

        On first connect and restart, the console tests whether the
        kernel is running and responsive by sending kernel_info_requests.
        This sets the timeout in seconds for how long the kernel can take
        before being presumed dead.
        """,
    )

    def parse_command_line(self, argv=None):
        """Parse the command line arguments."""
        super().parse_command_line(argv)
        self.build_kernel_argv(self.extra_args)
        self.filenames_to_run = self.extra_args[:]

    @catch_config_error
    def initialize(self, argv=None):
        """Initialize the app."""
        self.log.debug("jupyter run: initialize...")
        super().initialize(argv)
        JupyterConsoleApp.initialize(self)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.init_kernel_info()

    def handle_sigint(self, *args):
        """Handle SIGINT."""
        if self.kernel_manager:
            self.kernel_manager.interrupt_kernel()
        else:
            self.log.error("Cannot interrupt kernels we didn't start.\n")

    def init_kernel_info(self):
        """Wait for a kernel to be ready, and store kernel info"""
        timeout = self.kernel_timeout
        tic = time.time()
        self.kernel_client.hb_channel.unpause()
        msg_id = self.kernel_client.kernel_info()
        while True:
            try:
                reply = self.kernel_client.get_shell_msg(timeout=1)
            except queue.Empty as e:
                if (time.time() - tic) > timeout:
                    msg = "Kernel didn't respond to kernel_info_request"
                    raise RuntimeError(msg) from e
            else:
                if reply["parent_header"].get("msg_id") == msg_id:
                    self.kernel_info = reply["content"]
                    return

    def start(self):
        """Start the application."""
        self.log.debug("jupyter run: starting...")
        super().start()
        if self.filenames_to_run:
            for filename in self.filenames_to_run:
                self.log.debug("jupyter run: executing `%s`", filename)
                with open(filename) as fp:
                    code = fp.read()
                    reply = self.kernel_client.execute_interactive(code, timeout=OUTPUT_TIMEOUT)
                    return_code = 0 if reply["content"]["status"] == "ok" else 1
                    if return_code:
                        raise Exception("jupyter-run error running '%s'" % filename)
        else:
            code = sys.stdin.read()
            reply = self.kernel_client.execute_interactive(code, timeout=OUTPUT_TIMEOUT)
            return_code = 0 if reply["content"]["status"] == "ok" else 1
            if return_code:
                msg = "jupyter-run error running 'stdin'"
                raise Exception(msg)


main = launch_new_instance = RunApp.launch_instance

if __name__ == "__main__":
    main()
