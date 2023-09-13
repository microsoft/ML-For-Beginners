# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

"""Provides facilities to dump all stacks of all threads in the process.
"""

import os
import sys
import time
import threading
import traceback

from debugpy.common import log


def dump():
    """Dump stacks of all threads in this process, except for the current thread."""

    tid = threading.current_thread().ident
    pid = os.getpid()

    log.info("Dumping stacks for process {0}...", pid)

    for t_ident, frame in sys._current_frames().items():
        if t_ident == tid:
            continue

        for t in threading.enumerate():
            if t.ident == tid:
                t_name = t.name
                t_daemon = t.daemon
                break
        else:
            t_name = t_daemon = "<unknown>"

        stack = "".join(traceback.format_stack(frame))
        log.info(
            "Stack of thread {0} (tid={1}, pid={2}, daemon={3}):\n\n{4}",
            t_name,
            t_ident,
            pid,
            t_daemon,
            stack,
        )

    log.info("Finished dumping stacks for process {0}.", pid)


def dump_after(secs):
    """Invokes dump() on a background thread after waiting for the specified time."""

    def dumper():
        time.sleep(secs)
        try:
            dump()
        except:
            log.swallow_exception()

    thread = threading.Thread(target=dumper)
    thread.daemon = True
    thread.start()
