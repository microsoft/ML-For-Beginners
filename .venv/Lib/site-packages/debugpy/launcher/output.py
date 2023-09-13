# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import codecs
import os
import threading

from debugpy import launcher
from debugpy.common import log


class CaptureOutput(object):
    """Captures output from the specified file descriptor, and tees it into another
    file descriptor while generating DAP "output" events for it.
    """

    instances = {}
    """Keys are output categories, values are CaptureOutput instances."""

    def __init__(self, whose, category, fd, stream):
        assert category not in self.instances
        self.instances[category] = self
        log.info("Capturing {0} of {1}.", category, whose)

        self.category = category
        self._whose = whose
        self._fd = fd
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="surrogateescape")

        if stream is None:
            # Can happen if running under pythonw.exe.
            self._stream = None
        else:
            self._stream = stream.buffer
            encoding = stream.encoding
            if encoding is None or encoding == "cp65001":
                encoding = "utf-8"
            try:
                self._encode = codecs.getencoder(encoding)
            except Exception:
                log.swallow_exception(
                    "Unsupported {0} encoding {1!r}; falling back to UTF-8.",
                    category,
                    encoding,
                    level="warning",
                )
                self._encode = codecs.getencoder("utf-8")
            else:
                log.info("Using encoding {0!r} for {1}", encoding, category)

        self._worker_thread = threading.Thread(target=self._worker, name=category)
        self._worker_thread.start()

    def __del__(self):
        fd = self._fd
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass

    def _worker(self):
        while self._fd is not None:
            try:
                s = os.read(self._fd, 0x1000)
            except Exception:
                break
            if not len(s):
                break
            self._process_chunk(s)

        # Flush any remaining data in the incremental decoder.
        self._process_chunk(b"", final=True)

    def _process_chunk(self, s, final=False):
        s = self._decoder.decode(s, final=final)
        if len(s) == 0:
            return

        try:
            launcher.channel.send_event(
                "output", {"category": self.category, "output": s.replace("\r\n", "\n")}
            )
        except Exception:
            pass  # channel to adapter is already closed

        if self._stream is None:
            return

        try:
            s, _ = self._encode(s, "surrogateescape")
            size = len(s)
            i = 0
            while i < size:
                written = self._stream.write(s[i:])
                self._stream.flush()
                if written == 0:
                    # This means that the output stream was closed from the other end.
                    # Do the same to the debuggee, so that it knows as well.
                    os.close(self._fd)
                    self._fd = None
                    break
                i += written
        except Exception:
            log.swallow_exception("Error printing {0!r} to {1}", s, self.category)


def wait_for_remaining_output():
    """Waits for all remaining output to be captured and propagated."""
    for category, instance in CaptureOutput.instances.items():
        log.info("Waiting for remaining {0} of {1}.", category, instance._whose)
        instance._worker_thread.join()
