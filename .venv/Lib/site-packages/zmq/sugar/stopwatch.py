"""Deprecated Stopwatch implementation"""

# Copyright (c) PyZMQ Development Team.
# Distributed under the terms of the Modified BSD License.


class Stopwatch:
    """Deprecated zmq.Stopwatch implementation

    You can use Python's builtin timers (time.monotonic, etc.).
    """

    def __init__(self):
        import warnings

        warnings.warn(
            "zmq.Stopwatch is deprecated. Use stdlib time.monotonic and friends instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._start = 0
        import time

        try:
            self._monotonic = time.monotonic
        except AttributeError:
            self._monotonic = time.time

    def start(self):
        """Start the counter"""
        self._start = self._monotonic()

    def stop(self):
        """Return time since start in microseconds"""
        stop = self._monotonic()
        return int(1e6 * (stop - self._start))
