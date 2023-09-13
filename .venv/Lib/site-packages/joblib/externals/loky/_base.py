###############################################################################
# Modification of concurrent.futures.Future
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from concurrent/futures/_base.py (17/02/2017)
#  * Do not use yield from
#  * Use old super syntax
#
# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

from concurrent.futures import Future as _BaseFuture
from concurrent.futures._base import LOGGER


# To make loky._base.Future instances awaitable  by concurrent.futures.wait,
# derive our custom Future class from _BaseFuture. _invoke_callback is the only
# modification made to this class in loky.
# TODO investigate why using `concurrent.futures.Future` directly does not
# always work in our test suite.
class Future(_BaseFuture):
    def _invoke_callbacks(self):
        for callback in self._done_callbacks:
            try:
                callback(self)
            except BaseException:
                LOGGER.exception(f"exception calling callback for {self!r}")
