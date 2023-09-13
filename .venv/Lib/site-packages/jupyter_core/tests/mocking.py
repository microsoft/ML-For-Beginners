"""General mocking utilities"""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import sys
from unittest.mock import patch


class MultiPatch:
    def __init__(self, *patchers):
        self.patchers = patchers

    def __enter__(self):
        for p in self.patchers:
            p.start()

    def __exit__(self, *args):
        for p in self.patchers:
            p.stop()


darwin = MultiPatch(
    patch.object(os, "name", "posix"),
    patch.object(sys, "platform", "darwin"),
)

linux = MultiPatch(
    patch.object(os, "name", "posix"),
    patch.object(sys, "platform", "linux2"),
)
