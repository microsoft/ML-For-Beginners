# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

"""Provides monotonic timestamps with a resetable zero.
"""

import time

__all__ = ["current", "reset"]


def current():
    return time.monotonic() - timestamp_zero


def reset():
    global timestamp_zero
    timestamp_zero = time.monotonic()


reset()
