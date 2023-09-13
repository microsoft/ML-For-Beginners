# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

from __future__ import annotations
import os
import typing

if typing.TYPE_CHECKING:
    __all__: list[str]

__all__ = []

# The lower time bound for assuming that the process hasn't spawned successfully.
PROCESS_SPAWN_TIMEOUT = float(os.getenv("DEBUGPY_PROCESS_SPAWN_TIMEOUT", 15)) or None

# The lower time bound for assuming that the process hasn't exited gracefully.
PROCESS_EXIT_TIMEOUT = float(os.getenv("DEBUGPY_PROCESS_EXIT_TIMEOUT", 5)) or None
