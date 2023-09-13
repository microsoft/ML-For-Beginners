# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    __all__: list[str]

__all__ = []

access_token = None
"""Access token used to authenticate with this adapter."""
