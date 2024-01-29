"""Sentinel class for constants with useful reprs"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import typing as t


class Sentinel:
    def __init__(self, name: str, module: t.Any, docstring: str | None = None) -> None:
        self.name = name
        self.module = module
        if docstring:
            self.__doc__ = docstring

    def __repr__(self) -> str:
        return str(self.module) + "." + self.name

    def __copy__(self) -> Sentinel:
        return self

    def __deepcopy__(self, memo: t.Any) -> Sentinel:
        return self
