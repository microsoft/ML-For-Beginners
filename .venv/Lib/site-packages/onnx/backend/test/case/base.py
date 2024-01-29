# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import inspect
from collections import defaultdict
from textwrap import dedent
from typing import Any, ClassVar, Dict, List, Tuple, Type

import numpy as np


def process_snippet(op_name: str, name: str, export: Any) -> Tuple[str, str]:
    snippet_name = name[len("export_") :] or op_name.lower()
    source_code = dedent(inspect.getsource(export))
    # remove the function signature line
    lines = source_code.splitlines()
    assert lines[0] == "@staticmethod"
    assert lines[1].startswith("def export")
    return snippet_name, dedent("\n".join(lines[2:]))


Snippets: Dict[str, List[Tuple[str, str]]] = defaultdict(list)


class _Exporter(type):
    exports: ClassVar[Dict[str, List[Tuple[str, str]]]] = defaultdict(list)

    def __init__(
        cls, name: str, bases: Tuple[Type[Any], ...], dct: Dict[str, Any]
    ) -> None:
        for k, v in dct.items():
            if k.startswith("export"):
                if not isinstance(v, staticmethod):
                    raise ValueError("Only staticmethods could be named as export.*")
                export = getattr(cls, k)
                Snippets[name].append(process_snippet(name, k, export))
                # export functions should call expect and so populate
                # TestCases
                np.random.seed(seed=0)
                export()
        super().__init__(name, bases, dct)


class Base(metaclass=_Exporter):
    pass
