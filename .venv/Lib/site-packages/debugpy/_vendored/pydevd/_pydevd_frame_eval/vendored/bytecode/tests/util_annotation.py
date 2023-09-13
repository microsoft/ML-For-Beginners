from __future__ import annotations

import textwrap
import types


def get_code(source, *, filename="<string>", function=False):
    source = textwrap.dedent(source).strip()
    code = compile(source, filename, "exec")
    if function:
        sub_code = [
            const for const in code.co_consts if isinstance(const, types.CodeType)
        ]
        if len(sub_code) != 1:
            raise ValueError("unable to find function code")
        code = sub_code[0]
    return code
