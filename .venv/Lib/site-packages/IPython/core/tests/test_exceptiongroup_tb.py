import unittest
import re
from IPython.utils.capture import capture_output
import sys
import pytest
from tempfile import TemporaryDirectory
from IPython.testing import tools as tt


def _exceptiongroup_common(
    outer_chain: str,
    inner_chain: str,
    native: bool,
) -> None:
    pre_raise = "exceptiongroup." if not native else ""
    pre_catch = pre_raise if sys.version_info < (3, 11) else ""
    filestr = f"""
    {"import exceptiongroup" if not native else ""}
    import pytest

    def f(): raise ValueError("From f()")
    def g(): raise BaseException("From g()")

    def inner(inner_chain):
        excs = []
        for callback in [f, g]:
            try:
                callback()
            except BaseException as err:
                excs.append(err)
        if excs:
            if inner_chain == "none":
                raise {pre_raise}BaseExceptionGroup("Oops", excs)
            try:
                raise SyntaxError()
            except SyntaxError as e:
                if inner_chain == "from":
                    raise {pre_raise}BaseExceptionGroup("Oops", excs) from e
                else:
                    raise {pre_raise}BaseExceptionGroup("Oops", excs)

    def outer(outer_chain, inner_chain):
        try:
            inner(inner_chain)
        except {pre_catch}BaseExceptionGroup as e:
            if outer_chain == "none":
                raise
            if outer_chain == "from":
                raise IndexError() from e
            else:
                raise IndexError


    outer("{outer_chain}", "{inner_chain}")
    """
    with capture_output() as cap:
        ip.run_cell(filestr)

    match_lines = []
    if inner_chain == "another":
        match_lines += [
            "During handling of the above exception, another exception occurred:",
        ]
    elif inner_chain == "from":
        match_lines += [
            "The above exception was the direct cause of the following exception:",
        ]

    match_lines += [
        "  + Exception Group Traceback (most recent call last):",
        f"  | {pre_catch}BaseExceptionGroup: Oops (2 sub-exceptions)",
        "    | ValueError: From f()",
        "    | BaseException: From g()",
    ]

    if outer_chain == "another":
        match_lines += [
            "During handling of the above exception, another exception occurred:",
            "IndexError",
        ]
    elif outer_chain == "from":
        match_lines += [
            "The above exception was the direct cause of the following exception:",
            "IndexError",
        ]

    error_lines = cap.stderr.split("\n")

    err_index = match_index = 0
    for expected in match_lines:
        for i, actual in enumerate(error_lines):
            if actual == expected:
                error_lines = error_lines[i + 1 :]
                break
        else:
            assert False, f"{expected} not found in cap.stderr"


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Native ExceptionGroup not implemented"
)
@pytest.mark.parametrize("outer_chain", ["none", "from", "another"])
@pytest.mark.parametrize("inner_chain", ["none", "from", "another"])
def test_native_exceptiongroup(outer_chain, inner_chain) -> None:
    _exceptiongroup_common(outer_chain, inner_chain, native=True)


@pytest.mark.parametrize("outer_chain", ["none", "from", "another"])
@pytest.mark.parametrize("inner_chain", ["none", "from", "another"])
def test_native_exceptiongroup(outer_chain, inner_chain) -> None:
    pytest.importorskip("exceptiongroup")
    _exceptiongroup_common(outer_chain, inner_chain, native=False)
