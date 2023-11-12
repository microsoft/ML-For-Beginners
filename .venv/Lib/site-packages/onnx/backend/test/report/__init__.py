# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

import _pytest
import pytest

from onnx.backend.test.report.coverage import Coverage

_coverage = Coverage()
_marks: Dict[str, Sequence[Any]] = {}


def _add_mark(mark: Any, bucket: str) -> None:
    proto = mark.args[0]
    if isinstance(proto, list):
        assert len(proto) == 1
        proto = proto[0]
    if proto is not None:
        _coverage.add_proto(proto, bucket, mark.args[1] == "RealModel")


def pytest_runtest_call(item: _pytest.nodes.Item) -> None:
    mark = item.get_closest_marker("onnx_coverage")
    if mark:
        assert item.nodeid not in _marks
        _marks[item.nodeid] = mark


def pytest_runtest_logreport(report: Any) -> None:
    if report.when == "call" and report.outcome == "passed" and report.nodeid in _marks:
        mark = _marks[report.nodeid]
        _add_mark(mark, "passed")


@pytest.hookimpl(trylast=True)  # type: ignore
def pytest_terminal_summary(
    terminalreporter: _pytest.terminal.TerminalReporter, exitstatus: int
) -> None:
    for mark in _marks.values():
        _add_mark(mark, "loaded")
    _coverage.report_text(terminalreporter)
