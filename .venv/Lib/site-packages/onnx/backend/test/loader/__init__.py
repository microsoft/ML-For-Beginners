# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import List, Optional

from onnx.backend.test.case.test_case import TestCase

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(os.path.dirname(__file__))), "data"
)


def load_model_tests(
    data_dir: str = DATA_DIR,
    kind: Optional[str] = None,
) -> List[TestCase]:
    """Load model test cases from on-disk data files."""

    supported_kinds = os.listdir(data_dir)
    if kind not in supported_kinds:
        raise ValueError(f"kind must be one of {supported_kinds}")

    testcases = []

    kind_dir = os.path.join(data_dir, kind)
    for test_name in os.listdir(kind_dir):
        case_dir = os.path.join(kind_dir, test_name)
        # skip the non-dir files, such as generated __init__.py.
        rtol = 1e-3
        atol = 1e-7
        if not os.path.isdir(case_dir):
            continue
        if os.path.exists(os.path.join(case_dir, "model.onnx")):
            url = None
            model_name = test_name[len("test_")]
            model_dir: Optional[str] = case_dir
        else:
            with open(os.path.join(case_dir, "data.json")) as f:
                data = json.load(f)
                url = data["url"]
                model_name = data["model_name"]
                rtol = data.get("rtol", 1e-3)
                atol = data.get("atol", 1e-7)
                model_dir = None
        testcases.append(
            TestCase(
                name=test_name,
                url=url,
                model_name=model_name,
                model_dir=model_dir,
                model=None,
                data_sets=None,
                kind=kind,
                rtol=rtol,
                atol=atol,
            )
        )

    return testcases
