# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import sys
from typing import List, Optional, Sequence

import numpy as np

from onnx import ModelProto
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.case.utils import import_recursive

_SimpleModelTestCases = []


def expect(
    model: ModelProto,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    name: Optional[str] = None,
) -> None:
    name = name or model.graph.name
    _SimpleModelTestCases.append(
        TestCase(
            name=name,
            model_name=model.graph.name,
            url=None,
            model_dir=None,
            model=model,
            data_sets=[(inputs, outputs)],
            kind="simple",
            rtol=1e-3,
            atol=1e-7,
        )
    )


# BASE_URL = "https://download.onnxruntime.ai/onnx/models"
BASE_URL = "onnx/backend/test/data/light/light_%s.onnx"


def collect_testcases() -> List[TestCase]:
    """Collect model test cases defined in python/numpy code."""

    real_model_testcases = []

    model_tests = [
        ("test_bvlc_alexnet", "bvlc_alexnet", 1e-3, 1e-7),
        ("test_densenet121", "densenet121", 2e-3, 1e-7),
        ("test_inception_v1", "inception_v1", 1e-3, 1e-7),
        ("test_inception_v2", "inception_v2", 1e-3, 1e-7),
        ("test_resnet50", "resnet50", 1e-3, 1e-7),
        ("test_shufflenet", "shufflenet", 1e-3, 1e-7),
        ("test_squeezenet", "squeezenet", 1e-3, 1e-7),
        ("test_vgg19", "vgg19", 1e-3, 1e-7),
        ("test_zfnet512", "zfnet512", 1e-3, 1e-7),
    ]

    for test_name, model_name, rtol, atol in model_tests:
        url = BASE_URL % model_name
        real_model_testcases.append(
            TestCase(
                name=test_name,
                model_name=model_name,
                url=url,
                model_dir=None,
                model=None,
                data_sets=None,
                kind="real",
                rtol=rtol,
                atol=atol,
            )
        )

    import_recursive(sys.modules[__name__])

    return real_model_testcases + _SimpleModelTestCases
