# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from typing import Any, Callable, List, Optional, Union

from onnx import ModelProto, NodeProto

# A container that hosts the test function and the associated
# test item (ModelProto)


@dataclasses.dataclass
class TestItem:
    func: Callable[..., Any]
    proto: List[Optional[Union[ModelProto, NodeProto]]]
