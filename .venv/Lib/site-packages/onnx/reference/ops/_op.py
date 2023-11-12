# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import numpy as np

from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError


class OpRunUnary(OpRun):  # pylint: disable=W0223
    """
    Ancestor to all unary operators in this subfolder.
    Checks that input and output types are the same.
    """

    def __init__(self, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRun.__init__(self, onnx_node, run_params)

    def run(self, x):  # type: ignore # pylint: disable=W0221
        """
        Calls method ``_run``, catches exceptions,
        displays a longer error message.
        Supports only unary operators.
        """
        self._log("-- begin %s.run(1 input)", self.__class__.__name__)
        try:
            res = self._run(x)
        except TypeError as e:
            raise TypeError(
                f"Issues with types {', '.join(str(type(_)) for _ in [x])} "
                f"(unary operator {self.__class__.__name__!r})."
            ) from e
        self._log("-- done %s.run -> %d outputs", self.__class__.__name__, len(res))
        return res


class OpRunUnaryNum(OpRunUnary):  # pylint: disable=W0223
    """
    Ancestor to all unary and numerical operators
    in this subfolder. Checks that input and output types
    are the same.
    """

    def __init__(self, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRunUnary.__init__(self, onnx_node, run_params)

    def run(self, x):  # type: ignore # pylint: disable=W0221
        """
        Calls method ``OpRunUnary.run``, catches exceptions,
        displays a longer error message.
        Checks that the result is not empty.
        """
        res = OpRunUnary.run(self, x)
        if len(res) == 0 or res[0] is None:
            return res
        if not isinstance(res[0], list) and res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                f"Output type mismatch: input '{x.dtype}' != output '{res[0].dtype}' "
                f"(operator {self.__class__.__name__!r})."
            )
        return res


class OpRunBinary(OpRun):  # pylint: disable=W0223
    """
    Ancestor to all binary operators in this subfolder.
    Checks that input and output types are the same.
    """

    def __init__(self, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRun.__init__(self, onnx_node, run_params)

    def run(self, x, y):  # type: ignore # pylint: disable=W0221
        """
        Calls method ``_run``, catches exceptions,
        displays a longer error message.
        Supports only binary operators.
        """
        self._log("-- begin %s.run(2 inputs)", self.__class__.__name__)
        if x is None or y is None:
            raise RuntimeError(
                f"x and y have different dtype: {type(x)} != {type(y)} ({type(self)})"
            )
        if x.dtype != y.dtype:
            raise RuntimeTypeError(
                f"Input type mismatch: {x.dtype} != {y.dtype} "
                f"(operator '{self.__class__.__name__!r}', "
                f"shapes {x.shape}, {y.shape})."
            )
        try:
            res = self._run(x, y)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Issues with types {', '.join(str(type(_)) for _ in [x, y])} "
                f"(binary operator {self.__class__.__name__!r})."
            ) from e
        self._log("-- done %s.run -> %d outputs", self.__class__.__name__, len(res))
        return res


class OpRunBinaryComparison(OpRunBinary):  # pylint: disable=W0223
    """
    Ancestor to all binary operators in this subfolder
    comparing tensors.
    """

    def __init__(self, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRunBinary.__init__(self, onnx_node, run_params)


class OpRunBinaryNum(OpRunBinary):  # pylint: disable=W0223
    """
    Ancestor to all binary operators in this subfolder.
    Checks that input oud output types are the same.
    """

    def __init__(self, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRunBinary.__init__(self, onnx_node, run_params)

    def run(self, x, y):  # type: ignore # pylint: disable=W0221
        """
        Calls method ``OpRunBinary.run``, catches exceptions,
        displays a longer error message.
        """
        res = OpRunBinary.run(self, x, y)
        if res[0].dtype != x.dtype:
            raise RuntimeTypeError(
                f"Output type mismatch: {x.dtype} != {res[0].dtype} or {y.dtype} "
                f"(operator {self.__class__.__name__!r})"
                f" type(x)={type(x)} type(y)={type(y)}"
            )
        return res


class OpRunBinaryNumpy(OpRunBinaryNum):
    """
    *numpy_fct* is a binary numpy function which
    takes two matrices.
    """

    def __init__(
        self, numpy_fct: Any, onnx_node: NodeProto, run_params: Dict[str, Any]
    ):
        OpRunBinaryNum.__init__(self, onnx_node, run_params)
        self.numpy_fct = numpy_fct

    def _run(self, a, b):  # type: ignore # pylint: disable=W0221
        return (self.numpy_fct(a, b),)


class OpRunReduceNumpy(OpRun):  # type: ignore
    """
    Implements the reduce logic.
    It must have a parameter *axes*.
    """

    def __init__(self, onnx_node: NodeProto, run_params: Dict[str, Any]):
        OpRun.__init__(self, onnx_node, run_params)
        if hasattr(self, "axes"):
            if isinstance(self.axes, np.ndarray):  # type: ignore # pylint: disable=E0203
                if len(self.axes.shape) == 0 or self.axes.shape[0] == 0:  # type: ignore # pylint: disable=E0203
                    self.axes = None
                else:
                    self.axes = tuple(self.axes)
            elif self.axes in [[], tuple()]:
                self.axes = None
            elif isinstance(self.axes, list):
                self.axes = tuple(self.axes)

    def is_axes_empty(self, axes):
        return axes is None

    def handle_axes(self, axes):
        if isinstance(axes, tuple):
            if len(axes) == 0:
                return None
            return axes
        if axes is None:
            return None
        if isinstance(axes, (int, tuple)):
            return axes
        if not isinstance(axes, np.ndarray):
            raise TypeError(f"axes must be an array, not {type(axes)}.")
        if len(axes.shape) == 0:
            return int(axes)
        if 0 in axes.shape:
            return None
        return tuple(axes.ravel().tolist())
