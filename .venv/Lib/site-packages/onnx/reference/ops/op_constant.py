# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.custom_element_types import (
    bfloat16,
    float8e4m3fn,
    float8e4m3fnuz,
    float8e5m2,
    float8e5m2fnuz,
)
from onnx.reference.op_run import OpRun, RefAttrName


def _check_dtype(val):  # type: ignore
    a = val.dtype
    if not isinstance(a, np.dtype) and a not in {
        bfloat16,
        float8e4m3fn,
        float8e4m3fnuz,
        float8e5m2,
        float8e5m2fnuz,
        np.int8,
        np.uint8,
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.int16,
        np.uint16,
        np.uint32,
        np.bool_,
        np.str_,
        np.uint64,
        bool,
        str,
    }:
        raise TypeError(
            f"Type ({a}, {type(a)}) is not a numpy type (operator 'Constant')"
        )


class ConstantCommon(OpRun):
    def _check(self, cst):  # type: ignore
        if isinstance(cst, tuple):
            raise TypeError(f"Unexpected type {type(cst)} for a constant.")
        return cst


class Constant_1(ConstantCommon):
    def __init__(self, onnx_node, run_params):  # type: ignore
        ConstantCommon.__init__(self, onnx_node, run_params)
        self.cst = self.value  # type: ignore
        _check_dtype(self.cst)

    def _run(self, **overridden_attributes):  # type: ignore
        if overridden_attributes and (
            len(overridden_attributes) > 1
            or "value" not in overridden_attributes
            or id(overridden_attributes["value"]) != id(self.value)
        ):
            raise RuntimeError(
                "Function attributes are not implemented for opset <= 11. Use opset > 12."
            )
        return (self._check(self.cst),)


class Constant_9(Constant_1):
    def __init__(self, onnx_node, run_params):  # type: ignore
        Constant_1.__init__(self, onnx_node, run_params)


class Constant_11(ConstantCommon):
    def __init__(self, onnx_node, run_params):  # type: ignore
        ConstantCommon.__init__(self, onnx_node, run_params)
        if getattr(self, "sparse_value", None) is None:
            self.cst = self.value  # type: ignore
        else:
            self.cst = self.sparse_value  # type: ignore
        _check_dtype(self.cst)

    def _run(self, **overridden_attributes):  # type: ignore
        if overridden_attributes and (
            len(overridden_attributes) > 1
            or "value" not in overridden_attributes
            or id(overridden_attributes["value"]) != id(self.value)
        ):
            raise RuntimeError(
                "Function attributes are not implemented for opset <= 11. Use opset > 12."
            )
        return (self._check(self.cst),)


class Constant_12(ConstantCommon):
    def __init__(self, onnx_node, run_params):  # type: ignore
        ConstantCommon.__init__(self, onnx_node, run_params)
        if hasattr(self, "sparse_value") and self.sparse_value is not None:  # type: ignore
            self.cst_name = "sparse_value"
            self.cst = self.sparse_value  # type: ignore
            self.cst_convert = lambda v: v
        elif hasattr(self, "value") and self.value is not None:  # type: ignore
            self.cst_name = "value"  # type: ignore
            self.cst = self.value if isinstance(self.value, RefAttrName) else self.value  # type: ignore
            self.cst_convert = lambda v: v
        else:
            for attr, np_dtype in {
                "value_float": np.float32,
                "value_floats": np.float32,
                "value_int": np.int64,
                "value_ints": np.int64,
                "value_string": np.str_,
                "value_strings": np.str_,
            }.items():
                if hasattr(self, attr) and getattr(self, attr) is not None:  # type: ignore
                    self.cst_name = attr
                    v = getattr(self, attr)
                    self.cst = (
                        v  # type: ignore
                        if isinstance(v, RefAttrName)  # type: ignore
                        else np.array(v, dtype=np_dtype)  # type: ignore
                    )
                    self.cst_convert = lambda v, np_dtype=np_dtype: np.array(  # type: ignore
                        v, dtype=np_dtype
                    )
                    break
        if not hasattr(self, "cst_name"):
            raise AttributeError("No constant is defined for operator 'Constant'.")

    def _run(self, **overridden_attributes):  # type: ignore
        if self.has_linked_attribute:
            if overridden_attributes is None:
                raise RuntimeError(
                    f"Attributes are empty, cannot retrieve value for {self.cst!r}."
                )
            if self.cst_name not in overridden_attributes:
                raise RuntimeError(
                    f"Cannot find attribute {self.cst_name!r} in {list(overridden_attributes)!r}."
                )
            value = overridden_attributes[self.cst_name]
            if isinstance(value, np.ndarray):
                return (value,)
            return (self.cst_convert(value),)
        return (self._check(self.cst),)
