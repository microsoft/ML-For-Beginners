# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Dict, NamedTuple, Union, cast

import numpy as np

from onnx import OptionalProto, SequenceProto, TensorProto

TensorDtypeMap = NamedTuple(
    "TensorDtypeMap", [("np_dtype", np.dtype), ("storage_dtype", int), ("name", str)]
)

# tensor_dtype: (numpy type, storage type, string name)
TENSOR_TYPE_MAP = {
    int(TensorProto.FLOAT): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.FLOAT), "TensorProto.FLOAT"
    ),
    int(TensorProto.UINT8): TensorDtypeMap(
        np.dtype("uint8"), int(TensorProto.INT32), "TensorProto.UINT8"
    ),
    int(TensorProto.INT8): TensorDtypeMap(
        np.dtype("int8"), int(TensorProto.INT32), "TensorProto.INT8"
    ),
    int(TensorProto.UINT16): TensorDtypeMap(
        np.dtype("uint16"), int(TensorProto.INT32), "TensorProto.UINT16"
    ),
    int(TensorProto.INT16): TensorDtypeMap(
        np.dtype("int16"), int(TensorProto.INT32), "TensorProto.INT16"
    ),
    int(TensorProto.INT32): TensorDtypeMap(
        np.dtype("int32"), int(TensorProto.INT32), "TensorProto.INT32"
    ),
    int(TensorProto.INT64): TensorDtypeMap(
        np.dtype("int64"), int(TensorProto.INT64), "TensorProto.INT64"
    ),
    int(TensorProto.BOOL): TensorDtypeMap(
        np.dtype("bool"), int(TensorProto.INT32), "TensorProto.BOOL"
    ),
    int(TensorProto.FLOAT16): TensorDtypeMap(
        np.dtype("float16"), int(TensorProto.UINT16), "TensorProto.FLOAT16"
    ),
    # Native numpy does not support bfloat16 so now use float32.
    int(TensorProto.BFLOAT16): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT16), "TensorProto.BFLOAT16"
    ),
    int(TensorProto.DOUBLE): TensorDtypeMap(
        np.dtype("float64"), int(TensorProto.DOUBLE), "TensorProto.DOUBLE"
    ),
    int(TensorProto.COMPLEX64): TensorDtypeMap(
        np.dtype("complex64"), int(TensorProto.FLOAT), "TensorProto.COMPLEX64"
    ),
    int(TensorProto.COMPLEX128): TensorDtypeMap(
        np.dtype("complex128"), int(TensorProto.DOUBLE), "TensorProto.COMPLEX128"
    ),
    int(TensorProto.UINT32): TensorDtypeMap(
        np.dtype("uint32"), int(TensorProto.UINT32), "TensorProto.UINT32"
    ),
    int(TensorProto.UINT64): TensorDtypeMap(
        np.dtype("uint64"), int(TensorProto.UINT64), "TensorProto.UINT64"
    ),
    int(TensorProto.STRING): TensorDtypeMap(
        np.dtype("object"), int(TensorProto.STRING), "TensorProto.STRING"
    ),
    # Native numpy does not support float8 types, so now use float32 for these types.
    int(TensorProto.FLOAT8E4M3FN): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E4M3FN"
    ),
    int(TensorProto.FLOAT8E4M3FNUZ): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E4M3FNUZ"
    ),
    int(TensorProto.FLOAT8E5M2): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E5M2"
    ),
    int(TensorProto.FLOAT8E5M2FNUZ): TensorDtypeMap(
        np.dtype("float32"), int(TensorProto.UINT8), "TensorProto.FLOAT8E5M2FNUZ"
    ),
}


class DeprecatedWarningDict(dict):  # type: ignore
    def __init__(
        self,
        dictionary: Dict[int, Union[int, str, np.dtype]],
        original_function: str,
        future_function: str = "",
    ) -> None:
        super().__init__(dictionary)
        self._origin_function = original_function
        self._future_function = future_function

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeprecatedWarningDict):
            return False
        return (
            self._origin_function == other._origin_function
            and self._future_function == other._future_function
        )

    def __getitem__(self, key: Union[int, str, np.dtype]) -> Any:
        if not self._future_function:
            warnings.warn(
                str(
                    f"`mapping.{self._origin_function}` is now deprecated and will be removed in a future release."
                    "To silence this warning, please simply use if-else statement to get the corresponding value."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                str(
                    f"`mapping.{self._origin_function}` is now deprecated and will be removed in a future release."
                    f"To silence this warning, please use `helper.{self._future_function}` instead."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getitem__(key)


# This map is used for converting TensorProto values into numpy arrays
TENSOR_TYPE_TO_NP_TYPE = DeprecatedWarningDict(
    {tensor_dtype: value.np_dtype for tensor_dtype, value in TENSOR_TYPE_MAP.items()},
    "TENSOR_TYPE_TO_NP_TYPE",
    "tensor_dtype_to_np_dtype",
)
# This is only used to get keys into STORAGE_TENSOR_TYPE_TO_FIELD.
# TODO(https://github.com/onnx/onnx/issues/4554): Move these variables into _mapping.py

TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE = DeprecatedWarningDict(
    {
        tensor_dtype: value.storage_dtype
        for tensor_dtype, value in TENSOR_TYPE_MAP.items()
    },
    "TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE",
    "tensor_dtype_to_storage_tensor_dtype",
)

# NP_TYPE_TO_TENSOR_TYPE will be eventually removed in the future
# and _NP_TYPE_TO_TENSOR_TYPE will only be used internally
_NP_TYPE_TO_TENSOR_TYPE = {
    v: k
    for k, v in TENSOR_TYPE_TO_NP_TYPE.items()
    if k
    not in (
        TensorProto.BFLOAT16,
        TensorProto.FLOAT8E4M3FN,
        TensorProto.FLOAT8E4M3FNUZ,
        TensorProto.FLOAT8E5M2,
        TensorProto.FLOAT8E5M2FNUZ,
    )
}

# Currently native numpy does not support bfloat16 so TensorProto.BFLOAT16 is ignored for now
# Numpy float32 array is only reversed to TensorProto.FLOAT
NP_TYPE_TO_TENSOR_TYPE = DeprecatedWarningDict(
    cast(Dict[int, Union[int, str, Any]], _NP_TYPE_TO_TENSOR_TYPE),
    "NP_TYPE_TO_TENSOR_TYPE",
    "np_dtype_to_tensor_dtype",
)

# STORAGE_TENSOR_TYPE_TO_FIELD will be eventually removed in the future
# and _STORAGE_TENSOR_TYPE_TO_FIELD will only be used internally
_STORAGE_TENSOR_TYPE_TO_FIELD = {
    int(TensorProto.FLOAT): "float_data",
    int(TensorProto.INT32): "int32_data",
    int(TensorProto.INT64): "int64_data",
    int(TensorProto.UINT8): "int32_data",
    int(TensorProto.UINT16): "int32_data",
    int(TensorProto.DOUBLE): "double_data",
    int(TensorProto.COMPLEX64): "float_data",
    int(TensorProto.COMPLEX128): "double_data",
    int(TensorProto.UINT32): "uint64_data",
    int(TensorProto.UINT64): "uint64_data",
    int(TensorProto.STRING): "string_data",
    int(TensorProto.BOOL): "int32_data",
}

STORAGE_TENSOR_TYPE_TO_FIELD = DeprecatedWarningDict(
    cast(Dict[int, Union[int, str, Any]], _STORAGE_TENSOR_TYPE_TO_FIELD),
    "STORAGE_TENSOR_TYPE_TO_FIELD",
)


# This map will be removed and there is no replacement for it
STORAGE_ELEMENT_TYPE_TO_FIELD = DeprecatedWarningDict(
    {
        int(SequenceProto.TENSOR): "tensor_values",
        int(SequenceProto.SPARSE_TENSOR): "sparse_tensor_values",
        int(SequenceProto.SEQUENCE): "sequence_values",
        int(SequenceProto.MAP): "map_values",
        int(OptionalProto.OPTIONAL): "optional_value",
    },
    "STORAGE_ELEMENT_TYPE_TO_FIELD",
)


# This map will be removed and there is no replacement for it
OPTIONAL_ELEMENT_TYPE_TO_FIELD = DeprecatedWarningDict(
    {
        int(OptionalProto.TENSOR): "tensor_value",
        int(OptionalProto.SPARSE_TENSOR): "sparse_tensor_value",
        int(OptionalProto.SEQUENCE): "sequence_value",
        int(OptionalProto.MAP): "map_value",
        int(OptionalProto.OPTIONAL): "optional_value",
    },
    "OPTIONAL_ELEMENT_TYPE_TO_FIELD",
)
