# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

"""
Every class imported in this module defines an implementation of
an operator of the main domain. Any class name uses `_` to specify a
version defined in a specific opset. The class name without `_`
defines the current implementation. If an operator has no class
with `_`, it means the implementation is valid for every opset.
The operator may have been updated to support more types but that
did not change the implementation.
"""
import textwrap
from typing import Any, Dict, List
from typing import Optional as TOptional
from typing import Union

from onnx import FunctionProto, NodeProto, TypeProto
from onnx.defs import get_schema, onnx_opset_version
from onnx.onnx_cpp2py_export.defs import SchemaError
from onnx.reference.op_run import (
    OpFunction,
    OpRun,
    RuntimeContextError,
    RuntimeImplementationError,
    _split_class_name,
)
from onnx.reference.ops._helpers import build_registered_operators_any_domain
from onnx.reference.ops.op_abs import Abs
from onnx.reference.ops.op_acos import Acos
from onnx.reference.ops.op_acosh import Acosh
from onnx.reference.ops.op_add import Add
from onnx.reference.ops.op_affine_grid import AffineGrid
from onnx.reference.ops.op_and import And
from onnx.reference.ops.op_argmax import ArgMax_1, ArgMax_12
from onnx.reference.ops.op_argmin import ArgMin_1, ArgMin_12
from onnx.reference.ops.op_asin import Asin
from onnx.reference.ops.op_asinh import Asinh
from onnx.reference.ops.op_atan import Atan
from onnx.reference.ops.op_atanh import Atanh
from onnx.reference.ops.op_attribute_has_value import AttributeHasValue
from onnx.reference.ops.op_average_pool import (
    AveragePool_1,
    AveragePool_7,
    AveragePool_11,
    AveragePool_19,
)
from onnx.reference.ops.op_batch_normalization import (
    BatchNormalization_6,
    BatchNormalization_9,
    BatchNormalization_14,
)
from onnx.reference.ops.op_bernoulli import Bernoulli
from onnx.reference.ops.op_bitshift import BitShift
from onnx.reference.ops.op_bitwise_and import BitwiseAnd
from onnx.reference.ops.op_bitwise_not import BitwiseNot
from onnx.reference.ops.op_bitwise_or import BitwiseOr
from onnx.reference.ops.op_bitwise_xor import BitwiseXor
from onnx.reference.ops.op_blackman_window import BlackmanWindow
from onnx.reference.ops.op_cast import Cast_1, Cast_19
from onnx.reference.ops.op_cast_like import CastLike_15, CastLike_19
from onnx.reference.ops.op_ceil import Ceil
from onnx.reference.ops.op_celu import Celu
from onnx.reference.ops.op_center_crop_pad import CenterCropPad
from onnx.reference.ops.op_clip import Clip_6, Clip_11
from onnx.reference.ops.op_col2im import Col2Im
from onnx.reference.ops.op_compress import Compress
from onnx.reference.ops.op_concat import Concat
from onnx.reference.ops.op_concat_from_sequence import ConcatFromSequence
from onnx.reference.ops.op_constant import (
    Constant_1,
    Constant_9,
    Constant_11,
    Constant_12,
)
from onnx.reference.ops.op_constant_of_shape import ConstantOfShape
from onnx.reference.ops.op_conv import Conv
from onnx.reference.ops.op_conv_integer import ConvInteger
from onnx.reference.ops.op_conv_transpose import ConvTranspose
from onnx.reference.ops.op_cos import Cos
from onnx.reference.ops.op_cosh import Cosh
from onnx.reference.ops.op_cum_sum import CumSum
from onnx.reference.ops.op_deform_conv import DeformConv
from onnx.reference.ops.op_depth_to_space import DepthToSpace
from onnx.reference.ops.op_dequantize_linear import DequantizeLinear
from onnx.reference.ops.op_det import Det
from onnx.reference.ops.op_dft import DFT_17, DFT_20
from onnx.reference.ops.op_div import Div
from onnx.reference.ops.op_dropout import Dropout_7, Dropout_12
from onnx.reference.ops.op_dynamic_quantize_linear import DynamicQuantizeLinear
from onnx.reference.ops.op_einsum import Einsum
from onnx.reference.ops.op_elu import Elu
from onnx.reference.ops.op_equal import Equal
from onnx.reference.ops.op_erf import Erf
from onnx.reference.ops.op_exp import Exp
from onnx.reference.ops.op_expand import Expand
from onnx.reference.ops.op_eyelike import EyeLike
from onnx.reference.ops.op_flatten import Flatten
from onnx.reference.ops.op_floor import Floor
from onnx.reference.ops.op_gather import Gather
from onnx.reference.ops.op_gather_elements import GatherElements
from onnx.reference.ops.op_gathernd import GatherND
from onnx.reference.ops.op_gemm import Gemm_6, Gemm_7
from onnx.reference.ops.op_global_average_pool import GlobalAveragePool
from onnx.reference.ops.op_global_max_pool import GlobalMaxPool
from onnx.reference.ops.op_greater import Greater
from onnx.reference.ops.op_greater_or_equal import GreaterOrEqual
from onnx.reference.ops.op_grid_sample import GridSample
from onnx.reference.ops.op_gru import GRU
from onnx.reference.ops.op_hamming_window import HammingWindow
from onnx.reference.ops.op_hann_window import HannWindow
from onnx.reference.ops.op_hard_sigmoid import HardSigmoid
from onnx.reference.ops.op_hardmax import Hardmax
from onnx.reference.ops.op_identity import Identity
from onnx.reference.ops.op_if import If
from onnx.reference.ops.op_image_decoder import ImageDecoder
from onnx.reference.ops.op_instance_normalization import InstanceNormalization
from onnx.reference.ops.op_isinf import IsInf
from onnx.reference.ops.op_isnan import IsNaN
from onnx.reference.ops.op_layer_normalization import LayerNormalization
from onnx.reference.ops.op_leaky_relu import LeakyRelu
from onnx.reference.ops.op_less import Less
from onnx.reference.ops.op_less_or_equal import LessOrEqual
from onnx.reference.ops.op_log import Log
from onnx.reference.ops.op_log_softmax import LogSoftmax
from onnx.reference.ops.op_loop import Loop
from onnx.reference.ops.op_lp_normalization import LpNormalization
from onnx.reference.ops.op_lp_pool import LpPool
from onnx.reference.ops.op_lrn import LRN
from onnx.reference.ops.op_lstm import LSTM
from onnx.reference.ops.op_matmul import MatMul
from onnx.reference.ops.op_matmul_integer import MatMulInteger
from onnx.reference.ops.op_max import Max
from onnx.reference.ops.op_max_pool import MaxPool
from onnx.reference.ops.op_max_unpool import MaxUnpool
from onnx.reference.ops.op_mean import Mean
from onnx.reference.ops.op_mel_weight_matrix import MelWeightMatrix
from onnx.reference.ops.op_min import Min
from onnx.reference.ops.op_mod import Mod
from onnx.reference.ops.op_mul import Mul
from onnx.reference.ops.op_neg import Neg
from onnx.reference.ops.op_negative_log_likelihood_loss import NegativeLogLikelihoodLoss
from onnx.reference.ops.op_non_max_suppression import NonMaxSuppression
from onnx.reference.ops.op_non_zero import NonZero
from onnx.reference.ops.op_not import Not
from onnx.reference.ops.op_one_hot import OneHot
from onnx.reference.ops.op_optional import Optional
from onnx.reference.ops.op_optional_get_element import OptionalGetElement
from onnx.reference.ops.op_optional_has_element import OptionalHasElement
from onnx.reference.ops.op_or import Or
from onnx.reference.ops.op_pad import Pad_1, Pad_2, Pad_11, Pad_18
from onnx.reference.ops.op_pow import Pow
from onnx.reference.ops.op_prelu import PRelu
from onnx.reference.ops.op_qlinear_conv import QLinearConv
from onnx.reference.ops.op_qlinear_matmul import QLinearMatMul
from onnx.reference.ops.op_quantize_linear import QuantizeLinear_10, QuantizeLinear_19
from onnx.reference.ops.op_random_normal import RandomNormal
from onnx.reference.ops.op_random_normal_like import RandomNormalLike
from onnx.reference.ops.op_random_uniform import RandomUniform
from onnx.reference.ops.op_random_uniform_like import RandomUniformLike
from onnx.reference.ops.op_range import Range
from onnx.reference.ops.op_reciprocal import Reciprocal
from onnx.reference.ops.op_reduce_l1 import ReduceL1_1, ReduceL1_18
from onnx.reference.ops.op_reduce_l2 import ReduceL2_1, ReduceL2_18
from onnx.reference.ops.op_reduce_log_sum import ReduceLogSum_1, ReduceLogSum_18
from onnx.reference.ops.op_reduce_log_sum_exp import (
    ReduceLogSumExp_1,
    ReduceLogSumExp_18,
)
from onnx.reference.ops.op_reduce_max import ReduceMax_1, ReduceMax_18
from onnx.reference.ops.op_reduce_mean import ReduceMean_1, ReduceMean_18
from onnx.reference.ops.op_reduce_min import ReduceMin_1, ReduceMin_18
from onnx.reference.ops.op_reduce_prod import ReduceProd_1, ReduceProd_18
from onnx.reference.ops.op_reduce_sum import ReduceSum_1, ReduceSum_13
from onnx.reference.ops.op_reduce_sum_square import (
    ReduceSumSquare_1,
    ReduceSumSquare_18,
)
from onnx.reference.ops.op_regex_full_match import RegexFullMatch
from onnx.reference.ops.op_relu import Relu
from onnx.reference.ops.op_reshape import Reshape_5, Reshape_14
from onnx.reference.ops.op_resize import Resize
from onnx.reference.ops.op_reverse_sequence import ReverseSequence
from onnx.reference.ops.op_rnn import RNN_7, RNN_14
from onnx.reference.ops.op_roi_align import RoiAlign
from onnx.reference.ops.op_round import Round
from onnx.reference.ops.op_scan import Scan
from onnx.reference.ops.op_scatter_elements import ScatterElements
from onnx.reference.ops.op_scatternd import ScatterND
from onnx.reference.ops.op_selu import Selu
from onnx.reference.ops.op_sequence_at import SequenceAt
from onnx.reference.ops.op_sequence_construct import SequenceConstruct
from onnx.reference.ops.op_sequence_empty import SequenceEmpty
from onnx.reference.ops.op_sequence_erase import SequenceErase
from onnx.reference.ops.op_sequence_insert import SequenceInsert
from onnx.reference.ops.op_sequence_length import SequenceLength
from onnx.reference.ops.op_sequence_map import SequenceMap
from onnx.reference.ops.op_shape import Shape_1, Shape_15
from onnx.reference.ops.op_shrink import Shrink
from onnx.reference.ops.op_sigmoid import Sigmoid
from onnx.reference.ops.op_sign import Sign
from onnx.reference.ops.op_sin import Sin
from onnx.reference.ops.op_sinh import Sinh
from onnx.reference.ops.op_size import Size
from onnx.reference.ops.op_slice import Slice_1, Slice_10
from onnx.reference.ops.op_softmax import Softmax
from onnx.reference.ops.op_softmax_cross_entropy_loss import SoftmaxCrossEntropyLoss
from onnx.reference.ops.op_softplus import Softplus
from onnx.reference.ops.op_softsign import Softsign
from onnx.reference.ops.op_space_to_depth import SpaceToDepth
from onnx.reference.ops.op_split import Split_2, Split_11, Split_13, Split_18
from onnx.reference.ops.op_split_to_sequence import SplitToSequence
from onnx.reference.ops.op_sqrt import Sqrt
from onnx.reference.ops.op_squeeze import Squeeze_1, Squeeze_11, Squeeze_13
from onnx.reference.ops.op_stft import STFT
from onnx.reference.ops.op_string_concat import StringConcat
from onnx.reference.ops.op_string_normalizer import StringNormalizer
from onnx.reference.ops.op_string_split import StringSplit
from onnx.reference.ops.op_sub import Sub
from onnx.reference.ops.op_sum import Sum
from onnx.reference.ops.op_tan import Tan
from onnx.reference.ops.op_tanh import Tanh
from onnx.reference.ops.op_tfidf_vectorizer import TfIdfVectorizer
from onnx.reference.ops.op_thresholded_relu import ThresholdedRelu
from onnx.reference.ops.op_tile import Tile
from onnx.reference.ops.op_topk import TopK_1, TopK_10, TopK_11
from onnx.reference.ops.op_transpose import Transpose
from onnx.reference.ops.op_trilu import Trilu
from onnx.reference.ops.op_unique import Unique
from onnx.reference.ops.op_unsqueeze import Unsqueeze_1, Unsqueeze_11, Unsqueeze_13
from onnx.reference.ops.op_upsample import Upsample
from onnx.reference.ops.op_where import Where
from onnx.reference.ops.op_xor import Xor


def _build_registered_operators() -> Dict[str, Dict[Union[int, None], OpRun]]:
    return build_registered_operators_any_domain(globals().copy())


def load_op(
    domain: str,
    op_type: str,
    version: Union[None, int] = None,
    custom: Any = None,
    node: Union[None, NodeProto] = None,
    input_types: Union[None, List[TypeProto]] = None,
    expand: bool = False,
) -> Any:
    """
    Loads the implemented for a specified operator.

    :param domain: domain
    :param op_type: oprator type
    :param version: requested version
    :param custom: custom implementation (like a function)
    :param node: used if no implementation was found and the operator defines a function
        which is context dependant
    :param input_types: used if no implementation was found and the operator defines a function
        which is context dependant
    :param expand: use the function implemented in the schema instead
        of its reference implementation
    :return: class
    """
    global _registered_operators  # noqa: PLW0603
    schema = None
    if _registered_operators is None:
        _registered_operators = _build_registered_operators()  # type: ignore[assignment]
    if custom is not None:
        return lambda *args: OpFunction(*args, impl=custom)  # type: ignore
    if version is None:
        version = onnx_opset_version()
    if domain != "":
        raise ValueError(f"Domain must be '' not {domain!r}.")
    if op_type in _registered_operators and not expand:  # type: ignore
        found = True
    else:
        # maybe the operator can be replacted by a function
        try:
            schema = get_schema(op_type, version, domain)  # type: ignore
        except SchemaError:
            raise NotImplementedError(
                f"No registered schema for operator {op_type!r} "
                f"and domain {domain!r}. Did you recompile the sources after updating the repository?"
            ) from None
        if schema.has_function:  # type: ignore
            from onnx.reference import ReferenceEvaluator

            body = schema.function_body  # type: ignore
            sess = ReferenceEvaluator(body)
            return lambda *args, sess=sess: OpFunction(*args, impl=sess)  # type: ignore
        if schema.has_context_dependent_function:  # type: ignore
            if node is None or input_types is None:
                raise RuntimeContextError(
                    f"No registered implementation for operator {op_type!r} "
                    f"and domain {domain!r}, the operator has a context dependent function. "
                    f"but argument node or input_types is not defined (input_types={input_types})."
                )
            from onnx.reference import ReferenceEvaluator

            body = schema.get_context_dependent_function(  # type: ignore
                node.SerializeToString(), [it.SerializeToString() for it in input_types]
            )
            proto = FunctionProto()
            proto.ParseFromString(body)
            sess = ReferenceEvaluator(proto)
            return lambda *args, sess=sess: OpFunction(*args, impl=sess)  # type: ignore
        found = False
    if not found:
        available = "\n".join(textwrap.wrap(", ".join(sorted(_registered_operators))))  # type: ignore
        has_function = schema.has_function if schema else None  # type: ignore
        has_context_dependent_function = (
            schema.has_context_dependent_function if schema else None  # type: ignore
        )
        raise RuntimeImplementationError(
            f"No registered implementation for operator {op_type!r} "
            f"and domain {domain!r}, schema.has_function is {has_function}, "
            f"schema.has_context_dependent_function is {has_context_dependent_function}. "
            f"You may either add one or skip the test in "
            f"'reference_evaluator_bakcend_test.py'. Available implementations:\n{available}"
        )
    impl = _registered_operators[op_type]  # type: ignore
    if None not in impl:
        raise RuntimeError(
            f"No default implementation for operator {op_type!r} "
            f"and domain {domain!r}, found "
            f"{', '.join(map(str, impl))}."
        )
    if version is None or len(impl) == 1:
        cl = impl[None]
    else:
        best = -1
        for v in impl:
            if v is None:
                continue
            if best < v <= version:
                best = v
        if best == -1:
            raise RuntimeError(
                f"No implementation for operator {op_type!r} "
                f"domain {domain!r} and version {version!r}, found "
                f"{', '.join(map(str, impl))}."
            )
        cl = impl[best]
    if cl is None:
        available = "\n".join(textwrap.wrap(", ".join(sorted(_registered_operators))))  # type: ignore
        raise ValueError(
            f"Not registered implementation for operator {op_type!r}, "
            f"domain {domain!r}, and {version!r} in\n{available}"
        )
    return cl


_registered_operators: TOptional[Dict[str, Dict[Union[int, None], OpRun]]] = None
