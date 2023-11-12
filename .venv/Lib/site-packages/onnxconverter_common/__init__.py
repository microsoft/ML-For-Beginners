# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
The entry point to onnxconverter-common.
This framework performs optimization for ONNX models and
includes common utilities for ONNX converters.
"""
__version__ = "1.14.0"
__author__ = "Microsoft"
__producer__ = "OnnxMLTools"
__producer_version__ = __version__
__domain__ = "onnxconverter-common"
__model_version__ = 0

from .data_types import *  # noqa F403
from .onnx_ops import *  # noqa F403
from .container import *  # noqa F403
from .registration import *  # noqa F403
from .topology import *  # noqa F403
from .interface import *  # noqa F403
from .shape_calculator import *  # noqa F403
from .tree_ensemble import *  # noqa F403
from .utils import *  # noqa F403
from .case_insensitive_dict import *  # noqa F403
from .metadata_props import add_metadata_props, set_denotation
from .float16 import convert_tensor_float_to_float16, convert_float_to_float16, convert_float_to_float16_model_path
from .optimizer import optimize_onnx, optimize_onnx_graph, optimize_onnx_model
from .auto_mixed_precision import auto_convert_mixed_precision
from .auto_mixed_precision_model_path import auto_convert_mixed_precision_model_path
