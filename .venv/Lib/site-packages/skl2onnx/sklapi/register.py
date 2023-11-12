# SPDX-License-Identifier: Apache-2.0

from .sklearn_text_onnx import register as register_text
from .woe_transformer_onnx import register as register_woe


register_text()
register_woe()
