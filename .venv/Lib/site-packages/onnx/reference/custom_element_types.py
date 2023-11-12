# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

bfloat16 = np.dtype((np.uint16, {"bfloat16": (np.uint16, 0)}))
float8e4m3fn = np.dtype((np.uint8, {"e4m3fn": (np.uint8, 0)}))
float8e4m3fnuz = np.dtype((np.uint8, {"e4m3fnuz": (np.uint8, 0)}))
float8e5m2 = np.dtype((np.uint8, {"e5m2": (np.uint8, 0)}))
float8e5m2fnuz = np.dtype((np.uint8, {"e5m2fnuz": (np.uint8, 0)}))
