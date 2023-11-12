/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/onnx_pb.h"
#ifdef ONNX_ML
#include "onnx/onnx-operators-ml.pb.h"
#else
#include "onnx/onnx-operators.pb.h"
#endif
