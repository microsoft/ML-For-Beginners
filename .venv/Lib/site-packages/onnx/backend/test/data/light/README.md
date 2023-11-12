<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Light models

The models in this folder were created by replacing
all float initializers by nodes `ConstantOfShape`
with function `replace_initializer_by_constant_of_shape`.
The models are lighter and can be added to the repository
for unit testing.

Expected outputs were obtained by using CReferenceEvaluator
implemented in [PR 4952](https://github.com/onnx/onnx/pull/4952).
