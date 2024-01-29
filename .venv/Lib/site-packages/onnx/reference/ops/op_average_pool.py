# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.ops.op_pool_common import CommonPool


class AveragePool_1(CommonPool):
    def _run(  # type: ignore
        self,
        x,
        auto_pad=None,
        ceil_mode=None,
        kernel_shape=None,
        pads=None,
        strides=None,
        count_include_pad=None,
    ):
        return CommonPool._run(
            self,
            "AVG",
            count_include_pad,
            x,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=None,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )


class AveragePool_7(CommonPool):
    def _run(  # type: ignore
        self,
        x,
        auto_pad=None,
        ceil_mode=None,
        kernel_shape=None,
        pads=None,
        strides=None,
        count_include_pad=None,
    ):
        return CommonPool._run(
            self,
            "AVG",
            count_include_pad,
            x,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=None,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )


class AveragePool_11(CommonPool):
    def _run(  # type: ignore
        self,
        x,
        auto_pad=None,
        ceil_mode=None,
        kernel_shape=None,
        pads=None,
        strides=None,
        count_include_pad=None,
    ):
        return CommonPool._run(
            self,
            "AVG",
            count_include_pad,
            x,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=None,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )


class AveragePool_19(CommonPool):
    def _run(  # type: ignore
        self,
        x,
        auto_pad=None,
        ceil_mode=None,
        dilations=None,
        kernel_shape=None,
        pads=None,
        strides=None,
        count_include_pad=None,
    ):
        return CommonPool._run(
            self,
            "AVG",
            count_include_pad,
            x,
            auto_pad=auto_pad,
            ceil_mode=ceil_mode,
            dilations=dilations,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
        )
