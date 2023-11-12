# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def compute_negative_log_likelihood_loss(input, target, weight=None, reduction="mean", ignore_index=None):  # type: ignore
    input_shape = input.shape
    if len(input_shape) == 1:
        raise RuntimeError("Unsupported shape")

    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]

    # initialize the positional weights when required
    gather_weight = None
    if weight is not None:
        # setting mode='clip' to deal with ignore_index > C or < 0 cases.
        # when the target value is > C or < 0, it doesn't matter which value we are
        # taking in gather_weight, since it will be set to 0 in the following if-block
        # use np.int32 to make it compatible with x86 machines
        gather_weight = np.take(weight, np.array(target, dtype=np.int32), mode="clip")
        # set `ignore_index`'s loss weight to 0.
        # The loss tensor will be multiplied by this weight tensor,
        # so `ingore_index`'s loss value will be eliminated.
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(
                dtype=np.float32
            )
    elif ignore_index is not None:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(dtype=np.float32)

    # if input is 4-d and above, make it 3-d
    if len(input_shape) != 3:
        input = input.reshape((N, C, -1))
        target = target.reshape((N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = input.shape[2]
    neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        for d in range(D):
            if target[i][d] != ignore_index:
                neg_gather_element_input[i][d] = -input[i][target[i][d]][d]

    loss = neg_gather_element_input

    # if the input was 4-d or above reshape to the right shape
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)

    # apply the weights when required
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = loss.sum() / gather_weight.sum()
            return loss

    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)
    return loss


class NegativeLogLikelihoodLoss(Base):
    @staticmethod
    def export_input_shape_is_NC() -> None:
        reduction = "none"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C = 3, 5
        np.random.seed(0)
        input = np.random.rand(N, C).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N,)).astype(np.int64)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=None, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NC",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2() -> None:
        reduction = "none"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=None, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2_reduction_mean() -> None:
        reduction = "mean"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=None, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2_reduction_mean",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2_reduction_sum() -> None:
        reduction = "sum"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=None, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2_reduction_sum",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2_with_weight() -> None:
        reduction = "none"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2_with_weight",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2_with_weight_reduction_mean() -> None:
        reduction = "mean"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2_with_weight_reduction_mean",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2_with_weight_reduction_sum() -> None:
        reduction = "sum"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2_with_weight_reduction_sum",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2_with_weight_reduction_sum_ii() -> None:
        reduction = "sum"
        ignore_index = np.int64(0)
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
        target[0][0][0] = np.int64(0)
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2_with_weight_reduction_sum_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2_no_weight_reduction_mean_ii() -> None:
        reduction = "mean"
        ignore_index = np.int64(1)
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, dim1, dim2 = 3, 5, 6, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
        target[0][0][0] = np.int64(1)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2_no_weight_reduction_mean_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1() -> None:
        reduction = "mean"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, d1 = 3, 5, 2
        np.random.seed(0)
        input = np.random.rand(N, C, d1).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=None, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1",
        )

    @staticmethod
    def export_input_shape_is_NCd1_weight() -> None:
        reduction = "mean"
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, d1 = 3, 5, 2
        np.random.seed(0)
        input = np.random.rand(N, C, d1).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1_weight",
        )

    @staticmethod
    def export_input_shape_is_NCd1_ii() -> None:
        reduction = "mean"
        ignore_index = np.int64(1)
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, d1 = 3, 5, 2
        np.random.seed(0)
        input = np.random.rand(N, C, d1).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
        target[0][0] = np.int64(1)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=None, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1_weight_ii() -> None:
        reduction = "mean"
        ignore_index = np.int64(1)
        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, d1 = 3, 5, 2
        np.random.seed(0)
        input = np.random.rand(N, C, d1).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
        target[0][0] = np.int64(1)
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1_weight_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3d4d5_mean_weight() -> None:
        reduction = "mean"

        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
        target = np.random.randint(
            0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)
        ).astype(np.int64)
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2d3d4d5_mean_weight",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3d4d5_none_no_weight() -> None:
        reduction = "none"

        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
        )

        N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
        target = np.random.randint(
            0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)
        ).astype(np.int64)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, reduction=reduction
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2d3d4d5_none_no_weight",
        )

    @staticmethod
    def export_input_shape_is_NCd1_mean_weight_negative_ii() -> None:
        reduction = "mean"
        ignore_index = np.int64(-1)

        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, dim1 = 3, 5, 6
        np.random.seed(0)
        input = np.random.rand(N, C, dim1).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
        target[0][0] = -1
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1_mean_weight_negative_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3_none_no_weight_negative_ii() -> None:
        reduction = "none"
        ignore_index = np.int64(-5)

        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target"],
            outputs=["loss"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
        np.random.seed(0)
        input = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(
            np.int64
        )
        target[0][0][0][0] = -5

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[input, target],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2d3_none_no_weight_negative_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3_sum_weight_high_ii() -> None:
        reduction = "sum"
        ignore_index = np.int64(10)

        node = onnx.helper.make_node(
            "NegativeLogLikelihoodLoss",
            inputs=["input", "target", "weight"],
            outputs=["loss"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C = 3, 5
        np.random.seed(0)
        input = np.random.rand(N, C).astype(np.float32)
        target = np.random.randint(0, high=C, size=(N)).astype(np.int64)
        target[0] = 10
        weight = np.random.rand(C).astype(np.float32)

        negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
            input, target, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[input, target, weight],
            outputs=[negative_log_likelihood_loss],
            name="test_nllloss_NCd1d2d3_sum_weight_high_ii",
        )
