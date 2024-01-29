# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def softmaxcrossentropy(x, target, weight=None, reduction="mean", ignore_index=None, get_log_prob=None):  # type: ignore
    input_shape = x.shape
    if len(input_shape) == 1:
        raise RuntimeError("Unsupported shape")

    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]

    # compute log_softmax
    max_x = np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x - max_x)
    p = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    inp = np.log(p)
    log_prob = None
    if get_log_prob is True:
        log_prob = np.copy(inp)

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
        inp = inp.reshape((N, C, -1))
        target = target.reshape((N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = inp.shape[2]
    neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        for d in range(D):
            if target[i][d] != ignore_index:
                neg_gather_element_input[i][d] = -inp[i][target[i][d]][d]

    loss = neg_gather_element_input

    # if the input was 4-d or above reshape to the right shape
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)

    # apply the weights when required
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = loss.sum() / gather_weight.sum()
            if get_log_prob is True:
                return loss, log_prob
            else:
                return loss

    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)

    if get_log_prob:
        return loss, log_prob
    return loss


class SoftmaxCrossEntropyLoss(Base):
    @staticmethod
    def export_softmaxcrossentropy_none() -> None:
        # Define operator attributes.
        reduction = "none"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, reduction="none")

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name="test_sce_none")

    @staticmethod
    def export_softmaxcrossentropy_none_log_prob() -> None:
        # Define operator attributes.
        reduction = "none"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, reduction="none", get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels],
            outputs=[loss, log_prob],
            name="test_sce_none_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_none_weights() -> None:
        # Define operator attributes.
        reduction = "none"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, weight=weights, reduction="none")

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[sce],
            name="test_sce_none_weights",
        )

    @staticmethod
    def export_softmaxcrossentropy_none_weights_log_prob() -> None:
        # Define operator attributes.
        reduction = "none"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z", "log_prob"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, weight=weights, reduction="none", get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[loss, log_prob],
            name="test_sce_none_weights_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_sum() -> None:
        # Define operator attributes.
        reduction = "sum"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, reduction="sum")

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name="test_sce_sum")

    @staticmethod
    def export_softmaxcrossentropy_sum_log_prob() -> None:
        # Define operator attributes.
        reduction = "sum"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, reduction="sum", get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels],
            outputs=[loss, log_prob],
            name="test_sce_sum_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean() -> None:
        # Define operator attributes.
        reduction = "mean"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels)

        # Check results
        expect(node, inputs=[x, labels], outputs=[sce], name="test_sce_mean")

    @staticmethod
    def export_softmaxcrossentropy_mean_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(x, labels, get_log_prob=True)

        # Check results
        expect(
            node,
            inputs=[x, labels],
            outputs=[loss, log_prob],
            name="test_sce_mean_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_3d() -> None:
        # Define operator attributes.
        reduction = "mean"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        y = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, y)

        # Check results
        expect(node, inputs=[x, y], outputs=[sce], name="test_sce_mean_3d")

    @staticmethod
    def export_softmaxcrossentropy_mean_3d_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        y = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(x, y, get_log_prob=True)

        # Check results
        expect(
            node,
            inputs=[x, y],
            outputs=[loss, log_prob],
            name="test_sce_mean_3d_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_weights() -> None:
        # Define operator attributes.
        reduction = "mean"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, weight=weights)

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[sce],
            name="test_sce_mean_weight",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z", "log_prob"],
            reduction=reduction,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, weight=weights, get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[loss, log_prob],
            name="test_sce_mean_weight_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ii() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(0)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
        labels[0] = np.int64(0)
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[sce],
            name="test_sce_mean_weight_ii",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ii_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(0)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
        labels[0] = np.int64(0)
        weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, weight=weights, ignore_index=ignore_index, get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[loss, log_prob],
            name="test_sce_mean_weight_ii_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ii() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
        labels[0] = np.int64(2)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, ignore_index=ignore_index)

        # Check results
        expect(
            node, inputs=[x, labels], outputs=[sce], name="test_sce_mean_no_weight_ii"
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ii_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3,)).astype(np.int64)
        labels[0] = np.int64(2)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, ignore_index=ignore_index, get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels],
            outputs=[loss, log_prob],
            name="test_sce_mean_no_weight_ii_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ii_3d() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(1)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
        labels[0][0] = np.int64(1)
        weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[sce],
            name="test_sce_mean_weight_ii_3d",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ii_3d_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(1)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
        labels[0][0] = np.int64(1)
        weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, weight=weights, ignore_index=ignore_index, get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[loss, log_prob],
            name="test_sce_mean_weight_ii_3d_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ii_3d() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
        labels[0][0] = np.int64(2)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(x, labels, ignore_index=ignore_index)

        # Check results
        expect(
            node,
            inputs=[x, labels],
            outputs=[sce],
            name="test_sce_mean_no_weight_ii_3d",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ii_3d_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
        labels[0][0] = np.int64(2)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, ignore_index=ignore_index, get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels],
            outputs=[loss, log_prob],
            name="test_sce_mean_no_weight_ii_3d_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ii_4d() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2, 7).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
        labels[0][0][0] = np.int64(2)
        weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(
            x, labels, reduction=reduction, weight=weights, ignore_index=ignore_index
        )

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[sce],
            name="test_sce_mean_weight_ii_4d",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_weights_ii_4d_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2, 7).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
        labels[0][0][0] = np.int64(2)
        weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x,
            labels,
            reduction=reduction,
            weight=weights,
            ignore_index=ignore_index,
            get_log_prob=True,
        )

        # Check results
        expect(
            node,
            inputs=[x, labels, weights],
            outputs=[loss, log_prob],
            name="test_sce_mean_weight_ii_4d_log_prob",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ii_4d() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2, 7).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
        labels[0][0][0] = np.int64(2)

        # Compute SoftmaxCrossEntropyLoss
        sce = softmaxcrossentropy(
            x, labels, reduction=reduction, ignore_index=ignore_index
        )

        # Check results
        expect(
            node,
            inputs=[x, labels],
            outputs=[sce],
            name="test_sce_mean_no_weight_ii_4d",
        )

    @staticmethod
    def export_softmaxcrossentropy_mean_no_weights_ii_4d_log_prob() -> None:
        # Define operator attributes.
        reduction = "mean"
        ignore_index = np.int64(2)

        # Create operator.
        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        # Define operator inputs.
        np.random.seed(0)
        x = np.random.rand(3, 5, 2, 7).astype(np.float32)
        labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
        labels[0][0][0] = np.int64(2)

        # Compute SoftmaxCrossEntropyLoss
        loss, log_prob = softmaxcrossentropy(
            x, labels, reduction=reduction, ignore_index=ignore_index, get_log_prob=True
        )

        # Check results
        expect(
            node,
            inputs=[x, labels],
            outputs=[loss, log_prob],
            name="test_sce_mean_no_weight_ii_4d_log_prob",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3d4d5_mean_weight() -> None:
        reduction = "mean"

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z"],
            reduction=reduction,
        )

        N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
        np.random.seed(0)
        x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
        labels = np.random.randint(
            0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)
        ).astype(np.int64)
        weight = np.random.rand(C).astype(np.float32)

        sce = softmaxcrossentropy(x, labels, weight=weight, reduction=reduction)

        expect(
            node,
            inputs=[x, labels, weight],
            outputs=[sce],
            name="test_sce_NCd1d2d3d4d5_mean_weight",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob() -> None:
        reduction = "mean"

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z", "log_prob"],
            reduction=reduction,
        )

        N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
        np.random.seed(0)
        x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
        labels = np.random.randint(
            0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)
        ).astype(np.int64)
        weight = np.random.rand(C).astype(np.float32)

        loss, log_prob = softmaxcrossentropy(
            x, labels, weight=weight, reduction=reduction, get_log_prob=True
        )

        expect(
            node,
            inputs=[x, labels, weight],
            outputs=[loss, log_prob],
            name="test_sce_NCd1d2d3d4d5_mean_weight_log_prob",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3d4d5_none_no_weight() -> None:
        reduction = "none"

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
        )

        N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
        np.random.seed(0)
        x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
        labels = np.random.randint(
            0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)
        ).astype(np.int64)

        sce = softmaxcrossentropy(x, labels, reduction=reduction)

        expect(
            node,
            inputs=[x, labels],
            outputs=[sce],
            name="test_sce_NCd1d2d3d4d5_none_no_weight",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob() -> None:
        reduction = "none"

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
        )

        N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
        np.random.seed(0)
        x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
        labels = np.random.randint(
            0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)
        ).astype(np.int64)

        loss, log_prob = softmaxcrossentropy(
            x, labels, reduction=reduction, get_log_prob=True
        )

        expect(
            node,
            inputs=[x, labels],
            outputs=[loss, log_prob],
            name="test_sce_NCd1d2d3d4d5_none_no_weight_log_prob",
        )

    @staticmethod
    def export_input_shape_is_NCd1_mean_weight_negative_ii() -> None:
        reduction = "mean"
        ignore_index = np.int64(-1)

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, dim1 = 3, 5, 6
        np.random.seed(0)
        x = np.random.rand(N, C, dim1).astype(np.float32)
        labels = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
        labels[0][0] = -1
        weight = np.random.rand(C).astype(np.float32)

        sce = softmaxcrossentropy(
            x, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[x, labels, weight],
            outputs=[sce],
            name="test_sce_NCd1_mean_weight_negative_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1_mean_weight_negative_ii_log_prob() -> None:
        reduction = "mean"
        ignore_index = np.int64(-1)

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, dim1 = 3, 5, 6
        np.random.seed(0)
        x = np.random.rand(N, C, dim1).astype(np.float32)
        labels = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
        labels[0][0] = -1
        weight = np.random.rand(C).astype(np.float32)

        loss, log_prob = softmaxcrossentropy(
            x,
            labels,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            get_log_prob=True,
        )

        expect(
            node,
            inputs=[x, labels, weight],
            outputs=[loss, log_prob],
            name="test_sce_NCd1_mean_weight_negative_ii_log_prob",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3_none_no_weight_negative_ii() -> None:
        reduction = "none"
        ignore_index = np.int64(-5)

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
        np.random.seed(0)
        x = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
        labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(
            np.int64
        )
        labels[0][0][0][0] = -5

        sce = softmaxcrossentropy(
            x, labels, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[x, labels],
            outputs=[sce],
            name="test_sce_NCd1d2d3_none_no_weight_negative_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3_none_no_weight_negative_ii_log_prob() -> None:
        reduction = "none"
        ignore_index = np.int64(-5)

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
        np.random.seed(0)
        x = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
        labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(
            np.int64
        )
        labels[0][0][0][0] = -5

        loss, log_prob = softmaxcrossentropy(
            x, labels, reduction=reduction, ignore_index=ignore_index, get_log_prob=True
        )

        expect(
            node,
            inputs=[x, labels],
            outputs=[loss, log_prob],
            name="test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3_sum_weight_high_ii() -> None:
        reduction = "sum"
        ignore_index = np.int64(10)

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C = 3, 5
        np.random.seed(0)
        x = np.random.rand(N, C).astype(np.float32)
        labels = np.random.randint(0, high=C, size=(N)).astype(np.int64)
        labels[0] = 10
        weight = np.random.rand(C).astype(np.float32)

        sce = softmaxcrossentropy(
            x, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
        )

        expect(
            node,
            inputs=[x, labels, weight],
            outputs=[sce],
            name="test_sce_NCd1d2d3_sum_weight_high_ii",
        )

    @staticmethod
    def export_input_shape_is_NCd1d2d3_sum_weight_high_ii_log_prob() -> None:
        reduction = "sum"
        ignore_index = np.int64(10)

        node = onnx.helper.make_node(
            "SoftmaxCrossEntropyLoss",
            inputs=["x", "y", "w"],
            outputs=["z", "log_prob"],
            reduction=reduction,
            ignore_index=ignore_index,
        )

        N, C = 3, 5
        np.random.seed(0)
        x = np.random.rand(N, C).astype(np.float32)
        labels = np.random.randint(0, high=C, size=(N)).astype(np.int64)
        labels[0] = 10
        weight = np.random.rand(C).astype(np.float32)

        loss, log_prob = softmaxcrossentropy(
            x,
            labels,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            get_log_prob=True,
        )

        expect(
            node,
            inputs=[x, labels, weight],
            outputs=[loss, log_prob],
            name="test_sce_NCd1d2d3_sum_weight_high_ii_log_prob",
        )
