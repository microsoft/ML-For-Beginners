# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.op_run import OpRun


def softmaxcrossentropy(  # type: ignore
    x, target, weight=None, reduction="mean", ignore_index=None, get_log_prob=None
):
    input_shape = x.shape
    if len(input_shape) == 1:
        raise RuntimeError(f"Unsupported shape {input_shape!r}.")

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
        gather_weight = np.take(weight, np.array(target, dtype=np.int32), mode="clip")
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(
                dtype=x.dtype
            )
    elif ignore_index is not None:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(dtype=x.dtype)

    # if input is 4-d and above, make it 3-d
    if len(input_shape) != 3:
        inp = inp.reshape((N, C, -1))
        target = target.reshape((N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = inp.shape[2]
    neg_gather_element_input = np.zeros((N, D), dtype=x.dtype)
    for i in range(N):
        for d in range(D):
            if target[i, d] != ignore_index:
                neg_gather_element_input[i, d] = -inp[i, target[i, d], d]

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
            return (loss,)

    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)

    loss = loss.astype(x.dtype)
    if get_log_prob is True:
        return loss, log_prob.astype(x.dtype)  # type: ignore[union-attr]
    return (loss,)


class SoftmaxCrossEntropyLoss(OpRun):
    def _run(self, x, target, weight=None, ignore_index=None, reduction=None):  # type: ignore
        n_outputs = len(self.onnx_node.output)  # type: ignore
        return softmaxcrossentropy(
            x,
            target,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            get_log_prob=n_outputs == 2,
        )
