# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,R0914,W0221

import numpy as np

from onnx.reference.ops.aionnxml._common_classifier import (
    compute_probit,
    compute_softmax_zero,
    expit,
)
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl


class LinearClassifier(OpRunAiOnnxMl):
    @staticmethod
    def _post_process_predicted_label(label, scores, classlabels_ints_string):  # type: ignore
        """
        Replaces int64 predicted labels by the corresponding
        strings.
        """
        if classlabels_ints_string is not None:
            label = np.array([classlabels_ints_string[i] for i in label])
        return label, scores

    def _run(  # type: ignore
        self,
        x,
        classlabels_ints=None,
        classlabels_strings=None,
        coefficients=None,
        intercepts=None,
        multi_class=None,  # pylint: disable=W0613
        post_transform=None,
    ):
        # multi_class is unused
        dtype = x.dtype
        if dtype != np.float64:
            x = x.astype(np.float32)
        coefficients = np.array(coefficients).astype(x.dtype)
        intercepts = np.array(intercepts).astype(x.dtype)
        coefficients = coefficients.reshape((-1, x.shape[1])).T
        scores = np.dot(x, coefficients)
        if intercepts is not None:
            scores += intercepts

        n_classes = max(len(classlabels_ints or []), len(classlabels_strings or []))
        if coefficients.shape[1] == 1 and n_classes == 2:
            new_scores = np.empty((scores.shape[0], 2), dtype=np.float32)
            new_scores[:, 0] = -scores[:, 0]
            new_scores[:, 1] = scores[:, 0]
            scores = new_scores

        if post_transform == "NONE":
            pass
        elif post_transform == "LOGISTIC":
            scores = expit(scores)
        elif post_transform == "SOFTMAX":
            np.subtract(
                scores,
                scores.max(axis=1, keepdims=1),  # pylint: disable=E1123
                out=scores,
            )
            scores = np.exp(scores)
            scores = np.divide(scores, scores.sum(axis=1, keepdims=1))
        elif post_transform == "SOFTMAX_ZERO":
            for i in range(scores.shape[0]):
                scores[i, :] = compute_softmax_zero(scores[i, :])
        elif post_transform == "PROBIT":
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    scores[i, j] = compute_probit(scores[i, j])
        else:
            raise NotImplementedError("Unknown post_transform: '{post_transform}'.")

        if scores.shape[1] > 1:
            labels = np.argmax(scores, axis=1)
            if classlabels_ints is not None:
                labels = np.array([classlabels_ints[i] for i in labels], dtype=np.int64)
            elif classlabels_strings is not None:
                labels = np.array([classlabels_strings[i] for i in labels])
        else:
            threshold = 0 if post_transform == "NONE" else 0.5
            if classlabels_ints is not None:
                labels = (
                    np.where(scores >= threshold, classlabels_ints[0], 0)
                    .astype(np.int64)
                    .ravel()
                )
            elif classlabels_strings is not None:
                labels = (
                    np.where(scores >= threshold, classlabels_strings[0], "")
                    .astype(np.int64)
                    .ravel()
                )
            else:
                labels = (scores >= threshold).astype(np.int64).ravel()
        return (labels, scores)
