# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


import numpy as np

from onnx.reference.ops.aionnxml._common_classifier import (
    logistic,
    probit,
    softmax,
    softmax_zero,
)
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_tree_ensemble_helper import TreeEnsemble


class TreeEnsembleClassifier(OpRunAiOnnxMl):
    def _run(  # type: ignore
        self,
        X,
        base_values=None,
        base_values_as_tensor=None,
        class_ids=None,
        class_nodeids=None,
        class_treeids=None,
        class_weights=None,
        class_weights_as_tensor=None,
        classlabels_int64s=None,
        classlabels_strings=None,
        nodes_falsenodeids=None,
        nodes_featureids=None,
        nodes_hitrates=None,
        nodes_hitrates_as_tensor=None,
        nodes_missing_value_tracks_true=None,
        nodes_modes=None,
        nodes_nodeids=None,
        nodes_treeids=None,
        nodes_truenodeids=None,
        nodes_values=None,
        nodes_values_as_tensor=None,
        post_transform=None,
    ):
        nmv = nodes_missing_value_tracks_true
        tr = TreeEnsemble(
            base_values=base_values,
            base_values_as_tensor=base_values_as_tensor,
            nodes_falsenodeids=nodes_falsenodeids,
            nodes_featureids=nodes_featureids,
            nodes_hitrates=nodes_hitrates,
            nodes_hitrates_as_tensor=nodes_hitrates_as_tensor,
            nodes_missing_value_tracks_true=nmv,
            nodes_modes=nodes_modes,
            nodes_nodeids=nodes_nodeids,
            nodes_treeids=nodes_treeids,
            nodes_truenodeids=nodes_truenodeids,
            nodes_values=nodes_values,
            nodes_values_as_tensor=nodes_values_as_tensor,
            class_weights=class_weights,
            class_weights_as_tensor=class_weights_as_tensor,
        )
        # unused unless for debugging purposes
        self._tree = tr
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float32)
        leaves_index = tr.leave_index_tree(X)
        n_classes = max(len(classlabels_int64s or []), len(classlabels_strings or []))
        res = np.empty((leaves_index.shape[0], n_classes), dtype=np.float32)
        if tr.atts.base_values is None:  # type: ignore
            res[:, :] = 0
        else:
            res[:, :] = np.array(tr.atts.base_values).reshape((1, -1))  # type: ignore

        class_index = {}  # type: ignore
        for i, (tid, nid) in enumerate(zip(class_treeids, class_nodeids)):
            if (tid, nid) not in class_index:
                class_index[tid, nid] = []
            class_index[tid, nid].append(i)
        for i in range(res.shape[0]):
            indices = leaves_index[i]
            t_index = [class_index[nodes_treeids[i], nodes_nodeids[i]] for i in indices]
            for its in t_index:
                for it in its:
                    res[i, class_ids[it]] += tr.atts.class_weights[it]  # type: ignore

        # post_transform
        binary = len(set(class_ids)) == 1
        classes = classlabels_int64s or classlabels_strings
        post_function = {
            None: lambda x: x,
            "NONE": lambda x: x,
            "LOGISTIC": logistic,
            "SOFTMAX": softmax,
            "SOFTMAX_ZERO": softmax_zero,
            "PROBIT": probit,
        }
        if binary:
            if res.shape[1] == len(classes) == 1:
                new_res = np.zeros((res.shape[0], 2), res.dtype)
                new_res[:, 1] = res[:, 0]
                res = new_res
            else:
                res[:, 1] = res[:, 0]
            if post_transform in (None, "NONE", "PROBIT"):
                res[:, 0] = 1 - res[:, 1]
            else:
                res[:, 0] = -res[:, 1]
        new_scores = post_function[post_transform](res)  # type: ignore
        labels = np.argmax(new_scores, axis=1)

        # labels
        if classlabels_int64s is not None:
            if len(classlabels_int64s) == 1:
                if classlabels_int64s[0] == 1:
                    d = {1: 1}
                    labels = np.array([d.get(i, 0) for i in labels], dtype=np.int64)
                else:
                    raise NotImplementedError(
                        f"classlabels_int64s={classlabels_int64s}, not supported."
                    )
            else:
                labels = np.array(
                    [classlabels_int64s[i] for i in labels], dtype=np.int64
                )
        elif classlabels_strings is not None:
            if len(classlabels_strings) == 1:
                raise NotImplementedError(
                    f"classlabels_strings={classlabels_strings}, not supported."
                )
            labels = np.array([classlabels_strings[i] for i in labels])

        return labels, new_scores
