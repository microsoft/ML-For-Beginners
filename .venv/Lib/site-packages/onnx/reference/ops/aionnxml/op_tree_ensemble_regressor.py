# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0912,R0913,R0914,W0221

import numpy as np

from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_tree_ensemble_helper import TreeEnsemble


class TreeEnsembleRegressor(OpRunAiOnnxMl):
    """
    `nodes_hitrates` and `nodes_hitrates_as_tensor` are not used.
    """

    def _run(  # type: ignore
        self,
        X,
        aggregate_function=None,
        base_values=None,
        base_values_as_tensor=None,
        n_targets=None,
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
        target_ids=None,
        target_nodeids=None,
        target_treeids=None,
        target_weights=None,
        target_weights_as_tensor=None,
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
            target_weights=target_weights,
            target_weights_as_tensor=target_weights_as_tensor,
        )
        # unused unless for debugging purposes
        self._tree = tr  # pylint: disable=W0201
        leaves_index = tr.leave_index_tree(X)
        res = np.zeros((leaves_index.shape[0], n_targets), dtype=X.dtype)
        n_trees = len(set(tr.atts.nodes_treeids))  # type: ignore

        target_index = {}  # type: ignore
        for i, (tid, nid) in enumerate(zip(target_treeids, target_nodeids)):
            if (tid, nid) not in target_index:
                target_index[tid, nid] = []
            target_index[tid, nid].append(i)
        for i in range(res.shape[0]):
            indices = leaves_index[i]
            t_index = [
                target_index[nodes_treeids[i], nodes_nodeids[i]] for i in indices
            ]
            if aggregate_function in ("SUM", "AVERAGE"):
                for its in t_index:
                    for it in its:
                        res[i, target_ids[it]] += tr.atts.target_weights[it]  # type: ignore
            elif aggregate_function == "MIN":
                res[i, :] = np.finfo(res.dtype).max
                for its in t_index:
                    for it in its:
                        res[i, target_ids[it]] = min(
                            res[i, target_ids[it]], tr.atts.target_weights[it]  # type: ignore
                        )
            elif aggregate_function == "MAX":
                res[i, :] = np.finfo(res.dtype).min
                for its in t_index:
                    for it in its:
                        res[i, target_ids[it]] = max(
                            res[i, target_ids[it]], tr.atts.target_weights[it]  # type: ignore
                        )
            else:
                raise NotImplementedError(
                    f"aggregate_transform={aggregate_function!r} " f"not supported yet."
                )
        if aggregate_function == "AVERAGE":
            res /= n_trees
        if base_values is not None:
            res[:, :] = np.array(base_values).reshape((1, -1))

        if post_transform in (None, "NONE"):
            return (res,)
        raise NotImplementedError(f"post_transform={post_transform!r} not implemented.")
