# SPDX-License-Identifier: Apache-2.0

"""
Common functions to convert any learner based on trees.
"""
import numpy as np


def get_default_tree_classifier_attribute_pairs():
    attrs = {}
    attrs["post_transform"] = "NONE"
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["nodes_missing_value_tracks_true"] = []
    attrs["nodes_hitrates"] = []
    attrs["class_treeids"] = []
    attrs["class_nodeids"] = []
    attrs["class_ids"] = []
    attrs["class_weights"] = []
    return attrs


def get_default_tree_regressor_attribute_pairs():
    attrs = {}
    attrs["post_transform"] = "NONE"
    attrs["n_targets"] = 0
    attrs["nodes_treeids"] = []
    attrs["nodes_nodeids"] = []
    attrs["nodes_featureids"] = []
    attrs["nodes_modes"] = []
    attrs["nodes_values"] = []
    attrs["nodes_truenodeids"] = []
    attrs["nodes_falsenodeids"] = []
    attrs["nodes_missing_value_tracks_true"] = []
    attrs["nodes_hitrates"] = []
    attrs["target_treeids"] = []
    attrs["target_nodeids"] = []
    attrs["target_ids"] = []
    attrs["target_weights"] = []
    return attrs


def find_switch_point(fy, nfy):
    """
    Finds the double so that
    ``(float)x != (float)(x + espilon)``.
    """
    a = np.float64(fy)
    b = np.float64(nfy)
    fa = np.float32(a)
    a0, b0 = a, a
    while a != a0 or b != b0:
        a0, b0 = a, b
        m = (a + b) / 2
        fm = np.float32(m)
        if fm == fa:
            a = m
            fa = fm
        else:
            b = m
    return a


def sklearn_threshold(dy, dtype, mode):
    """
    *scikit-learn* does not compare x to a threshold
    but (float)x to a double threshold. As we need a float
    threshold, we need a different value than the threshold
    rounded to float. For floats, it finds float *w* which
    verifies::

        (float)x <= y <=> (float)x <= w

    For doubles, it finds double *w* which verifies::

        (float)x <= y <=> x <= w
    """
    if mode == "BRANCH_LEQ":
        if dtype == np.float32:
            fy = np.float32(dy)
            if fy == dy:
                return np.float64(fy)
            if fy < dy:
                return np.float64(fy)
            eps = max(abs(fy), np.finfo(np.float32).eps) * 10
            nfy = np.nextafter([fy], [fy - eps], dtype=np.float32)[0]
            return np.float64(nfy)
        elif dtype == np.float64:
            fy = np.float32(dy)
            eps = max(abs(fy), np.finfo(np.float32).eps) * 10
            afy = np.nextafter([fy], [fy - eps], dtype=np.float32)[0]
            afy2 = find_switch_point(afy, fy)
            if fy > dy > afy2:
                return afy2
            bfy = np.nextafter([fy], [fy + eps], dtype=np.float32)[0]
            bfy2 = find_switch_point(fy, bfy)
            if fy <= dy <= bfy2:
                return bfy2
            return np.float64(fy)
        raise TypeError("Unexpected dtype {}.".format(dtype))
    raise RuntimeError(
        "Threshold is not changed for other mode and "
        "'BRANCH_LEQ' (actually '{}').".format(mode)
    )


def add_node(
    attr_pairs,
    is_classifier,
    tree_id,
    tree_weight,
    node_id,
    feature_id,
    mode,
    value,
    true_child_id,
    false_child_id,
    weights,
    weight_id_bias,
    leaf_weights_are_counts,
    adjust_threshold_for_sklearn,
    dtype,
    nodes_missing_value_tracks_true=False,
):
    attr_pairs["nodes_treeids"].append(tree_id)
    attr_pairs["nodes_nodeids"].append(node_id)
    attr_pairs["nodes_featureids"].append(feature_id)
    attr_pairs["nodes_modes"].append(mode)
    if adjust_threshold_for_sklearn and mode != "LEAF":
        attr_pairs["nodes_values"].append(sklearn_threshold(value, dtype, mode))
    else:
        attr_pairs["nodes_values"].append(value)
    attr_pairs["nodes_truenodeids"].append(true_child_id)
    attr_pairs["nodes_falsenodeids"].append(false_child_id)
    attr_pairs["nodes_missing_value_tracks_true"].append(
        nodes_missing_value_tracks_true
    )
    attr_pairs["nodes_hitrates"].append(1.0)

    # Add leaf information for making prediction
    if mode == "LEAF":
        flattened_weights = weights.flatten()
        factor = tree_weight
        # If the values stored at leaves are counts of possible classes, we
        # need convert them to probabilities by doing a normalization.
        if leaf_weights_are_counts:
            s = sum(flattened_weights)
            factor /= float(s) if s != 0.0 else 1.0
        flattened_weights = [w * factor for w in flattened_weights]
        if len(flattened_weights) == 2 and is_classifier:
            flattened_weights = [flattened_weights[1]]

        # Note that attribute names for making prediction are different for
        # classifiers and regressors
        if is_classifier:
            for i, w in enumerate(flattened_weights):
                attr_pairs["class_treeids"].append(tree_id)
                attr_pairs["class_nodeids"].append(node_id)
                attr_pairs["class_ids"].append(i + weight_id_bias)
                attr_pairs["class_weights"].append(w)
        else:
            for i, w in enumerate(flattened_weights):
                attr_pairs["target_treeids"].append(tree_id)
                attr_pairs["target_nodeids"].append(node_id)
                attr_pairs["target_ids"].append(i + weight_id_bias)
                attr_pairs["target_weights"].append(w)


def add_tree_to_attribute_pairs(
    attr_pairs,
    is_classifier,
    tree,
    tree_id,
    tree_weight,
    weight_id_bias,
    leaf_weights_are_counts,
    adjust_threshold_for_sklearn=False,
    dtype=None,
):
    for i in range(tree.node_count):
        node_id = i
        weight = tree.value[i]

        if tree.children_left[i] > i or tree.children_right[i] > i:
            mode = "BRANCH_LEQ"
            feat_id = tree.feature[i]
            threshold = tree.threshold[i]
            left_child_id = int(tree.children_left[i])
            right_child_id = int(tree.children_right[i])
        else:
            mode = "LEAF"
            feat_id = 0
            threshold = 0.0
            left_child_id = 0
            right_child_id = 0

        add_node(
            attr_pairs,
            is_classifier,
            tree_id,
            tree_weight,
            node_id,
            feat_id,
            mode,
            threshold,
            left_child_id,
            right_child_id,
            weight,
            weight_id_bias,
            leaf_weights_are_counts,
            adjust_threshold_for_sklearn=adjust_threshold_for_sklearn,
            dtype=dtype,
        )


def add_tree_to_attribute_pairs_hist_gradient_boosting(
    attr_pairs,
    is_classifier,
    tree,
    tree_id,
    tree_weight,
    weight_id_bias,
    leaf_weights_are_counts,
    adjust_threshold_for_sklearn=False,
    dtype=None,
):
    for i, node in enumerate(tree.nodes):
        node_id = i
        weight = node["value"]

        if node["is_leaf"]:
            mode = "LEAF"
            feat_id = 0
            threshold = 0.0
            left_child_id = 0
            right_child_id = 0
            missing = False
        else:
            mode = "BRANCH_LEQ"
            feat_id = node["feature_idx"]
            try:
                threshold = node["threshold"]
            except ValueError:
                threshold = node["num_threshold"]
            left_child_id = node["left"]
            right_child_id = node["right"]
            missing = node["missing_go_to_left"]

        add_node(
            attr_pairs,
            is_classifier,
            tree_id,
            tree_weight,
            node_id,
            feat_id,
            mode,
            threshold,
            left_child_id,
            right_child_id,
            weight,
            weight_id_bias,
            leaf_weights_are_counts,
            adjust_threshold_for_sklearn=adjust_threshold_for_sklearn,
            dtype=dtype,
            nodes_missing_value_tracks_true=missing,
        )
