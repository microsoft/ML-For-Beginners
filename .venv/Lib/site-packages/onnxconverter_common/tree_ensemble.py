# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
Common functions to convert any learner based on trees.
"""

from .registration import register_converter  # noqa F401


def get_default_tree_classifier_attribute_pairs():
    attrs = {}
    attrs['post_transform'] = 'NONE'
    attrs['nodes_treeids'] = []
    attrs['nodes_nodeids'] = []
    attrs['nodes_featureids'] = []
    attrs['nodes_modes'] = []
    attrs['nodes_values'] = []
    attrs['nodes_truenodeids'] = []
    attrs['nodes_falsenodeids'] = []
    attrs['nodes_missing_value_tracks_true'] = []
    attrs['nodes_hitrates'] = []
    attrs['class_treeids'] = []
    attrs['class_nodeids'] = []
    attrs['class_ids'] = []
    attrs['class_weights'] = []
    return attrs


def get_default_tree_regressor_attribute_pairs():
    attrs = {}
    attrs['post_transform'] = 'NONE'
    attrs['n_targets'] = 0
    attrs['nodes_treeids'] = []
    attrs['nodes_nodeids'] = []
    attrs['nodes_featureids'] = []
    attrs['nodes_modes'] = []
    attrs['nodes_values'] = []
    attrs['nodes_truenodeids'] = []
    attrs['nodes_falsenodeids'] = []
    attrs['nodes_missing_value_tracks_true'] = []
    attrs['nodes_hitrates'] = []
    attrs['target_treeids'] = []
    attrs['target_nodeids'] = []
    attrs['target_ids'] = []
    attrs['target_weights'] = []
    return attrs


def add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id, feature_id, mode, value, true_child_id,
             false_child_id, weights, weight_id_bias, leaf_weights_are_counts):
    attr_pairs['nodes_treeids'].append(tree_id)
    attr_pairs['nodes_nodeids'].append(node_id)
    attr_pairs['nodes_featureids'].append(feature_id)
    attr_pairs['nodes_modes'].append(mode)
    attr_pairs['nodes_values'].append(value)
    attr_pairs['nodes_truenodeids'].append(true_child_id)
    attr_pairs['nodes_falsenodeids'].append(false_child_id)
    attr_pairs['nodes_missing_value_tracks_true'].append(False)
    attr_pairs['nodes_hitrates'].append(1.)

    # Add leaf information for making prediction
    if mode == 'LEAF':
        flattened_weights = weights.flatten()
        factor = tree_weight
        # If the values stored at leaves are counts of possible classes, we need convert them to probabilities by
        # doing a normalization.
        if leaf_weights_are_counts:
            s = sum(flattened_weights)
            factor /= float(s) if s != 0. else 1.
        flattened_weights = [w * factor for w in flattened_weights]
        if len(flattened_weights) == 2 and is_classifier:
            flattened_weights = [flattened_weights[1]]

        # Note that attribute names for making prediction are different for classifiers and regressors
        if is_classifier:
            for i, w in enumerate(flattened_weights):
                attr_pairs['class_treeids'].append(tree_id)
                attr_pairs['class_nodeids'].append(node_id)
                attr_pairs['class_ids'].append(i + weight_id_bias)
                attr_pairs['class_weights'].append(w)
        else:
            for i, w in enumerate(flattened_weights):
                attr_pairs['target_treeids'].append(tree_id)
                attr_pairs['target_nodeids'].append(node_id)
                attr_pairs['target_ids'].append(i + weight_id_bias)
                attr_pairs['target_weights'].append(w)


def add_tree_to_attribute_pairs(attr_pairs, is_classifier, tree, tree_id, tree_weight,
                                weight_id_bias, leaf_weights_are_counts):
    for i in range(tree.node_count):
        node_id = i
        weight = tree.value[i]

        if tree.children_left[i] > i or tree.children_right[i] > i:
            mode = 'BRANCH_LEQ'
            feat_id = tree.feature[i]
            threshold = tree.threshold[i]
            left_child_id = int(tree.children_left[i])
            right_child_id = int(tree.children_right[i])
        else:
            mode = 'LEAF'
            feat_id = 0
            threshold = 0.
            left_child_id = 0
            right_child_id = 0

        add_node(attr_pairs, is_classifier, tree_id, tree_weight, node_id, feat_id, mode, threshold,
                 left_child_id, right_child_id, weight, weight_id_bias, leaf_weights_are_counts)
