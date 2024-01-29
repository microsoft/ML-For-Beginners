/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {
static const char* LabelEncoder_ver1_doc = R"DOC(
    Converts strings to integers and vice versa.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.<br>
    Each operator converts either integers to strings or strings to integers, depending
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    When converting from integers to strings, the string is fetched from the
    'classes_strings' list, by simple indexing.<br>
    When converting from strings to integers, the string is looked up in the list
    and the index at which it is found is used as the converted value.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    1,
    OpSchema()
        .SetDoc(LabelEncoder_ver1_doc)
        .Input(0, "X", "Input data.", "T1")
        .Output(0, "Y", "Output data. If strings are input, the output values are integers, and vice versa.", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)", "tensor(int64)"},
            "The input type must be a tensor of integers or strings, of any shape.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output type will be a tensor of strings or integers, and will have the same shape as the input.")
        .Attr("classes_strings", "A list of labels.", AttributeProto::STRINGS, OPTIONAL_VALUE)
        .Attr(
            "default_int64",
            "An integer to use when an input string value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Attr(
            "default_string",
            "A string to use when an input integer value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
            AttributeProto::STRING,
            std::string("_Unused"))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (TensorProto::STRING == input_elem_type) {
            output_elem_type->set_elem_type(TensorProto::INT64);
          } else if (TensorProto::INT64 == input_elem_type) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          }
        }));

static const char* TreeEnsembleClassifier_ver1_doc = R"DOC(
    Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    1,
    OpSchema()
        .SetDoc(TreeEnsembleClassifier_ver1_doc)
        .Input(0, "X", "Input of shape [N,F]", "T1")
        .Output(0, "Y", "N, Top class for each point", "T2")
        .Output(1, "Z", "The class score for each class, for each point, a tensor of shape [N,E].", "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input type must be a tensor of a numeric type.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output type will be a tensor of strings or integers, depending on which of the classlabels_* attributes is used.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Ids may restart at zero for each tree, but it not required to.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the 'true' or 'false' branch based on the value in this array.<br>This attribute may be left undefined, and the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("class_treeids", "The id of the tree that this node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_nodeids", "node id that this weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_ids", "The index of the class list that each weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_weights", "The weight for the class in class_id.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "classlabels_strings",
            "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_int64s",
            "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br> One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "base_values",
            "Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<std::string> label_strs;
          auto result = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
          bool using_strings = (result && !label_strs.empty());
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (using_strings) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          } else {
            output_elem_type->set_elem_type(TensorProto::INT64);
          }
        }));

static const char* TreeEnsembleRegressor_ver1_doc = R"DOC(
    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    1,
    OpSchema()
        .SetDoc(TreeEnsembleRegressor_ver1_doc)
        .Input(0, "X", "Input of shape [N,F]", "T")
        .Output(0, "Y", "N classes", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input type must be a tensor of a numeric type.")
        .Attr("nodes_treeids", "Tree id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_nodeids",
            "Node id for each node. Node ids must restart at zero for each tree and increase sequentially.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("nodes_featureids", "Feature id for each node.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_values",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_modes",
            "The node kind, that is, the comparison to make at the node. There is no comparison to make at a leaf node.<br>One of 'BRANCH_LEQ', 'BRANCH_LT', 'BRANCH_GTE', 'BRANCH_GT', 'BRANCH_EQ', 'BRANCH_NEQ', 'LEAF'",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("nodes_truenodeids", "Child node if expression is true", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("nodes_falsenodeids", "Child node if expression is false", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr(
            "nodes_missing_value_tracks_true",
            "For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) or 'false' (if the attribute value is 0) branch based on the value in this array.<br>This attribute may be left undefined and the default value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("target_treeids", "The id of the tree that each node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_nodeids", "The node id of each weight", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_ids", "The index of the target that each weight is for", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_weights", "The weight for each target", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("n_targets", "The total number of targets.", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "aggregate_function",
            "Defines how to aggregate leaf values within a target. <br>One of 'AVERAGE,' 'SUM,' 'MIN,' 'MAX.'",
            AttributeProto::STRING,
            std::string("SUM"))
        .Attr(
            "base_values",
            "Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE));

static const char* LabelEncoder_ver2_doc = R"DOC(
    Maps each element in the input tensor to another value.<br>
    The mapping is determined by the two parallel attributes, 'keys_*' and
    'values_*' attribute. The i-th value in the specified 'keys_*' attribute
    would be mapped to the i-th value in the specified 'values_*' attribute. It
    implies that input's element type and the element type of the specified
    'keys_*' should be identical while the output type is identical to the
    specified 'values_*' attribute. If an input element can not be found in the
    specified 'keys_*' attribute, the 'default_*' that matches the specified
    'values_*' attribute may be used as its output value.<br>
    Let's consider an example which maps a string tensor to an integer tensor.
    Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
    and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
    "Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
    Since this operator is an one-to-one mapping, its input and output shapes
    are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
    For key look-up, bit-wise comparison is used so even a float NaN can be
    mapped to a value in 'values_*' attribute.<br>
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LabelEncoder,
    2,
    OpSchema()
        .SetDoc(LabelEncoder_ver2_doc)
        .Input(0, "X", "Input data. It can be either tensor or scalar.", "T1")
        .Output(0, "Y", "Output data.", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)", "tensor(int64)", "tensor(float)"},
            "The input type is a tensor of any shape.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)", "tensor(float)"},
            "Output type is determined by the specified 'values_*' attribute.")
        .Attr(
            "keys_strings",
            "A list of strings. One and only one of 'keys_*'s should be set.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("keys_int64s", "A list of ints.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("keys_floats", "A list of floats.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "values_strings",
            "A list of strings. One and only one of 'value_*'s should be set.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("values_int64s", "A list of ints.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("values_floats", "A list of floats.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("default_string", "A string.", AttributeProto::STRING, std::string("_Unused"))
        .Attr("default_int64", "An integer.", AttributeProto::INT, static_cast<int64_t>(-1))
        .Attr("default_float", "A float.", AttributeProto::FLOAT, -0.f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Label encoder is one-to-one mapping.
          if (ctx.getNumInputs() != 1) {
            fail_shape_inference("Label encoder has only one input.");
          }
          if (ctx.getNumOutputs() != 1) {
            fail_shape_inference("Label encoder has only one output.");
          }

          // Load all key_* attributes.
          std::vector<std::string> keys_strings;
          bool keys_strings_result = getRepeatedAttribute(ctx, "keys_strings", keys_strings);
          std::vector<int64_t> keys_int64s;
          bool keys_int64s_result = getRepeatedAttribute(ctx, "keys_int64s", keys_int64s);
          std::vector<float> keys_floats;
          bool keys_floats_result = getRepeatedAttribute(ctx, "keys_floats", keys_floats);

          // Check if only one keys_* attribute is set.
          if (static_cast<int>(keys_strings_result) + static_cast<int>(keys_int64s_result) +
                  static_cast<int>(keys_floats_result) !=
              1) {
            fail_shape_inference("Only one of keys_*'s can be set in label encoder.");
          }

          // Check if the specified keys_* matches input type.
          auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
          if (keys_strings_result && input_elem_type != TensorProto::STRING) {
            fail_shape_inference("Input type is not string tensor but key_strings is set");
          }
          if (keys_int64s_result && input_elem_type != TensorProto::INT64) {
            fail_shape_inference("Input type is not int64 tensor but keys_int64s is set");
          }
          if (keys_floats_result && input_elem_type != TensorProto::FLOAT) {
            fail_shape_inference("Input type is not float tensor but keys_floats is set");
          }

          // Load all values_* attributes.
          std::vector<std::string> values_strings;
          bool values_strings_result = getRepeatedAttribute(ctx, "values_strings", values_strings);
          std::vector<int64_t> values_int64s;
          bool values_int64s_result = getRepeatedAttribute(ctx, "values_int64s", values_int64s);
          std::vector<float> values_floats;
          bool values_floats_result = getRepeatedAttribute(ctx, "values_floats", values_floats);

          // Check if only one values_* attribute is set.
          if (static_cast<int>(values_strings_result) + static_cast<int>(values_int64s_result) +
                  static_cast<int>(values_floats_result) !=
              1) {
            fail_shape_inference("Only one of values_*'s can be set in label encoder.");
          }

          // Assign output type based on the specified values_*.
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (values_strings_result)
            output_elem_type->set_elem_type(TensorProto::STRING);
          if (values_int64s_result)
            output_elem_type->set_elem_type(TensorProto::INT64);
          if (values_floats_result)
            output_elem_type->set_elem_type(TensorProto::FLOAT);

          // Input and output shapes are the same.
          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

} // namespace ONNX_NAMESPACE
#endif
