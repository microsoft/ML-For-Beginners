/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"

#ifdef ONNX_ML
namespace ONNX_NAMESPACE {
static const char* ArrayFeatureExtractor_ver1_doc = R"DOC(
    Select elements of the input tensor based on the indices passed.<br>
    The indices are applied to the last axes of the tensor.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    ArrayFeatureExtractor,
    1,
    OpSchema()
        .SetDoc(ArrayFeatureExtractor_ver1_doc)
        .Input(0, "X", "Data to be selected", "T")
        .Input(1, "Y", "The indices, based on 0 as the first index of any dimension.", "tensor(int64)")
        .Output(0, "Z", "Selected output data as an array", "T")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          if (input_ndim == 1) {
            return;
          }
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          // This operator only applies to the last dimension; thus -1
          for (int i = 0; i < input_ndim - 1; ++i) {
            *output_shape->add_dim() = input_shape.dim(i);
          }

          // value of the output's last dimension is the total amount of indices
          // set Unknown length for the last dimension if it cannot be calculated
          auto last_dim = output_shape->add_dim();
          if (hasInputShape(ctx, 1)) {
            const auto& indices_shape = getInputShape(ctx, 1);
            if (indices_shape.dim_size() > 0) {
              int64_t num_indices = 1;
              std::string single_symbolic_dim;
              for (int i = 0; i < indices_shape.dim_size(); i++) {
                if (indices_shape.dim(i).has_dim_value()) {
                  num_indices *= indices_shape.dim(i).dim_value();
                } else if (indices_shape.dim(i).has_dim_param()) {
                  if (single_symbolic_dim.empty()) {
                    // it is possible to set symbolic dimension param if the rest dim values are all value 1
                    single_symbolic_dim = indices_shape.dim(i).dim_param();
                  } else {
                    return;
                  }
                } else {
                  return;
                }
              }
              if (single_symbolic_dim.empty()) {
                last_dim->set_dim_value(num_indices);
              } else if (num_indices == 1) {
                last_dim->set_dim_param(single_symbolic_dim);
              }
            }
          }
        })
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)", "tensor(string)"},
            "The input must be a tensor of a numeric type or string. The output will be of the same tensor type."));

static const char* Binarizer_ver1_doc = R"DOC(
    Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Binarizer,
    1,
    OpSchema()
        .SetDoc(Binarizer_ver1_doc)
        .Input(0, "X", "Data to be binarized", "T")
        .Output(0, "Y", "Binarized output data", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input must be a tensor of a numeric type. The output will be of the same tensor type.")
        .Attr("threshold", "Values greater than this are mapped to 1, others to 0.", AttributeProto::FLOAT, 0.f)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { propagateShapeAndTypeFromFirstInput(ctx); }));

static const char* CastMap_ver1_doc = R"DOC(
    Converts a map to a tensor.<br>The map key must be an int64 and the values will be ordered
    in ascending order based on this key.<br>The operator supports dense packing or sparse packing.
    If using sparse packing, the key cannot exceed the max_map-1 value.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    CastMap,
    1,
    OpSchema()
        .SetDoc(CastMap_ver1_doc)
        .Input(0, "X", "The input map that is to be cast to a tensor", "T1")
        .Output(0, "Y", "A tensor representing the same data as the input map, ordered by their keys", "T2")
        .TypeConstraint(
            "T1",
            {"map(int64, string)", "map(int64, float)"},
            "The input must be an integer map to either string or float.")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(float)", "tensor(int64)"},
            "The output is a 1-D tensor of string, float, or integer.")
        .Attr(
            "cast_to",
            "A string indicating the desired element type of the output tensor, one of 'TO_FLOAT', 'TO_STRING', 'TO_INT64'.",
            AttributeProto::STRING,
            std::string("TO_FLOAT"))
        .Attr(
            "map_form",
            "Indicates whether to only output as many values as are in the input (dense), or position the input based on using the key of the map as the index of the output (sparse).<br>One of 'DENSE', 'SPARSE'.",
            AttributeProto::STRING,
            std::string("DENSE"))
        .Attr(
            "max_map",
            "If the value of map_form is 'SPARSE,' this attribute indicates the total length of the output tensor.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto cast_to_attr = ctx.getAttribute("cast_to");
          auto output_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (nullptr == cast_to_attr) {
            output_type->set_elem_type(TensorProto::FLOAT);
            return;
          }
          auto& cast_to = cast_to_attr->s();
          if (0 == cast_to.compare("TO_FLOAT")) {
            output_type->set_elem_type(TensorProto::FLOAT);
          } else if (0 == cast_to.compare("TO_INT64")) {
            output_type->set_elem_type(TensorProto::INT64);
          } else if (0 == cast_to.compare("TO_STRING")) {
            output_type->set_elem_type(TensorProto::STRING);
          }
        }));

static const char* CategoryMapper_ver1_doc = R"DOC(
    Converts strings to integers and vice versa.<br>
    Two sequences of equal length are used to map between integers and strings,
    with strings and integers at the same index detailing the mapping.<br>
    Each operator converts either integers to strings or strings to integers, depending
    on which default value attribute is provided. Only one default value attribute
    should be defined.<br>
    If the string default value is set, it will convert integers to strings.
    If the int default value is set, it will convert strings to integers.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    CategoryMapper,
    1,
    OpSchema()
        .SetDoc(CategoryMapper_ver1_doc)
        .Input(0, "X", "Input data", "T1")
        .Output(0, "Y", "Output data. If strings are input, the output values are integers, and vice versa.", "T2")
        .TypeConstraint(
            "T1",
            {"tensor(string)", "tensor(int64)"},
            "The input must be a tensor of strings or integers, either [N,C] or [C].")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output is a tensor of strings or integers. Its shape will be the same as the input shape.")
        .Attr(
            "cats_strings",
            "The strings of the map. This sequence must be the same length as the 'cats_int64s' sequence",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "cats_int64s",
            "The integers of the map. This sequence must be the same length as the 'cats_strings' sequence.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "default_string",
            "A string to use when an input integer value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
            AttributeProto::STRING,
            std::string("_Unused"))
        .Attr(
            "default_int64",
            "An integer to use when an input string value is not found in the map.<br>One and only one of the 'default_*' attributes must be defined.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (nullptr == ctx.getInputType(0))
            return;
          auto input_elem_type = ctx.getInputType(0)->tensor_type().elem_type();
          if (TensorProto::STRING == input_elem_type) {
            updateOutputElemType(ctx, 0, TensorProto::INT64);
          } else if (TensorProto::INT64 == input_elem_type) {
            updateOutputElemType(ctx, 0, TensorProto::STRING);
          }
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

static const char* DictVectorizer_ver1_doc = R"DOC(
    Uses an index mapping to convert a dictionary to an array.<br>
    Given a dictionary, each key is looked up in the vocabulary attribute corresponding to
    the key type. The index into the vocabulary array at which the key is found is then
    used to index the output 1-D tensor 'Y' and insert into it the value found in the dictionary 'X'.<br>
    The key type of the input map must correspond to the element type of the defined vocabulary attribute.
    Therefore, the output array will be equal in length to the index mapping vector parameter.
    All keys in the input dictionary must be present in the index mapping vector.
    For each item in the input dictionary, insert its value in the output array.
    Any keys not present in the input dictionary, will be zero in the output array.<br>
    For example: if the ``string_vocabulary`` parameter is set to ``["a", "c", "b", "z"]``,
    then an input of ``{"a": 4, "c": 8}`` will produce an output of ``[4, 8, 0, 0]``.
    )DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    DictVectorizer,
    1,
    OpSchema()
        .SetDoc(DictVectorizer_ver1_doc)
        .Input(0, "X", "A dictionary.", "T1")
        .Output(0, "Y", "A 1-D tensor holding values from the input dictionary.", "T2")
        .TypeConstraint(
            "T1",
            {"map(string, int64)",
             "map(int64, string)",
             "map(int64, float)",
             "map(int64, double)",
             "map(string, float)",
             "map(string, double)"},
            "The input must be a map from strings or integers to either strings or a numeric type. The key and value types cannot be the same.")
        .TypeConstraint(
            "T2",
            {"tensor(int64)", "tensor(float)", "tensor(double)", "tensor(string)"},
            "The output will be a tensor of the value type of the input map. It's shape will be [1,C], where C is the length of the input dictionary.")
        .Attr(
            "string_vocabulary",
            "A string vocabulary array.<br>One and only one of the vocabularies must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "int64_vocabulary",
            "An integer vocabulary array.<br>One and only one of the vocabularies must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto input_elem_type = ctx.getInputType(0)->map_type().value_type().tensor_type().elem_type();
          auto output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          output_elem_type->set_elem_type(input_elem_type);
        }));

static const char* FeatureVectorizer_ver1_doc = R"DOC(
    Concatenates input tensors into one continuous output.<br>
    All input shapes are 2-D and are concatenated along the second dimention. 1-D tensors are treated as [1,C].
    Inputs are copied to the output maintaining the order of the input arguments.<br>
    All inputs must be integers or floats, while the output will be all floating point values.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    FeatureVectorizer,
    1,
    OpSchema()
        .SetDoc(FeatureVectorizer_ver1_doc)
        .Input(0, "X", "An ordered collection of tensors, all with the same element type.", "T1", OpSchema::Variadic)
        .Output(0, "Y", "The output array, elements ordered as the inputs.", "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(int32)", "tensor(int64)", "tensor(float)", "tensor(double)"},
            "The input type must be a tensor of a numeric type.")
        .Attr("inputdimensions", "The size of each input in the input list", AttributeProto::INTS, OPTIONAL_VALUE));

static const char* Imputer_ver1_doc = R"DOC(
    Replaces inputs that equal one value with another, leaving all other elements alone.<br>
    This operator is typically used to replace missing values in situations where they have a canonical
    representation, such as -1, 0, NaN, or some extreme value.<br>
    One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
    holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
    width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
    which one depends on whether floats or integers are being processed.<br>
    The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Imputer,
    1,
    OpSchema()
        .SetDoc(Imputer_ver1_doc)
        .Input(0, "X", "Data to be processed.", "T")
        .Output(0, "Y", "Imputed output data", "T")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input type must be a tensor of a numeric type, either [N,C] or [C]. The output type will be of the same tensor type and shape.")
        .Attr("imputed_value_floats", "Value(s) to change to", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("replaced_value_float", "A value that needs replacing.", AttributeProto::FLOAT, 0.f)
        .Attr("imputed_value_int64s", "Value(s) to change to.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("replaced_value_int64", "A value that needs replacing.", AttributeProto::INT, static_cast<int64_t>(0)));

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

static const char* LinearClassifier_ver1_doc = R"DOC(
    Linear classifier
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LinearClassifier,
    1,
    OpSchema()
        .SetDoc(LinearClassifier_ver1_doc)
        .Input(0, "X", "Data to be classified.", "T1")
        .Output(0, "Y", "Classification outputs (one class per example).", "T2")
        .Output(1, "Z", "Classification scores ([N,E] - one score for each class and example", "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input must be a tensor of a numeric type, and of shape [N,C] or [C]. In the latter case, it will be treated as [1,C]")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output will be a tensor of strings or integers.")
        .Attr("coefficients", "A collection of weights of the model(s).", AttributeProto::FLOATS)
        .Attr("intercepts", "A collection of intercepts.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "multi_class",
            "Indicates whether to do OvR or multinomial (0=OvR is the default).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "classlabels_strings",
            "Class labels when using string labels. One and only one 'classlabels' attribute must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_ints",
            "Class labels when using integer labels. One and only one 'classlabels' attribute must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the scores vector.<br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<std::string> label_strs;
          std::vector<int64_t> label_ints;

          auto labels_strings_present = getRepeatedAttribute(ctx, "classlabels_strings", label_strs);
          bool using_strings = (labels_strings_present && !label_strs.empty());

          if (!using_strings) {
            getRepeatedAttribute(ctx, "classlabels_ints", label_ints);
          }

          // Type inference
          auto* output_elem_type = ctx.getOutputType(0)->mutable_tensor_type();
          if (using_strings) {
            output_elem_type->set_elem_type(TensorProto::STRING);
          } else {
            output_elem_type->set_elem_type(TensorProto::INT64);
          }

          // second output is always of float type
          ctx.getOutputType(1)->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);

          // Shape/Rank inference begins

          // establish the number of classes
          std::vector<float> intercepts;
          getRepeatedAttribute(ctx, "intercepts", intercepts);
          int class_count = static_cast<int>(intercepts.size());
          if (intercepts.size() == 1 &&
              ((using_strings && label_strs.size() == 2) || (!using_strings && label_ints.size() == 2))) {
            class_count = 2;
          }

          TensorShapeProto_Dimension batch_size_dim, class_count_dim;
          class_count_dim.set_dim_value(class_count);

          if (hasNInputShapes(ctx, 1)) {
            const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            const auto input_rank = input_shape.dim_size();
            if (input_rank == 1) {
              // if input_rank is 1, batch_size is interpreted to be 1
              batch_size_dim.set_dim_value(1);
            } else if (input_rank == 2) {
              batch_size_dim = input_shape.dim((int)0);
            } else {
              fail_shape_inference("Input's shape should be 1D or 2D");
            }
          }

          updateOutputShape(ctx, 0, {batch_size_dim});
          updateOutputShape(ctx, 1, {batch_size_dim, class_count_dim});
        }));

static const char* LinearRegressor_ver1_doc = R"DOC(
    Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    LinearRegressor,
    1,
    OpSchema()
        .SetDoc(LinearRegressor_ver1_doc)
        .Input(0, "X", "Data to be regressed.", "T")
        .Output(0, "Y", "Regression outputs (one per target, per example).", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input must be a tensor of a numeric type.")
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the regression output vector.<br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr("coefficients", "Weights of the model(s).", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("intercepts", "Weights of the intercepts, if used.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "targets",
            "The total number of regression targets, 1 if not defined.",
            AttributeProto::INT,
            static_cast<int64_t>(1)));

static const char* Normalizer_ver1_doc = R"DOC(
    Normalize the input.  There are three normalization modes, which have the corresponding formulas,
    defined using element-wise infix operators '/' and '^' and tensor-wide functions 'max' and 'sum':<br>
<br>
    Max: Y = X / max(X)<br>
    L1:  Y = X / sum(X)<br>
    L2:  Y = sqrt(X^2 / sum(X^2)}<br>
    In all modes, if the divisor is zero, Y == X.
<br>
    For batches, that is, [N,C] tensors, normalization is done along the C axis. In other words, each row
    of the batch is normalized independently.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Normalizer,
    1,
    OpSchema()
        .SetDoc(Normalizer_ver1_doc)
        .Input(0, "X", "Data to be encoded, a tensor of shape [N,C] or [C]", "T")
        .Output(0, "Y", "Encoded output data", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input must be a tensor of a numeric type.")
        .Attr("norm", "One of 'MAX,' 'L1,' 'L2'", AttributeProto::STRING, std::string("MAX")));

static const char* OneHotEncoder_ver1_doc = R"DOC(
    Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count
    will determine the size of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8,
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    This operator assumes every input feature is from the same set of categories.<br>
    If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    OneHotEncoder,
    1,
    OpSchema()
        .SetDoc(OneHotEncoder_ver1_doc)
        .Input(0, "X", "Data to be encoded.", "T")
        .Output(0, "Y", "Encoded output data, having one more dimension than X.", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(string)", "tensor(int64)", "tensor(int32)", "tensor(float)", "tensor(double)"},
            "The input must be a tensor of a numeric type.")
        .Attr(
            "cats_int64s",
            "List of categories, ints.<br>One and only one of the 'cats_*' attributes must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr(
            "cats_strings",
            "List of categories, strings.<br>One and only one of the 'cats_*' attributes must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "zeros",
            "If true and category is not present, will return all zeros; if false and a category if not found, the operator will fail.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<int64_t> cats_int64s;
          bool has_int64s = getRepeatedAttribute(ctx, "cats_int64s", cats_int64s);
          std::vector<std::string> cats_strings;
          bool has_strings = getRepeatedAttribute(ctx, "cats_strings", cats_strings);
          if (has_int64s == has_strings) {
            fail_shape_inference("Exactly one of 'cats_*' attributes must be provided.");
          }
          const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          TensorShapeProto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          for (int i = 0; i < input_shape.dim_size(); i++) {
            *shape->add_dim() = input_shape.dim(i);
          }
          shape->add_dim()->set_dim_value(std::max(cats_int64s.size(), cats_strings.size()));
          updateOutputElemType(ctx, 0, TensorProto::FLOAT);
        }));

static const char* Scaler_ver1_doc = R"DOC(
    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    Scaler,
    1,
    OpSchema()
        .SetDoc(Scaler_ver1_doc)
        .Input(0, "X", "Data to be scaled.", "T")
        .Output(0, "Y", "Scaled output data.", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input must be a tensor of a numeric type.")
        .Attr(
            "offset",
            "First, offset by this.<br>Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "scale",
            "Second, multiply by this.<br>Can be length of features in an [N,F] tensor or length 1, in which case it applies to all features, regardless of dimension count.<br>Must be same length as 'offset'",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE));

static const char* SVMClassifier_ver1_doc = R"DOC(
    Support Vector Machine classifier
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    SVMClassifier,
    1,
    OpSchema()
        .SetDoc(SVMClassifier_ver1_doc)
        .Input(0, "X", "Data to be classified.", "T1")
        .Output(0, "Y", "Classification outputs (one class per example).", "T2")
        .Output(
            1,
            "Z",
            "Class scores (one per class per example), if prob_a and prob_b are provided they are probabilities for each class, otherwise they are raw scores.",
            "tensor(float)")
        .TypeConstraint(
            "T1",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input must be a tensor of a numeric type, either [C] or [N,C].")
        .TypeConstraint(
            "T2",
            {"tensor(string)", "tensor(int64)"},
            "The output type will be a tensor of strings or integers, depending on which of the classlabels_* attributes is used. Its size will match the bactch size of the input.")
        .Attr(
            "kernel_type",
            "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.",
            AttributeProto::STRING,
            std::string("LINEAR"))
        .Attr(
            "kernel_params",
            "List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr("vectors_per_class", "", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("support_vectors", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("coefficients", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("prob_a", "First set of probability coefficients.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "prob_b",
            "Second set of probability coefficients. This array must be same size as prob_a.<br>If these are provided then output Z are probability estimates, otherwise they are raw scores.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr("rho", "", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr(
            "classlabels_strings",
            "Class labels if using string labels.<br>One and only one of the 'classlabels_*' attributes must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_ints",
            "Class labels if using integer labels.<br>One and only one of the 'classlabels_*' attributes must be defined.",
            AttributeProto::INTS,
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

static const char* SVMRegressor_ver1_doc = R"DOC(
    Support Vector Machine regression prediction and one-class SVM anomaly detection.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    SVMRegressor,
    1,
    OpSchema()
        .SetDoc(SVMRegressor_ver1_doc)
        .Input(0, "X", "Data to be regressed.", "T")
        .Output(0, "Y", "Regression outputs (one score per target per example).", "tensor(float)")
        .TypeConstraint(
            "T",
            {"tensor(float)", "tensor(double)", "tensor(int64)", "tensor(int32)"},
            "The input type must be a tensor of a numeric type, either [C] or [N,C].")
        .Attr(
            "kernel_type",
            "The kernel type, one of 'LINEAR,' 'POLY,' 'RBF,' 'SIGMOID'.",
            AttributeProto::STRING,
            std::string("LINEAR"))
        .Attr(
            "kernel_params",
            "List of 3 elements containing gamma, coef0, and degree, in that order. Zero if unused for the kernel.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr("support_vectors", "Chosen support vectors", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "one_class",
            "Flag indicating whether the regression is a one-class SVM or not.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr("coefficients", "Support vector coefficients.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("n_supports", "The number of support vectors.", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr(
            "post_transform",
            "Indicates the transform to apply to the score. <br>One of 'NONE,' 'SOFTMAX,' 'LOGISTIC,' 'SOFTMAX_ZERO,' or 'PROBIT.'",
            AttributeProto::STRING,
            std::string("NONE"))
        .Attr("rho", "", AttributeProto::FLOATS, OPTIONAL_VALUE));

static const char* TreeEnsembleClassifier_ver3_doc = R"DOC(
    Tree Ensemble classifier. Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with 'class_' are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleClassifier,
    3,
    OpSchema()
        .SetDoc(TreeEnsembleClassifier_ver3_doc)
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
            "nodes_values_as_tensor",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates_as_tensor",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::TENSOR,
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
            "For each node, define what to do in the presence of a missing value: if a value is missing (NaN), use the 'true' or 'false' branch based on the value in this array.<br>This attribute may be left undefined, and the defalt value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("class_treeids", "The id of the tree that this node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_nodeids", "node id that this weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_ids", "The index of the class list that each weight is for.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("class_weights", "The weight for the class in class_id.", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr(
            "class_weights_as_tensor",
            "The weight for the class in class_id.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
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
        .Attr(
            "base_values_as_tensor",
            "Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* nodes_values = ctx.getAttribute("nodes_values");
          auto* nodes_values_as_tensor = ctx.getAttribute("nodes_values_as_tensor");
          auto* nodes_hitrates = ctx.getAttribute("nodes_hitrates");
          auto* nodes_hitrates_as_tensor = ctx.getAttribute("nodes_hitrates_as_tensor");
          auto* class_weights = ctx.getAttribute("class_weights");
          auto* class_weights_as_tensor = ctx.getAttribute("class_weights_as_tensor");
          auto* base_values = ctx.getAttribute("base_values");
          auto* base_values_as_tensor = ctx.getAttribute("base_values_as_tensor");

          if (nullptr != nodes_values && nullptr != nodes_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_values', 'nodes_values_as_tensor' should be specified.");
          }
          if (nullptr != nodes_hitrates && nullptr != nodes_hitrates_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_hitrates', 'nodes_hitrates_as_tensor' should be specified.");
          }
          if (nullptr != class_weights && nullptr != class_weights_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'class_weights', 'class_weights_as_tensor' should be specified.");
          }
          if (nullptr != base_values && nullptr != base_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'base_values', 'base_values_as_tensor' should be specified.");
          }

          std::vector<std::string> classlabels_strings;
          auto result = getRepeatedAttribute(ctx, "classlabels_strings", classlabels_strings);
          bool using_strings = (result && !classlabels_strings.empty());
          if (using_strings) {
            updateOutputElemType(ctx, 0, TensorProto::STRING);
          } else {
            updateOutputElemType(ctx, 0, TensorProto::INT64);
          }
          updateOutputElemType(ctx, 1, TensorProto::FLOAT);

          checkInputRank(ctx, 0, 2);
          Dim N, E;
          unifyInputDim(ctx, 0, 0, N);

          if (using_strings) {
            unifyDim(E, classlabels_strings.size());
          } else {
            std::vector<int64_t> classlabels_int64s;
            result = getRepeatedAttribute(ctx, "classlabels_int64s", classlabels_int64s);
            if (!result || classlabels_int64s.empty()) {
              fail_shape_inference("Non of classlabels_int64s or classlabels_strings is set.");
            }
            unifyDim(E, classlabels_int64s.size());
          }
          updateOutputShape(ctx, 0, {N});
          updateOutputShape(ctx, 1, {N, E});
        }));

static const char* TreeEnsembleRegressor_ver3_doc = R"DOC(
    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    TreeEnsembleRegressor,
    3,
    OpSchema()
        .SetDoc(TreeEnsembleRegressor_ver3_doc)
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
            "nodes_values_as_tensor",
            "Thresholds to do the splitting on for each node.",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::FLOATS,
            OPTIONAL_VALUE)
        .Attr(
            "nodes_hitrates_as_tensor",
            "Popularity of each node, used for performance and may be omitted.",
            AttributeProto::TENSOR,
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
            "For each node, define what to do in the presence of a NaN: use the 'true' (if the attribute value is 1) or 'false' (if the attribute value is 0) branch based on the value in this array.<br>This attribute may be left undefined and the defalt value is false (0) for all nodes.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("target_treeids", "The id of the tree that each node is in.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_nodeids", "The node id of each weight", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_ids", "The index of the target that each weight is for", AttributeProto::INTS, OPTIONAL_VALUE)
        .Attr("target_weights", "The weight for each target", AttributeProto::FLOATS, OPTIONAL_VALUE)
        .Attr("target_weights_as_tensor", "The weight for each target", AttributeProto::TENSOR, OPTIONAL_VALUE)
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
            OPTIONAL_VALUE)
        .Attr(
            "base_values_as_tensor",
            "Base values for classification, added to final class score; the size must be the same as the classes or can be left unassigned (assumed 0)",
            AttributeProto::TENSOR,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto* nodes_values = ctx.getAttribute("nodes_values");
          auto* nodes_values_as_tensor = ctx.getAttribute("nodes_values_as_tensor");
          auto* nodes_hitrates = ctx.getAttribute("nodes_hitrates");
          auto* nodes_hitrates_as_tensor = ctx.getAttribute("nodes_hitrates_as_tensor");
          auto* target_weights = ctx.getAttribute("target_weights");
          auto* target_weights_as_tensor = ctx.getAttribute("target_weights_as_tensor");
          auto* base_values = ctx.getAttribute("base_values");
          auto* base_values_as_tensor = ctx.getAttribute("base_values_as_tensor");

          if (nullptr != nodes_values && nullptr != nodes_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_values', 'nodes_values_as_tensor' should be specified.");
          }
          if (nullptr != nodes_hitrates && nullptr != nodes_hitrates_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'nodes_hitrates', 'nodes_hitrates_as_tensor' should be specified.");
          }
          if (nullptr != target_weights && nullptr != target_weights_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'target_weights', 'target_weights_as_tensor' should be specified.");
          }
          if (nullptr != base_values && nullptr != base_values_as_tensor) {
            fail_shape_inference(
                "Only one of the attributes 'base_values', 'base_values_as_tensor' should be specified.");
          }

          checkInputRank(ctx, 0, 2);
          Dim N, E;
          unifyInputDim(ctx, 0, 0, N);
          if (nullptr != ctx.getAttribute("n_targets")) {
            unifyDim(E, ctx.getAttribute("n_targets")->i());
          }
          updateOutputElemType(ctx, 0, TensorProto::FLOAT);
          updateOutputShape(ctx, 0, {N, E});
        }));

static const char* ZipMap_ver1_doc = R"DOC(
    Creates a map from the input and the attributes.<br>
    The values are provided by the input tensor, while the keys are specified by the attributes.
    Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
    The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
)DOC";

ONNX_ML_OPERATOR_SET_SCHEMA(
    ZipMap,
    1,
    OpSchema()
        .SetDoc(ZipMap_ver1_doc)
        .Input(0, "X", "The input values", "tensor(float)")
        .Output(0, "Z", "The output map", "T")
        .TypeConstraint(
            "T",
            {"seq(map(string, float))", "seq(map(int64, float))"},
            "The output will be a sequence of string or integer maps to float.")
        .Attr(
            "classlabels_strings",
            "The keys when using string keys.<br>One and only one of the 'classlabels_*' attributes must be defined.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "classlabels_int64s",
            "The keys when using int keys.<br>One and only one of the 'classlabels_*' attributes must be defined.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          std::vector<std::string> classlabels_strings;
          bool result = getRepeatedAttribute(ctx, "classlabels_strings", classlabels_strings);
          auto output_map_type = ctx.getOutputType(0)->mutable_sequence_type()->mutable_elem_type()->mutable_map_type();
          output_map_type->mutable_value_type()->mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
          if (result && !classlabels_strings.empty()) {
            output_map_type->set_key_type(TensorProto::STRING);
          }
          std::vector<int64_t> classlabels_int64s;
          result = getRepeatedAttribute(ctx, "classlabels_int64s", classlabels_int64s);
          if (result && !classlabels_int64s.empty()) {
            output_map_type->set_key_type(TensorProto::INT64);
          }
        }));

} // namespace ONNX_NAMESPACE
#endif
