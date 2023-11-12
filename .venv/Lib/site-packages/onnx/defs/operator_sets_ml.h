/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef ONNX_ML

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Forward declarations for ai.onnx.ml version 1
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, ArrayFeatureExtractor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, Binarizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, CastMap);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, CategoryMapper);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, DictVectorizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, FeatureVectorizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, Imputer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, LabelEncoder);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, LinearClassifier);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, LinearRegressor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, Normalizer);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, OneHotEncoder);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, SVMClassifier);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, SVMRegressor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, Scaler);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, TreeEnsembleClassifier);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, TreeEnsembleRegressor);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, ZipMap);

// Iterate over schema from ai.onnx.ml version 1
class OpSet_OnnxML_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, ArrayFeatureExtractor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, Binarizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, CastMap)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, CategoryMapper)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, DictVectorizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, FeatureVectorizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, Imputer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, LabelEncoder)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, LinearClassifier)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, LinearRegressor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, Normalizer)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, OneHotEncoder)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, SVMClassifier)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, SVMRegressor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, Scaler)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, TreeEnsembleClassifier)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, TreeEnsembleRegressor)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 1, ZipMap)>());
  }
};

class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 2, LabelEncoder);

class OpSet_OnnxML_ver2 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 2, LabelEncoder)>());
  }
};

class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 3, TreeEnsembleClassifier);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 3, TreeEnsembleRegressor);

class OpSet_OnnxML_ver3 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 3, TreeEnsembleClassifier)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxML, 3, TreeEnsembleRegressor)>());
  }
};

inline void RegisterOnnxMLOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_OnnxML_ver1>();
  RegisterOpSetSchema<OpSet_OnnxML_ver2>();
  RegisterOpSetSchema<OpSet_OnnxML_ver3>();
}
} // namespace ONNX_NAMESPACE

#endif
