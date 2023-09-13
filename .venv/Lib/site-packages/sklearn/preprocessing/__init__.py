"""
The :mod:`sklearn.preprocessing` module includes scaling, centering,
normalization, binarization methods.
"""

from ._data import (
    Binarizer,
    KernelCenterer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    add_dummy_feature,
    binarize,
    maxabs_scale,
    minmax_scale,
    normalize,
    power_transform,
    quantile_transform,
    robust_scale,
    scale,
)
from ._discretization import KBinsDiscretizer
from ._encoders import OneHotEncoder, OrdinalEncoder
from ._function_transformer import FunctionTransformer
from ._label import LabelBinarizer, LabelEncoder, MultiLabelBinarizer, label_binarize
from ._polynomial import PolynomialFeatures, SplineTransformer
from ._target_encoder import TargetEncoder

__all__ = [
    "Binarizer",
    "FunctionTransformer",
    "KBinsDiscretizer",
    "KernelCenterer",
    "LabelBinarizer",
    "LabelEncoder",
    "MultiLabelBinarizer",
    "MinMaxScaler",
    "MaxAbsScaler",
    "QuantileTransformer",
    "Normalizer",
    "OneHotEncoder",
    "OrdinalEncoder",
    "PowerTransformer",
    "RobustScaler",
    "SplineTransformer",
    "StandardScaler",
    "TargetEncoder",
    "add_dummy_feature",
    "PolynomialFeatures",
    "binarize",
    "normalize",
    "scale",
    "robust_scale",
    "maxabs_scale",
    "minmax_scale",
    "label_binarize",
    "quantile_transform",
    "power_transform",
]
