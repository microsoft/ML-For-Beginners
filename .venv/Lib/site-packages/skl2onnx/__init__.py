# SPDX-License-Identifier: Apache-2.0

"""
Main entry point to the converter from the *scikit-learn* to *onnx*.
"""
__version__ = "1.16.0"
__author__ = "Microsoft"
__producer__ = "skl2onnx"
__producer_version__ = __version__
__domain__ = "ai.onnx"
__model_version__ = 0
__max_supported_opset__ = 19  # Converters are tested up to this version.


from .convert import convert_sklearn, to_onnx, wrap_as_onnx_mixin  # noqa
from ._supported_operators import update_registered_converter, get_model_alias  # noqa
from ._parse import update_registered_parser  # noqa
from .proto import get_latest_tested_opset_version  # noqa


def supported_converters(from_sklearn=False):
    """
    Returns the list of supported converters.
    To find the converter associated to a specific model,
    the library gets the name of the model class,
    adds ``'Sklearn'`` as a prefix and retrieves
    the associated converter if available.

    :param from_sklearn: every supported model is mapped to converter
        by a name prefixed with ``'Sklearn'``, the prefix is removed
        if this parameter is False but the function only returns converters
        whose name is prefixed by ``'Sklearn'``
    :return: list of supported models as string
    """
    from .common._registration import _converter_pool  # noqa

    # The two following lines populates the list of supported converters.
    from . import shape_calculators  # noqa
    from . import operator_converters  # noqa

    names = sorted(_converter_pool.keys())
    if from_sklearn:
        return [_[7:] for _ in names if _.startswith("Sklearn")]
    return list(names)
