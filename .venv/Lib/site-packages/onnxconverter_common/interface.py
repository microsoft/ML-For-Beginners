# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

"""
This file defines the interface of the converter internal object for callback,
So the usage of the methods and properties list here will not be affected among the different versions.
"""

import abc


class ModelContainer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_initializer(self, name, onnx_type, shape, content):
        """
        Add a TensorProto into the initializer list of the final ONNX model

        :param name: Variable name in the produced ONNX model.
        :param onnx_type: Element types allowed in ONNX tensor, e.g., TensorProto.FLOAT and TensorProto.STRING.
        :param shape: Tensor shape, a list of integers.
        :param content: Flattened tensor values (i.e., a float list or a float array).
        """
        return

    @abc.abstractmethod
    def add_node(self, op_type, inputs, outputs, op_domain='', op_version=1, **attrs):
        """
        Add a NodeProto into the node list of the final ONNX model. If the input operator's domain-version information
        cannot be found in our domain-version pool (a Python set), we may add it.

        :param op_type: A string (e.g., Pool and Conv) indicating the type of the NodeProto
        :param inputs: A list of strings. They are the input variables' names of the considered NodeProto
        :param outputs: A list of strings. They are the output variables' names of the considered NodeProto
        :param op_domain: The domain name (e.g., ai.onnx.ml) of the operator we are trying to add.
        :param op_version: The version number (e.g., 0 and 1) of the operator we are trying to add.
        :param attrs: A Python dictionary. Keys and values are attributes' names and attributes' values, respectively.
        """
        return


class OperatorBase:
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def full_name(self):
        """
        Return a globally unique operator ID
        """
        pass

    @property
    @abc.abstractmethod
    def input_full_names(self):
        """
        Return all input variables' names
        """
        pass

    @property
    @abc.abstractmethod
    def output_full_names(self):
        """
        Return all outpu variables' names
        """
        pass

    @property
    @abc.abstractmethod
    def original_operator(self):
        """
        Return the original operator/layer
        """
        pass


class ScopeBase:
    __metaclass__ = abc.ABCMeta

    pass
