# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###########################################################################

from collections import OrderedDict
import math
import numpy as np


def indent(s):
    return "\n".join("    " + line for line in s.split("\n"))


class NoPyObjException(Exception):
    def __init__(self):
        super().__init__("Tracing object has no associated python object")


class TracingObject:
    """
    Used by onnx2py to mock a module like numpy or onnx.helper and record calls on that module
    Ex:
        np = TracingObject("np")
        x = np.array(np.product([1, 2, 3]), np.int32)
        assert repr(x) == "np.array(np.product([1, 2, 3]), np.int32)"
    """
    def __init__(self, trace, py_obj=NoPyObjException):
        self._trace = trace
        self._py_obj = py_obj
        self._cnt = 0

    @staticmethod
    def reset_cnt(o):
        o._cnt = 0

    @staticmethod
    def get_cnt(o):
        return o._cnt

    @staticmethod
    def from_repr(o):
        return TracingObject(TracingObject.get_repr(o), o)

    @staticmethod
    def get_repr(x):
        if isinstance(x, np.ndarray):
            return "np.array(%s, dtype='%s')" % (TracingObject.get_repr(x.tolist()), x.dtype)
        if isinstance(x, float) and not math.isfinite(x):
            return "float('%r')" % x   # handle nan/inf/-inf
        if not isinstance(x, list):
            return repr(x)
        ls = [TracingObject.get_repr(o) for o in x]
        code = "[" + ", ".join(ls) + "]"
        if len(code) <= 200:
            return code
        return "[\n" + "".join(indent(s) + ",\n" for s in ls) + "]"

    @staticmethod
    def get_py_obj(o):
        if isinstance(o, list):
            return [TracingObject.get_py_obj(x) for x in o]
        if isinstance(o, TracingObject):
            if o._py_obj is NoPyObjException:
                raise NoPyObjException()
            return o._py_obj
        return o

    def __getattr__(self, attr):
        self._cnt += 1
        trace = self._trace + "." + attr
        if self._py_obj is NoPyObjException:
            return TracingObject(trace)
        return TracingObject(trace, getattr(self._py_obj, attr))

    def __call__(self, *args, **kwargs):
        self._cnt += 1
        arg_s = [TracingObject.get_repr(o) for o in args]
        arg_s += [k + "=" + TracingObject.get_repr(o) for k, o in kwargs.items()]
        trace = self._trace + "(" + ", ".join(arg_s) + ")"
        if len(trace) > 200:
            trace = self._trace + "(\n" + "".join(indent(s) + ",\n" for s in arg_s) + ")"
        try:
            arg_o = [TracingObject.get_py_obj(a) for a in args]
            kwarg_o = OrderedDict((k, TracingObject.get_py_obj(v)) for k, v in kwargs.items())
            py_obj = TracingObject.get_py_obj(self)(*arg_o, **kwarg_o)
        except NoPyObjException:
            py_obj = NoPyObjException
        return TracingObject(trace, py_obj)

    def __repr__(self):
        return self._trace
