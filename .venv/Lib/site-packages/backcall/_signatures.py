"""Function signature objects for callables

Back port of Python 3.3's function signature tools from the inspect module,
modified to be compatible with Python 2.6, 2.7 and 3.2+.
"""

#-----------------------------------------------------------------------------
#  Python 3.3 stdlib inspect.py is public domain
# 
#  Backports Copyright (C) 2013 Aaron Iles
#  Used under Apache License Version 2.0
#
#  Further Changes are Copyright (C) 2013 The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function
import itertools
import functools
import re
import types


# patch for single-file
# we don't support 2.6, so we can just import OrderedDict
from collections import OrderedDict

__version__ = '0.3'
# end patch

__all__ = ['BoundArguments', 'Parameter', 'Signature', 'signature']


_WrapperDescriptor = type(type.__call__)
_MethodWrapper = type(all.__call__)

_NonUserDefinedCallables = (_WrapperDescriptor,
                            _MethodWrapper,
                            types.BuiltinFunctionType)


def formatannotation(annotation, base_module=None):
    if isinstance(annotation, type):
        if annotation.__module__ in ('builtins', '__builtin__', base_module):
            return annotation.__name__
        return annotation.__module__+'.'+annotation.__name__
    return repr(annotation)


def _get_user_defined_method(cls, method_name, *nested):
    try:
        if cls is type:
            return
        meth = getattr(cls, method_name)
        for name in nested:
            meth = getattr(meth, name, meth)
    except AttributeError:
        return
    else:
        if not isinstance(meth, _NonUserDefinedCallables):
            # Once '__signature__' will be added to 'C'-level
            # callables, this check won't be necessary
            return meth


def signature(obj):
    '''Get a signature object for the passed callable.'''

    if not callable(obj):
        raise TypeError('{0!r} is not a callable object'.format(obj))

    if isinstance(obj, types.MethodType):
        # In this case we skip the first parameter of the underlying
        # function (usually `self` or `cls`).
        sig = signature(obj.__func__)
        return sig.replace(parameters=tuple(sig.parameters.values())[1:])

    try:
        sig = obj.__signature__
    except AttributeError:
        pass
    else:
        if sig is not None:
            return sig

    try:
        # Was this function wrapped by a decorator?
        wrapped = obj.__wrapped__
    except AttributeError:
        pass
    else:
        return signature(wrapped)

    if isinstance(obj, types.FunctionType):
        return Signature.from_function(obj)

    if isinstance(obj, functools.partial):
        sig = signature(obj.func)

        new_params = OrderedDict(sig.parameters.items())

        partial_args = obj.args or ()
        partial_keywords = obj.keywords or {}
        try:
            ba = sig.bind_partial(*partial_args, **partial_keywords)
        except TypeError as ex:
            msg = 'partial object {0!r} has incorrect arguments'.format(obj)
            raise ValueError(msg)

        for arg_name, arg_value in ba.arguments.items():
            param = new_params[arg_name]
            if arg_name in partial_keywords:
                # We set a new default value, because the following code
                # is correct:
                #
                #   >>> def foo(a): print(a)
                #   >>> print(partial(partial(foo, a=10), a=20)())
                #   20
                #   >>> print(partial(partial(foo, a=10), a=20)(a=30))
                #   30
                #
                # So, with 'partial' objects, passing a keyword argument is
                # like setting a new default value for the corresponding
                # parameter
                #
                # We also mark this parameter with '_partial_kwarg'
                # flag.  Later, in '_bind', the 'default' value of this
                # parameter will be added to 'kwargs', to simulate
                # the 'functools.partial' real call.
                new_params[arg_name] = param.replace(default=arg_value,
                                                     _partial_kwarg=True)

            elif (param.kind not in (_VAR_KEYWORD, _VAR_POSITIONAL) and
                            not param._partial_kwarg):
                new_params.pop(arg_name)

        return sig.replace(parameters=new_params.values())

    sig = None
    if isinstance(obj, type):
        # obj is a class or a metaclass

        # First, let's see if it has an overloaded __call__ defined
        # in its metaclass
        call = _get_user_defined_method(type(obj), '__call__')
        if call is not None:
            sig = signature(call)
        else:
            # Now we check if the 'obj' class has a '__new__' method
            new = _get_user_defined_method(obj, '__new__')
            if new is not None:
                sig = signature(new)
            else:
                # Finally, we should have at least __init__ implemented
                init = _get_user_defined_method(obj, '__init__')
                if init is not None:
                    sig = signature(init)
    elif not isinstance(obj, _NonUserDefinedCallables):
        # An object with __call__
        # We also check that the 'obj' is not an instance of
        # _WrapperDescriptor or _MethodWrapper to avoid
        # infinite recursion (and even potential segfault)
        call = _get_user_defined_method(type(obj), '__call__', 'im_func')
        if call is not None:
            sig = signature(call)

    if sig is not None:
        return sig

    if isinstance(obj, types.BuiltinFunctionType):
        # Raise a nicer error message for builtins
        msg = 'no signature found for builtin function {0!r}'.format(obj)
        raise ValueError(msg)

    raise ValueError('callable {0!r} is not supported by signature'.format(obj))


class _void(object):
    '''A private marker - used in Parameter & Signature'''


class _empty(object):
    pass


class _ParameterKind(int):
    def __new__(self, *args, **kwargs):
        obj = int.__new__(self, *args)
        obj._name = kwargs['name']
        return obj

    def __str__(self):
        return self._name

    def __repr__(self):
        return '<_ParameterKind: {0!r}>'.format(self._name)


_POSITIONAL_ONLY        = _ParameterKind(0, name='POSITIONAL_ONLY')
_POSITIONAL_OR_KEYWORD  = _ParameterKind(1, name='POSITIONAL_OR_KEYWORD')
_VAR_POSITIONAL         = _ParameterKind(2, name='VAR_POSITIONAL')
_KEYWORD_ONLY           = _ParameterKind(3, name='KEYWORD_ONLY')
_VAR_KEYWORD            = _ParameterKind(4, name='VAR_KEYWORD')


class Parameter(object):
    '''Represents a parameter in a function signature.

    Has the following public attributes:

    * name : str
        The name of the parameter as a string.
    * default : object
        The default value for the parameter if specified.  If the
        parameter has no default value, this attribute is not set.
    * annotation
        The annotation for the parameter if specified.  If the
        parameter has no annotation, this attribute is not set.
    * kind : str
        Describes how argument values are bound to the parameter.
        Possible values: `Parameter.POSITIONAL_ONLY`,
        `Parameter.POSITIONAL_OR_KEYWORD`, `Parameter.VAR_POSITIONAL`,
        `Parameter.KEYWORD_ONLY`, `Parameter.VAR_KEYWORD`.
    '''

    __slots__ = ('_name', '_kind', '_default', '_annotation', '_partial_kwarg')

    POSITIONAL_ONLY         = _POSITIONAL_ONLY
    POSITIONAL_OR_KEYWORD   = _POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL          = _VAR_POSITIONAL
    KEYWORD_ONLY            = _KEYWORD_ONLY
    VAR_KEYWORD             = _VAR_KEYWORD

    empty = _empty

    def __init__(self, name, kind, default=_empty, annotation=_empty,
                 _partial_kwarg=False):

        if kind not in (_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD,
                        _VAR_POSITIONAL, _KEYWORD_ONLY, _VAR_KEYWORD):
            raise ValueError("invalid value for 'Parameter.kind' attribute")
        self._kind = kind

        if default is not _empty:
            if kind in (_VAR_POSITIONAL, _VAR_KEYWORD):
                msg = '{0} parameters cannot have default values'.format(kind)
                raise ValueError(msg)
        self._default = default
        self._annotation = annotation

        if name is None:
            if kind != _POSITIONAL_ONLY:
                raise ValueError("None is not a valid name for a "
                                 "non-positional-only parameter")
            self._name = name
        else:
            name = str(name)
            if kind != _POSITIONAL_ONLY and not re.match(r'[a-z_]\w*$', name, re.I):
                msg = '{0!r} is not a valid parameter name'.format(name)
                raise ValueError(msg)
            self._name = name

        self._partial_kwarg = _partial_kwarg

    @property
    def name(self):
        return self._name

    @property
    def default(self):
        return self._default

    @property
    def annotation(self):
        return self._annotation

    @property
    def kind(self):
        return self._kind

    def replace(self, name=_void, kind=_void, annotation=_void,
                default=_void, _partial_kwarg=_void):
        '''Creates a customized copy of the Parameter.'''

        if name is _void:
            name = self._name

        if kind is _void:
            kind = self._kind

        if annotation is _void:
            annotation = self._annotation

        if default is _void:
            default = self._default

        if _partial_kwarg is _void:
            _partial_kwarg = self._partial_kwarg

        return type(self)(name, kind, default=default, annotation=annotation,
                          _partial_kwarg=_partial_kwarg)

    def __str__(self):
        kind = self.kind

        formatted = self._name
        if kind == _POSITIONAL_ONLY:
            if formatted is None:
                formatted = ''
            formatted = '<{0}>'.format(formatted)

        # Add annotation and default value
        if self._annotation is not _empty:
            formatted = '{0}:{1}'.format(formatted,
                                       formatannotation(self._annotation))

        if self._default is not _empty:
            formatted = '{0}={1}'.format(formatted, repr(self._default))

        if kind == _VAR_POSITIONAL:
            formatted = '*' + formatted
        elif kind == _VAR_KEYWORD:
            formatted = '**' + formatted

        return formatted

    def __repr__(self):
        return '<{0} at {1:#x} {2!r}>'.format(self.__class__.__name__,
                                           id(self), self.name)

    def __hash__(self):
        msg = "unhashable type: '{0}'".format(self.__class__.__name__)
        raise TypeError(msg)

    def __eq__(self, other):
        return (issubclass(other.__class__, Parameter) and
                self._name == other._name and
                self._kind == other._kind and
                self._default == other._default and
                self._annotation == other._annotation)

    def __ne__(self, other):
        return not self.__eq__(other)


class BoundArguments(object):
    '''Result of `Signature.bind` call.  Holds the mapping of arguments
    to the function's parameters.

    Has the following public attributes:

    * arguments : OrderedDict
        An ordered mutable mapping of parameters' names to arguments' values.
        Does not contain arguments' default values.
    * signature : Signature
        The Signature object that created this instance.
    * args : tuple
        Tuple of positional arguments values.
    * kwargs : dict
        Dict of keyword arguments values.
    '''

    def __init__(self, signature, arguments):
        self.arguments = arguments
        self._signature = signature

    @property
    def signature(self):
        return self._signature

    @property
    def args(self):
        args = []
        for param_name, param in self._signature.parameters.items():
            if (param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY) or
                                                    param._partial_kwarg):
                # Keyword arguments mapped by 'functools.partial'
                # (Parameter._partial_kwarg is True) are mapped
                # in 'BoundArguments.kwargs', along with VAR_KEYWORD &
                # KEYWORD_ONLY
                break

            try:
                arg = self.arguments[param_name]
            except KeyError:
                # We're done here. Other arguments
                # will be mapped in 'BoundArguments.kwargs'
                break
            else:
                if param.kind == _VAR_POSITIONAL:
                    # *args
                    args.extend(arg)
                else:
                    # plain argument
                    args.append(arg)

        return tuple(args)

    @property
    def kwargs(self):
        kwargs = {}
        kwargs_started = False
        for param_name, param in self._signature.parameters.items():
            if not kwargs_started:
                if (param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY) or
                                                param._partial_kwarg):
                    kwargs_started = True
                else:
                    if param_name not in self.arguments:
                        kwargs_started = True
                        continue

            if not kwargs_started:
                continue

            try:
                arg = self.arguments[param_name]
            except KeyError:
                pass
            else:
                if param.kind == _VAR_KEYWORD:
                    # **kwargs
                    kwargs.update(arg)
                else:
                    # plain keyword argument
                    kwargs[param_name] = arg

        return kwargs

    def __hash__(self):
        msg = "unhashable type: '{0}'".format(self.__class__.__name__)
        raise TypeError(msg)

    def __eq__(self, other):
        return (issubclass(other.__class__, BoundArguments) and
                self.signature == other.signature and
                self.arguments == other.arguments)

    def __ne__(self, other):
        return not self.__eq__(other)


class Signature(object):
    '''A Signature object represents the overall signature of a function.
    It stores a Parameter object for each parameter accepted by the
    function, as well as information specific to the function itself.

    A Signature object has the following public attributes and methods:

    * parameters : OrderedDict
        An ordered mapping of parameters' names to the corresponding
        Parameter objects (keyword-only arguments are in the same order
        as listed in `code.co_varnames`).
    * return_annotation : object
        The annotation for the return type of the function if specified.
        If the function has no annotation for its return type, this
        attribute is not set.
    * bind(*args, **kwargs) -> BoundArguments
        Creates a mapping from positional and keyword arguments to
        parameters.
    * bind_partial(*args, **kwargs) -> BoundArguments
        Creates a partial mapping from positional and keyword arguments
        to parameters (simulating 'functools.partial' behavior.)
    '''

    __slots__ = ('_return_annotation', '_parameters')

    _parameter_cls = Parameter
    _bound_arguments_cls = BoundArguments

    empty = _empty

    def __init__(self, parameters=None, return_annotation=_empty,
                 __validate_parameters__=True):
        '''Constructs Signature from the given list of Parameter
        objects and 'return_annotation'.  All arguments are optional.
        '''

        if parameters is None:
            params = OrderedDict()
        else:
            if __validate_parameters__:
                params = OrderedDict()
                top_kind = _POSITIONAL_ONLY

                for idx, param in enumerate(parameters):
                    kind = param.kind
                    if kind < top_kind:
                        msg = 'wrong parameter order: {0} before {1}'
                        msg = msg.format(top_kind, param.kind)
                        raise ValueError(msg)
                    else:
                        top_kind = kind

                    name = param.name
                    if name is None:
                        name = str(idx)
                        param = param.replace(name=name)

                    if name in params:
                        msg = 'duplicate parameter name: {0!r}'.format(name)
                        raise ValueError(msg)
                    params[name] = param
            else:
                params = OrderedDict(((param.name, param)
                                                for param in parameters))

        self._parameters = params
        self._return_annotation = return_annotation

    @classmethod
    def from_function(cls, func):
        '''Constructs Signature for the given python function'''

        if not isinstance(func, types.FunctionType):
            raise TypeError('{0!r} is not a Python function'.format(func))

        Parameter = cls._parameter_cls

        # Parameter information.
        func_code = func.__code__
        pos_count = func_code.co_argcount
        arg_names = func_code.co_varnames
        positional = tuple(arg_names[:pos_count])
        keyword_only_count = getattr(func_code, 'co_kwonlyargcount', 0)
        keyword_only = arg_names[pos_count:(pos_count + keyword_only_count)]
        annotations = getattr(func, '__annotations__', {})
        defaults = func.__defaults__
        kwdefaults = getattr(func, '__kwdefaults__', None)

        if defaults:
            pos_default_count = len(defaults)
        else:
            pos_default_count = 0

        parameters = []

        # Non-keyword-only parameters w/o defaults.
        non_default_count = pos_count - pos_default_count
        for name in positional[:non_default_count]:
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation,
                                        kind=_POSITIONAL_OR_KEYWORD))

        # ... w/ defaults.
        for offset, name in enumerate(positional[non_default_count:]):
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation,
                                        kind=_POSITIONAL_OR_KEYWORD,
                                        default=defaults[offset]))

        # *args
        if func_code.co_flags & 0x04:
            name = arg_names[pos_count + keyword_only_count]
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation,
                                        kind=_VAR_POSITIONAL))

        # Keyword-only parameters.
        for name in keyword_only:
            default = _empty
            if kwdefaults is not None:
                default = kwdefaults.get(name, _empty)

            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation,
                                        kind=_KEYWORD_ONLY,
                                        default=default))
        # **kwargs
        if func_code.co_flags & 0x08:
            index = pos_count + keyword_only_count
            if func_code.co_flags & 0x04:
                index += 1

            name = arg_names[index]
            annotation = annotations.get(name, _empty)
            parameters.append(Parameter(name, annotation=annotation,
                                        kind=_VAR_KEYWORD))

        return cls(parameters,
                   return_annotation=annotations.get('return', _empty),
                   __validate_parameters__=False)

    @property
    def parameters(self):
        try:
            return types.MappingProxyType(self._parameters)
        except AttributeError:
            return OrderedDict(self._parameters.items())

    @property
    def return_annotation(self):
        return self._return_annotation

    def replace(self, parameters=_void, return_annotation=_void):
        '''Creates a customized copy of the Signature.
        Pass 'parameters' and/or 'return_annotation' arguments
        to override them in the new copy.
        '''

        if parameters is _void:
            parameters = self.parameters.values()

        if return_annotation is _void:
            return_annotation = self._return_annotation

        return type(self)(parameters,
                          return_annotation=return_annotation)

    def __hash__(self):
        msg = "unhashable type: '{0}'".format(self.__class__.__name__)
        raise TypeError(msg)

    def __eq__(self, other):
        if (not issubclass(type(other), Signature) or
                    self.return_annotation != other.return_annotation or
                    len(self.parameters) != len(other.parameters)):
            return False

        other_positions = dict((param, idx)
                           for idx, param in enumerate(other.parameters.keys()))

        for idx, (param_name, param) in enumerate(self.parameters.items()):
            if param.kind == _KEYWORD_ONLY:
                try:
                    other_param = other.parameters[param_name]
                except KeyError:
                    return False
                else:
                    if param != other_param:
                        return False
            else:
                try:
                    other_idx = other_positions[param_name]
                except KeyError:
                    return False
                else:
                    if (idx != other_idx or
                                    param != other.parameters[param_name]):
                        return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def _bind(self, args, kwargs, partial=False):
        '''Private method.  Don't use directly.'''

        arguments = OrderedDict()

        parameters = iter(self.parameters.values())
        parameters_ex = ()
        arg_vals = iter(args)

        if partial:
            # Support for binding arguments to 'functools.partial' objects.
            # See 'functools.partial' case in 'signature()' implementation
            # for details.
            for param_name, param in self.parameters.items():
                if (param._partial_kwarg and param_name not in kwargs):
                    # Simulating 'functools.partial' behavior
                    kwargs[param_name] = param.default

        while True:
            # Let's iterate through the positional arguments and corresponding
            # parameters
            try:
                arg_val = next(arg_vals)
            except StopIteration:
                # No more positional arguments
                try:
                    param = next(parameters)
                except StopIteration:
                    # No more parameters. That's it. Just need to check that
                    # we have no `kwargs` after this while loop
                    break
                else:
                    if param.kind == _VAR_POSITIONAL:
                        # That's OK, just empty *args.  Let's start parsing
                        # kwargs
                        break
                    elif param.name in kwargs:
                        if param.kind == _POSITIONAL_ONLY:
                            msg = '{arg!r} parameter is positional only, ' \
                                  'but was passed as a keyword'
                            msg = msg.format(arg=param.name)
                            raise TypeError(msg)
                        parameters_ex = (param,)
                        break
                    elif (param.kind == _VAR_KEYWORD or
                                                param.default is not _empty):
                        # That's fine too - we have a default value for this
                        # parameter.  So, lets start parsing `kwargs`, starting
                        # with the current parameter
                        parameters_ex = (param,)
                        break
                    else:
                        if partial:
                            parameters_ex = (param,)
                            break
                        else:
                            msg = '{arg!r} parameter lacking default value'
                            msg = msg.format(arg=param.name)
                            raise TypeError(msg)
            else:
                # We have a positional argument to process
                try:
                    param = next(parameters)
                except StopIteration:
                    raise TypeError('too many positional arguments')
                else:
                    if param.kind in (_VAR_KEYWORD, _KEYWORD_ONLY):
                        # Looks like we have no parameter for this positional
                        # argument
                        raise TypeError('too many positional arguments')

                    if param.kind == _VAR_POSITIONAL:
                        # We have an '*args'-like argument, let's fill it with
                        # all positional arguments we have left and move on to
                        # the next phase
                        values = [arg_val]
                        values.extend(arg_vals)
                        arguments[param.name] = tuple(values)
                        break

                    if param.name in kwargs:
                        raise TypeError('multiple values for argument '
                                        '{arg!r}'.format(arg=param.name))

                    arguments[param.name] = arg_val

        # Now, we iterate through the remaining parameters to process
        # keyword arguments
        kwargs_param = None
        for param in itertools.chain(parameters_ex, parameters):
            if param.kind == _POSITIONAL_ONLY:
                # This should never happen in case of a properly built
                # Signature object (but let's have this check here
                # to ensure correct behaviour just in case)
                raise TypeError('{arg!r} parameter is positional only, '
                                'but was passed as a keyword'. \
                                format(arg=param.name))

            if param.kind == _VAR_KEYWORD:
                # Memorize that we have a '**kwargs'-like parameter
                kwargs_param = param
                continue

            param_name = param.name
            try:
                arg_val = kwargs.pop(param_name)
            except KeyError:
                # We have no value for this parameter.  It's fine though,
                # if it has a default value, or it is an '*args'-like
                # parameter, left alone by the processing of positional
                # arguments.
                if (not partial and param.kind != _VAR_POSITIONAL and
                                                    param.default is _empty):
                    raise TypeError('{arg!r} parameter lacking default value'. \
                                    format(arg=param_name))

            else:
                arguments[param_name] = arg_val

        if kwargs:
            if kwargs_param is not None:
                # Process our '**kwargs'-like parameter
                arguments[kwargs_param.name] = kwargs
            else:
                raise TypeError('too many keyword arguments')

        return self._bound_arguments_cls(self, arguments)

    def bind(self, *args, **kwargs):
        '''Get a BoundArguments object, that maps the passed `args`
        and `kwargs` to the function's signature.  Raises `TypeError`
        if the passed arguments can not be bound.
        '''
        return self._bind(args, kwargs)

    def bind_partial(self, *args, **kwargs):
        '''Get a BoundArguments object, that partially maps the
        passed `args` and `kwargs` to the function's signature.
        Raises `TypeError` if the passed arguments can not be bound.
        '''
        return self._bind(args, kwargs, partial=True)

    def __str__(self):
        result = []
        render_kw_only_separator = True
        for idx, param in enumerate(self.parameters.values()):
            formatted = str(param)

            kind = param.kind
            if kind == _VAR_POSITIONAL:
                # OK, we have an '*args'-like parameter, so we won't need
                # a '*' to separate keyword-only arguments
                render_kw_only_separator = False
            elif kind == _KEYWORD_ONLY and render_kw_only_separator:
                # We have a keyword-only parameter to render and we haven't
                # rendered an '*args'-like parameter before, so add a '*'
                # separator to the parameters list ("foo(arg1, *, arg2)" case)
                result.append('*')
                # This condition should be only triggered once, so
                # reset the flag
                render_kw_only_separator = False

            result.append(formatted)

        rendered = '({0})'.format(', '.join(result))

        if self.return_annotation is not _empty:
            anno = formatannotation(self.return_annotation)
            rendered += ' -> {0}'.format(anno)

        return rendered
