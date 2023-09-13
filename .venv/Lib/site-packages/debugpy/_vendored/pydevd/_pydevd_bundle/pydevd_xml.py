from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_extension_utils
from _pydevd_bundle import pydevd_resolver
import sys
from _pydevd_bundle.pydevd_constants import BUILTINS_MODULE_NAME, MAXIMUM_VARIABLE_REPRESENTATION_SIZE, \
    RETURN_VALUES_DICT, LOAD_VALUES_ASYNC, DEFAULT_VALUE
from _pydev_bundle.pydev_imports import quote
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_utils import isinstance_checked, hasattr_checked, DAPGrouper
from _pydevd_bundle.pydevd_resolver import get_var_scope, MoreItems, MoreItemsRange
from typing import Optional

try:
    import types

    frame_type = types.FrameType
except:
    frame_type = None


def make_valid_xml_value(s):
    # Same thing as xml.sax.saxutils.escape but also escaping double quotes.
    return s.replace("&", "&amp;").replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


class ExceptionOnEvaluate:

    def __init__(self, result, etype, tb):
        self.result = result
        self.etype = etype
        self.tb = tb


_IS_JYTHON = sys.platform.startswith("java")


def _create_default_type_map():
    default_type_map = [
        # None means that it should not be treated as a compound variable

        # isintance does not accept a tuple on some versions of python, so, we must declare it expanded
        (type(None), None,),
        (int, None),
        (float, None),
        (complex, None),
        (str, None),
        (tuple, pydevd_resolver.tupleResolver),
        (list, pydevd_resolver.tupleResolver),
        (dict, pydevd_resolver.dictResolver),
    ]
    try:
        from collections import OrderedDict
        default_type_map.insert(0, (OrderedDict, pydevd_resolver.orderedDictResolver))
        # we should put it before dict
    except:
        pass

    try:
        default_type_map.append((long, None))  # @UndefinedVariable
    except:
        pass  # not available on all python versions

    default_type_map.append((DAPGrouper, pydevd_resolver.dapGrouperResolver))
    default_type_map.append((MoreItems, pydevd_resolver.forwardInternalResolverToObject))
    default_type_map.append((MoreItemsRange, pydevd_resolver.forwardInternalResolverToObject))

    try:
        default_type_map.append((set, pydevd_resolver.setResolver))
    except:
        pass  # not available on all python versions

    try:
        default_type_map.append((frozenset, pydevd_resolver.setResolver))
    except:
        pass  # not available on all python versions

    try:
        from django.utils.datastructures import MultiValueDict
        default_type_map.insert(0, (MultiValueDict, pydevd_resolver.multiValueDictResolver))
        # we should put it before dict
    except:
        pass  # django may not be installed

    try:
        from django.forms import BaseForm
        default_type_map.insert(0, (BaseForm, pydevd_resolver.djangoFormResolver))
        # we should put it before instance resolver
    except:
        pass  # django may not be installed

    try:
        from collections import deque
        default_type_map.append((deque, pydevd_resolver.dequeResolver))
    except:
        pass

    try:
        from ctypes import Array
        default_type_map.append((Array, pydevd_resolver.tupleResolver))
    except:
        pass

    if frame_type is not None:
        default_type_map.append((frame_type, pydevd_resolver.frameResolver))

    if _IS_JYTHON:
        from org.python import core  # @UnresolvedImport
        default_type_map.append((core.PyNone, None))
        default_type_map.append((core.PyInteger, None))
        default_type_map.append((core.PyLong, None))
        default_type_map.append((core.PyFloat, None))
        default_type_map.append((core.PyComplex, None))
        default_type_map.append((core.PyString, None))
        default_type_map.append((core.PyTuple, pydevd_resolver.tupleResolver))
        default_type_map.append((core.PyList, pydevd_resolver.tupleResolver))
        default_type_map.append((core.PyDictionary, pydevd_resolver.dictResolver))
        default_type_map.append((core.PyStringMap, pydevd_resolver.dictResolver))

        if hasattr(core, 'PyJavaInstance'):
            # Jython 2.5b3 removed it.
            default_type_map.append((core.PyJavaInstance, pydevd_resolver.instanceResolver))

    return default_type_map


class TypeResolveHandler(object):
    NO_PROVIDER = []  # Sentinel value (any mutable object to be used as a constant would be valid).

    def __init__(self):
        # Note: don't initialize with the types we already know about so that the extensions can override
        # the default resolvers that are already available if they want.
        self._type_to_resolver_cache = {}
        self._type_to_str_provider_cache = {}
        self._initialized = False

    def _initialize(self):
        self._default_type_map = _create_default_type_map()
        self._resolve_providers = pydevd_extension_utils.extensions_of_type(TypeResolveProvider)
        self._str_providers = pydevd_extension_utils.extensions_of_type(StrPresentationProvider)
        self._initialized = True

    def get_type(self, o):
        try:
            try:
                # Faster than type(o) as we don't need the function call.
                type_object = o.__class__  # could fail here
                type_name = type_object.__name__
                return self._get_type(o, type_object, type_name)  # could fail here
            except:
                # Not all objects have __class__ (i.e.: there are bad bindings around).
                type_object = type(o)
                type_name = type_object.__name__

                try:
                    return self._get_type(o, type_object, type_name)
                except:
                    if isinstance(type_object, type):
                        # If it's still something manageable, use the default resolver, otherwise
                        # fallback to saying that it wasn't possible to get any info on it.
                        return type_object, str(type_name), pydevd_resolver.defaultResolver

                    return 'Unable to get Type', 'Unable to get Type', None
        except:
            # This happens for org.python.core.InitModule
            return 'Unable to get Type', 'Unable to get Type', None

    def _get_type(self, o, type_object, type_name):
        # Note: we could have an exception here if the type_object is not hashable...
        resolver = self._type_to_resolver_cache.get(type_object)
        if resolver is not None:
            return type_object, type_name, resolver

        if not self._initialized:
            self._initialize()

        try:
            for resolver in self._resolve_providers:
                if resolver.can_provide(type_object, type_name):
                    # Cache it
                    self._type_to_resolver_cache[type_object] = resolver
                    return type_object, type_name, resolver

            for t in self._default_type_map:
                if isinstance_checked(o, t[0]):
                    # Cache it
                    resolver = t[1]
                    self._type_to_resolver_cache[type_object] = resolver
                    return (type_object, type_name, resolver)
        except:
            pydev_log.exception()

        # No match return default (and cache it).
        resolver = pydevd_resolver.defaultResolver
        self._type_to_resolver_cache[type_object] = resolver
        return type_object, type_name, resolver

    if _IS_JYTHON:
        _base_get_type = _get_type

        def _get_type(self, o, type_object, type_name):
            if type_name == 'org.python.core.PyJavaInstance':
                return type_object, type_name, pydevd_resolver.instanceResolver

            if type_name == 'org.python.core.PyArray':
                return type_object, type_name, pydevd_resolver.jyArrayResolver

            return self._base_get_type(o, type_object, type_name)

    def _get_str_from_provider(self, provider, o, context: Optional[str]=None):
        if context is not None:
            get_str_in_context = getattr(provider, 'get_str_in_context', None)
            if get_str_in_context is not None:
                return get_str_in_context(o, context)

        return provider.get_str(o)

    def str_from_providers(self, o, type_object, type_name, context: Optional[str]=None):
        provider = self._type_to_str_provider_cache.get(type_object)

        if provider is self.NO_PROVIDER:
            return None

        if provider is not None:
            return self._get_str_from_provider(provider, o, context)

        if not self._initialized:
            self._initialize()

        for provider in self._str_providers:
            if provider.can_provide(type_object, type_name):
                self._type_to_str_provider_cache[type_object] = provider
                try:
                    return self._get_str_from_provider(provider, o, context)
                except:
                    pydev_log.exception("Error when getting str with custom provider: %s." % (provider,))

        self._type_to_str_provider_cache[type_object] = self.NO_PROVIDER
        return None


_TYPE_RESOLVE_HANDLER = TypeResolveHandler()

"""
def get_type(o):
    Receives object and returns a triple (type_object, type_string, resolver).

    resolver != None means that variable is a container, and should be displayed as a hierarchy.

    Use the resolver to get its attributes.

    All container objects (i.e.: dict, list, tuple, object, etc) should have a resolver.
"""
get_type = _TYPE_RESOLVE_HANDLER.get_type

_str_from_providers = _TYPE_RESOLVE_HANDLER.str_from_providers


def is_builtin(x):
    return getattr(x, '__module__', None) == BUILTINS_MODULE_NAME


def should_evaluate_full_value(val):
    return not LOAD_VALUES_ASYNC or (is_builtin(type(val)) and not isinstance_checked(val, (list, tuple, dict)))


def return_values_from_dict_to_xml(return_dict):
    res = []
    for name, val in return_dict.items():
        res.append(var_to_xml(val, name, additional_in_xml=' isRetVal="True"'))
    return ''.join(res)


def frame_vars_to_xml(frame_f_locals, hidden_ns=None):
    """ dumps frame variables to XML
    <var name="var_name" scope="local" type="type" value="value"/>
    """
    xml = []

    keys = sorted(frame_f_locals)

    return_values_xml = []

    for k in keys:
        try:
            v = frame_f_locals[k]
            eval_full_val = should_evaluate_full_value(v)

            if k == '_pydev_stop_at_break':
                continue

            if k == RETURN_VALUES_DICT:
                for name, val in v.items():
                    return_values_xml.append(var_to_xml(val, name, additional_in_xml=' isRetVal="True"'))

            else:
                if hidden_ns is not None and k in hidden_ns:
                    xml.append(var_to_xml(v, str(k), additional_in_xml=' isIPythonHidden="True"',
                                      evaluate_full_value=eval_full_val))
                else:
                    xml.append(var_to_xml(v, str(k), evaluate_full_value=eval_full_val))
        except Exception:
            pydev_log.exception("Unexpected error, recovered safely.")

    # Show return values as the first entry.
    return_values_xml.extend(xml)
    return ''.join(return_values_xml)


def get_variable_details(val, evaluate_full_value=True, to_string=None, context: Optional[str]=None):
    '''
    :param context:
        This is the context in which the variable is being requested. Valid values:
            "watch",
            "repl",
            "hover",
            "clipboard"
    '''
    try:
        # This should be faster than isinstance (but we have to protect against not having a '__class__' attribute).
        is_exception_on_eval = val.__class__ == ExceptionOnEvaluate
    except:
        is_exception_on_eval = False

    if is_exception_on_eval:
        v = val.result
    else:
        v = val

    _type, type_name, resolver = get_type(v)
    type_qualifier = getattr(_type, "__module__", "")
    if not evaluate_full_value:
        value = DEFAULT_VALUE
    else:
        try:
            str_from_provider = _str_from_providers(v, _type, type_name, context)
            if str_from_provider is not None:
                value = str_from_provider

            elif to_string is not None:
                value = to_string(v)

            elif hasattr_checked(v, '__class__'):
                if v.__class__ == frame_type:
                    value = pydevd_resolver.frameResolver.get_frame_name(v)

                elif v.__class__ in (list, tuple):
                    if len(v) > 300:
                        value = '%s: %s' % (str(v.__class__), '<Too big to print. Len: %s>' % (len(v),))
                    else:
                        value = '%s: %s' % (str(v.__class__), v)
                else:
                    try:
                        cName = str(v.__class__)
                        if cName.find('.') != -1:
                            cName = cName.split('.')[-1]

                        elif cName.find("'") != -1:  # does not have '.' (could be something like <type 'int'>)
                            cName = cName[cName.index("'") + 1:]

                        if cName.endswith("'>"):
                            cName = cName[:-2]
                    except:
                        cName = str(v.__class__)

                    value = '%s: %s' % (cName, v)
            else:
                value = str(v)
        except:
            try:
                value = repr(v)
            except:
                value = 'Unable to get repr for %s' % v.__class__

    # fix to work with unicode values
    try:
        if value.__class__ == bytes:
            value = value.decode('utf-8', 'replace')
    except TypeError:
        pass

    return type_name, type_qualifier, is_exception_on_eval, resolver, value


def var_to_xml(val, name, trim_if_too_big=True, additional_in_xml='', evaluate_full_value=True):
    """ single variable or dictionary to xml representation """

    type_name, type_qualifier, is_exception_on_eval, resolver, value = get_variable_details(
        val, evaluate_full_value)

    scope = get_var_scope(name, val, '', True)
    try:
        name = quote(name, '/>_= ')  # TODO: Fix PY-5834 without using quote
    except:
        pass

    xml = '<var name="%s" type="%s" ' % (make_valid_xml_value(name), make_valid_xml_value(type_name))

    if type_qualifier:
        xml_qualifier = 'qualifier="%s"' % make_valid_xml_value(type_qualifier)
    else:
        xml_qualifier = ''

    if value:
        # cannot be too big... communication may not handle it.
        if len(value) > MAXIMUM_VARIABLE_REPRESENTATION_SIZE and trim_if_too_big:
            value = value[0:MAXIMUM_VARIABLE_REPRESENTATION_SIZE]
            value += '...'

        xml_value = ' value="%s"' % (make_valid_xml_value(quote(value, '/>_= ')))
    else:
        xml_value = ''

    if is_exception_on_eval:
        xml_container = ' isErrorOnEval="True"'
    else:
        if resolver is not None:
            xml_container = ' isContainer="True"'
        else:
            xml_container = ''

    if scope:
        return ''.join((xml, xml_qualifier, xml_value, xml_container, additional_in_xml, ' scope="', scope, '"', ' />\n'))
    else:
        return ''.join((xml, xml_qualifier, xml_value, xml_container, additional_in_xml, ' />\n'))
