from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename

from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
    MethodWrapperType, RETURN_VALUES_DICT, DebugInfoHolder, IS_PYPY, GENERATED_LEN_ATTR_NAME
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants

TOO_LARGE_MSG = 'Maximum number of items (%s) reached. To show more items customize the value of the PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS environment variable.'
TOO_LARGE_ATTR = 'Unable to handle:'


#=======================================================================================================================
# UnableToResolveVariableException
#=======================================================================================================================
class UnableToResolveVariableException(Exception):
    pass


try:
    from collections import OrderedDict
except:
    OrderedDict = dict

try:
    import java.lang  # @UnresolvedImport
except:
    pass

#=======================================================================================================================
# See: pydevd_extension_api module for resolver interface
#=======================================================================================================================


def sorted_attributes_key(attr_name):
    if attr_name.startswith('__'):
        if attr_name.endswith('__'):
            # __ double under before and after __
            return (3, attr_name)
        else:
            # __ double under before
            return (2, attr_name)
    elif attr_name.startswith('_'):
        # _ single under
        return (1, attr_name)
    else:
        # Regular (Before anything)
        return (0, attr_name)


#=======================================================================================================================
# DefaultResolver
#=======================================================================================================================
class DefaultResolver:
    '''
        DefaultResolver is the class that'll actually resolve how to show some variable.
    '''

    def resolve(self, var, attribute):
        return getattr(var, attribute)

    def get_contents_debug_adapter_protocol(self, obj, fmt=None):
        if MethodWrapperType:
            dct, used___dict__ = self._get_py_dictionary(obj)
        else:
            dct = self._get_jy_dictionary(obj)[0]

        lst = sorted(dct.items(), key=lambda tup: sorted_attributes_key(tup[0]))
        if used___dict__:
            eval_name = '.__dict__[%s]'
        else:
            eval_name = '.%s'

        ret = []
        for attr_name, attr_value in lst:
            entry = (attr_name, attr_value, eval_name % attr_name)
            ret.append(entry)

        return ret

    def get_dictionary(self, var, names=None, used___dict__=False):
        if MethodWrapperType:
            return self._get_py_dictionary(var, names, used___dict__=used___dict__)[0]
        else:
            return self._get_jy_dictionary(var)[0]

    def _get_jy_dictionary(self, obj):
        ret = {}
        found = java.util.HashMap()

        original = obj
        if hasattr_checked(obj, '__class__') and obj.__class__ == java.lang.Class:

            # get info about superclasses
            classes = []
            classes.append(obj)
            c = obj.getSuperclass()
            while c != None:
                classes.append(c)
                c = c.getSuperclass()

            # get info about interfaces
            interfs = []
            for obj in classes:
                interfs.extend(obj.getInterfaces())
            classes.extend(interfs)

            # now is the time when we actually get info on the declared methods and fields
            for obj in classes:

                declaredMethods = obj.getDeclaredMethods()
                declaredFields = obj.getDeclaredFields()
                for i in range(len(declaredMethods)):
                    name = declaredMethods[i].getName()
                    ret[name] = declaredMethods[i].toString()
                    found.put(name, 1)

                for i in range(len(declaredFields)):
                    name = declaredFields[i].getName()
                    found.put(name, 1)
                    # if declaredFields[i].isAccessible():
                    declaredFields[i].setAccessible(True)
                    # ret[name] = declaredFields[i].get( declaredFields[i] )
                    try:
                        ret[name] = declaredFields[i].get(original)
                    except:
                        ret[name] = declaredFields[i].toString()

        # this simple dir does not always get all the info, that's why we have the part before
        # (e.g.: if we do a dir on String, some methods that are from other interfaces such as
        # charAt don't appear)
        try:
            d = dir(original)
            for name in d:
                if found.get(name) != 1:
                    ret[name] = getattr(original, name)
        except:
            # sometimes we're unable to do a dir
            pass

        return ret

    def get_names(self, var):
        used___dict__ = False
        try:
            names = dir(var)
        except Exception:
            names = []
        if not names:
            if hasattr_checked(var, '__dict__'):
                names = list(var.__dict__)
                used___dict__ = True
        return names, used___dict__

    def _get_py_dictionary(self, var, names=None, used___dict__=False):
        '''
        :return tuple(names, used___dict__), where used___dict__ means we have to access
        using obj.__dict__[name] instead of getattr(obj, name)
        '''

        # On PyPy we never show functions. This is because of a corner case where PyPy becomes
        # absurdly slow -- it takes almost half a second to introspect a single numpy function (so,
        # the related test, "test_case_16_resolve_numpy_array", times out... this probably isn't
        # specific to numpy, but to any library where the CPython bridge is used, but as we
        # can't be sure in the debugger, we play it safe and don't show it at all).
        filter_function = IS_PYPY

        if not names:
            names, used___dict__ = self.get_names(var)
        d = {}

        # Be aware that the order in which the filters are applied attempts to
        # optimize the operation by removing as many items as possible in the
        # first filters, leaving fewer items for later filters

        timer = Timer()
        cls = type(var)
        for name in names:
            try:
                name_as_str = name
                if name_as_str.__class__ != str:
                    name_as_str = '%r' % (name_as_str,)

                if not used___dict__:
                    attr = getattr(var, name)
                else:
                    attr = var.__dict__[name]

                # filter functions?
                if filter_function:
                    if inspect.isroutine(attr) or isinstance(attr, MethodWrapperType):
                        continue
            except:
                # if some error occurs getting it, let's put it to the user.
                strIO = StringIO()
                traceback.print_exc(file=strIO)
                attr = strIO.getvalue()

            finally:
                timer.report_if_getting_attr_slow(cls, name_as_str)

            d[name_as_str] = attr

        return d, used___dict__


class DAPGrouperResolver:

    def get_contents_debug_adapter_protocol(self, obj, fmt=None):
        return obj.get_contents_debug_adapter_protocol()


_basic_immutable_types = (int, float, complex, str, bytes, type(None), bool, frozenset)


def _does_obj_repr_evaluate_to_obj(obj):
    '''
    If obj is an object where evaluating its representation leads to
    the same object, return True, otherwise, return False.
    '''
    try:
        if isinstance(obj, tuple):
            for o in obj:
                if not _does_obj_repr_evaluate_to_obj(o):
                    return False
            return True
        else:
            return isinstance(obj, _basic_immutable_types)
    except:
        return False


#=======================================================================================================================
# DictResolver
#=======================================================================================================================
class DictResolver:

    sort_keys = not IS_PY36_OR_GREATER

    def resolve(self, dct, key):
        if key in (GENERATED_LEN_ATTR_NAME, TOO_LARGE_ATTR):
            return None

        if '(' not in key:
            # we have to treat that because the dict resolver is also used to directly resolve the global and local
            # scopes (which already have the items directly)
            try:
                return dct[key]
            except:
                return getattr(dct, key)

        # ok, we have to iterate over the items to find the one that matches the id, because that's the only way
        # to actually find the reference from the string we have before.
        expected_id = int(key.split('(')[-1][:-1])
        for key, val in dct.items():
            if id(key) == expected_id:
                return val

        raise UnableToResolveVariableException()

    def key_to_str(self, key, fmt=None):
        if fmt is not None:
            if fmt.get('hex', False):
                safe_repr = SafeRepr()
                safe_repr.convert_to_hex = True
                return safe_repr(key)
        return '%r' % (key,)

    def init_dict(self):
        return {}

    def get_contents_debug_adapter_protocol(self, dct, fmt=None):
        '''
        This method is to be used in the case where the variables are all saved by its id (and as
        such don't need to have the `resolve` method called later on, so, keys don't need to
        embed the reference in the key).

        Note that the return should be ordered.

        :return list(tuple(name:str, value:object, evaluateName:str))
        '''
        ret = []

        i = 0

        found_representations = set()

        for key, val in dct.items():
            i += 1
            key_as_str = self.key_to_str(key, fmt)

            if key_as_str not in found_representations:
                found_representations.add(key_as_str)
            else:
                # If the key would be a duplicate, add the key id (otherwise
                # VSCode won't show all keys correctly).
                # See: https://github.com/microsoft/debugpy/issues/148
                key_as_str = '%s (id: %s)' % (key_as_str, id(key))
                found_representations.add(key_as_str)

            if _does_obj_repr_evaluate_to_obj(key):
                s = self.key_to_str(key)  # do not format the key
                eval_key_str = '[%s]' % (s,)
            else:
                eval_key_str = None
            ret.append((key_as_str, val, eval_key_str))
            if i >= pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS:
                ret.append((TOO_LARGE_ATTR, TOO_LARGE_MSG % (pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS,), None))
                break

        # in case the class extends built-in type and has some additional fields
        from_default_resolver = defaultResolver.get_contents_debug_adapter_protocol(dct, fmt)

        if from_default_resolver:
            ret = from_default_resolver + ret

        if self.sort_keys:
            ret = sorted(ret, key=lambda tup: sorted_attributes_key(tup[0]))

        ret.append((GENERATED_LEN_ATTR_NAME, len(dct), partial(_apply_evaluate_name, evaluate_name='len(%s)')))
        return ret

    def get_dictionary(self, dct):
        ret = self.init_dict()

        i = 0
        for key, val in dct.items():
            i += 1
            # we need to add the id because otherwise we cannot find the real object to get its contents later on.
            key = '%s (%s)' % (self.key_to_str(key), id(key))
            ret[key] = val
            if i >= pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS:
                ret[TOO_LARGE_ATTR] = TOO_LARGE_MSG % (pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS,)
                break

        # in case if the class extends built-in type and has some additional fields
        additional_fields = defaultResolver.get_dictionary(dct)
        ret.update(additional_fields)
        ret[GENERATED_LEN_ATTR_NAME] = len(dct)
        return ret


def _apply_evaluate_name(parent_name, evaluate_name):
    return evaluate_name % (parent_name,)


class MoreItemsRange:

    def __init__(self, value, from_i, to_i):
        self.value = value
        self.from_i = from_i
        self.to_i = to_i

    def get_contents_debug_adapter_protocol(self, _self, fmt=None):
        l = len(self.value)
        ret = []

        format_str = '%0' + str(int(len(str(l - 1)))) + 'd'
        if fmt is not None and fmt.get('hex', False):
            format_str = '0x%0' + str(int(len(hex(l).lstrip('0x')))) + 'x'

        for i, item in enumerate(self.value[self.from_i:self.to_i]):
            i += self.from_i
            ret.append((format_str % i, item, '[%s]' % i))
        return ret

    def get_dictionary(self, _self, fmt=None):
        dct = {}
        for key, obj, _ in self.get_contents_debug_adapter_protocol(self, fmt):
            dct[key] = obj
        return dct

    def resolve(self, attribute):
        '''
        :param var: that's the original object we're dealing with.
        :param attribute: that's the key to resolve
            -- either the dict key in get_dictionary or the name in the dap protocol.
        '''
        return self.value[int(attribute)]

    def __eq__(self, o):
        return isinstance(o, MoreItemsRange) and self.value is o.value and \
                self.from_i == o.from_i and self.to_i == o.to_i

    def __str__(self):
        return '[%s:%s]' % (self.from_i, self.to_i)

    __repr__ = __str__


class MoreItems:

    def __init__(self, value, handled_items):
        self.value = value
        self.handled_items = handled_items

    def get_contents_debug_adapter_protocol(self, _self, fmt=None):
        total_items = len(self.value)
        remaining = total_items - self.handled_items
        bucket_size = pydevd_constants.PYDEVD_CONTAINER_BUCKET_SIZE

        from_i = self.handled_items
        to_i = from_i + min(bucket_size, remaining)

        ret = []
        while remaining > 0:
            remaining -= bucket_size
            more_items_range = MoreItemsRange(self.value, from_i, to_i)
            ret.append((str(more_items_range), more_items_range, None))

            from_i = to_i
            to_i = from_i + min(bucket_size, remaining)

        return ret

    def get_dictionary(self, _self, fmt=None):
        dct = {}
        for key, obj, _ in self.get_contents_debug_adapter_protocol(self, fmt):
            dct[key] = obj
        return dct

    def resolve(self, attribute):
        from_i, to_i = attribute[1:-1].split(':')
        from_i = int(from_i)
        to_i = int(to_i)
        return MoreItemsRange(self.value, from_i, to_i)

    def __eq__(self, o):
        return isinstance(o, MoreItems) and self.value is o.value

    def __str__(self):
        return '...'

    __repr__ = __str__


class ForwardInternalResolverToObject:
    '''
    To be used when we provide some internal object that'll actually do the resolution.
    '''

    def get_contents_debug_adapter_protocol(self, obj, fmt=None):
        return obj.get_contents_debug_adapter_protocol(fmt)

    def get_dictionary(self, var, fmt={}):
        return var.get_dictionary(var, fmt)

    def resolve(self, var, attribute):
        return var.resolve(attribute)


class TupleResolver:  # to enumerate tuples and lists

    def resolve(self, var, attribute):
        '''
        :param var: that's the original object we're dealing with.
        :param attribute: that's the key to resolve
            -- either the dict key in get_dictionary or the name in the dap protocol.
        '''
        if attribute in (GENERATED_LEN_ATTR_NAME, TOO_LARGE_ATTR):
            return None
        try:
            return var[int(attribute)]
        except:
            if attribute == 'more':
                return MoreItems(var, pydevd_constants.PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS)

            return getattr(var, attribute)

    def get_contents_debug_adapter_protocol(self, lst, fmt=None):
        '''
        This method is to be used in the case where the variables are all saved by its id (and as
        such don't need to have the `resolve` method called later on, so, keys don't need to
        embed the reference in the key).

        Note that the return should be ordered.

        :return list(tuple(name:str, value:object, evaluateName:str))
        '''
        lst_len = len(lst)
        ret = []

        format_str = '%0' + str(int(len(str(lst_len - 1)))) + 'd'
        if fmt is not None and fmt.get('hex', False):
            format_str = '0x%0' + str(int(len(hex(lst_len).lstrip('0x')))) + 'x'

        initial_expanded = pydevd_constants.PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS
        for i, item in enumerate(lst):
            ret.append((format_str % i, item, '[%s]' % i))

            if i >= initial_expanded - 1:
                if (lst_len - initial_expanded) < pydevd_constants.PYDEVD_CONTAINER_BUCKET_SIZE:
                    # Special case: if we have just 1 more bucket just put it inline.
                    item = MoreItemsRange(lst, initial_expanded, lst_len)

                else:
                    # Multiple buckets
                    item = MoreItems(lst, initial_expanded)
                ret.append(('more', item, None))
                break

        # Needed in case the class extends the built-in type and has some additional fields.
        from_default_resolver = defaultResolver.get_contents_debug_adapter_protocol(lst, fmt=fmt)
        if from_default_resolver:
            ret = from_default_resolver + ret

        ret.append((GENERATED_LEN_ATTR_NAME, len(lst), partial(_apply_evaluate_name, evaluate_name='len(%s)')))
        return ret

    def get_dictionary(self, var, fmt={}):
        l = len(var)
        d = {}

        format_str = '%0' + str(int(len(str(l - 1)))) + 'd'
        if fmt is not None and fmt.get('hex', False):
            format_str = '0x%0' + str(int(len(hex(l).lstrip('0x')))) + 'x'

        initial_expanded = pydevd_constants.PYDEVD_CONTAINER_INITIAL_EXPANDED_ITEMS
        for i, item in enumerate(var):
            d[format_str % i] = item

            if i >= initial_expanded - 1:
                item = MoreItems(var, initial_expanded)
                d['more'] = item
                break

        # in case if the class extends built-in type and has some additional fields
        additional_fields = defaultResolver.get_dictionary(var)
        d.update(additional_fields)
        d[GENERATED_LEN_ATTR_NAME] = len(var)
        return d


#=======================================================================================================================
# SetResolver
#=======================================================================================================================
class SetResolver:
    '''
        Resolves a set as dict id(object)->object
    '''

    def get_contents_debug_adapter_protocol(self, obj, fmt=None):
        ret = []

        for i, item in enumerate(obj):
            ret.append((str(id(item)), item, None))

            if i >= pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS:
                ret.append((TOO_LARGE_ATTR, TOO_LARGE_MSG % (pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS,), None))
                break

        # Needed in case the class extends the built-in type and has some additional fields.
        from_default_resolver = defaultResolver.get_contents_debug_adapter_protocol(obj, fmt=fmt)
        if from_default_resolver:
            ret = from_default_resolver + ret
        ret.append((GENERATED_LEN_ATTR_NAME, len(obj), partial(_apply_evaluate_name, evaluate_name='len(%s)')))
        return ret

    def resolve(self, var, attribute):
        if attribute in (GENERATED_LEN_ATTR_NAME, TOO_LARGE_ATTR):
            return None

        try:
            attribute = int(attribute)
        except:
            return getattr(var, attribute)

        for v in var:
            if id(v) == attribute:
                return v

        raise UnableToResolveVariableException('Unable to resolve %s in %s' % (attribute, var))

    def get_dictionary(self, var):
        d = {}
        for i, item in enumerate(var):
            d[str(id(item))] = item

            if i >= pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS:
                d[TOO_LARGE_ATTR] = TOO_LARGE_MSG % (pydevd_constants.PYDEVD_CONTAINER_RANDOM_ACCESS_MAX_ITEMS,)
                break

        # in case if the class extends built-in type and has some additional fields
        additional_fields = defaultResolver.get_dictionary(var)
        d.update(additional_fields)
        d[GENERATED_LEN_ATTR_NAME] = len(var)
        return d

    def change_var_from_name(self, container, name, new_value):
        # The name given in this case must be the id(item), so, we can actually
        # iterate in the set and see which item matches the given id.

        try:
            # Check that the new value can actually be added to a set (i.e.: it's hashable/comparable).
            set().add(new_value)
        except:
            return None

        for item in container:
            if str(id(item)) == name:
                container.remove(item)
                container.add(new_value)
                return str(id(new_value))

        return None


#=======================================================================================================================
# InstanceResolver
#=======================================================================================================================
class InstanceResolver:

    def resolve(self, var, attribute):
        field = var.__class__.getDeclaredField(attribute)
        field.setAccessible(True)
        return field.get(var)

    def get_dictionary(self, obj):
        ret = {}

        declaredFields = obj.__class__.getDeclaredFields()
        for i in range(len(declaredFields)):
            name = declaredFields[i].getName()
            try:
                declaredFields[i].setAccessible(True)
                ret[name] = declaredFields[i].get(obj)
            except:
                pydev_log.exception()

        return ret


#=======================================================================================================================
# JyArrayResolver
#=======================================================================================================================
class JyArrayResolver:
    '''
        This resolves a regular Object[] array from java
    '''

    def resolve(self, var, attribute):
        if attribute == GENERATED_LEN_ATTR_NAME:
            return None
        return var[int(attribute)]

    def get_dictionary(self, obj):
        ret = {}

        for i in range(len(obj)):
            ret[ i ] = obj[i]

        ret[GENERATED_LEN_ATTR_NAME] = len(obj)
        return ret


#=======================================================================================================================
# MultiValueDictResolver
#=======================================================================================================================
class MultiValueDictResolver(DictResolver):

    def resolve(self, dct, key):
        if key in (GENERATED_LEN_ATTR_NAME, TOO_LARGE_ATTR):
            return None

        # ok, we have to iterate over the items to find the one that matches the id, because that's the only way
        # to actually find the reference from the string we have before.
        expected_id = int(key.split('(')[-1][:-1])
        for key in list(dct.keys()):
            val = dct.getlist(key)
            if id(key) == expected_id:
                return val

        raise UnableToResolveVariableException()


#=======================================================================================================================
# DjangoFormResolver
#=======================================================================================================================
class DjangoFormResolver(DefaultResolver):

    def get_dictionary(self, var, names=None):
        # Do not call self.errors because it is a property and has side effects.
        names, used___dict__ = self.get_names(var)

        has_errors_attr = False
        if "errors" in names:
            has_errors_attr = True
            names.remove("errors")

        d = defaultResolver.get_dictionary(var, names=names, used___dict__=used___dict__)
        if has_errors_attr:
            try:
                errors_attr = getattr(var, "_errors")
            except:
                errors_attr = None
            d["errors"] = errors_attr
        return d


#=======================================================================================================================
# DequeResolver
#=======================================================================================================================
class DequeResolver(TupleResolver):

    def get_dictionary(self, var):
        d = TupleResolver.get_dictionary(self, var)
        d['maxlen'] = getattr(var, 'maxlen', None)
        return d


#=======================================================================================================================
# OrderedDictResolver
#=======================================================================================================================
class OrderedDictResolver(DictResolver):

    sort_keys = False

    def init_dict(self):
        return OrderedDict()


#=======================================================================================================================
# FrameResolver
#=======================================================================================================================
class FrameResolver:
    '''
    This resolves a frame.
    '''

    def resolve(self, obj, attribute):
        if attribute == '__internals__':
            return defaultResolver.get_dictionary(obj)

        if attribute == 'stack':
            return self.get_frame_stack(obj)

        if attribute == 'f_locals':
            return obj.f_locals

        return None

    def get_dictionary(self, obj):
        ret = {}
        ret['__internals__'] = defaultResolver.get_dictionary(obj)
        ret['stack'] = self.get_frame_stack(obj)
        ret['f_locals'] = obj.f_locals
        return ret

    def get_frame_stack(self, frame):
        ret = []
        if frame is not None:
            ret.append(self.get_frame_name(frame))

            while frame.f_back:
                frame = frame.f_back
                ret.append(self.get_frame_name(frame))

        return ret

    def get_frame_name(self, frame):
        if frame is None:
            return 'None'
        try:
            name = basename(frame.f_code.co_filename)
            return 'frame: %s [%s:%s]  id:%s' % (frame.f_code.co_name, name, frame.f_lineno, id(frame))
        except:
            return 'frame object'


defaultResolver = DefaultResolver()
dictResolver = DictResolver()
tupleResolver = TupleResolver()
instanceResolver = InstanceResolver()
jyArrayResolver = JyArrayResolver()
setResolver = SetResolver()
multiValueDictResolver = MultiValueDictResolver()
djangoFormResolver = DjangoFormResolver()
dequeResolver = DequeResolver()
orderedDictResolver = OrderedDictResolver()
frameResolver = FrameResolver()
dapGrouperResolver = DAPGrouperResolver()
forwardInternalResolverToObject = ForwardInternalResolverToObject()


class InspectStub:

    def isbuiltin(self, _args):
        return False

    def isroutine(self, object):
        return False


try:
    import inspect
except:
    inspect = InspectStub()


def get_var_scope(attr_name, attr_value, evaluate_name, handle_return_values):
    if attr_name.startswith("'"):
        if attr_name.endswith("'"):
            # i.e.: strings denote that it is a regular value in some container.
            return ''
        else:
            i = attr_name.find("__' (")
            if i >= 0:
                # Handle attr_name such as: >>'__name__' (1732494379184)<<
                attr_name = attr_name[1: i + 2]

    if handle_return_values and attr_name == RETURN_VALUES_DICT:
        return ''

    elif attr_name == GENERATED_LEN_ATTR_NAME:
        return ''

    if attr_name.startswith('__') and attr_name.endswith('__'):
        return DAPGrouper.SCOPE_SPECIAL_VARS

    if attr_name.startswith('_') or attr_name.endswith('__'):
        return DAPGrouper.SCOPE_PROTECTED_VARS

    try:
        if inspect.isroutine(attr_value) or isinstance(attr_value, MethodWrapperType):
            return DAPGrouper.SCOPE_FUNCTION_VARS

        elif inspect.isclass(attr_value):
            return DAPGrouper.SCOPE_CLASS_VARS
    except:
        # It's possible that isinstance throws an exception when dealing with user-code.
        if DebugInfoHolder.DEBUG_TRACE_LEVEL > 0:
            pydev_log.exception()

    return ''
