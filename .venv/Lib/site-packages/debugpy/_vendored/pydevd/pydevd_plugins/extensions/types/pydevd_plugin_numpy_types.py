from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider
from _pydevd_bundle.pydevd_resolver import defaultResolver
from .pydevd_helpers import find_mod_attr
from _pydevd_bundle import pydevd_constants

TOO_LARGE_MSG = 'Maximum number of items (%s) reached. To show more items customize the value of the PYDEVD_CONTAINER_NUMPY_MAX_ITEMS environment variable.'
TOO_LARGE_ATTR = 'Unable to handle:'


class NdArrayItemsContainer(object):
    pass


class NDArrayTypeResolveProvider(object):
    '''
    This resolves a numpy ndarray returning some metadata about the NDArray
    '''

    def can_provide(self, type_object, type_name):
        nd_array = find_mod_attr('numpy', 'ndarray')
        return nd_array is not None and issubclass(type_object, nd_array)

    def is_numeric(self, obj):
        if not hasattr(obj, 'dtype'):
            return False
        return obj.dtype.kind in 'biufc'

    def resolve(self, obj, attribute):
        if attribute == '__internals__':
            return defaultResolver.get_dictionary(obj)
        if attribute == 'min':
            if self.is_numeric(obj) and obj.size > 0:
                return obj.min()
            else:
                return None
        if attribute == 'max':
            if self.is_numeric(obj) and obj.size > 0:
                return obj.max()
            else:
                return None
        if attribute == 'shape':
            return obj.shape
        if attribute == 'dtype':
            return obj.dtype
        if attribute == 'size':
            return obj.size
        if attribute.startswith('['):
            container = NdArrayItemsContainer()
            i = 0
            format_str = '%0' + str(int(len(str(len(obj))))) + 'd'
            for item in obj:
                setattr(container, format_str % i, item)
                i += 1
                if i >= pydevd_constants.PYDEVD_CONTAINER_NUMPY_MAX_ITEMS:
                    setattr(container, TOO_LARGE_ATTR, TOO_LARGE_MSG % (pydevd_constants.PYDEVD_CONTAINER_NUMPY_MAX_ITEMS,))
                    break
            return container
        return None

    def get_dictionary(self, obj):
        ret = dict()
        ret['__internals__'] = defaultResolver.get_dictionary(obj)
        if obj.size > 1024 * 1024:
            ret['min'] = 'ndarray too big, calculating min would slow down debugging'
            ret['max'] = 'ndarray too big, calculating max would slow down debugging'
        elif obj.size == 0:
            ret['min'] = 'array is empty'
            ret['max'] = 'array is empty'
        else:
            if self.is_numeric(obj):
                ret['min'] = obj.min()
                ret['max'] = obj.max()
            else:
                ret['min'] = 'not a numeric object'
                ret['max'] = 'not a numeric object'
        ret['shape'] = obj.shape
        ret['dtype'] = obj.dtype
        ret['size'] = obj.size
        try:
            ret['[0:%s] ' % (len(obj))] = list(obj[0:pydevd_constants.PYDEVD_CONTAINER_NUMPY_MAX_ITEMS])
        except:
            # This may not work depending on the array shape.
            pass
        return ret


import sys

if not sys.platform.startswith("java"):
    TypeResolveProvider.register(NDArrayTypeResolveProvider)
