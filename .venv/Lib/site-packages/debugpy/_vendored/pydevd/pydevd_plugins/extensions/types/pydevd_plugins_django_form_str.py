from _pydevd_bundle.pydevd_extension_api import StrPresentationProvider
from .pydevd_helpers import find_mod_attr, find_class_name


class DjangoFormStr(object):
    def can_provide(self, type_object, type_name):
        form_class = find_mod_attr('django.forms', 'Form')
        return form_class is not None and issubclass(type_object, form_class)

    def get_str(self, val):
        return '%s: %r' % (find_class_name(val), val)

import sys

if not sys.platform.startswith("java"):
    StrPresentationProvider.register(DjangoFormStr)
