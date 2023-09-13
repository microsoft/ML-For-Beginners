from _pydevd_bundle.pydevd_extension_api import StrPresentationProvider, TypeResolveProvider


class RectResolver(TypeResolveProvider):
    def get_dictionary(self, var):
        return {'length': var.length, 'width': var.width, 'area': var.length * var.width}

    def resolve(self, var, attribute):
        return getattr(var, attribute, None) if attribute != 'area' else var.length * var.width

    def can_provide(self, type_object, type_name):
        return type_name.endswith('Rect')


class RectToString(StrPresentationProvider):
    def get_str(self, val):
        return "Rectangle[Length: %s, Width: %s , Area: %s]" % (val.length, val.width, val.length * val.width)

    def can_provide(self, type_object, type_name):
        return type_name.endswith('Rect')
