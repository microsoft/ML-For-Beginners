import sys


def find_cached_module(mod_name):
    return sys.modules.get(mod_name, None)

def find_mod_attr(mod_name, attr):
    mod = find_cached_module(mod_name)
    if mod is None:
        return None
    return getattr(mod, attr, None)


def find_class_name(val):
    class_name = str(val.__class__)
    if class_name.find('.') != -1:
        class_name = class_name.split('.')[-1]

    elif class_name.find("'") != -1: #does not have '.' (could be something like <type 'int'>)
        class_name = class_name[class_name.index("'") + 1:]

    if class_name.endswith("'>"):
        class_name = class_name[:-2]

    return class_name

