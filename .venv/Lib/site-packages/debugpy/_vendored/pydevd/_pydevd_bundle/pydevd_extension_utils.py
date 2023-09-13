import pkgutil
import sys
from _pydev_bundle import pydev_log
try:
    import pydevd_plugins.extensions as extensions
except:
    pydev_log.exception()
    extensions = None


class ExtensionManager(object):

    def __init__(self):
        self.loaded_extensions = None
        self.type_to_instance = {}

    def _load_modules(self):
        self.loaded_extensions = []
        if extensions:
            for module_loader, name, ispkg in pkgutil.walk_packages(extensions.__path__,
                                                                    extensions.__name__ + '.'):
                mod_name = name.split('.')[-1]
                if not ispkg and mod_name.startswith('pydevd_plugin'):
                    try:
                        __import__(name)
                        module = sys.modules[name]
                        self.loaded_extensions.append(module)
                    except ImportError:
                        pydev_log.critical('Unable to load extension: %s', name)

    def _ensure_loaded(self):
        if self.loaded_extensions is None:
            self._load_modules()

    def _iter_attr(self):
        for extension in self.loaded_extensions:
            dunder_all = getattr(extension, '__all__', None)
            for attr_name in dir(extension):
                if not attr_name.startswith('_'):
                    if dunder_all is None or attr_name in dunder_all:
                        yield attr_name, getattr(extension, attr_name)

    def get_extension_classes(self, extension_type):
        self._ensure_loaded()
        if extension_type in self.type_to_instance:
            return self.type_to_instance[extension_type]
        handlers = self.type_to_instance.setdefault(extension_type, [])
        for attr_name, attr in self._iter_attr():
            if isinstance(attr, type) and issubclass(attr, extension_type) and attr is not extension_type:
                try:
                    handlers.append(attr())
                except:
                    pydev_log.exception('Unable to load extension class: %s', attr_name)
        return handlers


EXTENSION_MANAGER_INSTANCE = ExtensionManager()


def extensions_of_type(extension_type):
    """

    :param T extension_type:  The type of the extension hook
    :rtype: list[T]
    """
    return EXTENSION_MANAGER_INSTANCE.get_extension_classes(extension_type)

