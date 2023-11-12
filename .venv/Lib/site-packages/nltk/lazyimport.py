# This module is from mx/DateTime/LazyModule.py and is
# distributed under the terms of the eGenix.com Public License Agreement
# https://www.egenix.com/products/eGenix.com-Public-License-1.1.0.pdf

""" Helper to enable simple lazy module import.

    'Lazy' means the actual import is deferred until an attribute is
    requested from the module's namespace. This has the advantage of
    allowing all imports to be done at the top of a script (in a
    prominent and visible place) without having a great impact
    on startup time.

    Copyright (c) 1999-2005, Marc-Andre Lemburg; mailto:mal@lemburg.com
    See the documentation for further information on copyrights,
    or contact the author. All Rights Reserved.
"""

### Constants

_debug = 0

###


class LazyModule:

    """Lazy module class.

    Lazy modules are imported into the given namespaces whenever a
    non-special attribute (there are some attributes like __doc__
    that class instances handle without calling __getattr__) is
    requested. The module is then registered under the given name
    in locals usually replacing the import wrapper instance. The
    import itself is done using globals as global namespace.

    Example of creating a lazy load module:

    ISO = LazyModule('ISO',locals(),globals())

    Later, requesting an attribute from ISO will load the module
    automatically into the locals() namespace, overriding the
    LazyModule instance:

    t = ISO.Week(1998,1,1)

    """

    # Flag which indicates whether the LazyModule is initialized or not
    __lazymodule_init = 0

    # Name of the module to load
    __lazymodule_name = ""

    # Flag which indicates whether the module was loaded or not
    __lazymodule_loaded = 0

    # Locals dictionary where to register the module
    __lazymodule_locals = None

    # Globals dictionary to use for the module import
    __lazymodule_globals = None

    def __init__(self, name, locals, globals=None):

        """Create a LazyModule instance wrapping module name.

        The module will later on be registered in locals under the
        given module name.

        globals is optional and defaults to locals.

        """
        self.__lazymodule_locals = locals
        if globals is None:
            globals = locals
        self.__lazymodule_globals = globals
        mainname = globals.get("__name__", "")
        if mainname:
            self.__name__ = mainname + "." + name
            self.__lazymodule_name = name
        else:
            self.__name__ = self.__lazymodule_name = name
        self.__lazymodule_init = 1

    def __lazymodule_import(self):

        """Import the module now."""
        # Load and register module
        local_name = self.__lazymodule_name  # e.g. "toolbox"
        full_name = self.__name__  # e.g. "nltk.toolbox"
        if self.__lazymodule_loaded:
            return self.__lazymodule_locals[local_name]
        if _debug:
            print("LazyModule: Loading module %r" % full_name)
        self.__lazymodule_locals[local_name] = module = __import__(
            full_name, self.__lazymodule_locals, self.__lazymodule_globals, "*"
        )

        # Fill namespace with all symbols from original module to
        # provide faster access.
        self.__dict__.update(module.__dict__)

        # Set import flag
        self.__dict__["__lazymodule_loaded"] = 1

        if _debug:
            print("LazyModule: Module %r loaded" % full_name)
        return module

    def __getattr__(self, name):

        """Import the module on demand and get the attribute."""
        if self.__lazymodule_loaded:
            raise AttributeError(name)
        if _debug:
            print(
                "LazyModule: "
                "Module load triggered by attribute %r read access" % name
            )
        module = self.__lazymodule_import()
        return getattr(module, name)

    def __setattr__(self, name, value):

        """Import the module on demand and set the attribute."""
        if not self.__lazymodule_init:
            self.__dict__[name] = value
            return
        if self.__lazymodule_loaded:
            self.__lazymodule_locals[self.__lazymodule_name] = value
            self.__dict__[name] = value
            return
        if _debug:
            print(
                "LazyModule: "
                "Module load triggered by attribute %r write access" % name
            )
        module = self.__lazymodule_import()
        setattr(module, name, value)

    def __repr__(self):
        return "<LazyModule '%s'>" % self.__name__
