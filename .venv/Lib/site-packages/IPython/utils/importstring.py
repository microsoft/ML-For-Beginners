# encoding: utf-8
"""
A simple utility to import something by its string name.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


def import_item(name):
    """Import and return ``bar`` given the string ``foo.bar``.

    Calling ``bar = import_item("foo.bar")`` is the functional equivalent of
    executing the code ``from foo import bar``.

    Parameters
    ----------
    name : string
        The fully qualified name of the module/package being imported.

    Returns
    -------
    mod : module object
        The module that was imported.
    """
    
    parts = name.rsplit('.', 1)
    if len(parts) == 2:
        # called with 'foo.bar....'
        package, obj = parts
        module = __import__(package, fromlist=[obj])
        try:
            pak = getattr(module, obj)
        except AttributeError as e:
            raise ImportError('No module named %s' % obj) from e
        return pak
    else:
        # called with un-dotted string
        return __import__(parts[0])
