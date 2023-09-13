"""Utility functions for finding modules

Utility functions for finding modules on sys.path.

"""
#-----------------------------------------------------------------------------
# Copyright (c) 2011, the IPython Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib imports
import importlib
import sys

# Third-party imports

# Our own imports


#-----------------------------------------------------------------------------
# Globals and constants
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Local utilities
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

def find_mod(module_name):
    """
    Find module `module_name` on sys.path, and return the path to module `module_name`.

    * If `module_name` refers to a module directory, then return path to `__init__` file.
        * If `module_name` is a directory without an __init__file, return None.

    * If module is missing or does not have a `.py` or `.pyw` extension, return None.
        * Note that we are not interested in running bytecode.

    * Otherwise, return the fill path of the module.

    Parameters
    ----------
    module_name : str

    Returns
    -------
    module_path : str
        Path to module `module_name`, its __init__.py, or None,
        depending on above conditions.
    """
    spec = importlib.util.find_spec(module_name)
    module_path = spec.origin
    if module_path is None:
        if spec.loader in sys.meta_path:
            return spec.loader
        return None
    else:
        split_path = module_path.split(".")
        if split_path[-1] in ["py", "pyw"]:
            return module_path
        else:
            return None
