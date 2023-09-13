"""Implementation of all the magic functions built into IPython.
"""
#-----------------------------------------------------------------------------
#  Copyright (c) 2012 The IPython Development Team.
#
#  Distributed under the terms of the Modified BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from ..magic import Magics, magics_class
from .auto import AutoMagics
from .basic import BasicMagics, AsyncMagics
from .code import CodeMagics, MacroToEdit
from .config import ConfigMagics
from .display import DisplayMagics
from .execution import ExecutionMagics
from .extension import ExtensionMagics
from .history import HistoryMagics
from .logging import LoggingMagics
from .namespace import NamespaceMagics
from .osm import OSMagics
from .packaging import PackagingMagics
from .pylab import PylabMagics
from .script import ScriptMagics

#-----------------------------------------------------------------------------
# Magic implementation classes
#-----------------------------------------------------------------------------

@magics_class
class UserMagics(Magics):
    """Placeholder for user-defined magics to be added at runtime.

    All magics are eventually merged into a single namespace at runtime, but we
    use this class to isolate the magics defined dynamically by the user into
    their own class.
    """
