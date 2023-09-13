#*****************************************************************************
# Copyright (C) 2016 The IPython Team <ipython-dev@scipy.org>
#
# Distributed under the terms of the BSD License.  The full license is in
# the file COPYING, distributed as part of this software.
#*****************************************************************************

"""
Color managing related utilities
"""

import pygments

from traitlets.config import Configurable
from traitlets import Unicode


available_themes = lambda : [s for s in pygments.styles.get_all_styles()]+['NoColor','LightBG','Linux', 'Neutral']

class Colorable(Configurable):
    """
    A subclass of configurable for all the classes that have a `default_scheme`
    """
    default_style=Unicode('LightBG').tag(config=True)

