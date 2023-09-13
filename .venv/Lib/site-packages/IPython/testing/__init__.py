"""Testing support (tools to test IPython itself).
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2009-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------


import os

#-----------------------------------------------------------------------------
# Constants
#-----------------------------------------------------------------------------

# We scale all timeouts via this factor, slow machines can increase it
IPYTHON_TESTING_TIMEOUT_SCALE = float(os.getenv(
                                    'IPYTHON_TESTING_TIMEOUT_SCALE', 1))
