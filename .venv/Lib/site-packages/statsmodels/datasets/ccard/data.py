"""Bill Greene's credit scoring data."""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission of the original author, who
retains all rights."""
TITLE       = __doc__
SOURCE      = """
William Greene's `Econometric Analysis`

More information can be found at the web site of the text:
http://pages.stern.nyu.edu/~wgreene/Text/econometricanalysis.htm
"""

DESCRSHORT  = """William Greene's credit scoring data"""

DESCRLONG   = """More information on this data can be found on the
homepage for Greene's `Econometric Analysis`. See source.
"""

NOTE        = """::

    Number of observations - 72
    Number of variables - 5
    Variable name definitions - See Source for more information on the
                                variables.
"""


def load_pandas():
    """Load the credit card data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx=0)


def load():
    """Load the credit card data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def _get_data():
    return du.load_csv(__file__, 'ccard.csv', convert_float=True)
