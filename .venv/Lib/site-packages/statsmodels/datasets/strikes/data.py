"""U.S. Strike Duration Data"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = __doc__
SOURCE      = """
This is a subset of the data used in Kennan (1985). It was originally
published by the Bureau of Labor Statistics.

::

    Kennan, J. 1985. "The duration of contract strikes in US manufacturing.
        `Journal of Econometrics` 28.1, 5-28.
"""

DESCRSHORT  = """Contains data on the length of strikes in US manufacturing and
unanticipated industrial production."""

DESCRLONG   = """Contains data on the length of strikes in US manufacturing and
unanticipated industrial production. The data is a subset of the data originally
used by Kennan. The data here is data for the months of June only to avoid
seasonal issues."""

#suggested notes
NOTE        = """::

    Number of observations - 62

    Number of variables - 2

    Variable name definitions::

                duration - duration of the strike in days
                iprod - unanticipated industrial production
"""



def load_pandas():
    """
    Load the strikes data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx=0)


def load():
    """
    Load the strikes data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def _get_data():
    return du.load_csv(__file__,'strikes.csv').astype(float)
