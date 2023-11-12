"""Nile River Flows."""
import pandas as pd

from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = """Nile River flows at Ashwan 1871-1970"""
SOURCE      = """
This data is first analyzed in:

    Cobb, G. W. 1978. "The Problem of the Nile: Conditional Solution to a
        Changepoint Problem." *Biometrika*. 65.2, 243-51.
"""

DESCRSHORT  = """This dataset contains measurements on the annual flow of
the Nile as measured at Ashwan for 100 years from 1871-1970."""

DESCRLONG   = DESCRSHORT + " There is an apparent changepoint near 1898."

#suggested notes
NOTE        = """::

    Number of observations: 100
    Number of variables: 2
    Variable name definitions:

        year - the year of the observations
        volumne - the discharge at Aswan in 10^8, m^3
"""


def load():
    """
    Load the Nile data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def load_pandas():
    data = _get_data()
    # TODO: time series
    endog = pd.Series(data['volume'], index=data['year'].astype(int))
    dataset = du.Dataset(data=data, names=list(data.columns), endog=endog, endog_name='volume')
    return dataset


def _get_data():
    return du.load_csv(__file__, 'nile.csv').astype(float)
