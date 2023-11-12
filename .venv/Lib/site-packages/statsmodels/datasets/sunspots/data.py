"""Yearly sunspots data 1700-2008"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This data is public domain."""
TITLE       = __doc__
SOURCE      = """
http://www.ngdc.noaa.gov/stp/solar/solarda3.html

The original dataset contains monthly data on sunspot activity in the file
./src/sunspots_yearly.dat.  There is also sunspots_monthly.dat.
"""

DESCRSHORT  = """Yearly (1700-2008) data on sunspots from the National
Geophysical Data Center."""

DESCRLONG   = DESCRSHORT

NOTE        = """::

    Number of Observations - 309 (Annual 1700 - 2008)
    Number of Variables - 1
    Variable name definitions::

        SUNACTIVITY - Number of sunspots for each year

    The data file contains a 'YEAR' variable that is not returned by load.
"""


def load_pandas():
    data = _get_data()
    # TODO: time series
    endog = data.set_index(data.YEAR).SUNACTIVITY
    dataset = du.Dataset(data=data, names=list(data.columns),
                         endog=endog, endog_name='volume')
    return dataset


def load():
    """
    Load the yearly sunspot data and returns a data class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    This dataset only contains data for one variable, so the attributes
    data, raw_data, and endog are all the same variable.  There is no exog
    attribute defined.
    """
    return load_pandas()


def _get_data():
    return du.load_csv(__file__, 'sunspots.csv').astype(float)
