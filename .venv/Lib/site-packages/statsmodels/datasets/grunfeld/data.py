"""Grunfeld (1950) Investment Data"""
import pandas as pd

from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = __doc__
SOURCE      = """This is the Grunfeld (1950) Investment Data.

The source for the data was the original 11-firm data set from Grunfeld's Ph.D.
thesis recreated by Kleiber and Zeileis (2008) "The Grunfeld Data at 50".
The data can be found here.
http://statmath.wu-wien.ac.at/~zeileis/grunfeld/

For a note on the many versions of the Grunfeld data circulating see:
http://www.stanford.edu/~clint/bench/grunfeld.htm
"""

DESCRSHORT  = """Grunfeld (1950) Investment Data for 11 U.S. Firms."""

DESCRLONG   = DESCRSHORT

NOTE        = """::

    Number of observations - 220 (20 years for 11 firms)

    Number of variables - 5

    Variables name definitions::

        invest  - Gross investment in 1947 dollars
        value   - Market value as of Dec. 31 in 1947 dollars
        capital - Stock of plant and equipment in 1947 dollars
        firm    - General Motors, US Steel, General Electric, Chrysler,
                Atlantic Refining, IBM, Union Oil, Westinghouse, Goodyear,
                Diamond Match, American Steel
        year    - 1935 - 1954

    Note that raw_data has firm expanded to dummy variables, since it is a
    string categorical variable.
"""

def load():
    """
    Loads the Grunfeld data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    raw_data has the firm variable expanded to dummy variables for each
    firm (ie., there is no reference dummy)
    """
    return load_pandas()

def load_pandas():
    """
    Loads the Grunfeld data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    raw_data has the firm variable expanded to dummy variables for each
    firm (ie., there is no reference dummy)
    """
    data = _get_data()
    data.year = data.year.astype(float)
    raw_data = pd.get_dummies(data)
    ds = du.process_pandas(data, endog_idx=0)
    ds.raw_data = raw_data
    return ds


def _get_data():
    data = du.load_csv(__file__, 'grunfeld.csv')
    return data
