"""US Capital Punishment dataset."""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission from the original author,
who retains all rights."""
TITLE       = __doc__
SOURCE      = """
Jeff Gill's `Generalized Linear Models: A Unified Approach`

http://jgill.wustl.edu/research/books.html
"""

DESCRSHORT  = """Number of state executions in 1997"""

DESCRLONG   = """This data describes the number of times capital punishment is implemented
at the state level for the year 1997.  The outcome variable is the number of
executions.  There were executions in 17 states.
Included in the data are explanatory variables for median per capita income
in dollars, the percent of the population classified as living in poverty,
the percent of Black citizens in the population, the rate of violent
crimes per 100,000 residents for 1996, a dummy variable indicating
whether the state is in the South, and (an estimate of) the proportion
of the population with a college degree of some kind.
"""

NOTE        = """::

    Number of Observations - 17
    Number of Variables - 7
    Variable name definitions::

        EXECUTIONS - Executions in 1996
        INCOME - Median per capita income in 1996 dollars
        PERPOVERTY - Percent of the population classified as living in poverty
        PERBLACK - Percent of black citizens in the population
        VC100k96 - Rate of violent crimes per 100,00 residents for 1996
        SOUTH - SOUTH == 1 indicates a state in the South
        DEGREE - An esimate of the proportion of the state population with a
            college degree of some kind

    State names are included in the data file, though not returned by load.
"""


def load_pandas():
    """
    Load the cpunish data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx=0)


def load():
    """
    Load the cpunish data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def _get_data():
    data = du.load_csv(__file__, 'cpunish.csv')
    data = data.iloc[:, 1:8].astype(float)
    return data
