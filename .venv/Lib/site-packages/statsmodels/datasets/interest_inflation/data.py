"""(West) German interest and inflation rate 1972-1998"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT = """..."""  # TODO
TITLE = __doc__
SOURCE = """
http://www.jmulti.de/download/datasets/e6.dat
"""

DESCRSHORT = """(West) German interest and inflation rate 1972Q2 - 1998Q4"""

DESCRLONG = """West German (until 1990) / German (afterwards) interest and
inflation rate 1972Q2 - 1998Q4
"""


NOTE = """::
    Number of Observations - 107

    Number of Variables - 2

    Variable name definitions::

        year      - 1972q2 - 1998q4
        quarter   - 1-4
        Dp        - Delta log gdp deflator
        R         - nominal long term interest rate
"""

variable_names = ["Dp", "R"]
first_season = 1  # 1 stands for: first observation in Q2 (0 would mean Q1)


def load():
    """
    Load the West German interest/inflation data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The interest_inflation Dataset instance does not contain endog and exog
    attributes.
    """
    return load_pandas()


def load_pandas():
    data = _get_data()
    names = data.columns
    dataset = du.Dataset(data=data, names=names)
    return dataset


def _get_data():
    return du.load_csv(__file__, 'E6.csv', convert_float=True)

def __str__():
    return "e6"
