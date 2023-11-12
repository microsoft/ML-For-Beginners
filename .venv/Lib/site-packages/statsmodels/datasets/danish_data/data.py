"""Danish Money Demand Data"""
import pandas as pd

from statsmodels.datasets import utils as du

__docformat__ = "restructuredtext"

COPYRIGHT = """This is public domain."""
TITLE = __doc__
SOURCE = """
Danish data used in S. Johansen and K. Juselius.  For estimating
estimating a money demand function::

    [1] Johansen, S. and Juselius, K. (1990), Maximum Likelihood Estimation
        and Inference on Cointegration - with Applications to the Demand
        for Money, Oxford Bulletin of Economics and Statistics, 52, 2,
        169-210.
"""

DESCRSHORT = """Danish Money Demand Data"""

DESCRLONG = DESCRSHORT

NOTE = """::
    Number of Observations - 55

    Number of Variables - 5

    Variable name definitions::

        lrm - Log real money
        lry - Log real income
        lpy - Log prices
        ibo - Bond rate
        ide - Deposit rate
"""


def load_pandas():
    data = _get_data()
    data.index.freq = "QS-JAN"
    return du.Dataset(data=data, names=list(data.columns))


def load():
    """
    Load the US macro data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The Dataset instance does not contain endog and exog attributes.
    """
    return load_pandas()


def _get_data():
    data = du.load_csv(__file__, "data.csv")
    for i, val in enumerate(data.period):
        parts = val.split("Q")
        month = (int(parts[1]) - 1) * 3 + 1

        data.loc[data.index[i], "period"] = f"{parts[0]}-{month:02d}-01"
    data["period"] = pd.to_datetime(data.period)
    return data.set_index("period").astype(float)


variable_names = ["lrm", "lry", "lpy", "ibo", "ide"]


def __str__():
    return "danish_data"
