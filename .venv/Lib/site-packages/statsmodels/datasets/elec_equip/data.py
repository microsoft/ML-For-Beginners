"""Euro area 18 - Total Turnover Index, Manufacture of electrical equipment"""
import os

import pandas as pd

from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT = """This is public domain."""
TITLE = __doc__
SOURCE = """
Data are from the Statistical Office of the European Commission (Eurostat)
"""

DESCRSHORT = """EU Manufacture of electrical equipment"""

DESCRLONG = DESCRSHORT

NOTE = """::
    Variable name definitions::

        date      - Date in format MMM-1-YYYY

        STS.M.I7.W.TOVT.NS0016.4.000   - Euro area 18 (fixed composition) -
            Total Turnover Index, NACE 26-27; Treatment and coating of metals;
            machining; Manufacture of electrical equipment - NACE Rev2;
            Eurostat; Working day adjusted, not seasonally adjusted
"""


def load_pandas():
    data = _get_data()
    return du.Dataset(data=data, names=list(data.columns))


def load():
    """
    Load the EU Electrical Equipment manufacturing data into a Dataset class

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
    curr_dir = os.path.split(os.path.abspath(__file__))[0]
    data = pd.read_csv(os.path.join(curr_dir, 'elec_equip.csv'))
    data.index = pd.to_datetime(data.pop('DATE'))
    return data


variable_names = ["elec_equip"]


def __str__():
    return "elec_equip"
