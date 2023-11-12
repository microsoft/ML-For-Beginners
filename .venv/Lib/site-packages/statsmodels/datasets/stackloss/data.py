"""Stack loss data"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain. """
TITLE       = __doc__
SOURCE      = """
Brownlee, K. A. (1965), "Statistical Theory and Methodology in
Science and Engineering", 2nd edition, New York:Wiley.
"""

DESCRSHORT  = """Stack loss plant data of Brownlee (1965)"""

DESCRLONG   = """The stack loss plant data of Brownlee (1965) contains
21 days of measurements from a plant's oxidation of ammonia to nitric acid.
The nitric oxide pollutants are captured in an absorption tower."""

NOTE        = """::

    Number of Observations - 21

    Number of Variables - 4

    Variable name definitions::

        STACKLOSS - 10 times the percentage of ammonia going into the plant
                    that escapes from the absoroption column
        AIRFLOW   - Rate of operation of the plant
        WATERTEMP - Cooling water temperature in the absorption tower
        ACIDCONC  - Acid concentration of circulating acid minus 50 times 10.
"""


def load():
    """
    Load the stack loss data and returns a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()

def load_pandas():
    """
    Load the stack loss data and returns a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx=0)


def _get_data():
    return du.load_csv(__file__, 'stackloss.csv').astype(float)
