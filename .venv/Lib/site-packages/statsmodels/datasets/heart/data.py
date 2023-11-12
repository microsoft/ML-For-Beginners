"""Heart Transplant Data, Miller 1976"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """???"""

TITLE       = """Transplant Survival Data"""

SOURCE      = """Miller, R. (1976). Least squares regression with censored data. Biometrica, 63 (3). 449-464.

"""

DESCRSHORT  = """Survival times after receiving a heart transplant"""

DESCRLONG   = """This data contains the survival time after receiving a heart transplant, the age of the patient and whether or not the survival time was censored.
"""

NOTE = """::

    Number of Observations - 69

    Number of Variables - 3

    Variable name definitions::
        death - Days after surgery until death
        age - age at the time of surgery
        censored - indicates if an observation is censored.  1 is uncensored
"""


def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def load_pandas():
    data = _get_data()
    dataset = du.process_pandas(data, endog_idx=0, exog_idx=None)
    dataset.censors = dataset.exog.iloc[:, 0]
    dataset.exog = dataset.exog.iloc[:, 1]
    return dataset


def _get_data():
    return du.load_csv(__file__, 'heart.csv')
