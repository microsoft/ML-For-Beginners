"""Smoking and lung cancer in eight cities in China."""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Intern. J. Epidemiol. (1992)"""
TITLE       = __doc__
SOURCE      = """
Transcribed from Z. Liu, Smoking and Lung Cancer Incidence in China,
Intern. J. Epidemiol., 21:197-201, (1992).
"""

DESCRSHORT  = """Co-occurrence of lung cancer and smoking in 8 Chinese cities."""

DESCRLONG   = """This is a series of 8 2x2 contingency tables showing the co-occurrence
of lung cancer and smoking in 8 Chinese cities.
"""

NOTE        = """::

    Number of Observations - 8
    Number of Variables - 3
    Variable name definitions::

        city_name - name of the city
        smoking - yes or no, according to a person's smoking behavior
        lung_cancer - yes or no, according to a person's lung cancer status
"""


def load_pandas():
    """
    Load the China smoking/lung cancer data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    raw_data = du.load_csv(__file__, 'china_smoking.csv')
    data = raw_data.set_index('Location')
    dset = du.Dataset(data=data, title="Smoking and lung cancer in Chinese regions")
    dset.raw_data = raw_data
    return dset


def load():
    """
    Load the China smoking/lung cancer data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()
