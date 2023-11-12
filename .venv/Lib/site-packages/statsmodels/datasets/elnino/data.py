"""El Nino dataset, 1950 - 2010"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This data is in the public domain."""

TITLE       = """El Nino - Sea Surface Temperatures"""

SOURCE      = """
National Oceanic and Atmospheric Administration's National Weather Service

ERSST.V3B dataset, Nino 1+2
http://www.cpc.ncep.noaa.gov/data/indices/
"""

DESCRSHORT  = """Averaged monthly sea surface temperature - Pacific Ocean."""

DESCRLONG   = """This data contains the averaged monthly sea surface
temperature in degrees Celcius of the Pacific Ocean, between 0-10 degrees South
and 90-80 degrees West, from 1950 to 2010.  This dataset was obtained from
NOAA.
"""

NOTE = """::

    Number of Observations - 61 x 12

    Number of Variables - 1

    Variable name definitions::

        TEMPERATURE - average sea surface temperature in degrees Celcius
                      (12 columns, one per month).
"""


def load_pandas():
    data = _get_data()
    dataset = du.Dataset(data=data, names=list(data.columns))
    return dataset


def load():
    """
    Load the El Nino data and return a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The elnino Dataset instance does not contain endog and exog attributes.
    """
    return load_pandas()


def _get_data():
    return du.load_csv(__file__, 'elnino.csv', convert_float=True)
