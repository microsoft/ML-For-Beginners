"""Mauna Loa Weekly Atmospheric CO2 Data"""
import pandas as pd

from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = """Mauna Loa Weekly Atmospheric CO2 Data"""
SOURCE      = """
Data obtained from http://cdiac.ornl.gov/trends/co2/sio-keel-flask/sio-keel-flaskmlo_c.html

Obtained on 3/15/2014.

Citation:

Keeling, C.D. and T.P. Whorf. 2004. Atmospheric CO2 concentrations derived from flask air samples at sites in the SIO network. In Trends: A Compendium of Data on Global Change. Carbon Dioxide Information Analysis Center, Oak Ridge National Laboratory, U.S. Department of Energy, Oak Ridge, Tennessee, U.S.A.
"""

DESCRSHORT  = """Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A."""

DESCRLONG   = """
Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.

Period of Record: March 1958 - December 2001

Methods: An Applied Physics Corporation (APC) nondispersive infrared gas analyzer was used to obtain atmospheric CO2 concentrations, based on continuous data (four measurements per hour) from atop intake lines on several towers. Steady data periods of not less than six hours per day are required; if no such six-hour periods are available on any given day, then no data are used that day. Weekly averages were calculated for most weeks throughout the approximately 44 years of record. The continuous data for year 2000 is compared with flask data from the same site in the graphics section."""

#suggested notes
NOTE        = """::

    Number of observations: 2225
    Number of variables: 2
    Variable name definitions:

        date - sample date in YYMMDD format
        co2 - CO2 Concentration ppmv

    The data returned by load_pandas contains the dates as the index.
"""


def load_pandas():
    data = _get_data()
    index = pd.date_range(start=str(data['date'][0]), periods=len(data), freq='W-SAT')
    dataset = data[['co2']]
    dataset.index = index
    return du.Dataset(data=dataset, names=list(data.columns))


def load():
    """
    Load the data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()

def _get_data():
    return du.load_csv(__file__, 'co2.csv')
