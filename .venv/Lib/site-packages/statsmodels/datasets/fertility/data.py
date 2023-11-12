"""World Bank Fertility Data."""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """This data is distributed according to the World Bank terms of use. See SOURCE."""
TITLE       = """World Bank Fertility Data"""
SOURCE      = """
This data has been acquired from

The World Bank: Fertility rate, total (births per woman): World Development Indicators

At the following URL: http://data.worldbank.org/indicator/SP.DYN.TFRT.IN

The sources for these statistics are listed as

(1) United Nations Population Division. World Population Prospects
(2) United Nations Statistical Division. Population and Vital Statistics Repot (various years)
(3) Census reports and other statistical publications from national statistical offices
(4) Eurostat: Demographic Statistics
(5) Secretariat of the Pacific Community: Statistics and Demography Programme
(6) U.S. Census Bureau: International Database

The World Bank Terms of Use can be found at the following URL

http://go.worldbank.org/OJC02YMLA0
"""

DESCRSHORT  = """Total fertility rate represents the number of children that would be born to a woman if she were to live to the end of her childbearing years and bear children in accordance with current age-specific fertility rates."""

DESCRLONG   = DESCRSHORT

#suggested notes
NOTE        = """
::

    This is panel data in wide-format

    Number of observations: 219
    Number of variables: 58
    Variable name definitions:
        Country Name
        Country Code
        Indicator Name - The World Bank Series indicator
        Indicator Code - The World Bank Series code
        1960 - 2013 - The fertility rate for the given year
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
    return du.Dataset(data=data)


def _get_data():
    return du.load_csv(__file__, 'fertility.csv')
