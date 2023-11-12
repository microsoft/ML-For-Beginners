"""Travel Mode Choice"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT = """This is public domain."""
TITLE = __doc__
SOURCE = """
Greene, W.H. and D. Hensher (1997) Multinomial logit and discrete choice models
in Greene, W. H. (1997) LIMDEP version 7.0 user's manual revised, Plainview,
New York econometric software, Inc.
Download from on-line complements to Greene, W.H. (2011) Econometric Analysis,
Prentice Hall, 7th Edition (data table F18-2)
http://people.stern.nyu.edu/wgreene/Text/Edition7/TableF18-2.csv
"""

DESCRSHORT = """Data used to study travel mode choice between Australian cities
"""

DESCRLONG = """The data, collected as part of a 1987 intercity mode choice
study, are a sub-sample of 210 non-business trips between Sydney, Canberra and
Melbourne in which the traveler chooses a mode from four alternatives (plane,
car, bus and train). The sample, 840 observations, is choice based with
over-sampling of the less popular modes (plane, train and bus) and under-sampling
of the more popular mode, car. The level of service data was derived from highway
and transport networks in Sydney, Melbourne, non-metropolitan N.S.W. and Victoria,
including the Australian Capital Territory."""

NOTE = """::

    Number of observations: 840 Observations On 4 Modes for 210 Individuals.
    Number of variables: 8
    Variable name definitions::

        individual = 1 to 210
        mode =
            1 - air
            2 - train
            3 - bus
            4 - car
        choice =
            0 - no
            1 - yes
        ttme = terminal waiting time for plane, train and bus (minutes); 0
               for car.
        invc = in vehicle cost for all stages (dollars).
        invt = travel time (in-vehicle time) for all stages (minutes).
        gc = generalized cost measure:invc+(invt*value of travel time savings)
            (dollars).
        hinc = household income ($1000s).
        psize = traveling group size in mode chosen (number)."""


def load():
    """
    Load the data modechoice data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def load_pandas():
    """
    Load the data modechoice data and return a Dataset class instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx = 2, exog_idx=[3,4,5,6,7,8])


def _get_data():
    return du.load_csv(__file__, 'modechoice.csv', sep=';', convert_float=True)
