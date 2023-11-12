"""Taxation Powers Vote for the Scottish Parliament 1997 dataset."""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission from the original author,
who retains all rights."""
TITLE       = "Taxation Powers Vote for the Scottish Parliament 1997"
SOURCE      = """
Jeff Gill's `Generalized Linear Models: A Unified Approach`

http://jgill.wustl.edu/research/books.html
"""
DESCRSHORT  = """Taxation Powers' Yes Vote for Scottish Parliamanet-1997"""

DESCRLONG   = """
This data is based on the example in Gill and describes the proportion of
voters who voted Yes to grant the Scottish Parliament taxation powers.
The data are divided into 32 council districts.  This example's explanatory
variables include the amount of council tax collected in pounds sterling as
of April 1997 per two adults before adjustments, the female percentage of
total claims for unemployment benefits as of January, 1998, the standardized
mortality rate (UK is 100), the percentage of labor force participation,
regional GDP, the percentage of children aged 5 to 15, and an interaction term
between female unemployment and the council tax.

The original source files and variable information are included in
/scotland/src/
"""

NOTE        = """::

    Number of Observations - 32 (1 for each Scottish district)

    Number of Variables - 8

    Variable name definitions::

        YES    - Proportion voting yes to granting taxation powers to the
                 Scottish parliament.
        COUTAX - Amount of council tax collected in pounds steling as of
                 April '97
        UNEMPF - Female percentage of total unemployment benefits claims as of
                January 1998
        MOR    - The standardized mortality rate (UK is 100)
        ACT    - Labor force participation (Short for active)
        GDP    - GDP per county
        AGE    - Percentage of children aged 5 to 15 in the county
        COUTAX_FEMALEUNEMP - Interaction between COUTAX and UNEMPF

    Council district names are included in the data file, though are not
    returned by load.
"""


def load():
    """
    Load the Scotvote data and returns a Dataset instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def load_pandas():
    """
    Load the Scotvote data and returns a Dataset instance.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx=0)


def _get_data():
    data = du.load_csv(__file__, 'scotvote.csv')
    data = data.iloc[:, 1:9]
    return data.astype(float)
