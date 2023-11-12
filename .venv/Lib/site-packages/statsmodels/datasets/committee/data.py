"""First 100 days of the US House of Representatives 1995"""
from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT   = """Used with express permission from the original author,
who retains all rights."""
TITLE       = __doc__
SOURCE      = """
Jeff Gill's `Generalized Linear Models: A Unifited Approach`

http://jgill.wustl.edu/research/books.html
"""

DESCRSHORT  = """Number of bill assignments in the 104th House in 1995"""

DESCRLONG   = """The example in Gill, seeks to explain the number of bill
assignments in the first 100 days of the US' 104th House of Representatives.
The response variable is the number of bill assignments in the first 100 days
over 20 Committees.  The explanatory variables in the example are the number of
assignments in the first 100 days of the 103rd House, the number of members on
the committee, the number of subcommittees, the log of the number of staff
assigned to the committee, a dummy variable indicating whether
the committee is a high prestige committee, and an interaction term between
the number of subcommittees and the log of the staff size.

The data returned by load are not cleaned to represent the above example.
"""

NOTE = """::

    Number of Observations - 20
    Number of Variables - 6
    Variable name definitions::

        BILLS104 - Number of bill assignments in the first 100 days of the
                   104th House of Representatives.
        SIZE     - Number of members on the committee.
        SUBS     - Number of subcommittees.
        STAFF    - Number of staff members assigned to the committee.
        PRESTIGE - PRESTIGE == 1 is a high prestige committee.
        BILLS103 - Number of bill assignments in the first 100 days of the
                   103rd House of Representatives.

    Committee names are included as a variable in the data file though not
    returned by load.
"""


def load_pandas():
    data = _get_data()
    return du.process_pandas(data, endog_idx=0)


def load():
    """Load the committee data and returns a data class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def _get_data():
    data = du.load_csv(__file__, 'committee.csv')
    data = data.iloc[:, 1:7].astype(float)
    return data
