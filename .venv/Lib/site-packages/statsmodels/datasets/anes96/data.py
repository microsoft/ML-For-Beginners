"""American National Election Survey 1996"""
from numpy import log

from statsmodels.datasets import utils as du

__docformat__ = 'restructuredtext'

COPYRIGHT = """This is public domain."""
TITLE = __doc__
SOURCE = """
http://www.electionstudies.org/

The American National Election Studies.
"""

DESCRSHORT = """This data is a subset of the American National Election Studies of 1996."""

DESCRLONG = DESCRSHORT

NOTE = """::

    Number of observations - 944
    Number of variables - 10

    Variables name definitions::

            popul - Census place population in 1000s
            TVnews - Number of times per week that respondent watches TV news.
            PID - Party identification of respondent.
                0 - Strong Democrat
                1 - Weak Democrat
                2 - Independent-Democrat
                3 - Independent-Indpendent
                4 - Independent-Republican
                5 - Weak Republican
                6 - Strong Republican
            age : Age of respondent.
            educ - Education level of respondent
                1 - 1-8 grades
                2 - Some high school
                3 - High school graduate
                4 - Some college
                5 - College degree
                6 - Master's degree
                7 - PhD
            income - Income of household
                1  - None or less than $2,999
                2  - $3,000-$4,999
                3  - $5,000-$6,999
                4  - $7,000-$8,999
                5  - $9,000-$9,999
                6  - $10,000-$10,999
                7  - $11,000-$11,999
                8  - $12,000-$12,999
                9  - $13,000-$13,999
                10 - $14,000-$14.999
                11 - $15,000-$16,999
                12 - $17,000-$19,999
                13 - $20,000-$21,999
                14 - $22,000-$24,999
                15 - $25,000-$29,999
                16 - $30,000-$34,999
                17 - $35,000-$39,999
                18 - $40,000-$44,999
                19 - $45,000-$49,999
                20 - $50,000-$59,999
                21 - $60,000-$74,999
                22 - $75,000-89,999
                23 - $90,000-$104,999
                24 - $105,000 and over
            vote - Expected vote
                0 - Clinton
                1 - Dole
            The following 3 variables all take the values:
                1 - Extremely liberal
                2 - Liberal
                3 - Slightly liberal
                4 - Moderate
                5 - Slightly conservative
                6 - Conservative
                7 - Extremely Conservative
            selfLR - Respondent's self-reported political leanings from "Left"
                to "Right".
            ClinLR - Respondents impression of Bill Clinton's political
                leanings from "Left" to "Right".
            DoleLR  - Respondents impression of Bob Dole's political leanings
                from "Left" to "Right".
            logpopul - log(popul + .1)
"""


def load_pandas():
    """Load the anes96 data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    data = _get_data()
    return du.process_pandas(data, endog_idx=5, exog_idx=[10, 2, 6, 7, 8])


def load():
    """Load the anes96 data and returns a Dataset class.

    Returns
    -------
    Dataset
        See DATASET_PROPOSAL.txt for more information.
    """
    return load_pandas()


def _get_data():
    data = du.load_csv(__file__, 'anes96.csv', sep=r'\s')
    data = du.strip_column_names(data)
    data['logpopul'] = log(data['popul'] + .1)
    return data.astype(float)
