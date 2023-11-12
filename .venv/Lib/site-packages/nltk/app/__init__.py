# Natural Language Toolkit: Applications package
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Interactive NLTK Applications:

chartparser:  Chart Parser
chunkparser:  Regular-Expression Chunk Parser
collocations: Find collocations in text
concordance:  Part-of-speech concordancer
nemo:         Finding (and Replacing) Nemo regular expression tool
rdparser:     Recursive Descent Parser
srparser:     Shift-Reduce Parser
wordnet:      WordNet Browser
"""


# Import Tkinter-based modules if Tkinter is installed
try:
    import tkinter
except ImportError:
    import warnings

    warnings.warn("nltk.app package not loaded (please install Tkinter library).")
else:
    from nltk.app.chartparser_app import app as chartparser
    from nltk.app.chunkparser_app import app as chunkparser
    from nltk.app.collocations_app import app as collocations
    from nltk.app.concordance_app import app as concordance
    from nltk.app.nemo_app import app as nemo
    from nltk.app.rdparser_app import app as rdparser
    from nltk.app.srparser_app import app as srparser
    from nltk.app.wordnet_app import app as wordnet

    try:
        from matplotlib import pylab
    except ImportError:
        import warnings

        warnings.warn("nltk.app.wordfreq not loaded (requires the matplotlib library).")
    else:
        from nltk.app.wordfreq_app import app as wordfreq
