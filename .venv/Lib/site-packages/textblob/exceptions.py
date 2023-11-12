# -*- coding: utf-8 -*-

MISSING_CORPUS_MESSAGE = """
Looks like you are missing some required data for this feature.

To download the necessary data, simply run

    python -m textblob.download_corpora

or use the NLTK downloader to download the missing data: http://nltk.org/data.html
If this doesn't fix the problem, file an issue at https://github.com/sloria/TextBlob/issues.
"""

class TextBlobError(Exception):
    """A TextBlob-related error."""
    pass


TextBlobException = TextBlobError  # Backwards compat

class MissingCorpusError(TextBlobError):
    """Exception thrown when a user tries to use a feature that requires a
    dataset or model that the user does not have on their system.
    """

    def __init__(self, message=MISSING_CORPUS_MESSAGE, *args, **kwargs):
        super(MissingCorpusError, self).__init__(message, *args, **kwargs)


MissingCorpusException = MissingCorpusError  # Backwards compat

class DeprecationError(TextBlobError):
    """Raised when user uses a deprecated feature."""
    pass

class TranslatorError(TextBlobError):
    """Raised when an error occurs during language translation or detection."""
    pass

class NotTranslated(TranslatorError):
    """Raised when text is unchanged after translation. This may be due to the language
    being unsupported by the translator.
    """
    pass

class FormatError(TextBlobError):
    """Raised if a data file with an unsupported format is passed to a classifier."""
    pass
