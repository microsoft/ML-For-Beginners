"""
UDHR corpus reader. It mostly deals with encodings.
"""

from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import find_corpus_fileids


class UdhrCorpusReader(PlaintextCorpusReader):

    ENCODINGS = [
        (".*-Latin1$", "latin-1"),
        (".*-Hebrew$", "hebrew"),
        (".*-Arabic$", "cp1256"),
        ("Czech_Cesky-UTF8", "cp1250"),  # yeah
        ("Polish-Latin2", "cp1250"),
        ("Polish_Polski-Latin2", "cp1250"),
        (".*-Cyrillic$", "cyrillic"),
        (".*-SJIS$", "SJIS"),
        (".*-GB2312$", "GB2312"),
        (".*-Latin2$", "ISO-8859-2"),
        (".*-Greek$", "greek"),
        (".*-UTF8$", "utf-8"),
        ("Hungarian_Magyar-Unicode", "utf-16-le"),
        ("Amahuaca", "latin1"),
        ("Turkish_Turkce-Turkish", "latin5"),
        ("Lithuanian_Lietuviskai-Baltic", "latin4"),
        ("Japanese_Nihongo-EUC", "EUC-JP"),
        ("Japanese_Nihongo-JIS", "iso2022_jp"),
        ("Chinese_Mandarin-HZ", "hz"),
        (r"Abkhaz\-Cyrillic\+Abkh", "cp1251"),
    ]

    SKIP = {
        # The following files are not fully decodable because they
        # were truncated at wrong bytes:
        "Burmese_Myanmar-UTF8",
        "Japanese_Nihongo-JIS",
        "Chinese_Mandarin-HZ",
        "Chinese_Mandarin-UTF8",
        "Gujarati-UTF8",
        "Hungarian_Magyar-Unicode",
        "Lao-UTF8",
        "Magahi-UTF8",
        "Marathi-UTF8",
        "Tamil-UTF8",
        # Unfortunately, encodings required for reading
        # the following files are not supported by Python:
        "Vietnamese-VPS",
        "Vietnamese-VIQR",
        "Vietnamese-TCVN",
        "Magahi-Agra",
        "Bhojpuri-Agra",
        "Esperanto-T61",  # latin3 raises an exception
        # The following files are encoded for specific fonts:
        "Burmese_Myanmar-WinResearcher",
        "Armenian-DallakHelv",
        "Tigrinya_Tigrigna-VG2Main",
        "Amharic-Afenegus6..60375",  # ?
        "Navaho_Dine-Navajo-Navaho-font",
        # What are these?
        "Azeri_Azerbaijani_Cyrillic-Az.Times.Cyr.Normal0117",
        "Azeri_Azerbaijani_Latin-Az.Times.Lat0117",
        # The following files are unintended:
        "Czech-Latin2-err",
        "Russian_Russky-UTF8~",
    }

    def __init__(self, root="udhr"):
        fileids = find_corpus_fileids(root, r"(?!README|\.).*")
        super().__init__(
            root,
            [fileid for fileid in fileids if fileid not in self.SKIP],
            encoding=self.ENCODINGS,
        )
