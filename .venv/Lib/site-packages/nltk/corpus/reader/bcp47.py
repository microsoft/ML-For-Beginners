# Natural Language Toolkit: BCP-47 language tags
#
# Copyright (C) 2022-2023 NLTK Project
# Author: Eric Kafe <kafe.eric@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import re
from warnings import warn
from xml.etree import ElementTree as et

from nltk.corpus.reader import CorpusReader


class BCP47CorpusReader(CorpusReader):
    """
    Parse BCP-47 composite language tags

    Supports all the main subtags, and the 'u-sd' extension:

    >>> from nltk.corpus import bcp47
    >>> bcp47.name('oc-gascon-u-sd-fr64')
    'Occitan (post 1500): Gascon: Pyrénées-Atlantiques'

    Can load a conversion table to Wikidata Q-codes:
    >>> bcp47.load_wiki_q()
    >>> bcp47.wiki_q['en-GI-spanglis']
    'Q79388'

    """

    def __init__(self, root, fileids):
        """Read the BCP-47 database"""
        super().__init__(root, fileids)
        self.langcode = {}
        with self.open("iana/language-subtag-registry.txt") as fp:
            self.db = self.data_dict(fp.read().split("%%\n"))
        with self.open("cldr/common-subdivisions-en.xml") as fp:
            self.subdiv = self.subdiv_dict(
                et.parse(fp).iterfind("localeDisplayNames/subdivisions/subdivision")
            )
        self.morphology()

    def load_wiki_q(self):
        """Load conversion table to Wikidata Q-codes (only if needed)"""
        with self.open("cldr/tools-cldr-rdf-external-entityToCode.tsv") as fp:
            self.wiki_q = self.wiki_dict(fp.read().strip().split("\n")[1:])

    def wiki_dict(self, lines):
        """Convert Wikidata list of Q-codes to a BCP-47 dictionary"""
        return {
            pair[1]: pair[0].split("/")[-1]
            for pair in [line.strip().split("\t") for line in lines]
        }

    def subdiv_dict(self, subdivs):
        """Convert the CLDR subdivisions list to a dictionary"""
        return {sub.attrib["type"]: sub.text for sub in subdivs}

    def morphology(self):
        self.casing = {
            "language": str.lower,
            "extlang": str.lower,
            "script": str.title,
            "region": str.upper,
            "variant": str.lower,
        }
        dig = "[0-9]"
        low = "[a-z]"
        up = "[A-Z]"
        alnum = "[a-zA-Z0-9]"
        self.format = {
            "language": re.compile(f"{low*3}?"),
            "extlang": re.compile(f"{low*3}"),
            "script": re.compile(f"{up}{low*3}"),
            "region": re.compile(f"({up*2})|({dig*3})"),
            "variant": re.compile(f"{alnum*4}{(alnum+'?')*4}"),
            "singleton": re.compile(f"{low}"),
        }

    def data_dict(self, records):
        """Convert the BCP-47 language subtag registry to a dictionary"""
        self.version = records[0].replace("File-Date:", "").strip()
        dic = {}
        dic["deprecated"] = {}
        for label in [
            "language",
            "extlang",
            "script",
            "region",
            "variant",
            "redundant",
            "grandfathered",
        ]:
            dic["deprecated"][label] = {}
        for record in records[1:]:
            fields = [field.split(": ") for field in record.strip().split("\n")]
            typ = fields[0][1]
            tag = fields[1][1]
            if typ not in dic:
                dic[typ] = {}
            subfields = {}
            for field in fields[2:]:
                if len(field) == 2:
                    [key, val] = field
                    if key not in subfields:
                        subfields[key] = [val]
                    else:  # multiple value
                        subfields[key].append(val)
                else:  # multiline field
                    subfields[key][-1] += " " + field[0].strip()
                if (
                    "Deprecated" not in record
                    and typ == "language"
                    and key == "Description"
                ):
                    self.langcode[subfields[key][-1]] = tag
            for key in subfields:
                if len(subfields[key]) == 1:  # single value
                    subfields[key] = subfields[key][0]
            if "Deprecated" in record:
                dic["deprecated"][typ][tag] = subfields
            else:
                dic[typ][tag] = subfields
        return dic

    def val2str(self, val):
        """Return only first value"""
        if type(val) == list:
            #            val = "/".join(val) # Concatenate all values
            val = val[0]
        return val

    def lang2str(self, lg_record):
        """Concatenate subtag values"""
        name = f"{lg_record['language']}"
        for label in ["extlang", "script", "region", "variant", "extension"]:
            if label in lg_record:
                name += f": {lg_record[label]}"
        return name

    def parse_tag(self, tag):
        """Convert a BCP-47 tag to a dictionary of labelled subtags"""
        subtags = tag.split("-")
        lang = {}
        labels = ["language", "extlang", "script", "region", "variant", "variant"]
        while subtags and labels:
            subtag = subtags.pop(0)
            found = False
            while labels:
                label = labels.pop(0)
                subtag = self.casing[label](subtag)
                if self.format[label].fullmatch(subtag):
                    if subtag in self.db[label]:
                        found = True
                        valstr = self.val2str(self.db[label][subtag]["Description"])
                        if label == "variant" and label in lang:
                            lang[label] += ": " + valstr
                        else:
                            lang[label] = valstr
                        break
                    elif subtag in self.db["deprecated"][label]:
                        found = True
                        note = f"The {subtag!r} {label} code is deprecated"
                        if "Preferred-Value" in self.db["deprecated"][label][subtag]:
                            prefer = self.db["deprecated"][label][subtag][
                                "Preferred-Value"
                            ]
                            note += f"', prefer '{self.val2str(prefer)}'"
                        lang[label] = self.val2str(
                            self.db["deprecated"][label][subtag]["Description"]
                        )
                        warn(note)
                        break
            if not found:
                if subtag == "u" and subtags[0] == "sd":  # CLDR regional subdivisions
                    sd = subtags[1]
                    if sd in self.subdiv:
                        ext = self.subdiv[sd]
                    else:
                        ext = f"<Unknown subdivision: {ext}>"
                else:  # other extension subtags are not supported yet
                    ext = f"{subtag}{''.join(['-'+ext for ext in subtags])}".lower()
                    if not self.format["singleton"].fullmatch(subtag):
                        ext = f"<Invalid extension: {ext}>"
                        warn(ext)
                lang["extension"] = ext
                subtags = []
        return lang

    def name(self, tag):
        """
        Convert a BCP-47 tag to a colon-separated string of subtag names

        >>> from nltk.corpus import bcp47
        >>> bcp47.name('ca-Latn-ES-valencia')
        'Catalan: Latin: Spain: Valencian'

        """
        for label in ["redundant", "grandfathered"]:
            val = None
            if tag in self.db[label]:
                val = f"{self.db[label][tag]['Description']}"
                note = f"The {tag!r} code is {label}"
            elif tag in self.db["deprecated"][label]:
                val = f"{self.db['deprecated'][label][tag]['Description']}"
                note = f"The {tag!r} code is {label} and deprecated"
                if "Preferred-Value" in self.db["deprecated"][label][tag]:
                    prefer = self.db["deprecated"][label][tag]["Preferred-Value"]
                    note += f", prefer {self.val2str(prefer)!r}"
            if val:
                warn(note)
                return val
        try:
            return self.lang2str(self.parse_tag(tag))
        except:
            warn(f"Tag {tag!r} was not recognized")
            return None
