# -*- coding: utf-8 -*-
'''The pluralize and singular methods from the pattern library.

Licenced under the BSD.
See here https://github.com/clips/pattern/blob/master/LICENSE.txt for
complete license information.
'''
import re

VERB, NOUN, ADJECTIVE, ADVERB = "VB", "NN", "JJ", "RB"

#### PLURALIZE #####################################################################################
# Based on "An Algorithmic Approach to English Pluralization" by Damian Conway:
# http://www.csse.monash.edu.au/~damian/papers/HTML/Plurals.html

# Prepositions are used to solve things like
# "mother-in-law" or "man at arms"
plural_prepositions = [
    "about", "above", "across", "after", "among", "around", "at", "athwart", "before", "behind",
    "below", "beneath", "beside", "besides", "between", "betwixt", "beyond", "but", "by", "during",
    "except", "for", "from", "in", "into", "near", "of", "off", "on", "onto", "out", "over",
    "since", "till", "to", "under", "until", "unto", "upon", "with"
]

# Inflection rules that are either general,
# or apply to a certain category of words,
# or apply to a certain category of words only in classical mode,
# or apply only in classical mode.
# Each rule consists of:
# suffix, inflection, category and classic flag.
plural_rules = [
    # 0) Indefinite articles and demonstratives.
    [["^a$|^an$", "some", None, False],
     ["^this$", "these", None, False],
     ["^that$", "those", None, False],
     ["^any$", "all", None, False]
    ],
    # 1) Possessive adjectives.
    # Overlaps with 1/ for "his" and "its".
    # Overlaps with 2/ for "her".
    [["^my$", "our", None, False],
     ["^your$|^thy$", "your", None, False],
     ["^her$|^his$|^its$|^their$", "their", None, False]
    ],
    # 2) Possessive pronouns.
    [["^mine$", "ours", None, False],
     ["^yours$|^thine$", "yours", None, False],
     ["^hers$|^his$|^its$|^theirs$", "theirs", None, False]
    ],
    # 3) Personal pronouns.
    [["^I$", "we", None, False],
     ["^me$", "us", None, False],
     ["^myself$", "ourselves", None, False],
     ["^you$", "you", None, False],
     ["^thou$|^thee$", "ye", None, False],
     ["^yourself$|^thyself$", "yourself", None, False],
     ["^she$|^he$|^it$|^they$", "they", None, False],
     ["^her$|^him$|^it$|^them$", "them", None, False],
     ["^herself$|^himself$|^itself$|^themself$", "themselves", None, False],
     ["^oneself$", "oneselves", None, False]
    ],
    # 4) Words that do not inflect.
    [["$", "", "uninflected", False],
     ["$", "", "uncountable", False],
     ["fish$", "fish", None, False],
     ["([- ])bass$", "\\1bass", None, False],
     ["ois$", "ois", None, False],
     ["sheep$", "sheep", None, False],
     ["deer$", "deer", None, False],
     ["pox$", "pox", None, False],
     ["([A-Z].*)ese$", "\\1ese", None, False],
     ["itis$", "itis", None, False],
     ["(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$", "\\1ose", None, False]
    ],
    # 5) Irregular plurals (mongoose, oxen).
    [["atlas$", "atlantes", None, True],
     ["atlas$", "atlases", None, False],
     ["beef$", "beeves", None, True],
     ["brother$", "brethren", None, True],
     ["child$", "children", None, False],
     ["corpus$", "corpora", None, True],
     ["corpus$", "corpuses", None, False],
     ["^cow$", "kine", None, True],
     ["ephemeris$", "ephemerides", None, False],
     ["ganglion$", "ganglia", None, True],
     ["genie$", "genii", None, True],
     ["genus$", "genera", None, False],
     ["graffito$", "graffiti", None, False],
     ["loaf$", "loaves", None, False],
     ["money$", "monies", None, True],
     ["mongoose$", "mongooses", None, False],
     ["mythos$", "mythoi", None, False],
     ["octopus$", "octopodes", None, True],
     ["opus$", "opera", None, True],
     ["opus$", "opuses", None, False],
     ["^ox$", "oxen", None, False],
     ["penis$", "penes", None, True],
     ["penis$", "penises", None, False],
     ["soliloquy$", "soliloquies", None, False],
     ["testis$", "testes", None, False],
     ["trilby$", "trilbys", None, False],
     ["turf$", "turves", None, True],
     ["numen$", "numena", None, False],
     ["occiput$", "occipita", None, True]
    ],
    # 6) Irregular inflections for common suffixes (synopses, mice, men).
    [["man$", "men", None, False],
     ["person$", "people", None, False],
     ["([lm])ouse$", "\\1ice", None, False],
     ["tooth$", "teeth", None, False],
     ["goose$", "geese", None, False],
     ["foot$", "feet", None, False],
     ["zoon$", "zoa", None, False],
     ["([csx])is$", "\\1es", None, False]
    ],
    # 7) Fully assimilated classical inflections (vertebrae, codices).
    [["ex$", "ices", "ex-ices", False],
     ["ex$", "ices", "ex-ices-classical", True],
     ["um$", "a", "um-a", False],
     ["um$", "a", "um-a-classical", True],
     ["on$", "a", "on-a", False],
     ["a$", "ae", "a-ae", False],
     ["a$", "ae", "a-ae-classical", True]
    ],
    # 8) Classical variants of modern inflections (stigmata, soprani).
    [["trix$", "trices", None, True],
     ["eau$", "eaux", None, True],
     ["ieu$", "ieu", None, True],
     ["([iay])nx$", "\\1nges", None, True],
     ["en$", "ina", "en-ina-classical", True],
     ["a$", "ata", "a-ata-classical", True],
     ["is$", "ides", "is-ides-classical", True],
     ["us$", "i", "us-i-classical", True],
     ["us$", "us", "us-us-classical", True],
     ["o$", "i", "o-i-classical", True],
     ["$", "i", "-i-classical", True],
     ["$", "im", "-im-classical", True]
    ],
    # 9) -ch, -sh and -ss and the s-singular group take -es in the plural (churches, classes, lenses).
    [["([cs])h$", "\\1hes", None, False],
     ["ss$", "sses", None, False],
     ["x$", "xes", None, False],
     ["s$", "ses", "s-singular", False]
    ],
    # 10) Certain words ending in -f or -fe take -ves in the plural (lives, wolves).
    [["([aeo]l)f$", "\\1ves", None, False],
     ["([^d]ea)f$", "\\1ves", None, False],
     ["arf$", "arves", None, False],
     ["([nlw]i)fe$", "\\1ves", None, False],
    ],
    # 11) -y takes -ys if preceded by a vowel or when a proper noun,
    # but -ies if preceded by a consonant (storeys, Marys, stories).
    [["([aeiou])y$", "\\1ys", None, False],
     ["([A-Z].*)y$", "\\1ys", None, False],
     ["y$", "ies", None, False]
    ],
    # 12) Some words ending in -o take -os, the rest take -oes.
    # Words in which the -o is preceded by a vowel always take -os (lassos, potatoes, bamboos).
    [["o$", "os", "o-os", False],
     ["([aeiou])o$", "\\1os", None, False],
     ["o$", "oes", None, False]
    ],
    # 13) Miltary stuff (Major Generals).
    [["l$", "ls", "general-generals", False]
    ],
    # 14) Otherwise, assume that the plural just adds -s (cats, programmes).
    [["$", "s", None, False]
    ],
]

# For performance, compile the regular expressions only once:
for ruleset in plural_rules:
    for rule in ruleset:
        rule[0] = re.compile(rule[0])

# Suffix categories.
plural_categories = {
    "uninflected": [
        "aircraft", "antelope", "bison", "bream", "breeches", "britches", "carp", "cattle", "chassis",
        "clippers", "cod", "contretemps", "corps", "debris", "diabetes", "djinn", "eland", "elk",
        "flounder", "gallows", "graffiti", "headquarters", "herpes", "high-jinks", "homework", "innings",
        "jackanapes", "mackerel", "measles", "mews", "moose", "mumps", "offspring", "news", "pincers",
        "pliers", "proceedings", "rabies", "salmon", "scissors", "series", "shears", "species", "swine",
        "trout", "tuna", "whiting", "wildebeest"],
    "uncountable": [
        "advice", "bread", "butter", "cannabis", "cheese", "electricity", "equipment", "fruit", "furniture",
        "garbage", "gravel", "happiness", "information", "ketchup", "knowledge", "love", "luggage",
        "mathematics", "mayonnaise", "meat", "mustard", "news", "progress", "research", "rice",
        "sand", "software", "understanding", "water"],
    "s-singular": [
        "acropolis", "aegis", "alias", "asbestos", "bathos", "bias", "bus", "caddis", "canvas",
        "chaos", "christmas", "cosmos", "dais", "digitalis", "epidermis", "ethos", "gas", "glottis",
        "ibis", "lens", "mantis", "marquis", "metropolis", "pathos", "pelvis", "polis", "rhinoceros",
        "sassafras", "trellis"],
    "ex-ices": ["codex", "murex", "silex"],
    "ex-ices-classical": [
        "apex", "cortex", "index", "latex", "pontifex", "simplex", "vertex", "vortex"],
    "um-a": [
        "agendum", "bacterium", "candelabrum", "datum", "desideratum", "erratum", "extremum",
        "ovum", "stratum"],
    "um-a-classical": [
        "aquarium", "compendium", "consortium", "cranium", "curriculum", "dictum", "emporium",
        "enconium", "gymnasium", "honorarium", "interregnum", "lustrum", "maximum", "medium",
        "memorandum", "millenium", "minimum", "momentum", "optimum", "phylum", "quantum", "rostrum",
        "spectrum", "speculum", "stadium", "trapezium", "ultimatum", "vacuum", "velum"],
    "on-a": [
        "aphelion", "asyndeton", "criterion", "hyperbaton", "noumenon", "organon", "perihelion",
        "phenomenon", "prolegomenon"],
    "a-ae": ["alga", "alumna", "vertebra"],
    "a-ae-classical": [
        "abscissa", "amoeba", "antenna", "aurora", "formula", "hydra", "hyperbola", "lacuna",
        "medusa", "nebula", "nova", "parabola"],
    "en-ina-classical": ["foramen", "lumen", "stamen"],
    "a-ata-classical": [
        "anathema", "bema", "carcinoma", "charisma", "diploma", "dogma", "drama", "edema", "enema",
        "enigma", "gumma", "lemma", "lymphoma", "magma", "melisma", "miasma", "oedema", "sarcoma",
        "schema", "soma", "stigma", "stoma", "trauma"],
    "is-ides-classical": ["clitoris", "iris"],
    "us-i-classical": [
        "focus", "fungus", "genius", "incubus", "nimbus", "nucleolus", "radius", "stylus", "succubus",
        "torus", "umbilicus", "uterus"],
    "us-us-classical": [
        "apparatus", "cantus", "coitus", "hiatus", "impetus", "nexus", "plexus", "prospectus",
        "sinus", "status"],
    "o-i-classical": ["alto", "basso", "canto", "contralto", "crescendo", "solo", "soprano", "tempo"],
    "-i-classical": ["afreet", "afrit", "efreet"],
    "-im-classical": ["cherub", "goy", "seraph"],
    "o-os": [
        "albino", "archipelago", "armadillo", "commando", "ditto", "dynamo", "embryo", "fiasco",
        "generalissimo", "ghetto", "guano", "inferno", "jumbo", "lingo", "lumbago", "magneto",
        "manifesto", "medico", "octavo", "photo", "pro", "quarto", "rhino", "stylo"],
    "general-generals": [
        "Adjutant", "Brigadier", "Lieutenant", "Major", "Quartermaster",
        "adjutant", "brigadier", "lieutenant", "major", "quartermaster"],
}

def pluralize(word, pos=NOUN, custom={}, classical=True):
    """ Returns the plural of a given word.
        For example: child -> children.
        Handles nouns and adjectives, using classical inflection by default
        (e.g. where "matrix" pluralizes to "matrices" instead of "matrixes").
        The custom dictionary is for user-defined replacements.
    """

    if word in custom:
        return custom[word]

    # Recursion of genitives.
    # Remove the apostrophe and any trailing -s,
    # form the plural of the resultant noun, and then append an apostrophe (dog's -> dogs').
    if word.endswith("'") or word.endswith("'s"):
        owner = word.rstrip("'s")
        owners = pluralize(owner, pos, custom, classical)
        if owners.endswith("s"):
            return owners + "'"
        else:
            return owners + "'s"

    # Recursion of compound words
    # (Postmasters General, mothers-in-law, Roman deities).
    words = word.replace("-", " ").split(" ")
    if len(words) > 1:
        if words[1] == "general" or words[1] == "General" and \
            words[0] not in plural_categories["general-generals"]:
            return word.replace(words[0], pluralize(words[0], pos, custom, classical))
        elif words[1] in plural_prepositions:
            return word.replace(words[0], pluralize(words[0], pos, custom, classical))
        else:
            return word.replace(words[-1], pluralize(words[-1], pos, custom, classical))

    # Only a very few number of adjectives inflect.
    n = list(range(len(plural_rules)))
    if pos.startswith(ADJECTIVE):
        n = [0, 1]

    # Apply pluralization rules.
    for i in n:
        ruleset = plural_rules[i]
        for rule in ruleset:
            suffix, inflection, category, classic = rule
            # A general rule, or a classic rule in classical mode.
            if category == None:
                if not classic or (classic and classical):
                    if suffix.search(word) is not None:
                        return suffix.sub(inflection, word)
            # A rule relating to a specific category of words.
            if category != None:
                if word in plural_categories[category] and (not classic or (classic and classical)):
                    if suffix.search(word) is not None:
                        return suffix.sub(inflection, word)

#### SINGULARIZE ###################################################################################
# Adapted from Bermi Ferrer's Inflector for Python:
# http://www.bermi.org/inflector/

# Copyright (c) 2006 Bermi Ferrer Martinez
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software to deal in this software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of this software, and to permit
# persons to whom this software is furnished to do so, subject to the following
# condition:
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THIS SOFTWARE.

singular_rules = [
    ['(?i)(.)ae$', '\\1a'],
    ['(?i)(.)itis$', '\\1itis'],
    ['(?i)(.)eaux$', '\\1eau'],
    ['(?i)(quiz)zes$', '\\1'],
    ['(?i)(matr)ices$', '\\1ix'],
    ['(?i)(ap|vert|ind)ices$', '\\1ex'],
    ['(?i)^(ox)en', '\\1'],
    ['(?i)(alias|status)es$', '\\1'],
    ['(?i)([octop|vir])i$', '\\1us'],
    ['(?i)(cris|ax|test)es$', '\\1is'],
    ['(?i)(shoe)s$', '\\1'],
    ['(?i)(o)es$', '\\1'],
    ['(?i)(bus)es$', '\\1'],
    ['(?i)([m|l])ice$', '\\1ouse'],
    ['(?i)(x|ch|ss|sh)es$', '\\1'],
    ['(?i)(m)ovies$', '\\1ovie'],
    ['(?i)(.)ombies$', '\\1ombie'],
    ['(?i)(s)eries$', '\\1eries'],
    ['(?i)([^aeiouy]|qu)ies$', '\\1y'],
    # Certain words ending in -f or -fe take -ves in the plural (lives, wolves).
    ["([aeo]l)ves$", "\\1f"],
    ["([^d]ea)ves$", "\\1f"],
    ["arves$", "arf"],
    ["erves$", "erve"],
    ["([nlw]i)ves$", "\\1fe"],
    ['(?i)([lr])ves$', '\\1f'],
    ["([aeo])ves$", "\\1ve"],
    ['(?i)(sive)s$', '\\1'],
    ['(?i)(tive)s$', '\\1'],
    ['(?i)(hive)s$', '\\1'],
    ['(?i)([^f])ves$', '\\1fe'],
    # -es suffix.
    ['(?i)(^analy)ses$', '\\1sis'],
    ['(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$', '\\1\\2sis'],
    ['(?i)(.)opses$', '\\1opsis'],
    ['(?i)(.)yses$', '\\1ysis'],
    ['(?i)(h|d|r|o|n|b|cl|p)oses$', '\\1ose'],
    ['(?i)(fruct|gluc|galact|lact|ket|malt|rib|sacchar|cellul)ose$', '\\1ose'],
    ['(?i)(.)oses$', '\\1osis'],
    # -a
    ['(?i)([ti])a$', '\\1um'],
    ['(?i)(n)ews$', '\\1ews'],
    ['(?i)s$', ''],
]

# For performance, compile the regular expressions only once:
for rule in singular_rules:
    rule[0] = re.compile(rule[0])

singular_uninflected = [
    "aircraft", "antelope", "bison", "bream", "breeches", "britches", "carp", "cattle", "chassis",
    "clippers", "cod", "contretemps", "corps", "debris", "diabetes", "djinn", "eland",
    "elk", "flounder", "gallows", "georgia", "graffiti", "headquarters", "herpes", "high-jinks",
    "homework", "innings", "jackanapes", "mackerel", "measles", "mews", "moose", "mumps", "news",
    "offspring", "pincers", "pliers", "proceedings", "rabies", "salmon", "scissors", "series",
    "shears", "species", "swine", "swiss", "trout", "tuna", "whiting", "wildebeest"
]
singular_uncountable = [
    "advice", "bread", "butter", "cannabis", "cheese", "electricity", "equipment", "fruit", "furniture",
    "garbage", "gravel", "happiness", "information", "ketchup", "knowledge", "love", "luggage",
    "mathematics", "mayonnaise", "meat", "mustard", "news", "progress", "research", "rice", "sand",
    "software", "understanding", "water"
]
singular_ie = [
    "algerie", "auntie", "beanie", "birdie", "bogie", "bombie", "bookie", "collie", "cookie", "cutie",
    "doggie", "eyrie", "freebie", "goonie", "groupie", "hankie", "hippie", "hoagie", "hottie",
    "indie", "junkie", "laddie", "laramie", "lingerie", "meanie", "nightie", "oldie", "^pie",
    "pixie", "quickie", "reverie", "rookie", "softie", "sortie", "stoolie", "sweetie", "techie",
    "^tie", "toughie", "valkyrie", "veggie", "weenie", "yuppie", "zombie"
]
singular_s = plural_categories['s-singular']

# key plural, value singular
singular_irregular = {
            "men": "man",
         "people": "person",
       "children": "child",
          "sexes": "sex",
           "axes": "axe",
          "moves": "move",
          "teeth": "tooth",
          "geese": "goose",
           "feet": "foot",
            "zoa": "zoon",
       "atlantes": "atlas",
        "atlases": "atlas",
         "beeves": "beef",
       "brethren": "brother",
       "children": "child",
        "corpora": "corpus",
       "corpuses": "corpus",
           "kine": "cow",
    "ephemerides": "ephemeris",
        "ganglia": "ganglion",
          "genii": "genie",
         "genera": "genus",
       "graffiti": "graffito",
         "helves": "helve",
         "leaves": "leaf",
         "loaves": "loaf",
         "monies": "money",
      "mongooses": "mongoose",
         "mythoi": "mythos",
      "octopodes": "octopus",
          "opera": "opus",
         "opuses": "opus",
           "oxen": "ox",
          "penes": "penis",
        "penises": "penis",
    "soliloquies": "soliloquy",
         "testes": "testis",
        "trilbys": "trilby",
         "turves": "turf",
         "numena": "numen",
       "occipita": "occiput",
            "our": "my",
}

def singularize(word, pos=NOUN, custom={}):

    if word in list(custom.keys()):
        return custom[word]

    # Recursion of compound words (e.g. mothers-in-law).
    if "-" in word:
        words = word.split("-")
        if len(words) > 1 and words[1] in plural_prepositions:
            return singularize(words[0], pos, custom)+"-"+"-".join(words[1:])
    # dogs' => dog's
    if word.endswith("'"):
        return singularize(word[:-1]) + "'s"

    lower = word.lower()
    for w in singular_uninflected:
        if w.endswith(lower):
            return word
    for w in singular_uncountable:
        if w.endswith(lower):
            return word
    for w in singular_ie:
        if lower.endswith(w+"s"):
            return w
    for w in singular_s:
        if lower.endswith(w + 'es'):
            return w
    for w in list(singular_irregular.keys()):
        if lower.endswith(w):
            return re.sub('(?i)'+w+'$', singular_irregular[w], word)

    for rule in singular_rules:
        suffix, inflection = rule
        match = suffix.search(word)
        if match:
            groups = match.groups()
            for k in range(0, len(groups)):
                if groups[k] == None:
                    inflection = inflection.replace('\\'+str(k+1), '')
            return suffix.sub(inflection, word)

    return word
