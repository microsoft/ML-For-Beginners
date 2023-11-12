# Natural Language Toolkit: Combinatory Categorial Grammar
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Graeme Gange <ggange@csse.unimelb.edu.au>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
"""
CCG Lexicons
"""

import re
from collections import defaultdict

from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression

# ------------
# Regular expressions used for parsing components of the lexicon
# ------------

# Parses a primitive category and subscripts
PRIM_RE = re.compile(r"""([A-Za-z]+)(\[[A-Za-z,]+\])?""")

# Separates the next primitive category from the remainder of the
# string
NEXTPRIM_RE = re.compile(r"""([A-Za-z]+(?:\[[A-Za-z,]+\])?)(.*)""")

# Separates the next application operator from the remainder
APP_RE = re.compile(r"""([\\/])([.,]?)([.,]?)(.*)""")

# Parses the definition of the right-hand side (rhs) of either a word or a family
LEX_RE = re.compile(r"""([\S_]+)\s*(::|[-=]+>)\s*(.+)""", re.UNICODE)

# Parses the right hand side that contains category and maybe semantic predicate
RHS_RE = re.compile(r"""([^{}]*[^ {}])\s*(\{[^}]+\})?""", re.UNICODE)

# Parses the semantic predicate
SEMANTICS_RE = re.compile(r"""\{([^}]+)\}""", re.UNICODE)

# Strips comments from a line
COMMENTS_RE = re.compile("""([^#]*)(?:#.*)?""")


class Token:
    """
    Class representing a token.

    token => category {semantics}
    e.g. eat => S\\var[pl]/var {\\x y.eat(x,y)}

    * `token` (string)
    * `categ` (string)
    * `semantics` (Expression)
    """

    def __init__(self, token, categ, semantics=None):
        self._token = token
        self._categ = categ
        self._semantics = semantics

    def categ(self):
        return self._categ

    def semantics(self):
        return self._semantics

    def __str__(self):
        semantics_str = ""
        if self._semantics is not None:
            semantics_str = " {" + str(self._semantics) + "}"
        return "" + str(self._categ) + semantics_str

    def __cmp__(self, other):
        if not isinstance(other, Token):
            return -1
        return cmp((self._categ, self._semantics), other.categ(), other.semantics())


class CCGLexicon:
    """
    Class representing a lexicon for CCG grammars.

    * `primitives`: The list of primitive categories for the lexicon
    * `families`: Families of categories
    * `entries`: A mapping of words to possible categories
    """

    def __init__(self, start, primitives, families, entries):
        self._start = PrimitiveCategory(start)
        self._primitives = primitives
        self._families = families
        self._entries = entries

    def categories(self, word):
        """
        Returns all the possible categories for a word
        """
        return self._entries[word]

    def start(self):
        """
        Return the target category for the parser
        """
        return self._start

    def __str__(self):
        """
        String representation of the lexicon. Used for debugging.
        """
        string = ""
        first = True
        for ident in sorted(self._entries):
            if not first:
                string = string + "\n"
            string = string + ident + " => "

            first = True
            for cat in self._entries[ident]:
                if not first:
                    string = string + " | "
                else:
                    first = False
                string = string + "%s" % cat
        return string


# -----------
# Parsing lexicons
# -----------


def matchBrackets(string):
    """
    Separate the contents matching the first set of brackets from the rest of
    the input.
    """
    rest = string[1:]
    inside = "("

    while rest != "" and not rest.startswith(")"):
        if rest.startswith("("):
            (part, rest) = matchBrackets(rest)
            inside = inside + part
        else:
            inside = inside + rest[0]
            rest = rest[1:]
    if rest.startswith(")"):
        return (inside + ")", rest[1:])
    raise AssertionError("Unmatched bracket in string '" + string + "'")


def nextCategory(string):
    """
    Separate the string for the next portion of the category from the rest
    of the string
    """
    if string.startswith("("):
        return matchBrackets(string)
    return NEXTPRIM_RE.match(string).groups()


def parseApplication(app):
    """
    Parse an application operator
    """
    return Direction(app[0], app[1:])


def parseSubscripts(subscr):
    """
    Parse the subscripts for a primitive category
    """
    if subscr:
        return subscr[1:-1].split(",")
    return []


def parsePrimitiveCategory(chunks, primitives, families, var):
    """
    Parse a primitive category

    If the primitive is the special category 'var', replace it with the
    correct `CCGVar`.
    """
    if chunks[0] == "var":
        if chunks[1] is None:
            if var is None:
                var = CCGVar()
            return (var, var)

    catstr = chunks[0]
    if catstr in families:
        (cat, cvar) = families[catstr]
        if var is None:
            var = cvar
        else:
            cat = cat.substitute([(cvar, var)])
        return (cat, var)

    if catstr in primitives:
        subscrs = parseSubscripts(chunks[1])
        return (PrimitiveCategory(catstr, subscrs), var)
    raise AssertionError(
        "String '" + catstr + "' is neither a family nor primitive category."
    )


def augParseCategory(line, primitives, families, var=None):
    """
    Parse a string representing a category, and returns a tuple with
    (possibly) the CCG variable for the category
    """
    (cat_string, rest) = nextCategory(line)

    if cat_string.startswith("("):
        (res, var) = augParseCategory(cat_string[1:-1], primitives, families, var)

    else:
        (res, var) = parsePrimitiveCategory(
            PRIM_RE.match(cat_string).groups(), primitives, families, var
        )

    while rest != "":
        app = APP_RE.match(rest).groups()
        direction = parseApplication(app[0:3])
        rest = app[3]

        (cat_string, rest) = nextCategory(rest)
        if cat_string.startswith("("):
            (arg, var) = augParseCategory(cat_string[1:-1], primitives, families, var)
        else:
            (arg, var) = parsePrimitiveCategory(
                PRIM_RE.match(cat_string).groups(), primitives, families, var
            )
        res = FunctionalCategory(res, arg, direction)

    return (res, var)


def fromstring(lex_str, include_semantics=False):
    """
    Convert string representation into a lexicon for CCGs.
    """
    CCGVar.reset_id()
    primitives = []
    families = {}
    entries = defaultdict(list)
    for line in lex_str.splitlines():
        # Strip comments and leading/trailing whitespace.
        line = COMMENTS_RE.match(line).groups()[0].strip()
        if line == "":
            continue

        if line.startswith(":-"):
            # A line of primitive categories.
            # The first one is the target category
            # ie, :- S, N, NP, VP
            primitives = primitives + [
                prim.strip() for prim in line[2:].strip().split(",")
            ]
        else:
            # Either a family definition, or a word definition
            (ident, sep, rhs) = LEX_RE.match(line).groups()
            (catstr, semantics_str) = RHS_RE.match(rhs).groups()
            (cat, var) = augParseCategory(catstr, primitives, families)

            if sep == "::":
                # Family definition
                # ie, Det :: NP/N
                families[ident] = (cat, var)
            else:
                semantics = None
                if include_semantics is True:
                    if semantics_str is None:
                        raise AssertionError(
                            line
                            + " must contain semantics because include_semantics is set to True"
                        )
                    else:
                        semantics = Expression.fromstring(
                            SEMANTICS_RE.match(semantics_str).groups()[0]
                        )
                # Word definition
                # ie, which => (N\N)/(S/NP)
                entries[ident].append(Token(ident, cat, semantics))
    return CCGLexicon(primitives[0], primitives, families, entries)


@deprecated("Use fromstring() instead.")
def parseLexicon(lex_str):
    return fromstring(lex_str)


openccg_tinytiny = fromstring(
    """
    # Rather minimal lexicon based on the openccg `tinytiny' grammar.
    # Only incorporates a subset of the morphological subcategories, however.
    :- S,NP,N                    # Primitive categories
    Det :: NP/N                  # Determiners
    Pro :: NP
    IntransVsg :: S\\NP[sg]    # Tensed intransitive verbs (singular)
    IntransVpl :: S\\NP[pl]    # Plural
    TransVsg :: S\\NP[sg]/NP   # Tensed transitive verbs (singular)
    TransVpl :: S\\NP[pl]/NP   # Plural

    the => NP[sg]/N[sg]
    the => NP[pl]/N[pl]

    I => Pro
    me => Pro
    we => Pro
    us => Pro

    book => N[sg]
    books => N[pl]

    peach => N[sg]
    peaches => N[pl]

    policeman => N[sg]
    policemen => N[pl]

    boy => N[sg]
    boys => N[pl]

    sleep => IntransVsg
    sleep => IntransVpl

    eat => IntransVpl
    eat => TransVpl
    eats => IntransVsg
    eats => TransVsg

    see => TransVpl
    sees => TransVsg
    """
)
