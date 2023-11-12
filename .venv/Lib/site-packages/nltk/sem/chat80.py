# Natural Language Toolkit: Chat-80 KB Reader
# See https://www.w3.org/TR/swbp-skos-core-guide/
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Ewan Klein <ewan@inf.ed.ac.uk>,
# URL: <https://www.nltk.org>
# For license information, see LICENSE.TXT

r"""
Overview
========

Chat-80 was a natural language system which allowed the user to
interrogate a Prolog knowledge base in the domain of world
geography. It was developed in the early '80s by Warren and Pereira; see
``https://www.aclweb.org/anthology/J82-3002.pdf`` for a description and
``http://www.cis.upenn.edu/~pereira/oldies.html`` for the source
files.

This module contains functions to extract data from the Chat-80
relation files ('the world database'), and convert then into a format
that can be incorporated in the FOL models of
``nltk.sem.evaluate``. The code assumes that the Prolog
input files are available in the NLTK corpora directory.

The Chat-80 World Database consists of the following files::

    world0.pl
    rivers.pl
    cities.pl
    countries.pl
    contain.pl
    borders.pl

This module uses a slightly modified version of ``world0.pl``, in which
a set of Prolog rules have been omitted. The modified file is named
``world1.pl``. Currently, the file ``rivers.pl`` is not read in, since
it uses a list rather than a string in the second field.

Reading Chat-80 Files
=====================

Chat-80 relations are like tables in a relational database. The
relation acts as the name of the table; the first argument acts as the
'primary key'; and subsequent arguments are further fields in the
table. In general, the name of the table provides a label for a unary
predicate whose extension is all the primary keys. For example,
relations in ``cities.pl`` are of the following form::

   'city(athens,greece,1368).'

Here, ``'athens'`` is the key, and will be mapped to a member of the
unary predicate *city*.

The fields in the table are mapped to binary predicates. The first
argument of the predicate is the primary key, while the second
argument is the data in the relevant field. Thus, in the above
example, the third field is mapped to the binary predicate
*population_of*, whose extension is a set of pairs such as
``'(athens, 1368)'``.

An exception to this general framework is required by the relations in
the files ``borders.pl`` and ``contains.pl``. These contain facts of the
following form::

    'borders(albania,greece).'

    'contains0(africa,central_africa).'

We do not want to form a unary concept out the element in
the first field of these records, and we want the label of the binary
relation just to be ``'border'``/``'contain'`` respectively.

In order to drive the extraction process, we use 'relation metadata bundles'
which are Python dictionaries such as the following::

  city = {'label': 'city',
          'closures': [],
          'schema': ['city', 'country', 'population'],
          'filename': 'cities.pl'}

According to this, the file ``city['filename']`` contains a list of
relational tuples (or more accurately, the corresponding strings in
Prolog form) whose predicate symbol is ``city['label']`` and whose
relational schema is ``city['schema']``. The notion of a ``closure`` is
discussed in the next section.

Concepts
========
In order to encapsulate the results of the extraction, a class of
``Concept`` objects is introduced.  A ``Concept`` object has a number of
attributes, in particular a ``prefLabel`` and ``extension``, which make
it easier to inspect the output of the extraction. In addition, the
``extension`` can be further processed: in the case of the ``'border'``
relation, we check that the relation is symmetric, and in the case
of the ``'contain'`` relation, we carry out the transitive
closure. The closure properties associated with a concept is
indicated in the relation metadata, as indicated earlier.

The ``extension`` of a ``Concept`` object is then incorporated into a
``Valuation`` object.

Persistence
===========
The functions ``val_dump`` and ``val_load`` are provided to allow a
valuation to be stored in a persistent database and re-loaded, rather
than having to be re-computed each time.

Individuals and Lexical Items
=============================
As well as deriving relations from the Chat-80 data, we also create a
set of individual constants, one for each entity in the domain. The
individual constants are string-identical to the entities. For
example, given a data item such as ``'zloty'``, we add to the valuation
a pair ``('zloty', 'zloty')``. In order to parse English sentences that
refer to these entities, we also create a lexical item such as the
following for each individual constant::

   PropN[num=sg, sem=<\P.(P zloty)>] -> 'Zloty'

The set of rules is written to the file ``chat_pnames.cfg`` in the
current directory.

"""

import os
import re
import shelve
import sys

import nltk.data

###########################################################################
# Chat-80 relation metadata bundles needed to build the valuation
###########################################################################

borders = {
    "rel_name": "borders",
    "closures": ["symmetric"],
    "schema": ["region", "border"],
    "filename": "borders.pl",
}

contains = {
    "rel_name": "contains0",
    "closures": ["transitive"],
    "schema": ["region", "contain"],
    "filename": "contain.pl",
}

city = {
    "rel_name": "city",
    "closures": [],
    "schema": ["city", "country", "population"],
    "filename": "cities.pl",
}

country = {
    "rel_name": "country",
    "closures": [],
    "schema": [
        "country",
        "region",
        "latitude",
        "longitude",
        "area",
        "population",
        "capital",
        "currency",
    ],
    "filename": "countries.pl",
}

circle_of_lat = {
    "rel_name": "circle_of_latitude",
    "closures": [],
    "schema": ["circle_of_latitude", "degrees"],
    "filename": "world1.pl",
}

circle_of_long = {
    "rel_name": "circle_of_longitude",
    "closures": [],
    "schema": ["circle_of_longitude", "degrees"],
    "filename": "world1.pl",
}

continent = {
    "rel_name": "continent",
    "closures": [],
    "schema": ["continent"],
    "filename": "world1.pl",
}

region = {
    "rel_name": "in_continent",
    "closures": [],
    "schema": ["region", "continent"],
    "filename": "world1.pl",
}

ocean = {
    "rel_name": "ocean",
    "closures": [],
    "schema": ["ocean"],
    "filename": "world1.pl",
}

sea = {"rel_name": "sea", "closures": [], "schema": ["sea"], "filename": "world1.pl"}


items = [
    "borders",
    "contains",
    "city",
    "country",
    "circle_of_lat",
    "circle_of_long",
    "continent",
    "region",
    "ocean",
    "sea",
]
items = tuple(sorted(items))

item_metadata = {
    "borders": borders,
    "contains": contains,
    "city": city,
    "country": country,
    "circle_of_lat": circle_of_lat,
    "circle_of_long": circle_of_long,
    "continent": continent,
    "region": region,
    "ocean": ocean,
    "sea": sea,
}

rels = item_metadata.values()

not_unary = ["borders.pl", "contain.pl"]

###########################################################################


class Concept:
    """
    A Concept class, loosely based on SKOS
    (https://www.w3.org/TR/swbp-skos-core-guide/).
    """

    def __init__(self, prefLabel, arity, altLabels=[], closures=[], extension=set()):
        """
        :param prefLabel: the preferred label for the concept
        :type prefLabel: str
        :param arity: the arity of the concept
        :type arity: int
        :param altLabels: other (related) labels
        :type altLabels: list
        :param closures: closure properties of the extension
            (list items can be ``symmetric``, ``reflexive``, ``transitive``)
        :type closures: list
        :param extension: the extensional value of the concept
        :type extension: set
        """
        self.prefLabel = prefLabel
        self.arity = arity
        self.altLabels = altLabels
        self.closures = closures
        # keep _extension internally as a set
        self._extension = extension
        # public access is via a list (for slicing)
        self.extension = sorted(list(extension))

    def __str__(self):
        # _extension = ''
        # for element in sorted(self.extension):
        # if isinstance(element, tuple):
        # element = '(%s, %s)' % (element)
        # _extension += element + ', '
        # _extension = _extension[:-1]

        return "Label = '{}'\nArity = {}\nExtension = {}".format(
            self.prefLabel,
            self.arity,
            self.extension,
        )

    def __repr__(self):
        return "Concept('%s')" % self.prefLabel

    def augment(self, data):
        """
        Add more data to the ``Concept``'s extension set.

        :param data: a new semantic value
        :type data: string or pair of strings
        :rtype: set

        """
        self._extension.add(data)
        self.extension = sorted(list(self._extension))
        return self._extension

    def _make_graph(self, s):
        """
        Convert a set of pairs into an adjacency linked list encoding of a graph.
        """
        g = {}
        for (x, y) in s:
            if x in g:
                g[x].append(y)
            else:
                g[x] = [y]
        return g

    def _transclose(self, g):
        """
        Compute the transitive closure of a graph represented as a linked list.
        """
        for x in g:
            for adjacent in g[x]:
                # check that adjacent is a key
                if adjacent in g:
                    for y in g[adjacent]:
                        if y not in g[x]:
                            g[x].append(y)
        return g

    def _make_pairs(self, g):
        """
        Convert an adjacency linked list back into a set of pairs.
        """
        pairs = []
        for node in g:
            for adjacent in g[node]:
                pairs.append((node, adjacent))
        return set(pairs)

    def close(self):
        """
        Close a binary relation in the ``Concept``'s extension set.

        :return: a new extension for the ``Concept`` in which the
                 relation is closed under a given property
        """
        from nltk.sem import is_rel

        assert is_rel(self._extension)
        if "symmetric" in self.closures:
            pairs = []
            for (x, y) in self._extension:
                pairs.append((y, x))
            sym = set(pairs)
            self._extension = self._extension.union(sym)
        if "transitive" in self.closures:
            all = self._make_graph(self._extension)
            closed = self._transclose(all)
            trans = self._make_pairs(closed)
            self._extension = self._extension.union(trans)
        self.extension = sorted(list(self._extension))


def clause2concepts(filename, rel_name, schema, closures=[]):
    """
    Convert a file of Prolog clauses into a list of ``Concept`` objects.

    :param filename: filename containing the relations
    :type filename: str
    :param rel_name: name of the relation
    :type rel_name: str
    :param schema: the schema used in a set of relational tuples
    :type schema: list
    :param closures: closure properties for the extension of the concept
    :type closures: list
    :return: a list of ``Concept`` objects
    :rtype: list
    """
    concepts = []
    # position of the subject of a binary relation
    subj = 0
    # label of the 'primary key'
    pkey = schema[0]
    # fields other than the primary key
    fields = schema[1:]

    # convert a file into a list of lists
    records = _str2records(filename, rel_name)

    # add a unary concept corresponding to the set of entities
    # in the primary key position
    # relations in 'not_unary' are more like ordinary binary relations
    if not filename in not_unary:
        concepts.append(unary_concept(pkey, subj, records))

    # add a binary concept for each non-key field
    for field in fields:
        obj = schema.index(field)
        concepts.append(binary_concept(field, closures, subj, obj, records))

    return concepts


def cities2table(filename, rel_name, dbname, verbose=False, setup=False):
    """
    Convert a file of Prolog clauses into a database table.

    This is not generic, since it doesn't allow arbitrary
    schemas to be set as a parameter.

    Intended usage::

        cities2table('cities.pl', 'city', 'city.db', verbose=True, setup=True)

    :param filename: filename containing the relations
    :type filename: str
    :param rel_name: name of the relation
    :type rel_name: str
    :param dbname: filename of persistent store
    :type schema: str
    """
    import sqlite3

    records = _str2records(filename, rel_name)
    connection = sqlite3.connect(dbname)
    cur = connection.cursor()
    if setup:
        cur.execute(
            """CREATE TABLE city_table
        (City text, Country text, Population int)"""
        )

    table_name = "city_table"
    for t in records:
        cur.execute("insert into %s values (?,?,?)" % table_name, t)
        if verbose:
            print("inserting values into %s: " % table_name, t)
    connection.commit()
    if verbose:
        print("Committing update to %s" % dbname)
    cur.close()


def sql_query(dbname, query):
    """
    Execute an SQL query over a database.
    :param dbname: filename of persistent store
    :type schema: str
    :param query: SQL query
    :type rel_name: str
    """
    import sqlite3

    try:
        path = nltk.data.find(dbname)
        connection = sqlite3.connect(str(path))
        cur = connection.cursor()
        return cur.execute(query)
    except (ValueError, sqlite3.OperationalError):
        import warnings

        warnings.warn(
            "Make sure the database file %s is installed and uncompressed." % dbname
        )
        raise


def _str2records(filename, rel):
    """
    Read a file into memory and convert each relation clause into a list.
    """
    recs = []
    contents = nltk.data.load("corpora/chat80/%s" % filename, format="text")
    for line in contents.splitlines():
        if line.startswith(rel):
            line = re.sub(rel + r"\(", "", line)
            line = re.sub(r"\)\.$", "", line)
            record = line.split(",")
            recs.append(record)
    return recs


def unary_concept(label, subj, records):
    """
    Make a unary concept out of the primary key in a record.

    A record is a list of entities in some relation, such as
    ``['france', 'paris']``, where ``'france'`` is acting as the primary
    key.

    :param label: the preferred label for the concept
    :type label: string
    :param subj: position in the record of the subject of the predicate
    :type subj: int
    :param records: a list of records
    :type records: list of lists
    :return: ``Concept`` of arity 1
    :rtype: Concept
    """
    c = Concept(label, arity=1, extension=set())
    for record in records:
        c.augment(record[subj])
    return c


def binary_concept(label, closures, subj, obj, records):
    """
    Make a binary concept out of the primary key and another field in a record.

    A record is a list of entities in some relation, such as
    ``['france', 'paris']``, where ``'france'`` is acting as the primary
    key, and ``'paris'`` stands in the ``'capital_of'`` relation to
    ``'france'``.

    More generally, given a record such as ``['a', 'b', 'c']``, where
    label is bound to ``'B'``, and ``obj`` bound to 1, the derived
    binary concept will have label ``'B_of'``, and its extension will
    be a set of pairs such as ``('a', 'b')``.


    :param label: the base part of the preferred label for the concept
    :type label: str
    :param closures: closure properties for the extension of the concept
    :type closures: list
    :param subj: position in the record of the subject of the predicate
    :type subj: int
    :param obj: position in the record of the object of the predicate
    :type obj: int
    :param records: a list of records
    :type records: list of lists
    :return: ``Concept`` of arity 2
    :rtype: Concept
    """
    if not label == "border" and not label == "contain":
        label = label + "_of"
    c = Concept(label, arity=2, closures=closures, extension=set())
    for record in records:
        c.augment((record[subj], record[obj]))
    # close the concept's extension according to the properties in closures
    c.close()
    return c


def process_bundle(rels):
    """
    Given a list of relation metadata bundles, make a corresponding
    dictionary of concepts, indexed by the relation name.

    :param rels: bundle of metadata needed for constructing a concept
    :type rels: list(dict)
    :return: a dictionary of concepts, indexed by the relation name.
    :rtype: dict(str): Concept
    """
    concepts = {}
    for rel in rels:
        rel_name = rel["rel_name"]
        closures = rel["closures"]
        schema = rel["schema"]
        filename = rel["filename"]

        concept_list = clause2concepts(filename, rel_name, schema, closures)
        for c in concept_list:
            label = c.prefLabel
            if label in concepts:
                for data in c.extension:
                    concepts[label].augment(data)
                concepts[label].close()
            else:
                concepts[label] = c
    return concepts


def make_valuation(concepts, read=False, lexicon=False):
    """
    Convert a list of ``Concept`` objects into a list of (label, extension) pairs;
    optionally create a ``Valuation`` object.

    :param concepts: concepts
    :type concepts: list(Concept)
    :param read: if ``True``, ``(symbol, set)`` pairs are read into a ``Valuation``
    :type read: bool
    :rtype: list or Valuation
    """
    vals = []

    for c in concepts:
        vals.append((c.prefLabel, c.extension))
    if lexicon:
        read = True
    if read:
        from nltk.sem import Valuation

        val = Valuation({})
        val.update(vals)
        # add labels for individuals
        val = label_indivs(val, lexicon=lexicon)
        return val
    else:
        return vals


def val_dump(rels, db):
    """
    Make a ``Valuation`` from a list of relation metadata bundles and dump to
    persistent database.

    :param rels: bundle of metadata needed for constructing a concept
    :type rels: list of dict
    :param db: name of file to which data is written.
               The suffix '.db' will be automatically appended.
    :type db: str
    """
    concepts = process_bundle(rels).values()
    valuation = make_valuation(concepts, read=True)
    db_out = shelve.open(db, "n")

    db_out.update(valuation)

    db_out.close()


def val_load(db):
    """
    Load a ``Valuation`` from a persistent database.

    :param db: name of file from which data is read.
               The suffix '.db' should be omitted from the name.
    :type db: str
    """
    dbname = db + ".db"

    if not os.access(dbname, os.R_OK):
        sys.exit("Cannot read file: %s" % dbname)
    else:
        db_in = shelve.open(db)
        from nltk.sem import Valuation

        val = Valuation(db_in)
        #        val.read(db_in.items())
        return val


# def alpha(str):
# """
# Utility to filter out non-alphabetic constants.

#:param str: candidate constant
#:type str: string
#:rtype: bool
# """
# try:
# int(str)
# return False
# except ValueError:
## some unknown values in records are labeled '?'
# if not str == '?':
# return True


def label_indivs(valuation, lexicon=False):
    """
    Assign individual constants to the individuals in the domain of a ``Valuation``.

    Given a valuation with an entry of the form ``{'rel': {'a': True}}``,
    add a new entry ``{'a': 'a'}``.

    :type valuation: Valuation
    :rtype: Valuation
    """
    # collect all the individuals into a domain
    domain = valuation.domain
    # convert the domain into a sorted list of alphabetic terms
    # use the same string as a label
    pairs = [(e, e) for e in domain]
    if lexicon:
        lex = make_lex(domain)
        with open("chat_pnames.cfg", "w") as outfile:
            outfile.writelines(lex)
    # read the pairs into the valuation
    valuation.update(pairs)
    return valuation


def make_lex(symbols):
    """
    Create lexical CFG rules for each individual symbol.

    Given a valuation with an entry of the form ``{'zloty': 'zloty'}``,
    create a lexical rule for the proper name 'Zloty'.

    :param symbols: a list of individual constants in the semantic representation
    :type symbols: sequence -- set(str)
    :rtype: list(str)
    """
    lex = []
    header = """
##################################################################
# Lexical rules automatically generated by running 'chat80.py -x'.
##################################################################

"""
    lex.append(header)
    template = r"PropN[num=sg, sem=<\P.(P %s)>] -> '%s'\n"

    for s in symbols:
        parts = s.split("_")
        caps = [p.capitalize() for p in parts]
        pname = "_".join(caps)
        rule = template % (s, pname)
        lex.append(rule)
    return lex


###########################################################################
# Interface function to emulate other corpus readers
###########################################################################


def concepts(items=items):
    """
    Build a list of concepts corresponding to the relation names in ``items``.

    :param items: names of the Chat-80 relations to extract
    :type items: list(str)
    :return: the ``Concept`` objects which are extracted from the relations
    :rtype: list(Concept)
    """
    if isinstance(items, str):
        items = (items,)

    rels = [item_metadata[r] for r in items]

    concept_map = process_bundle(rels)
    return concept_map.values()


###########################################################################


def main():
    import sys
    from optparse import OptionParser

    description = """
Extract data from the Chat-80 Prolog files and convert them into a
Valuation object for use in the NLTK semantics package.
    """

    opts = OptionParser(description=description)
    opts.set_defaults(verbose=True, lex=False, vocab=False)
    opts.add_option(
        "-s", "--store", dest="outdb", help="store a valuation in DB", metavar="DB"
    )
    opts.add_option(
        "-l",
        "--load",
        dest="indb",
        help="load a stored valuation from DB",
        metavar="DB",
    )
    opts.add_option(
        "-c",
        "--concepts",
        action="store_true",
        help="print concepts instead of a valuation",
    )
    opts.add_option(
        "-r",
        "--relation",
        dest="label",
        help="print concept with label REL (check possible labels with '-v' option)",
        metavar="REL",
    )
    opts.add_option(
        "-q",
        "--quiet",
        action="store_false",
        dest="verbose",
        help="don't print out progress info",
    )
    opts.add_option(
        "-x",
        "--lex",
        action="store_true",
        dest="lex",
        help="write a file of lexical entries for country names, then exit",
    )
    opts.add_option(
        "-v",
        "--vocab",
        action="store_true",
        dest="vocab",
        help="print out the vocabulary of concept labels and their arity, then exit",
    )

    (options, args) = opts.parse_args()
    if options.outdb and options.indb:
        opts.error("Options --store and --load are mutually exclusive")

    if options.outdb:
        # write the valuation to a persistent database
        if options.verbose:
            outdb = options.outdb + ".db"
            print("Dumping a valuation to %s" % outdb)
        val_dump(rels, options.outdb)
        sys.exit(0)
    else:
        # try to read in a valuation from a database
        if options.indb is not None:
            dbname = options.indb + ".db"
            if not os.access(dbname, os.R_OK):
                sys.exit("Cannot read file: %s" % dbname)
            else:
                valuation = val_load(options.indb)
        # we need to create the valuation from scratch
        else:
            # build some concepts
            concept_map = process_bundle(rels)
            concepts = concept_map.values()
            # just print out the vocabulary
            if options.vocab:
                items = sorted((c.arity, c.prefLabel) for c in concepts)
                for (arity, label) in items:
                    print(label, arity)
                sys.exit(0)
            # show all the concepts
            if options.concepts:
                for c in concepts:
                    print(c)
                    print()
            if options.label:
                print(concept_map[options.label])
                sys.exit(0)
            else:
                # turn the concepts into a Valuation
                if options.lex:
                    if options.verbose:
                        print("Writing out lexical rules")
                    make_valuation(concepts, lexicon=True)
                else:
                    valuation = make_valuation(concepts, read=True)
                    print(valuation)


def sql_demo():
    """
    Print out every row from the 'city.db' database.
    """
    print()
    print("Using SQL to extract rows from 'city.db' RDB.")
    for row in sql_query("corpora/city_database/city.db", "SELECT * FROM city_table"):
        print(row)


if __name__ == "__main__":
    main()
    sql_demo()
