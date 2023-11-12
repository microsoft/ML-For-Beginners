# Natural Language Toolkit: Glue Semantics
#
# Author: Dan Garrette <dhgarrette@gmail.com>
#
# Copyright (C) 2001-2023 NLTK Project
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import os
from itertools import chain

import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
    AbstractVariableExpression,
    Expression,
    LambdaExpression,
    Variable,
    VariableExpression,
)
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger

SPEC_SEMTYPES = {
    "a": "ex_quant",
    "an": "ex_quant",
    "every": "univ_quant",
    "the": "def_art",
    "no": "no_quant",
    "default": "ex_quant",
}

OPTIONAL_RELATIONSHIPS = ["nmod", "vmod", "punct"]


class GlueFormula:
    def __init__(self, meaning, glue, indices=None):
        if not indices:
            indices = set()

        if isinstance(meaning, str):
            self.meaning = Expression.fromstring(meaning)
        elif isinstance(meaning, Expression):
            self.meaning = meaning
        else:
            raise RuntimeError(
                "Meaning term neither string or expression: %s, %s"
                % (meaning, meaning.__class__)
            )

        if isinstance(glue, str):
            self.glue = linearlogic.LinearLogicParser().parse(glue)
        elif isinstance(glue, linearlogic.Expression):
            self.glue = glue
        else:
            raise RuntimeError(
                "Glue term neither string or expression: %s, %s"
                % (glue, glue.__class__)
            )

        self.indices = indices

    def applyto(self, arg):
        """self = (\\x.(walk x), (subj -o f))
        arg  = (john        ,  subj)
        returns ((walk john),          f)
        """
        if self.indices & arg.indices:  # if the sets are NOT disjoint
            raise linearlogic.LinearLogicApplicationException(
                f"'{self}' applied to '{arg}'.  Indices are not disjoint."
            )
        else:  # if the sets ARE disjoint
            return_indices = self.indices | arg.indices

        try:
            return_glue = linearlogic.ApplicationExpression(
                self.glue, arg.glue, arg.indices
            )
        except linearlogic.LinearLogicApplicationException as e:
            raise linearlogic.LinearLogicApplicationException(
                f"'{self.simplify()}' applied to '{arg.simplify()}'"
            ) from e

        arg_meaning_abstracted = arg.meaning
        if return_indices:
            for dep in self.glue.simplify().antecedent.dependencies[
                ::-1
            ]:  # if self.glue is (A -o B), dep is in A.dependencies
                arg_meaning_abstracted = self.make_LambdaExpression(
                    Variable("v%s" % dep), arg_meaning_abstracted
                )
        return_meaning = self.meaning.applyto(arg_meaning_abstracted)

        return self.__class__(return_meaning, return_glue, return_indices)

    def make_VariableExpression(self, name):
        return VariableExpression(name)

    def make_LambdaExpression(self, variable, term):
        return LambdaExpression(variable, term)

    def lambda_abstract(self, other):
        assert isinstance(other, GlueFormula)
        assert isinstance(other.meaning, AbstractVariableExpression)
        return self.__class__(
            self.make_LambdaExpression(other.meaning.variable, self.meaning),
            linearlogic.ImpExpression(other.glue, self.glue),
        )

    def compile(self, counter=None):
        """From Iddo Lev's PhD Dissertation p108-109"""
        if not counter:
            counter = Counter()
        (compiled_glue, new_forms) = self.glue.simplify().compile_pos(
            counter, self.__class__
        )
        return new_forms + [
            self.__class__(self.meaning, compiled_glue, {counter.get()})
        ]

    def simplify(self):
        return self.__class__(
            self.meaning.simplify(), self.glue.simplify(), self.indices
        )

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__
            and self.meaning == other.meaning
            and self.glue == other.glue
        )

    def __ne__(self, other):
        return not self == other

    # sorting for use in doctests which must be deterministic
    def __lt__(self, other):
        return str(self) < str(other)

    def __str__(self):
        assert isinstance(self.indices, set)
        accum = f"{self.meaning} : {self.glue}"
        if self.indices:
            accum += (
                " : {" + ", ".join(str(index) for index in sorted(self.indices)) + "}"
            )
        return accum

    def __repr__(self):
        return "%s" % self


class GlueDict(dict):
    def __init__(self, filename, encoding=None):
        self.filename = filename
        self.file_encoding = encoding
        self.read_file()

    def read_file(self, empty_first=True):
        if empty_first:
            self.clear()

        try:
            contents = nltk.data.load(
                self.filename, format="text", encoding=self.file_encoding
            )
            # TODO: the above can't handle zip files, but this should anyway be fixed in nltk.data.load()
        except LookupError as e:
            try:
                contents = nltk.data.load(
                    "file:" + self.filename, format="text", encoding=self.file_encoding
                )
            except LookupError:
                raise e
        lines = contents.splitlines()

        for line in lines:  # example: 'n : (\\x.(<word> x), (v-or))'
            #     lambdacalc -^  linear logic -^
            line = line.strip()  # remove trailing newline
            if not len(line):
                continue  # skip empty lines
            if line[0] == "#":
                continue  # skip commented out lines

            parts = line.split(
                " : ", 2
            )  # ['verb', '(\\x.(<word> x), ( subj -o f ))', '[subj]']

            glue_formulas = []
            paren_count = 0
            tuple_start = 0
            tuple_comma = 0

            relationships = None

            if len(parts) > 1:
                for (i, c) in enumerate(parts[1]):
                    if c == "(":
                        if paren_count == 0:  # if it's the first '(' of a tuple
                            tuple_start = i + 1  # then save the index
                        paren_count += 1
                    elif c == ")":
                        paren_count -= 1
                        if paren_count == 0:  # if it's the last ')' of a tuple
                            meaning_term = parts[1][
                                tuple_start:tuple_comma
                            ]  # '\\x.(<word> x)'
                            glue_term = parts[1][tuple_comma + 1 : i]  # '(v-r)'
                            glue_formulas.append(
                                [meaning_term, glue_term]
                            )  # add the GlueFormula to the list
                    elif c == ",":
                        if (
                            paren_count == 1
                        ):  # if it's a comma separating the parts of the tuple
                            tuple_comma = i  # then save the index
                    elif c == "#":  # skip comments at the ends of lines
                        if (
                            paren_count != 0
                        ):  # if the line hasn't parsed correctly so far
                            raise RuntimeError(
                                "Formula syntax is incorrect for entry " + line
                            )
                        break  # break to the next line

            if len(parts) > 2:  # if there is a relationship entry at the end
                rel_start = parts[2].index("[") + 1
                rel_end = parts[2].index("]")
                if rel_start == rel_end:
                    relationships = frozenset()
                else:
                    relationships = frozenset(
                        r.strip() for r in parts[2][rel_start:rel_end].split(",")
                    )

            try:
                start_inheritance = parts[0].index("(")
                end_inheritance = parts[0].index(")")
                sem = parts[0][:start_inheritance].strip()
                supertype = parts[0][start_inheritance + 1 : end_inheritance]
            except:
                sem = parts[0].strip()
                supertype = None

            if sem not in self:
                self[sem] = {}

            if (
                relationships is None
            ):  # if not specified for a specific relationship set
                # add all relationship entries for parents
                if supertype:
                    for rels in self[supertype]:
                        if rels not in self[sem]:
                            self[sem][rels] = []
                        glue = self[supertype][rels]
                        self[sem][rels].extend(glue)
                        self[sem][rels].extend(
                            glue_formulas
                        )  # add the glue formulas to every rel entry
                else:
                    if None not in self[sem]:
                        self[sem][None] = []
                    self[sem][None].extend(
                        glue_formulas
                    )  # add the glue formulas to every rel entry
            else:
                if relationships not in self[sem]:
                    self[sem][relationships] = []
                if supertype:
                    self[sem][relationships].extend(self[supertype][relationships])
                self[sem][relationships].extend(
                    glue_formulas
                )  # add the glue entry to the dictionary

    def __str__(self):
        accum = ""
        for pos in self:
            str_pos = "%s" % pos
            for relset in self[pos]:
                i = 1
                for gf in self[pos][relset]:
                    if i == 1:
                        accum += str_pos + ": "
                    else:
                        accum += " " * (len(str_pos) + 2)
                    accum += "%s" % gf
                    if relset and i == len(self[pos][relset]):
                        accum += " : %s" % relset
                    accum += "\n"
                    i += 1
        return accum

    def to_glueformula_list(self, depgraph, node=None, counter=None, verbose=False):
        if node is None:
            # TODO: should it be depgraph.root? Is this code tested?
            top = depgraph.nodes[0]
            depList = list(chain.from_iterable(top["deps"].values()))
            root = depgraph.nodes[depList[0]]

            return self.to_glueformula_list(depgraph, root, Counter(), verbose)

        glueformulas = self.lookup(node, depgraph, counter)
        for dep_idx in chain.from_iterable(node["deps"].values()):
            dep = depgraph.nodes[dep_idx]
            glueformulas.extend(
                self.to_glueformula_list(depgraph, dep, counter, verbose)
            )
        return glueformulas

    def lookup(self, node, depgraph, counter):
        semtype_names = self.get_semtypes(node)

        semtype = None
        for name in semtype_names:
            if name in self:
                semtype = self[name]
                break
        if semtype is None:
            # raise KeyError, "There is no GlueDict entry for sem type '%s' (for '%s')" % (sem, word)
            return []

        self.add_missing_dependencies(node, depgraph)

        lookup = self._lookup_semtype_option(semtype, node, depgraph)

        if not len(lookup):
            raise KeyError(
                "There is no GlueDict entry for sem type of '%s' "
                "with tag '%s', and rel '%s'" % (node["word"], node["tag"], node["rel"])
            )

        return self.get_glueformulas_from_semtype_entry(
            lookup, node["word"], node, depgraph, counter
        )

    def add_missing_dependencies(self, node, depgraph):
        rel = node["rel"].lower()

        if rel == "main":
            headnode = depgraph.nodes[node["head"]]
            subj = self.lookup_unique("subj", headnode, depgraph)
            relation = subj["rel"]
            node["deps"].setdefault(relation, [])
            node["deps"][relation].append(subj["address"])
            # node['deps'].append(subj['address'])

    def _lookup_semtype_option(self, semtype, node, depgraph):
        relationships = frozenset(
            depgraph.nodes[dep]["rel"].lower()
            for dep in chain.from_iterable(node["deps"].values())
            if depgraph.nodes[dep]["rel"].lower() not in OPTIONAL_RELATIONSHIPS
        )

        try:
            lookup = semtype[relationships]
        except KeyError:
            # An exact match is not found, so find the best match where
            # 'best' is defined as the glue entry whose relationship set has the
            # most relations of any possible relationship set that is a subset
            # of the actual depgraph
            best_match = frozenset()
            for relset_option in set(semtype) - {None}:
                if (
                    len(relset_option) > len(best_match)
                    and relset_option < relationships
                ):
                    best_match = relset_option
            if not best_match:
                if None in semtype:
                    best_match = None
                else:
                    return None
            lookup = semtype[best_match]

        return lookup

    def get_semtypes(self, node):
        """
        Based on the node, return a list of plausible semtypes in order of
        plausibility.
        """
        rel = node["rel"].lower()
        word = node["word"].lower()

        if rel == "spec":
            if word in SPEC_SEMTYPES:
                return [SPEC_SEMTYPES[word]]
            else:
                return [SPEC_SEMTYPES["default"]]
        elif rel in ["nmod", "vmod"]:
            return [node["tag"], rel]
        else:
            return [node["tag"]]

    def get_glueformulas_from_semtype_entry(
        self, lookup, word, node, depgraph, counter
    ):
        glueformulas = []

        glueFormulaFactory = self.get_GlueFormula_factory()
        for meaning, glue in lookup:
            gf = glueFormulaFactory(self.get_meaning_formula(meaning, word), glue)
            if not len(glueformulas):
                gf.word = word
            else:
                gf.word = f"{word}{len(glueformulas) + 1}"

            gf.glue = self.initialize_labels(gf.glue, node, depgraph, counter.get())

            glueformulas.append(gf)
        return glueformulas

    def get_meaning_formula(self, generic, word):
        """
        :param generic: A meaning formula string containing the
            parameter "<word>"
        :param word: The actual word to be replace "<word>"
        """
        word = word.replace(".", "")
        return generic.replace("<word>", word)

    def initialize_labels(self, expr, node, depgraph, unique_index):
        if isinstance(expr, linearlogic.AtomicExpression):
            name = self.find_label_name(expr.name.lower(), node, depgraph, unique_index)
            if name[0].isupper():
                return linearlogic.VariableExpression(name)
            else:
                return linearlogic.ConstantExpression(name)
        else:
            return linearlogic.ImpExpression(
                self.initialize_labels(expr.antecedent, node, depgraph, unique_index),
                self.initialize_labels(expr.consequent, node, depgraph, unique_index),
            )

    def find_label_name(self, name, node, depgraph, unique_index):
        try:
            dot = name.index(".")

            before_dot = name[:dot]
            after_dot = name[dot + 1 :]
            if before_dot == "super":
                return self.find_label_name(
                    after_dot, depgraph.nodes[node["head"]], depgraph, unique_index
                )
            else:
                return self.find_label_name(
                    after_dot,
                    self.lookup_unique(before_dot, node, depgraph),
                    depgraph,
                    unique_index,
                )
        except ValueError:
            lbl = self.get_label(node)
            if name == "f":
                return lbl
            elif name == "v":
                return "%sv" % lbl
            elif name == "r":
                return "%sr" % lbl
            elif name == "super":
                return self.get_label(depgraph.nodes[node["head"]])
            elif name == "var":
                return f"{lbl.upper()}{unique_index}"
            elif name == "a":
                return self.get_label(self.lookup_unique("conja", node, depgraph))
            elif name == "b":
                return self.get_label(self.lookup_unique("conjb", node, depgraph))
            else:
                return self.get_label(self.lookup_unique(name, node, depgraph))

    def get_label(self, node):
        """
        Pick an alphabetic character as identifier for an entity in the model.

        :param value: where to index into the list of characters
        :type value: int
        """
        value = node["address"]

        letter = [
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "a",
            "b",
            "c",
            "d",
            "e",
        ][value - 1]
        num = int(value) // 26
        if num > 0:
            return letter + str(num)
        else:
            return letter

    def lookup_unique(self, rel, node, depgraph):
        """
        Lookup 'key'. There should be exactly one item in the associated relation.
        """
        deps = [
            depgraph.nodes[dep]
            for dep in chain.from_iterable(node["deps"].values())
            if depgraph.nodes[dep]["rel"].lower() == rel.lower()
        ]

        if len(deps) == 0:
            raise KeyError(
                "'{}' doesn't contain a feature '{}'".format(node["word"], rel)
            )
        elif len(deps) > 1:
            raise KeyError(
                "'{}' should only have one feature '{}'".format(node["word"], rel)
            )
        else:
            return deps[0]

    def get_GlueFormula_factory(self):
        return GlueFormula


class Glue:
    def __init__(
        self, semtype_file=None, remove_duplicates=False, depparser=None, verbose=False
    ):
        self.verbose = verbose
        self.remove_duplicates = remove_duplicates
        self.depparser = depparser

        from nltk import Prover9

        self.prover = Prover9()

        if semtype_file:
            self.semtype_file = semtype_file
        else:
            self.semtype_file = os.path.join(
                "grammars", "sample_grammars", "glue.semtype"
            )

    def train_depparser(self, depgraphs=None):
        if depgraphs:
            self.depparser.train(depgraphs)
        else:
            self.depparser.train_from_file(
                nltk.data.find(
                    os.path.join("grammars", "sample_grammars", "glue_train.conll")
                )
            )

    def parse_to_meaning(self, sentence):
        readings = []
        for agenda in self.parse_to_compiled(sentence):
            readings.extend(self.get_readings(agenda))
        return readings

    def get_readings(self, agenda):
        readings = []
        agenda_length = len(agenda)
        atomics = dict()
        nonatomics = dict()
        while agenda:  # is not empty
            cur = agenda.pop()
            glue_simp = cur.glue.simplify()
            if isinstance(
                glue_simp, linearlogic.ImpExpression
            ):  # if cur.glue is non-atomic
                for key in atomics:
                    try:
                        if isinstance(cur.glue, linearlogic.ApplicationExpression):
                            bindings = cur.glue.bindings
                        else:
                            bindings = linearlogic.BindingDict()
                        glue_simp.antecedent.unify(key, bindings)
                        for atomic in atomics[key]:
                            if not (
                                cur.indices & atomic.indices
                            ):  # if the sets of indices are disjoint
                                try:
                                    agenda.append(cur.applyto(atomic))
                                except linearlogic.LinearLogicApplicationException:
                                    pass
                    except linearlogic.UnificationException:
                        pass
                try:
                    nonatomics[glue_simp.antecedent].append(cur)
                except KeyError:
                    nonatomics[glue_simp.antecedent] = [cur]

            else:  # else cur.glue is atomic
                for key in nonatomics:
                    for nonatomic in nonatomics[key]:
                        try:
                            if isinstance(
                                nonatomic.glue, linearlogic.ApplicationExpression
                            ):
                                bindings = nonatomic.glue.bindings
                            else:
                                bindings = linearlogic.BindingDict()
                            glue_simp.unify(key, bindings)
                            if not (
                                cur.indices & nonatomic.indices
                            ):  # if the sets of indices are disjoint
                                try:
                                    agenda.append(nonatomic.applyto(cur))
                                except linearlogic.LinearLogicApplicationException:
                                    pass
                        except linearlogic.UnificationException:
                            pass
                try:
                    atomics[glue_simp].append(cur)
                except KeyError:
                    atomics[glue_simp] = [cur]

        for entry in atomics:
            for gf in atomics[entry]:
                if len(gf.indices) == agenda_length:
                    self._add_to_reading_list(gf, readings)
        for entry in nonatomics:
            for gf in nonatomics[entry]:
                if len(gf.indices) == agenda_length:
                    self._add_to_reading_list(gf, readings)
        return readings

    def _add_to_reading_list(self, glueformula, reading_list):
        add_reading = True
        if self.remove_duplicates:
            for reading in reading_list:
                try:
                    if reading.equiv(glueformula.meaning, self.prover):
                        add_reading = False
                        break
                except Exception as e:
                    # if there is an exception, the syntax of the formula
                    # may not be understandable by the prover, so don't
                    # throw out the reading.
                    print("Error when checking logical equality of statements", e)

        if add_reading:
            reading_list.append(glueformula.meaning)

    def parse_to_compiled(self, sentence):
        gfls = [self.depgraph_to_glue(dg) for dg in self.dep_parse(sentence)]
        return [self.gfl_to_compiled(gfl) for gfl in gfls]

    def dep_parse(self, sentence):
        """
        Return a dependency graph for the sentence.

        :param sentence: the sentence to be parsed
        :type sentence: list(str)
        :rtype: DependencyGraph
        """

        # Lazy-initialize the depparser
        if self.depparser is None:
            from nltk.parse import MaltParser

            self.depparser = MaltParser(tagger=self.get_pos_tagger())
        if not self.depparser._trained:
            self.train_depparser()
        return self.depparser.parse(sentence, verbose=self.verbose)

    def depgraph_to_glue(self, depgraph):
        return self.get_glue_dict().to_glueformula_list(depgraph)

    def get_glue_dict(self):
        return GlueDict(self.semtype_file)

    def gfl_to_compiled(self, gfl):
        index_counter = Counter()
        return_list = []
        for gf in gfl:
            return_list.extend(gf.compile(index_counter))

        if self.verbose:
            print("Compiled Glue Premises:")
            for cgf in return_list:
                print(cgf)

        return return_list

    def get_pos_tagger(self):
        from nltk.corpus import brown

        regexp_tagger = RegexpTagger(
            [
                (r"^-?[0-9]+(\.[0-9]+)?$", "CD"),  # cardinal numbers
                (r"(The|the|A|a|An|an)$", "AT"),  # articles
                (r".*able$", "JJ"),  # adjectives
                (r".*ness$", "NN"),  # nouns formed from adjectives
                (r".*ly$", "RB"),  # adverbs
                (r".*s$", "NNS"),  # plural nouns
                (r".*ing$", "VBG"),  # gerunds
                (r".*ed$", "VBD"),  # past tense verbs
                (r".*", "NN"),  # nouns (default)
            ]
        )
        brown_train = brown.tagged_sents(categories="news")
        unigram_tagger = UnigramTagger(brown_train, backoff=regexp_tagger)
        bigram_tagger = BigramTagger(brown_train, backoff=unigram_tagger)
        trigram_tagger = TrigramTagger(brown_train, backoff=bigram_tagger)

        # Override particular words
        main_tagger = RegexpTagger(
            [(r"(A|a|An|an)$", "ex_quant"), (r"(Every|every|All|all)$", "univ_quant")],
            backoff=trigram_tagger,
        )

        return main_tagger


class DrtGlueFormula(GlueFormula):
    def __init__(self, meaning, glue, indices=None):
        if not indices:
            indices = set()

        if isinstance(meaning, str):
            self.meaning = drt.DrtExpression.fromstring(meaning)
        elif isinstance(meaning, drt.DrtExpression):
            self.meaning = meaning
        else:
            raise RuntimeError(
                "Meaning term neither string or expression: %s, %s"
                % (meaning, meaning.__class__)
            )

        if isinstance(glue, str):
            self.glue = linearlogic.LinearLogicParser().parse(glue)
        elif isinstance(glue, linearlogic.Expression):
            self.glue = glue
        else:
            raise RuntimeError(
                "Glue term neither string or expression: %s, %s"
                % (glue, glue.__class__)
            )

        self.indices = indices

    def make_VariableExpression(self, name):
        return drt.DrtVariableExpression(name)

    def make_LambdaExpression(self, variable, term):
        return drt.DrtLambdaExpression(variable, term)


class DrtGlueDict(GlueDict):
    def get_GlueFormula_factory(self):
        return DrtGlueFormula


class DrtGlue(Glue):
    def __init__(
        self, semtype_file=None, remove_duplicates=False, depparser=None, verbose=False
    ):
        if not semtype_file:
            semtype_file = os.path.join(
                "grammars", "sample_grammars", "drt_glue.semtype"
            )
        Glue.__init__(self, semtype_file, remove_duplicates, depparser, verbose)

    def get_glue_dict(self):
        return DrtGlueDict(self.semtype_file)


def demo(show_example=-1):
    from nltk.parse import MaltParser

    examples = [
        "David sees Mary",
        "David eats a sandwich",
        "every man chases a dog",
        "every man believes a dog sleeps",
        "John gives David a sandwich",
        "John chases himself",
    ]
    #                'John persuades David to order a pizza',
    #                'John tries to go',
    #                'John tries to find a unicorn',
    #                'John seems to vanish',
    #                'a unicorn seems to approach',
    #                'every big cat leaves',
    #                'every gray cat leaves',
    #                'every big gray cat leaves',
    #                'a former senator leaves',

    print("============== DEMO ==============")

    tagger = RegexpTagger(
        [
            ("^(David|Mary|John)$", "NNP"),
            (
                "^(sees|eats|chases|believes|gives|sleeps|chases|persuades|tries|seems|leaves)$",
                "VB",
            ),
            ("^(go|order|vanish|find|approach)$", "VB"),
            ("^(a)$", "ex_quant"),
            ("^(every)$", "univ_quant"),
            ("^(sandwich|man|dog|pizza|unicorn|cat|senator)$", "NN"),
            ("^(big|gray|former)$", "JJ"),
            ("^(him|himself)$", "PRP"),
        ]
    )

    depparser = MaltParser(tagger=tagger)
    glue = Glue(depparser=depparser, verbose=False)

    for (i, sentence) in enumerate(examples):
        if i == show_example or show_example == -1:
            print(f"[[[Example {i}]]]  {sentence}")
            for reading in glue.parse_to_meaning(sentence.split()):
                print(reading.simplify())
            print("")


if __name__ == "__main__":
    demo()
