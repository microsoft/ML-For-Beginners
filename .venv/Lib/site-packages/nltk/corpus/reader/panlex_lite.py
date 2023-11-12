# Natural Language Toolkit: PanLex Corpus Reader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: David Kamholz <kamholz@panlex.org>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
CorpusReader for PanLex Lite, a stripped down version of PanLex distributed
as an SQLite database. See the README.txt in the panlex_lite corpus directory
for more information on PanLex Lite.
"""

import os
import sqlite3

from nltk.corpus.reader.api import CorpusReader


class PanLexLiteCorpusReader(CorpusReader):
    MEANING_Q = """
        SELECT dnx2.mn, dnx2.uq, dnx2.ap, dnx2.ui, ex2.tt, ex2.lv
        FROM dnx
        JOIN ex ON (ex.ex = dnx.ex)
        JOIN dnx dnx2 ON (dnx2.mn = dnx.mn)
        JOIN ex ex2 ON (ex2.ex = dnx2.ex)
        WHERE dnx.ex != dnx2.ex AND ex.tt = ? AND ex.lv = ?
        ORDER BY dnx2.uq DESC
    """

    TRANSLATION_Q = """
        SELECT s.tt, sum(s.uq) AS trq FROM (
            SELECT ex2.tt, max(dnx.uq) AS uq
            FROM dnx
            JOIN ex ON (ex.ex = dnx.ex)
            JOIN dnx dnx2 ON (dnx2.mn = dnx.mn)
            JOIN ex ex2 ON (ex2.ex = dnx2.ex)
            WHERE dnx.ex != dnx2.ex AND ex.lv = ? AND ex.tt = ? AND ex2.lv = ?
            GROUP BY ex2.tt, dnx.ui
        ) s
        GROUP BY s.tt
        ORDER BY trq DESC, s.tt
    """

    def __init__(self, root):
        self._c = sqlite3.connect(os.path.join(root, "db.sqlite")).cursor()

        self._uid_lv = {}
        self._lv_uid = {}

        for row in self._c.execute("SELECT uid, lv FROM lv"):
            self._uid_lv[row[0]] = row[1]
            self._lv_uid[row[1]] = row[0]

    def language_varieties(self, lc=None):
        """
        Return a list of PanLex language varieties.

        :param lc: ISO 639 alpha-3 code. If specified, filters returned varieties
            by this code. If unspecified, all varieties are returned.
        :return: the specified language varieties as a list of tuples. The first
            element is the language variety's seven-character uniform identifier,
            and the second element is its default name.
        :rtype: list(tuple)
        """

        if lc is None:
            return self._c.execute("SELECT uid, tt FROM lv ORDER BY uid").fetchall()
        else:
            return self._c.execute(
                "SELECT uid, tt FROM lv WHERE lc = ? ORDER BY uid", (lc,)
            ).fetchall()

    def meanings(self, expr_uid, expr_tt):
        """
        Return a list of meanings for an expression.

        :param expr_uid: the expression's language variety, as a seven-character
            uniform identifier.
        :param expr_tt: the expression's text.
        :return: a list of Meaning objects.
        :rtype: list(Meaning)
        """

        expr_lv = self._uid_lv[expr_uid]

        mn_info = {}

        for i in self._c.execute(self.MEANING_Q, (expr_tt, expr_lv)):
            mn = i[0]
            uid = self._lv_uid[i[5]]

            if not mn in mn_info:
                mn_info[mn] = {
                    "uq": i[1],
                    "ap": i[2],
                    "ui": i[3],
                    "ex": {expr_uid: [expr_tt]},
                }

            if not uid in mn_info[mn]["ex"]:
                mn_info[mn]["ex"][uid] = []

            mn_info[mn]["ex"][uid].append(i[4])

        return [Meaning(mn, mn_info[mn]) for mn in mn_info]

    def translations(self, from_uid, from_tt, to_uid):
        """
        Return a list of translations for an expression into a single language
        variety.

        :param from_uid: the source expression's language variety, as a
            seven-character uniform identifier.
        :param from_tt: the source expression's text.
        :param to_uid: the target language variety, as a seven-character
            uniform identifier.
        :return: a list of translation tuples. The first element is the expression
            text and the second element is the translation quality.
        :rtype: list(tuple)
        """

        from_lv = self._uid_lv[from_uid]
        to_lv = self._uid_lv[to_uid]

        return self._c.execute(self.TRANSLATION_Q, (from_lv, from_tt, to_lv)).fetchall()


class Meaning(dict):
    """
    Represents a single PanLex meaning. A meaning is a translation set derived
    from a single source.
    """

    def __init__(self, mn, attr):
        super().__init__(**attr)
        self["mn"] = mn

    def id(self):
        """
        :return: the meaning's id.
        :rtype: int
        """
        return self["mn"]

    def quality(self):
        """
        :return: the meaning's source's quality (0=worst, 9=best).
        :rtype: int
        """
        return self["uq"]

    def source(self):
        """
        :return: the meaning's source id.
        :rtype: int
        """
        return self["ap"]

    def source_group(self):
        """
        :return: the meaning's source group id.
        :rtype: int
        """
        return self["ui"]

    def expressions(self):
        """
        :return: the meaning's expressions as a dictionary whose keys are language
            variety uniform identifiers and whose values are lists of expression
            texts.
        :rtype: dict
        """
        return self["ex"]
