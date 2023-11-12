# Natural Language Toolkit: Agreement Metrics
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Lauri Hallila <laurihallila@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

"""Counts Paice's performance statistics for evaluating stemming algorithms.

What is required:
 - A dictionary of words grouped by their real lemmas
 - A dictionary of words grouped by stems from a stemming algorithm

When these are given, Understemming Index (UI), Overstemming Index (OI),
Stemming Weight (SW) and Error-rate relative to truncation (ERRT) are counted.

References:
Chris D. Paice (1994). An evaluation method for stemming algorithms.
In Proceedings of SIGIR, 42--50.
"""

from math import sqrt


def get_words_from_dictionary(lemmas):
    """
    Get original set of words used for analysis.

    :param lemmas: A dictionary where keys are lemmas and values are sets
        or lists of words corresponding to that lemma.
    :type lemmas: dict(str): list(str)
    :return: Set of words that exist as values in the dictionary
    :rtype: set(str)
    """
    words = set()
    for lemma in lemmas:
        words.update(set(lemmas[lemma]))
    return words


def _truncate(words, cutlength):
    """Group words by stems defined by truncating them at given length.

    :param words: Set of words used for analysis
    :param cutlength: Words are stemmed by cutting at this length.
    :type words: set(str) or list(str)
    :type cutlength: int
    :return: Dictionary where keys are stems and values are sets of words
    corresponding to that stem.
    :rtype: dict(str): set(str)
    """
    stems = {}
    for word in words:
        stem = word[:cutlength]
        try:
            stems[stem].update([word])
        except KeyError:
            stems[stem] = {word}
    return stems


# Reference: https://en.wikipedia.org/wiki/Line-line_intersection
def _count_intersection(l1, l2):
    """Count intersection between two line segments defined by coordinate pairs.

    :param l1: Tuple of two coordinate pairs defining the first line segment
    :param l2: Tuple of two coordinate pairs defining the second line segment
    :type l1: tuple(float, float)
    :type l2: tuple(float, float)
    :return: Coordinates of the intersection
    :rtype: tuple(float, float)
    """
    x1, y1 = l1[0]
    x2, y2 = l1[1]
    x3, y3 = l2[0]
    x4, y4 = l2[1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0.0:  # lines are parallel
        if x1 == x2 == x3 == x4 == 0.0:
            # When lines are parallel, they must be on the y-axis.
            # We can ignore x-axis because we stop counting the
            # truncation line when we get there.
            # There are no other options as UI (x-axis) grows and
            # OI (y-axis) diminishes when we go along the truncation line.
            return (0.0, y4)

    x = (
        (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    y = (
        (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return (x, y)


def _get_derivative(coordinates):
    """Get derivative of the line from (0,0) to given coordinates.

    :param coordinates: A coordinate pair
    :type coordinates: tuple(float, float)
    :return: Derivative; inf if x is zero
    :rtype: float
    """
    try:
        return coordinates[1] / coordinates[0]
    except ZeroDivisionError:
        return float("inf")


def _calculate_cut(lemmawords, stems):
    """Count understemmed and overstemmed pairs for (lemma, stem) pair with common words.

    :param lemmawords: Set or list of words corresponding to certain lemma.
    :param stems: A dictionary where keys are stems and values are sets
    or lists of words corresponding to that stem.
    :type lemmawords: set(str) or list(str)
    :type stems: dict(str): set(str)
    :return: Amount of understemmed and overstemmed pairs contributed by words
    existing in both lemmawords and stems.
    :rtype: tuple(float, float)
    """
    umt, wmt = 0.0, 0.0
    for stem in stems:
        cut = set(lemmawords) & set(stems[stem])
        if cut:
            cutcount = len(cut)
            stemcount = len(stems[stem])
            # Unachieved merge total
            umt += cutcount * (len(lemmawords) - cutcount)
            # Wrongly merged total
            wmt += cutcount * (stemcount - cutcount)
    return (umt, wmt)


def _calculate(lemmas, stems):
    """Calculate actual and maximum possible amounts of understemmed and overstemmed word pairs.

    :param lemmas: A dictionary where keys are lemmas and values are sets
    or lists of words corresponding to that lemma.
    :param stems: A dictionary where keys are stems and values are sets
    or lists of words corresponding to that stem.
    :type lemmas: dict(str): list(str)
    :type stems: dict(str): set(str)
    :return: Global unachieved merge total (gumt),
    global desired merge total (gdmt),
    global wrongly merged total (gwmt) and
    global desired non-merge total (gdnt).
    :rtype: tuple(float, float, float, float)
    """

    n = sum(len(lemmas[word]) for word in lemmas)

    gdmt, gdnt, gumt, gwmt = (0.0, 0.0, 0.0, 0.0)

    for lemma in lemmas:
        lemmacount = len(lemmas[lemma])

        # Desired merge total
        gdmt += lemmacount * (lemmacount - 1)

        # Desired non-merge total
        gdnt += lemmacount * (n - lemmacount)

        # For each (lemma, stem) pair with common words, count how many
        # pairs are understemmed and overstemmed.
        umt, wmt = _calculate_cut(lemmas[lemma], stems)

        # Add to total undesired and wrongly-merged totals
        gumt += umt
        gwmt += wmt

    # Each object is counted twice, so divide by two
    return (gumt / 2, gdmt / 2, gwmt / 2, gdnt / 2)


def _indexes(gumt, gdmt, gwmt, gdnt):
    """Count Understemming Index (UI), Overstemming Index (OI) and Stemming Weight (SW).

    :param gumt, gdmt, gwmt, gdnt: Global unachieved merge total (gumt),
    global desired merge total (gdmt),
    global wrongly merged total (gwmt) and
    global desired non-merge total (gdnt).
    :type gumt, gdmt, gwmt, gdnt: float
    :return: Understemming Index (UI),
    Overstemming Index (OI) and
    Stemming Weight (SW).
    :rtype: tuple(float, float, float)
    """
    # Calculate Understemming Index (UI),
    # Overstemming Index (OI) and Stemming Weight (SW)
    try:
        ui = gumt / gdmt
    except ZeroDivisionError:
        # If GDMT (max merge total) is 0, define UI as 0
        ui = 0.0
    try:
        oi = gwmt / gdnt
    except ZeroDivisionError:
        # IF GDNT (max non-merge total) is 0, define OI as 0
        oi = 0.0
    try:
        sw = oi / ui
    except ZeroDivisionError:
        if oi == 0.0:
            # OI and UI are 0, define SW as 'not a number'
            sw = float("nan")
        else:
            # UI is 0, define SW as infinity
            sw = float("inf")
    return (ui, oi, sw)


class Paice:
    """Class for storing lemmas, stems and evaluation metrics."""

    def __init__(self, lemmas, stems):
        """
        :param lemmas: A dictionary where keys are lemmas and values are sets
            or lists of words corresponding to that lemma.
        :param stems: A dictionary where keys are stems and values are sets
            or lists of words corresponding to that stem.
        :type lemmas: dict(str): list(str)
        :type stems: dict(str): set(str)
        """
        self.lemmas = lemmas
        self.stems = stems
        self.coords = []
        self.gumt, self.gdmt, self.gwmt, self.gdnt = (None, None, None, None)
        self.ui, self.oi, self.sw = (None, None, None)
        self.errt = None
        self.update()

    def __str__(self):
        text = ["Global Unachieved Merge Total (GUMT): %s\n" % self.gumt]
        text.append("Global Desired Merge Total (GDMT): %s\n" % self.gdmt)
        text.append("Global Wrongly-Merged Total (GWMT): %s\n" % self.gwmt)
        text.append("Global Desired Non-merge Total (GDNT): %s\n" % self.gdnt)
        text.append("Understemming Index (GUMT / GDMT): %s\n" % self.ui)
        text.append("Overstemming Index (GWMT / GDNT): %s\n" % self.oi)
        text.append("Stemming Weight (OI / UI): %s\n" % self.sw)
        text.append("Error-Rate Relative to Truncation (ERRT): %s\r\n" % self.errt)
        coordinates = " ".join(["(%s, %s)" % item for item in self.coords])
        text.append("Truncation line: %s" % coordinates)
        return "".join(text)

    def _get_truncation_indexes(self, words, cutlength):
        """Count (UI, OI) when stemming is done by truncating words at \'cutlength\'.

        :param words: Words used for the analysis
        :param cutlength: Words are stemmed by cutting them at this length
        :type words: set(str) or list(str)
        :type cutlength: int
        :return: Understemming and overstemming indexes
        :rtype: tuple(int, int)
        """

        truncated = _truncate(words, cutlength)
        gumt, gdmt, gwmt, gdnt = _calculate(self.lemmas, truncated)
        ui, oi = _indexes(gumt, gdmt, gwmt, gdnt)[:2]
        return (ui, oi)

    def _get_truncation_coordinates(self, cutlength=0):
        """Count (UI, OI) pairs for truncation points until we find the segment where (ui, oi) crosses the truncation line.

        :param cutlength: Optional parameter to start counting from (ui, oi)
        coordinates gotten by stemming at this length. Useful for speeding up
        the calculations when you know the approximate location of the
        intersection.
        :type cutlength: int
        :return: List of coordinate pairs that define the truncation line
        :rtype: list(tuple(float, float))
        """
        words = get_words_from_dictionary(self.lemmas)
        maxlength = max(len(word) for word in words)

        # Truncate words from different points until (0, 0) - (ui, oi) segment crosses the truncation line
        coords = []
        while cutlength <= maxlength:
            # Get (UI, OI) pair of current truncation point
            pair = self._get_truncation_indexes(words, cutlength)

            # Store only new coordinates so we'll have an actual
            # line segment when counting the intersection point
            if pair not in coords:
                coords.append(pair)
            if pair == (0.0, 0.0):
                # Stop counting if truncation line goes through origo;
                # length from origo to truncation line is 0
                return coords
            if len(coords) >= 2 and pair[0] > 0.0:
                derivative1 = _get_derivative(coords[-2])
                derivative2 = _get_derivative(coords[-1])
                # Derivative of the truncation line is a decreasing value;
                # when it passes Stemming Weight, we've found the segment
                # of truncation line intersecting with (0, 0) - (ui, oi) segment
                if derivative1 >= self.sw >= derivative2:
                    return coords
            cutlength += 1
        return coords

    def _errt(self):
        """Count Error-Rate Relative to Truncation (ERRT).

        :return: ERRT, length of the line from origo to (UI, OI) divided by
        the length of the line from origo to the point defined by the same
        line when extended until the truncation line.
        :rtype: float
        """
        # Count (UI, OI) pairs for truncation points until we find the segment where (ui, oi) crosses the truncation line
        self.coords = self._get_truncation_coordinates()
        if (0.0, 0.0) in self.coords:
            # Truncation line goes through origo, so ERRT cannot be counted
            if (self.ui, self.oi) != (0.0, 0.0):
                return float("inf")
            else:
                return float("nan")
        if (self.ui, self.oi) == (0.0, 0.0):
            # (ui, oi) is origo; define errt as 0.0
            return 0.0
        # Count the intersection point
        # Note that (self.ui, self.oi) cannot be (0.0, 0.0) and self.coords has different coordinates
        # so we have actual line segments instead of a line segment and a point
        intersection = _count_intersection(
            ((0, 0), (self.ui, self.oi)), self.coords[-2:]
        )
        # Count OP (length of the line from origo to (ui, oi))
        op = sqrt(self.ui**2 + self.oi**2)
        # Count OT (length of the line from origo to truncation line that goes through (ui, oi))
        ot = sqrt(intersection[0] ** 2 + intersection[1] ** 2)
        # OP / OT tells how well the stemming algorithm works compared to just truncating words
        return op / ot

    def update(self):
        """Update statistics after lemmas and stems have been set."""
        self.gumt, self.gdmt, self.gwmt, self.gdnt = _calculate(self.lemmas, self.stems)
        self.ui, self.oi, self.sw = _indexes(self.gumt, self.gdmt, self.gwmt, self.gdnt)
        self.errt = self._errt()


def demo():
    """Demonstration of the module."""
    # Some words with their real lemmas
    lemmas = {
        "kneel": ["kneel", "knelt"],
        "range": ["range", "ranged"],
        "ring": ["ring", "rang", "rung"],
    }
    # Same words with stems from a stemming algorithm
    stems = {
        "kneel": ["kneel"],
        "knelt": ["knelt"],
        "rang": ["rang", "range", "ranged"],
        "ring": ["ring"],
        "rung": ["rung"],
    }
    print("Words grouped by their lemmas:")
    for lemma in sorted(lemmas):
        print("{} => {}".format(lemma, " ".join(lemmas[lemma])))
    print()
    print("Same words grouped by a stemming algorithm:")
    for stem in sorted(stems):
        print("{} => {}".format(stem, " ".join(stems[stem])))
    print()
    p = Paice(lemmas, stems)
    print(p)
    print()
    # Let's "change" results from a stemming algorithm
    stems = {
        "kneel": ["kneel"],
        "knelt": ["knelt"],
        "rang": ["rang"],
        "range": ["range", "ranged"],
        "ring": ["ring"],
        "rung": ["rung"],
    }
    print("Counting stats after changing stemming results:")
    for stem in sorted(stems):
        print("{} => {}".format(stem, " ".join(stems[stem])))
    print()
    p.stems = stems
    p.update()
    print(p)


if __name__ == "__main__":
    demo()
