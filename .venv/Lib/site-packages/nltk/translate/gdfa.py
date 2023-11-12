# Natural Language Toolkit: GDFA word alignment symmetrization
#
# Copyright (C) 2001-2023 NLTK Project
# Authors: Liling Tan
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

from collections import defaultdict


def grow_diag_final_and(srclen, trglen, e2f, f2e):
    """
    This module symmetrisatizes the source-to-target and target-to-source
    word alignment output and produces, aka. GDFA algorithm (Koehn, 2005).

    Step 1: Find the intersection of the bidirectional alignment.

    Step 2: Search for additional neighbor alignment points to be added, given
            these criteria: (i) neighbor alignments points are not in the
            intersection and (ii) neighbor alignments are in the union.

    Step 3: Add all other alignment points that are not in the intersection, not in
            the neighboring alignments that met the criteria but in the original
            forward/backward alignment outputs.

        >>> forw = ('0-0 2-1 9-2 21-3 10-4 7-5 11-6 9-7 12-8 1-9 3-10 '
        ...         '4-11 17-12 17-13 25-14 13-15 24-16 11-17 28-18')
        >>> back = ('0-0 1-9 2-9 3-10 4-11 5-12 6-6 7-5 8-6 9-7 10-4 '
        ...         '11-6 12-8 13-12 15-12 17-13 18-13 19-12 20-13 '
        ...         '21-3 22-12 23-14 24-17 25-15 26-17 27-18 28-18')
        >>> srctext = ("この よう な ハロー 白色 わい 星 の Ｌ 関数 "
        ...            "は Ｌ と 共 に 不連続 に 増加 する こと が "
        ...            "期待 さ れる こと を 示し た 。")
        >>> trgtext = ("Therefore , we expect that the luminosity function "
        ...            "of such halo white dwarfs increases discontinuously "
        ...            "with the luminosity .")
        >>> srclen = len(srctext.split())
        >>> trglen = len(trgtext.split())
        >>>
        >>> gdfa = grow_diag_final_and(srclen, trglen, forw, back)
        >>> gdfa == sorted(set([(28, 18), (6, 6), (24, 17), (2, 1), (15, 12), (13, 12),
        ...         (2, 9), (3, 10), (26, 17), (25, 15), (8, 6), (9, 7), (20,
        ...         13), (18, 13), (0, 0), (10, 4), (13, 15), (23, 14), (7, 5),
        ...         (25, 14), (1, 9), (17, 13), (4, 11), (11, 17), (9, 2), (22,
        ...         12), (27, 18), (24, 16), (21, 3), (19, 12), (17, 12), (5,
        ...         12), (11, 6), (12, 8)]))
        True

    References:
    Koehn, P., A. Axelrod, A. Birch, C. Callison, M. Osborne, and D. Talbot.
    2005. Edinburgh System Description for the 2005 IWSLT Speech
    Translation Evaluation. In MT Eval Workshop.

    :type srclen: int
    :param srclen: the number of tokens in the source language
    :type trglen: int
    :param trglen: the number of tokens in the target language
    :type e2f: str
    :param e2f: the forward word alignment outputs from source-to-target
                language (in pharaoh output format)
    :type f2e: str
    :param f2e: the backward word alignment outputs from target-to-source
                language (in pharaoh output format)
    :rtype: set(tuple(int))
    :return: the symmetrized alignment points from the GDFA algorithm
    """

    # Converts pharaoh text format into list of tuples.
    e2f = [tuple(map(int, a.split("-"))) for a in e2f.split()]
    f2e = [tuple(map(int, a.split("-"))) for a in f2e.split()]

    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    alignment = set(e2f).intersection(set(f2e))  # Find the intersection.
    union = set(e2f).union(set(f2e))

    # *aligned* is used to check if neighbors are aligned in grow_diag()
    aligned = defaultdict(set)
    for i, j in alignment:
        aligned["e"].add(i)
        aligned["f"].add(j)

    def grow_diag():
        """
        Search for the neighbor points and them to the intersected alignment
        points if criteria are met.
        """
        prev_len = len(alignment) - 1
        # iterate until no new points added
        while prev_len < len(alignment):
            no_new_points = True
            # for english word e = 0 ... en
            for e in range(srclen):
                # for foreign word f = 0 ... fn
                for f in range(trglen):
                    # if ( e aligned with f)
                    if (e, f) in alignment:
                        # for each neighboring point (e-new, f-new)
                        for neighbor in neighbors:
                            neighbor = tuple(i + j for i, j in zip((e, f), neighbor))
                            e_new, f_new = neighbor
                            # if ( ( e-new not aligned and f-new not aligned)
                            # and (e-new, f-new in union(e2f, f2e) )
                            if (
                                e_new not in aligned and f_new not in aligned
                            ) and neighbor in union:
                                alignment.add(neighbor)
                                aligned["e"].add(e_new)
                                aligned["f"].add(f_new)
                                prev_len += 1
                                no_new_points = False
            # iterate until no new points added
            if no_new_points:
                break

    def final_and(a):
        """
        Adds remaining points that are not in the intersection, not in the
        neighboring alignments but in the original *e2f* and *f2e* alignments
        """
        # for english word e = 0 ... en
        for e_new in range(srclen):
            # for foreign word f = 0 ... fn
            for f_new in range(trglen):
                # if ( ( e-new not aligned and f-new not aligned)
                # and (e-new, f-new in union(e2f, f2e) )
                if (
                    e_new not in aligned
                    and f_new not in aligned
                    and (e_new, f_new) in union
                ):
                    alignment.add((e_new, f_new))
                    aligned["e"].add(e_new)
                    aligned["f"].add(f_new)

    grow_diag()
    final_and(e2f)
    final_and(f2e)
    return sorted(alignment)
