"""
Tool to find wrong contour order between different masters, and
other interpolatability (or lack thereof) issues.

Call as:
$ fonttools varLib.interpolatable font1 font2 ...
"""

from fontTools.pens.basePen import AbstractPen, BasePen
from fontTools.pens.pointPen import SegmentToPointPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.statisticsPen import StatisticsPen
from fontTools.pens.momentsPen import OpenContourError
from collections import OrderedDict
import math
import itertools
import sys


def _rot_list(l, k):
    """Rotate list by k items forward.  Ie. item at position 0 will be
    at position k in returned list.  Negative k is allowed."""
    n = len(l)
    k %= n
    if not k:
        return l
    return l[n - k :] + l[: n - k]


class PerContourPen(BasePen):
    def __init__(self, Pen, glyphset=None):
        BasePen.__init__(self, glyphset)
        self._glyphset = glyphset
        self._Pen = Pen
        self._pen = None
        self.value = []

    def _moveTo(self, p0):
        self._newItem()
        self._pen.moveTo(p0)

    def _lineTo(self, p1):
        self._pen.lineTo(p1)

    def _qCurveToOne(self, p1, p2):
        self._pen.qCurveTo(p1, p2)

    def _curveToOne(self, p1, p2, p3):
        self._pen.curveTo(p1, p2, p3)

    def _closePath(self):
        self._pen.closePath()
        self._pen = None

    def _endPath(self):
        self._pen.endPath()
        self._pen = None

    def _newItem(self):
        self._pen = pen = self._Pen()
        self.value.append(pen)


class PerContourOrComponentPen(PerContourPen):
    def addComponent(self, glyphName, transformation):
        self._newItem()
        self.value[-1].addComponent(glyphName, transformation)


class RecordingPointPen(BasePen):
    def __init__(self):
        self.value = []

    def beginPath(self, identifier=None, **kwargs):
        pass

    def endPath(self) -> None:
        pass

    def addPoint(self, pt, segmentType=None):
        self.value.append((pt, False if segmentType is None else True))


def _vdiff(v0, v1):
    return tuple(b - a for a, b in zip(v0, v1))


def _vlen(vec):
    v = 0
    for x in vec:
        v += x * x
    return v


def _complex_vlen(vec):
    v = 0
    for x in vec:
        v += abs(x) * abs(x)
    return v


def _matching_cost(G, matching):
    return sum(G[i][j] for i, j in enumerate(matching))


def min_cost_perfect_bipartite_matching(G):
    n = len(G)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(G)
        assert (rows == list(range(n))).all()
        return list(cols), _matching_cost(G, cols)
    except ImportError:
        pass

    try:
        from munkres import Munkres

        cols = [None] * n
        for row, col in Munkres().compute(G):
            cols[row] = col
        return cols, _matching_cost(G, cols)
    except ImportError:
        pass

    if n > 6:
        raise Exception("Install Python module 'munkres' or 'scipy >= 0.17.0'")

    # Otherwise just brute-force
    permutations = itertools.permutations(range(n))
    best = list(next(permutations))
    best_cost = _matching_cost(G, best)
    for p in permutations:
        cost = _matching_cost(G, p)
        if cost < best_cost:
            best, best_cost = list(p), cost
    return best, best_cost


def test(glyphsets, glyphs=None, names=None, ignore_missing=False):
    if names is None:
        names = glyphsets
    if glyphs is None:
        # `glyphs = glyphsets[0].keys()` is faster, certainly, but doesn't allow for sparse TTFs/OTFs given out of order
        # ... risks the sparse master being the first one, and only processing a subset of the glyphs
        glyphs = {g for glyphset in glyphsets for g in glyphset.keys()}

    hist = []
    problems = OrderedDict()

    def add_problem(glyphname, problem):
        problems.setdefault(glyphname, []).append(problem)

    for glyph_name in glyphs:
        try:
            m0idx = 0
            allVectors = []
            allNodeTypes = []
            allContourIsomorphisms = []
            for glyphset, name in zip(glyphsets, names):
                glyph = glyphset[glyph_name]

                if glyph is None:
                    if not ignore_missing:
                        add_problem(glyph_name, {"type": "missing", "master": name})
                    allNodeTypes.append(None)
                    allVectors.append(None)
                    allContourIsomorphisms.append(None)
                    continue

                perContourPen = PerContourOrComponentPen(
                    RecordingPen, glyphset=glyphset
                )
                try:
                    glyph.draw(perContourPen, outputImpliedClosingLine=True)
                except TypeError:
                    glyph.draw(perContourPen)
                contourPens = perContourPen.value
                del perContourPen

                contourVectors = []
                contourIsomorphisms = []
                nodeTypes = []
                allNodeTypes.append(nodeTypes)
                allVectors.append(contourVectors)
                allContourIsomorphisms.append(contourIsomorphisms)
                for ix, contour in enumerate(contourPens):
                    nodeVecs = tuple(instruction[0] for instruction in contour.value)
                    nodeTypes.append(nodeVecs)

                    stats = StatisticsPen(glyphset=glyphset)
                    try:
                        contour.replay(stats)
                    except OpenContourError as e:
                        add_problem(
                            glyph_name,
                            {"master": name, "contour": ix, "type": "open_path"},
                        )
                        continue
                    size = math.sqrt(abs(stats.area)) * 0.5
                    vector = (
                        int(size),
                        int(stats.meanX),
                        int(stats.meanY),
                        int(stats.stddevX * 2),
                        int(stats.stddevY * 2),
                        int(stats.correlation * size),
                    )
                    contourVectors.append(vector)
                    # print(vector)

                    # Check starting point
                    if nodeVecs[0] == "addComponent":
                        continue
                    assert nodeVecs[0] == "moveTo"
                    assert nodeVecs[-1] in ("closePath", "endPath")
                    points = RecordingPointPen()
                    converter = SegmentToPointPen(points, False)
                    contour.replay(converter)
                    # points.value is a list of pt,bool where bool is true if on-curve and false if off-curve;
                    # now check all rotations and mirror-rotations of the contour and build list of isomorphic
                    # possible starting points.
                    bits = 0
                    for pt, b in points.value:
                        bits = (bits << 1) | b
                    n = len(points.value)
                    mask = (1 << n) - 1
                    isomorphisms = []
                    contourIsomorphisms.append(isomorphisms)
                    for i in range(n):
                        b = ((bits << i) & mask) | ((bits >> (n - i)))
                        if b == bits:
                            isomorphisms.append(
                                _rot_list([complex(*pt) for pt, bl in points.value], i)
                            )
                    # Add mirrored rotations
                    mirrored = list(reversed(points.value))
                    reversed_bits = 0
                    for pt, b in mirrored:
                        reversed_bits = (reversed_bits << 1) | b
                    for i in range(n):
                        b = ((reversed_bits << i) & mask) | ((reversed_bits >> (n - i)))
                        if b == bits:
                            isomorphisms.append(
                                _rot_list([complex(*pt) for pt, bl in mirrored], i)
                            )

            # m0idx should be the index of the first non-None item in allNodeTypes,
            # else give it the first index of None, which is likely 0
            m0idx = allNodeTypes.index(
                next((x for x in allNodeTypes if x is not None), None)
            )
            # m0 is the first non-None item in allNodeTypes, or the first item if all are None
            m0 = allNodeTypes[m0idx]
            for i, m1 in enumerate(allNodeTypes[m0idx + 1 :]):
                if m1 is None:
                    continue
                if len(m0) != len(m1):
                    add_problem(
                        glyph_name,
                        {
                            "type": "path_count",
                            "master_1": names[m0idx],
                            "master_2": names[m0idx + i + 1],
                            "value_1": len(m0),
                            "value_2": len(m1),
                        },
                    )
                if m0 == m1:
                    continue
                for pathIx, (nodes1, nodes2) in enumerate(zip(m0, m1)):
                    if nodes1 == nodes2:
                        continue
                    if len(nodes1) != len(nodes2):
                        add_problem(
                            glyph_name,
                            {
                                "type": "node_count",
                                "path": pathIx,
                                "master_1": names[m0idx],
                                "master_2": names[m0idx + i + 1],
                                "value_1": len(nodes1),
                                "value_2": len(nodes2),
                            },
                        )
                        continue
                    for nodeIx, (n1, n2) in enumerate(zip(nodes1, nodes2)):
                        if n1 != n2:
                            add_problem(
                                glyph_name,
                                {
                                    "type": "node_incompatibility",
                                    "path": pathIx,
                                    "node": nodeIx,
                                    "master_1": names[m0idx],
                                    "master_2": names[m0idx + i + 1],
                                    "value_1": n1,
                                    "value_2": n2,
                                },
                            )
                            continue

            # m0idx should be the index of the first non-None item in allVectors,
            # else give it the first index of None, which is likely 0
            m0idx = allVectors.index(
                next((x for x in allVectors if x is not None), None)
            )
            # m0 is the first non-None item in allVectors, or the first item if all are None
            m0 = allVectors[m0idx]
            for i, m1 in enumerate(allVectors[m0idx + 1 :]):
                if m1 is None:
                    continue
                if len(m0) != len(m1):
                    # We already reported this
                    continue
                if not m0:
                    continue
                costs = [[_vlen(_vdiff(v0, v1)) for v1 in m1] for v0 in m0]
                matching, matching_cost = min_cost_perfect_bipartite_matching(costs)
                identity_matching = list(range(len(m0)))
                identity_cost = sum(costs[i][i] for i in range(len(m0)))
                if (
                    matching != identity_matching
                    and matching_cost < identity_cost * 0.95
                ):
                    add_problem(
                        glyph_name,
                        {
                            "type": "contour_order",
                            "master_1": names[m0idx],
                            "master_2": names[m0idx + i + 1],
                            "value_1": list(range(len(m0))),
                            "value_2": matching,
                        },
                    )
                    break

            # m0idx should be the index of the first non-None item in allContourIsomorphisms,
            # else give it the first index of None, which is likely 0
            m0idx = allContourIsomorphisms.index(
                next((x for x in allContourIsomorphisms if x is not None), None)
            )
            # m0 is the first non-None item in allContourIsomorphisms, or the first item if all are None
            m0 = allContourIsomorphisms[m0idx]
            for i, m1 in enumerate(allContourIsomorphisms[m0idx + 1 :]):
                if m1 is None:
                    continue
                if len(m0) != len(m1):
                    # We already reported this
                    continue
                if not m0:
                    continue
                for ix, (contour0, contour1) in enumerate(zip(m0, m1)):
                    c0 = contour0[0]
                    costs = [
                        v for v in (_complex_vlen(_vdiff(c0, c1)) for c1 in contour1)
                    ]
                    min_cost = min(costs)
                    first_cost = costs[0]
                    if min_cost < first_cost * 0.95:
                        add_problem(
                            glyph_name,
                            {
                                "type": "wrong_start_point",
                                "contour": ix,
                                "master_1": names[m0idx],
                                "master_2": names[m0idx + i + 1],
                            },
                        )

        except ValueError as e:
            add_problem(
                glyph_name,
                {"type": "math_error", "master": name, "error": e},
            )
    return problems


def main(args=None):
    """Test for interpolatability issues between fonts"""
    import argparse

    parser = argparse.ArgumentParser(
        "fonttools varLib.interpolatable",
        description=main.__doc__,
    )
    parser.add_argument(
        "--glyphs",
        action="store",
        help="Space-separate name of glyphs to check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report in JSON format",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only exit with code 1 or 0, no output",
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Will not report glyphs missing from sparse masters as errors",
    )
    parser.add_argument(
        "inputs",
        metavar="FILE",
        type=str,
        nargs="+",
        help="Input a single DesignSpace/Glyphs file, or multiple TTF/UFO files",
    )

    args = parser.parse_args(args)

    glyphs = set(args.glyphs.split()) if args.glyphs else None

    from os.path import basename

    fonts = []
    names = []

    if len(args.inputs) == 1:
        if args.inputs[0].endswith(".designspace"):
            from fontTools.designspaceLib import DesignSpaceDocument

            designspace = DesignSpaceDocument.fromfile(args.inputs[0])
            args.inputs = [master.path for master in designspace.sources]

        elif args.inputs[0].endswith(".glyphs"):
            from glyphsLib import GSFont, to_ufos

            gsfont = GSFont(args.inputs[0])
            fonts.extend(to_ufos(gsfont))
            names = ["%s-%s" % (f.info.familyName, f.info.styleName) for f in fonts]
            args.inputs = []

        elif args.inputs[0].endswith(".ttf"):
            from fontTools.ttLib import TTFont

            font = TTFont(args.inputs[0])
            if "gvar" in font:
                # Is variable font
                gvar = font["gvar"]
                # Gather all "master" locations
                locs = set()
                for variations in gvar.variations.values():
                    for var in variations:
                        loc = []
                        for tag, val in sorted(var.axes.items()):
                            loc.append((tag, val[1]))
                        locs.add(tuple(loc))
                # Rebuild locs as dictionaries
                new_locs = [{}]
                names.append("()")
                for loc in sorted(locs, key=lambda v: (len(v), v)):
                    names.append(str(loc))
                    l = {}
                    for tag, val in loc:
                        l[tag] = val
                    new_locs.append(l)
                locs = new_locs
                del new_locs
                # locs is all master locations now

                for loc in locs:
                    fonts.append(font.getGlyphSet(location=loc, normalized=True))

                args.inputs = []

    for filename in args.inputs:
        if filename.endswith(".ufo"):
            from fontTools.ufoLib import UFOReader

            fonts.append(UFOReader(filename))
        else:
            from fontTools.ttLib import TTFont

            fonts.append(TTFont(filename))

        names.append(basename(filename).rsplit(".", 1)[0])

    glyphsets = []
    for font in fonts:
        if hasattr(font, "getGlyphSet"):
            glyphset = font.getGlyphSet()
        else:
            glyphset = font
        glyphsets.append({k: glyphset[k] for k in glyphset.keys()})

    if not glyphs:
        glyphs = set([gn for glyphset in glyphsets for gn in glyphset.keys()])

    for glyphset in glyphsets:
        glyphSetGlyphNames = set(glyphset.keys())
        diff = glyphs - glyphSetGlyphNames
        if diff:
            for gn in diff:
                glyphset[gn] = None

    problems = test(
        glyphsets, glyphs=glyphs, names=names, ignore_missing=args.ignore_missing
    )

    if not args.quiet:
        if args.json:
            import json

            print(json.dumps(problems))
        else:
            for glyph, glyph_problems in problems.items():
                print(f"Glyph {glyph} was not compatible: ")
                for p in glyph_problems:
                    if p["type"] == "missing":
                        print("    Glyph was missing in master %s" % p["master"])
                    if p["type"] == "open_path":
                        print("    Glyph has an open path in master %s" % p["master"])
                    if p["type"] == "path_count":
                        print(
                            "    Path count differs: %i in %s, %i in %s"
                            % (p["value_1"], p["master_1"], p["value_2"], p["master_2"])
                        )
                    if p["type"] == "node_count":
                        print(
                            "    Node count differs in path %i: %i in %s, %i in %s"
                            % (
                                p["path"],
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            )
                        )
                    if p["type"] == "node_incompatibility":
                        print(
                            "    Node %o incompatible in path %i: %s in %s, %s in %s"
                            % (
                                p["node"],
                                p["path"],
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            )
                        )
                    if p["type"] == "contour_order":
                        print(
                            "    Contour order differs: %s in %s, %s in %s"
                            % (
                                p["value_1"],
                                p["master_1"],
                                p["value_2"],
                                p["master_2"],
                            )
                        )
                    if p["type"] == "wrong_start_point":
                        print(
                            "    Contour %d start point differs: %s, %s"
                            % (
                                p["contour"],
                                p["master_1"],
                                p["master_2"],
                            )
                        )
                    if p["type"] == "math_error":
                        print(
                            "    Miscellaneous error in %s: %s"
                            % (
                                p["master"],
                                p["error"],
                            )
                        )
    if problems:
        return problems


if __name__ == "__main__":
    import sys

    problems = main()
    sys.exit(int(bool(problems)))
