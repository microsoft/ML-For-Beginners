"""Pen calculating area, center of mass, variance and standard-deviation,
covariance and correlation, and slant, of glyph shapes."""
import math
from fontTools.pens.momentsPen import MomentsPen

__all__ = ["StatisticsPen"]


class StatisticsPen(MomentsPen):

    """Pen calculating area, center of mass, variance and
    standard-deviation, covariance and correlation, and slant,
    of glyph shapes.

    Note that all the calculated values are 'signed'. Ie. if the
    glyph shape is self-intersecting, the values are not correct
    (but well-defined). As such, area will be negative if contour
    directions are clockwise.  Moreover, variance might be negative
    if the shapes are self-intersecting in certain ways."""

    def __init__(self, glyphset=None):
        MomentsPen.__init__(self, glyphset=glyphset)
        self.__zero()

    def _closePath(self):
        MomentsPen._closePath(self)
        self.__update()

    def __zero(self):
        self.meanX = 0
        self.meanY = 0
        self.varianceX = 0
        self.varianceY = 0
        self.stddevX = 0
        self.stddevY = 0
        self.covariance = 0
        self.correlation = 0
        self.slant = 0

    def __update(self):
        area = self.area
        if not area:
            self.__zero()
            return

        # Center of mass
        # https://en.wikipedia.org/wiki/Center_of_mass#A_continuous_volume
        self.meanX = meanX = self.momentX / area
        self.meanY = meanY = self.momentY / area

        #  Var(X) = E[X^2] - E[X]^2
        self.varianceX = varianceX = self.momentXX / area - meanX**2
        self.varianceY = varianceY = self.momentYY / area - meanY**2

        self.stddevX = stddevX = math.copysign(abs(varianceX) ** 0.5, varianceX)
        self.stddevY = stddevY = math.copysign(abs(varianceY) ** 0.5, varianceY)

        #  Covariance(X,Y) = ( E[X.Y] - E[X]E[Y] )
        self.covariance = covariance = self.momentXY / area - meanX * meanY

        #  Correlation(X,Y) = Covariance(X,Y) / ( stddev(X) * stddev(Y) )
        # https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
        if stddevX * stddevY == 0:
            correlation = float("NaN")
        else:
            correlation = covariance / (stddevX * stddevY)
        self.correlation = correlation if abs(correlation) > 1e-3 else 0

        slant = covariance / varianceY if varianceY != 0 else float("NaN")
        self.slant = slant if abs(slant) > 1e-3 else 0


def _test(glyphset, upem, glyphs, quiet=False):
    from fontTools.pens.transformPen import TransformPen
    from fontTools.misc.transform import Scale

    wght_sum = 0
    wght_sum_perceptual = 0
    wdth_sum = 0
    slnt_sum = 0
    slnt_sum_perceptual = 0
    for glyph_name in glyphs:
        glyph = glyphset[glyph_name]
        pen = StatisticsPen(glyphset=glyphset)
        transformer = TransformPen(pen, Scale(1.0 / upem))
        glyph.draw(transformer)

        wght_sum += abs(pen.area)
        wght_sum_perceptual += abs(pen.area) * glyph.width
        wdth_sum += glyph.width
        slnt_sum += pen.slant
        slnt_sum_perceptual += pen.slant * glyph.width

        if quiet:
            continue

        print()
        print("glyph:", glyph_name)

        for item in [
            "area",
            "momentX",
            "momentY",
            "momentXX",
            "momentYY",
            "momentXY",
            "meanX",
            "meanY",
            "varianceX",
            "varianceY",
            "stddevX",
            "stddevY",
            "covariance",
            "correlation",
            "slant",
        ]:
            print("%s: %g" % (item, getattr(pen, item)))

    if not quiet:
        print()
        print("font:")

    print("weight: %g" % (wght_sum * upem / wdth_sum))
    print("weight (perceptual): %g" % (wght_sum_perceptual / wdth_sum))
    print("width:  %g" % (wdth_sum / upem / len(glyphs)))
    slant = slnt_sum / len(glyphs)
    print("slant:  %g" % slant)
    print("slant angle:  %g" % -math.degrees(math.atan(slant)))
    slant_perceptual = slnt_sum_perceptual / wdth_sum
    print("slant (perceptual):  %g" % slant_perceptual)
    print("slant (perceptual) angle:  %g" % -math.degrees(math.atan(slant_perceptual)))


def main(args):
    """Report font glyph shape geometricsl statistics"""

    if args is None:
        import sys

        args = sys.argv[1:]

    import argparse

    parser = argparse.ArgumentParser(
        "fonttools pens.statisticsPen",
        description="Report font glyph shape geometricsl statistics",
    )
    parser.add_argument("font", metavar="font.ttf", help="Font file.")
    parser.add_argument("glyphs", metavar="glyph-name", help="Glyph names.", nargs="*")
    parser.add_argument(
        "-y",
        metavar="<number>",
        help="Face index into a collection to open. Zero based.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only report font-wide statistics."
    )
    parser.add_argument(
        "--variations",
        metavar="AXIS=LOC",
        default="",
        help="List of space separated locations. A location consist in "
        "the name of a variation axis, followed by '=' and a number. E.g.: "
        "wght=700 wdth=80. The default is the location of the base master.",
    )

    options = parser.parse_args(args)

    glyphs = options.glyphs
    fontNumber = int(options.y) if options.y is not None else 0

    location = {}
    for tag_v in options.variations.split():
        fields = tag_v.split("=")
        tag = fields[0].strip()
        v = int(fields[1])
        location[tag] = v

    from fontTools.ttLib import TTFont

    font = TTFont(options.font, fontNumber=fontNumber)
    if not glyphs:
        glyphs = font.getGlyphOrder()
    _test(
        font.getGlyphSet(location=location),
        font["head"].unitsPerEm,
        glyphs,
        quiet=options.quiet,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
