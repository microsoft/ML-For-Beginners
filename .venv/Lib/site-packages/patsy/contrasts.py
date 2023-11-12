# This file is part of Patsy
# Copyright (C) 2011-2012 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm
# http://www.ats.ucla.edu/stat/sas/webbooks/reg/chapter5/sasreg5.htm

from __future__ import print_function

# These are made available in the patsy.* namespace
__all__ = ["ContrastMatrix", "Treatment", "Poly", "Sum", "Helmert", "Diff"]

import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
                        safe_issubdtype,
                        no_pickling, assert_no_pickling)

class ContrastMatrix(object):
    """A simple container for a matrix used for coding categorical factors.

    Attributes:

    .. attribute:: matrix

       A 2d ndarray, where each column corresponds to one column of the
       resulting design matrix, and each row contains the entries for a single
       categorical variable level. Usually n-by-n for a full rank coding or
       n-by-(n-1) for a reduced rank coding, though other options are
       possible.

    .. attribute:: column_suffixes

       A list of strings to be appended to the factor name, to produce the
       final column names. E.g. for treatment coding the entries will look
       like ``"[T.level1]"``.
    """
    def __init__(self, matrix, column_suffixes):
        self.matrix = np.asarray(matrix)
        self.column_suffixes = column_suffixes
        if self.matrix.shape[1] != len(column_suffixes):
            raise PatsyError("matrix and column_suffixes don't conform")

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        repr_pretty_impl(p, self, [self.matrix, self.column_suffixes])

    __getstate__ = no_pickling

def test_ContrastMatrix():
    cm = ContrastMatrix([[1, 0], [0, 1]], ["a", "b"])
    assert np.array_equal(cm.matrix, np.eye(2))
    assert cm.column_suffixes == ["a", "b"]
    # smoke test
    repr(cm)

    import pytest
    pytest.raises(PatsyError, ContrastMatrix, [[1], [0]], ["a", "b"])

    assert_no_pickling(cm)

# This always produces an object of the type that Python calls 'str' (whether
# that be a Python 2 string-of-bytes or a Python 3 string-of-unicode). It does
# *not* make any particular guarantees about being reversible or having other
# such useful programmatic properties -- it just produces something that will
# be nice for users to look at.
def _obj_to_readable_str(obj):
    if isinstance(obj, str):
        return obj
    elif sys.version_info >= (3,) and isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return repr(obj)
    elif sys.version_info < (3,) and isinstance(obj, unicode):
        try:
            return obj.encode("ascii")
        except UnicodeEncodeError:
            return repr(obj)
    else:
        return repr(obj)

def test__obj_to_readable_str():
    def t(obj, expected):
        got = _obj_to_readable_str(obj)
        assert type(got) is str
        assert got == expected
    t(1, "1")
    t(1.0, "1.0")
    t("asdf", "asdf")
    t(six.u("asdf"), "asdf")
    if sys.version_info >= (3,):
        # we can use "foo".encode here b/c this is python 3!
        # a utf-8 encoded euro-sign comes out as a real euro sign.
        t("\u20ac".encode("utf-8"), six.u("\u20ac"))
        # but a iso-8859-15 euro sign can't be decoded, and we fall back on
        # repr()
        t("\u20ac".encode("iso-8859-15"), "b'\\xa4'")
    else:
        t(six.u("\u20ac"), "u'\\u20ac'")

def _name_levels(prefix, levels):
    return ["[%s%s]" % (prefix, _obj_to_readable_str(level)) for level in levels]

def test__name_levels():
    assert _name_levels("a", ["b", "c"]) == ["[ab]", "[ac]"]

def _dummy_code(levels):
    return ContrastMatrix(np.eye(len(levels)), _name_levels("", levels))

def _get_level(levels, level_ref):
    if level_ref in levels:
        return levels.index(level_ref)
    if isinstance(level_ref, six.integer_types):
        if level_ref < 0:
            level_ref += len(levels)
        if not (0 <= level_ref < len(levels)):
            raise PatsyError("specified level %r is out of range"
                                % (level_ref,))
        return level_ref
    raise PatsyError("specified level %r not found" % (level_ref,))

def test__get_level():
    assert _get_level(["a", "b", "c"], 0) == 0
    assert _get_level(["a", "b", "c"], -1) == 2
    assert _get_level(["a", "b", "c"], "b") == 1
    # For integer levels, we check identity before treating it as an index
    assert _get_level([2, 1, 0], 0) == 2
    import pytest
    pytest.raises(PatsyError, _get_level, ["a", "b"], 2)
    pytest.raises(PatsyError, _get_level, ["a", "b"], -3)
    pytest.raises(PatsyError, _get_level, ["a", "b"], "c")

    if not six.PY3:
        assert _get_level(["a", "b", "c"], long(0)) == 0
        assert _get_level(["a", "b", "c"], long(-1)) == 2
        assert _get_level([2, 1, 0], long(0)) == 2


class Treatment(object):
    """Treatment coding (also known as dummy coding).

    This is the default coding.

    For reduced-rank coding, one level is chosen as the "reference", and its
    mean behaviour is represented by the intercept. Each column of the
    resulting matrix represents the difference between the mean of one level
    and this reference level.

    For full-rank coding, classic "dummy" coding is used, and each column of
    the resulting matrix represents the mean of the corresponding level.

    The reference level defaults to the first level, or can be specified
    explicitly.

    .. ipython:: python

       # reduced rank
       dmatrix("C(a, Treatment)", balanced(a=3))
       # full rank
       dmatrix("0 + C(a, Treatment)", balanced(a=3))
       # Setting a reference level
       dmatrix("C(a, Treatment(1))", balanced(a=3))
       dmatrix("C(a, Treatment('a2'))", balanced(a=3))

    Equivalent to R ``contr.treatment``. The R documentation suggests that
    using ``Treatment(reference=-1)`` will produce contrasts that are
    "equivalent to those produced by many (but not all) SAS procedures".
    """
    def __init__(self, reference=None):
        self.reference = reference

    def code_with_intercept(self, levels):
        return _dummy_code(levels)

    def code_without_intercept(self, levels):
        if self.reference is None:
            reference = 0
        else:
            reference = _get_level(levels, self.reference)
        eye = np.eye(len(levels) - 1)
        contrasts = np.vstack((eye[:reference, :],
                                np.zeros((1, len(levels) - 1)),
                                eye[reference:, :]))
        names = _name_levels("T.", levels[:reference] + levels[reference + 1:])
        return ContrastMatrix(contrasts, names)

    __getstate__ = no_pickling

def test_Treatment():
    t1 = Treatment()
    matrix = t1.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[a]", "[b]", "[c]"]
    assert np.allclose(matrix.matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    matrix = t1.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[T.b]", "[T.c]"]
    assert np.allclose(matrix.matrix, [[0, 0], [1, 0], [0, 1]])
    matrix = Treatment(reference=1).code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[T.a]", "[T.c]"]
    assert np.allclose(matrix.matrix, [[1, 0], [0, 0], [0, 1]])
    matrix = Treatment(reference=-2).code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[T.a]", "[T.c]"]
    assert np.allclose(matrix.matrix, [[1, 0], [0, 0], [0, 1]])
    matrix = Treatment(reference="b").code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[T.a]", "[T.c]"]
    assert np.allclose(matrix.matrix, [[1, 0], [0, 0], [0, 1]])
    # Make sure the default is always the first level, even if there is a
    # different level called 0.
    matrix = Treatment().code_without_intercept([2, 1, 0])
    assert matrix.column_suffixes == ["[T.1]", "[T.0]"]
    assert np.allclose(matrix.matrix, [[0, 0], [1, 0], [0, 1]])

class Poly(object):
    """Orthogonal polynomial contrast coding.

    This coding scheme treats the levels as ordered samples from an underlying
    continuous scale, whose effect takes an unknown functional form which is
    `Taylor-decomposed`__ into the sum of a linear, quadratic, etc. components.

    .. __: https://en.wikipedia.org/wiki/Taylor_series

    For reduced-rank coding, you get a linear column, a quadratic column,
    etc., up to the number of levels provided.

    For full-rank coding, the same scheme is used, except that the zero-order
    constant polynomial is also included. I.e., you get an intercept column
    included as part of your categorical term.

    By default the levels are treated as equally spaced, but you can override
    this by providing a value for the `scores` argument.

    Examples:

    .. ipython:: python

       # Reduced rank
       dmatrix("C(a, Poly)", balanced(a=4))
       # Full rank
       dmatrix("0 + C(a, Poly)", balanced(a=3))
       # Explicit scores
       dmatrix("C(a, Poly([1, 2, 10]))", balanced(a=3))

    This is equivalent to R's ``contr.poly``. (But note that in R, reduced
    rank encodings are always dummy-coded, regardless of what contrast you
    have set.)
    """
    def __init__(self, scores=None):
        self.scores = scores

    def _code_either(self, intercept, levels):
        n = len(levels)
        scores = self.scores
        if scores is None:
            scores = np.arange(n)
        scores = np.asarray(scores, dtype=float)
        if len(scores) != n:
            raise PatsyError("number of levels (%s) does not match"
                                " number of scores (%s)"
                                % (n, len(scores)))
        # Strategy: just make a matrix whose columns are naive linear,
        # quadratic, etc., functions of the raw scores, and then use 'qr' to
        # orthogonalize each column against those to its left.
        scores -= scores.mean()
        raw_poly = scores.reshape((-1, 1)) ** np.arange(n).reshape((1, -1))
        q, r = np.linalg.qr(raw_poly)
        q *= np.sign(np.diag(r))
        q /= np.sqrt(np.sum(q ** 2, axis=1))
        # The constant term is always all 1's -- we don't normalize it.
        q[:, 0] = 1
        names = [".Constant", ".Linear", ".Quadratic", ".Cubic"]
        names += ["^%s" % (i,) for i in range(4, n)]
        names = names[:n]
        if intercept:
            return ContrastMatrix(q, names)
        else:
            # We always include the constant/intercept column as something to
            # orthogonalize against, but we don't always return it:
            return ContrastMatrix(q[:, 1:], names[1:])

    def code_with_intercept(self, levels):
        return self._code_either(True, levels)

    def code_without_intercept(self, levels):
        return self._code_either(False, levels)

    __getstate__ = no_pickling

def test_Poly():
    t1 = Poly()
    matrix = t1.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == [".Constant", ".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3)'
    expected = [[1, -7.07106781186548e-01, 0.408248290463863],
                [1, 0, -0.816496580927726],
                [1, 7.07106781186547e-01, 0.408248290463863]]
    print(matrix.matrix)
    assert np.allclose(matrix.matrix, expected)
    matrix = t1.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == [".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3)'
    print(matrix.matrix)
    assert np.allclose(matrix.matrix,
                       [[-7.07106781186548e-01, 0.408248290463863],
                        [0, -0.816496580927726],
                        [7.07106781186547e-01, 0.408248290463863]])

    matrix = Poly(scores=[0, 10, 11]).code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == [".Constant", ".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3, scores=c(0, 10, 11))'
    print(matrix.matrix)
    assert np.allclose(matrix.matrix,
                       [[1, -0.813733471206735, 0.0671156055214024],
                        [1, 0.348742916231458, -0.7382716607354268],
                        [1, 0.464990554975277, 0.6711560552140243]])

    # we had an integer/float handling bug for score vectors whose mean was
    # non-integer, so check one of those:
    matrix = Poly(scores=[0, 10, 12]).code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == [".Constant", ".Linear", ".Quadratic"]
    # Values from R 'options(digits=15); contr.poly(3, scores=c(0, 10, 12))'
    print(matrix.matrix)
    assert np.allclose(matrix.matrix,
                       [[1, -0.806559132617443, 0.127000127000191],
                        [1, 0.293294230042706, -0.762000762001143],
                        [1, 0.513264902574736, 0.635000635000952]])

    import pytest
    pytest.raises(PatsyError,
                  Poly(scores=[0, 1]).code_with_intercept,
                  ["a", "b", "c"])

    matrix = t1.code_with_intercept(list(range(6)))
    assert matrix.column_suffixes == [".Constant", ".Linear", ".Quadratic",
                                      ".Cubic", "^4", "^5"]


class Sum(object):
    """Deviation coding (also known as sum-to-zero coding).

    Compares the mean of each level to the mean-of-means. (In a balanced
    design, compares the mean of each level to the overall mean.)

    For full-rank coding, a standard intercept term is added.

    One level must be omitted to avoid redundancy; by default this is the last
    level, but this can be adjusted via the `omit` argument.

    .. warning:: There are multiple definitions of 'deviation coding' in
       use. Make sure this is the one you expect before trying to interpret
       your results!

    Examples:

    .. ipython:: python

       # Reduced rank
       dmatrix("C(a, Sum)", balanced(a=4))
       # Full rank
       dmatrix("0 + C(a, Sum)", balanced(a=4))
       # Omit a different level
       dmatrix("C(a, Sum(1))", balanced(a=3))
       dmatrix("C(a, Sum('a1'))", balanced(a=3))

    This is equivalent to R's `contr.sum`.
    """
    def __init__(self, omit=None):
        self.omit = omit

    def _omit_i(self, levels):
        if self.omit is None:
            # We assume below that this is positive
            return len(levels) - 1
        else:
            return _get_level(levels, self.omit)

    def _sum_contrast(self, levels):
        n = len(levels)
        omit_i = self._omit_i(levels)
        eye = np.eye(n - 1)
        out = np.empty((n, n - 1))
        out[:omit_i, :] = eye[:omit_i, :]
        out[omit_i, :] = -1
        out[omit_i + 1:, :] = eye[omit_i:, :]
        return out

    def code_with_intercept(self, levels):
        contrast = self.code_without_intercept(levels)
        matrix = np.column_stack((np.ones(len(levels)),
                                  contrast.matrix))
        column_suffixes = ["[mean]"] + contrast.column_suffixes
        return ContrastMatrix(matrix, column_suffixes)

    def code_without_intercept(self, levels):
        matrix = self._sum_contrast(levels)
        omit_i = self._omit_i(levels)
        included_levels = levels[:omit_i] + levels[omit_i + 1:]
        return ContrastMatrix(matrix, _name_levels("S.", included_levels))

    __getstate__ = no_pickling

def test_Sum():
    t1 = Sum()
    matrix = t1.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[mean]", "[S.a]", "[S.b]"]
    assert np.allclose(matrix.matrix, [[1, 1, 0], [1, 0, 1], [1, -1, -1]])
    matrix = t1.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[S.a]", "[S.b]"]
    assert np.allclose(matrix.matrix, [[1, 0], [0, 1], [-1, -1]])
    # Check that it's not thrown off by negative integer term names
    matrix = t1.code_without_intercept([-1, -2, -3])
    assert matrix.column_suffixes == ["[S.-1]", "[S.-2]"]
    assert np.allclose(matrix.matrix, [[1, 0], [0, 1], [-1, -1]])
    t2 = Sum(omit=1)
    matrix = t2.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[mean]", "[S.a]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[1, 1, 0], [1, -1, -1], [1, 0, 1]])
    matrix = t2.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[S.a]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[1, 0], [-1, -1], [0, 1]])
    matrix = t2.code_without_intercept([1, 0, 2])
    assert matrix.column_suffixes == ["[S.0]", "[S.2]"]
    assert np.allclose(matrix.matrix, [[-1, -1], [1, 0], [0, 1]])
    t3 = Sum(omit=-3)
    matrix = t3.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[mean]", "[S.b]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    matrix = t3.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[S.b]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[-1, -1], [1, 0], [0, 1]])
    t4 = Sum(omit="a")
    matrix = t3.code_with_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[mean]", "[S.b]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[1, -1, -1], [1, 1, 0], [1, 0, 1]])
    matrix = t3.code_without_intercept(["a", "b", "c"])
    assert matrix.column_suffixes == ["[S.b]", "[S.c]"]
    assert np.allclose(matrix.matrix, [[-1, -1], [1, 0], [0, 1]])

class Helmert(object):
    """Helmert contrasts.

    Compares the second level with the first, the third with the average of
    the first two, and so on.

    For full-rank coding, a standard intercept term is added.

    .. warning:: There are multiple definitions of 'Helmert coding' in
       use. Make sure this is the one you expect before trying to interpret
       your results!

    Examples:

    .. ipython:: python

       # Reduced rank
       dmatrix("C(a, Helmert)", balanced(a=4))
       # Full rank
       dmatrix("0 + C(a, Helmert)", balanced(a=4))

    This is equivalent to R's `contr.helmert`.
    """
    def _helmert_contrast(self, levels):
        n = len(levels)
        #http://www.ats.ucla.edu/stat/sas/webbooks/reg/chapter5/sasreg5.htm#HELMERT
        #contr = np.eye(n - 1)
        #int_range = np.arange(n - 1., 1, -1)
        #denom = np.repeat(int_range, np.arange(n - 2, 0, -1))
        #contr[np.tril_indices(n - 1, -1)] = -1. / denom

        #http://www.ats.ucla.edu/stat/r/library/contrast_coding.htm#HELMERT
        #contr = np.zeros((n - 1., n - 1))
        #int_range = np.arange(n, 1, -1)
        #denom = np.repeat(int_range[:-1], np.arange(n - 2, 0, -1))
        #contr[np.diag_indices(n - 1)] = (int_range - 1.) / int_range
        #contr[np.tril_indices(n - 1, -1)] = -1. / denom
        #contr = np.vstack((contr, -1./int_range))

        #r-like
        contr = np.zeros((n, n - 1))
        contr[1:][np.diag_indices(n - 1)] = np.arange(1, n)
        contr[np.triu_indices(n - 1)] = -1
        return contr

    def code_with_intercept(self, levels):
        contrast = np.column_stack((np.ones(len(levels)),
                                    self._helmert_contrast(levels)))
        column_suffixes = _name_levels("H.", ["intercept"] + list(levels[1:]))
        return ContrastMatrix(contrast, column_suffixes)

    def code_without_intercept(self, levels):
        contrast = self._helmert_contrast(levels)
        return ContrastMatrix(contrast,
                              _name_levels("H.", levels[1:]))

    __getstate__ = no_pickling

def test_Helmert():
    t1 = Helmert()
    for levels in (["a", "b", "c", "d"], ("a", "b", "c", "d")):
        matrix = t1.code_with_intercept(levels)
        assert matrix.column_suffixes == ["[H.intercept]",
                                          "[H.b]",
                                          "[H.c]",
                                          "[H.d]"]
        assert np.allclose(matrix.matrix, [[1, -1, -1, -1],
                                           [1, 1, -1, -1],
                                           [1, 0, 2, -1],
                                           [1, 0, 0, 3]])
        matrix = t1.code_without_intercept(levels)
        assert matrix.column_suffixes == ["[H.b]", "[H.c]", "[H.d]"]
        assert np.allclose(matrix.matrix, [[-1, -1, -1],
                                           [1, -1, -1],
                                           [0, 2, -1],
                                           [0, 0, 3]])

class Diff(object):
    """Backward difference coding.

    This coding scheme is useful for ordered factors, and compares the mean of
    each level with the preceding level. So you get the second level minus the
    first, the third level minus the second, etc.

    For full-rank coding, a standard intercept term is added (which gives the
    mean value for the first level).

    Examples:

    .. ipython:: python

       # Reduced rank
       dmatrix("C(a, Diff)", balanced(a=3))
       # Full rank
       dmatrix("0 + C(a, Diff)", balanced(a=3))
    """
    def _diff_contrast(self, levels):
        nlevels = len(levels)
        contr = np.zeros((nlevels, nlevels-1))
        int_range = np.arange(1, nlevels)
        upper_int = np.repeat(int_range, int_range)
        row_i, col_i = np.triu_indices(nlevels-1)
        # we want to iterate down the columns not across the rows
        # it would be nice if the index functions had a row/col order arg
        col_order = np.argsort(col_i)
        contr[row_i[col_order],
              col_i[col_order]] = (upper_int-nlevels)/float(nlevels)
        lower_int = np.repeat(int_range, int_range[::-1])
        row_i, col_i = np.tril_indices(nlevels-1)
        # we want to iterate down the columns not across the rows
        col_order = np.argsort(col_i)
        contr[row_i[col_order]+1, col_i[col_order]] = lower_int/float(nlevels)
        return contr

    def code_with_intercept(self, levels):
        contrast = np.column_stack((np.ones(len(levels)),
                                    self._diff_contrast(levels)))
        return ContrastMatrix(contrast, _name_levels("D.", levels))

    def code_without_intercept(self, levels):
        contrast = self._diff_contrast(levels)
        return ContrastMatrix(contrast, _name_levels("D.", levels[:-1]))

    __getstate__ = no_pickling

def test_diff():
    t1 = Diff()
    matrix = t1.code_with_intercept(["a", "b", "c", "d"])
    assert matrix.column_suffixes == ["[D.a]", "[D.b]", "[D.c]",
                                      "[D.d]"]
    assert np.allclose(matrix.matrix, [[1, -3/4., -1/2., -1/4.],
                                        [1, 1/4., -1/2., -1/4.],
                                        [1, 1/4., 1./2, -1/4.],
                                        [1, 1/4., 1/2., 3/4.]])
    matrix = t1.code_without_intercept(["a", "b", "c", "d"])
    assert matrix.column_suffixes == ["[D.a]", "[D.b]", "[D.c]"]
    assert np.allclose(matrix.matrix, [[-3/4., -1/2., -1/4.],
                                        [1/4., -1/2., -1/4.],
                                        [1/4., 2./4, -1/4.],
                                        [1/4., 1/2., 3/4.]])

# contrast can be:
#   -- a ContrastMatrix
#   -- a simple np.ndarray
#   -- an object with code_with_intercept and code_without_intercept methods
#   -- a function returning one of the above
#   -- None, in which case the above rules are applied to 'default'
# This function always returns a ContrastMatrix.
def code_contrast_matrix(intercept, levels, contrast, default=None):
    if contrast is None:
        contrast = default
    if callable(contrast):
        contrast = contrast()
    if isinstance(contrast, ContrastMatrix):
        return contrast
    as_array = np.asarray(contrast)
    if safe_issubdtype(as_array.dtype, np.number):
        return ContrastMatrix(as_array,
                              _name_levels("custom", range(as_array.shape[1])))
    if intercept:
        return contrast.code_with_intercept(levels)
    else:
        return contrast.code_without_intercept(levels)

