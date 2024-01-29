# This file is part of Patsy
# Copyright (C) 2011-2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# This file defines the main class for storing metadata about a model
# design. It also defines a 'value-added' design matrix type -- a subclass of
# ndarray that represents a design matrix and holds metadata about its
# columns.  The intent is that these are useful and usable data structures
# even if you're not using *any* of the rest of patsy to actually build
# your matrices.


# XX TMP TODO:
#
# - update design_matrix_builders and build_design_matrices docs
# - add tests and docs for new design info stuff
# - consider renaming design_matrix_builders (and I guess
#   build_design_matrices too). Ditto for highlevel dbuilder functions.


from __future__ import print_function

# These are made available in the patsy.* namespace
__all__ = ["DesignInfo", "FactorInfo", "SubtermInfo", "DesignMatrix"]

import warnings
import numbers
import six
import numpy as np
from patsy import PatsyError
from patsy.util import atleast_2d_column_default
from patsy.compat import OrderedDict
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
                        safe_issubdtype,
                        no_pickling, assert_no_pickling)
from patsy.constraint import linear_constraint
from patsy.contrasts import ContrastMatrix
from patsy.desc import ModelDesc, Term

class FactorInfo(object):
    """A FactorInfo object is a simple class that provides some metadata about
    the role of a factor within a model. :attr:`DesignInfo.factor_infos` is
    a dictionary which maps factor objects to FactorInfo objects for each
    factor in the model.

    .. versionadded:: 0.4.0

    Attributes:

    .. attribute:: factor

       The factor object being described.

    .. attribute:: type

       The type of the factor -- either the string ``"numerical"`` or the
       string ``"categorical"``.

    .. attribute:: state

       An opaque object which holds the state needed to evaluate this
       factor on new data (e.g., for prediction). See
       :meth:`factor_protocol.eval`.

    .. attribute:: num_columns

       For numerical factors, the number of columns this factor produces. For
       categorical factors, this attribute will always be ``None``.

    .. attribute:: categories

       For categorical factors, a tuple of the possible categories this factor
       takes on, in order. For numerical factors, this attribute will always be
       ``None``.
    """

    def __init__(self, factor, type, state,
                 num_columns=None, categories=None):
        self.factor = factor
        self.type = type
        if self.type not in ["numerical", "categorical"]:
            raise ValueError("FactorInfo.type must be "
                             "'numerical' or 'categorical', not %r"
                             % (self.type,))
        self.state = state
        if self.type == "numerical":
            if not isinstance(num_columns, six.integer_types):
                raise ValueError("For numerical factors, num_columns "
                                 "must be an integer")
            if categories is not None:
                raise ValueError("For numerical factors, categories "
                                 "must be None")
        else:
            assert self.type == "categorical"
            if num_columns is not None:
                raise ValueError("For categorical factors, num_columns "
                                 "must be None")
            categories = tuple(categories)
        self.num_columns = num_columns
        self.categories = categories

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        class FactorState(object):
            def __repr__(self):
                return "<factor state>"
        kwlist = [("factor", self.factor),
                  ("type", self.type),
                  # Don't put the state in people's faces, it will
                  # just encourage them to pay attention to the
                  # contents :-). Plus it's a bunch of gobbledygook
                  # they don't care about. They can always look at
                  # self.state if they want to know...
                  ("state", FactorState()),
                  ]
        if self.type == "numerical":
            kwlist.append(("num_columns", self.num_columns))
        else:
            kwlist.append(("categories", self.categories))
        repr_pretty_impl(p, self, [], kwlist)

    __getstate__ = no_pickling

def test_FactorInfo():
    fi1 = FactorInfo("asdf", "numerical", {"a": 1}, num_columns=10)
    assert fi1.factor == "asdf"
    assert fi1.state == {"a": 1}
    assert fi1.type == "numerical"
    assert fi1.num_columns == 10
    assert fi1.categories is None

    # smoke test
    repr(fi1)

    fi2 = FactorInfo("asdf", "categorical", {"a": 2}, categories=["z", "j"])
    assert fi2.factor == "asdf"
    assert fi2.state == {"a": 2}
    assert fi2.type == "categorical"
    assert fi2.num_columns is None
    assert fi2.categories == ("z", "j")

    # smoke test
    repr(fi2)

    import pytest
    pytest.raises(ValueError, FactorInfo, "asdf", "non-numerical", {})
    pytest.raises(ValueError, FactorInfo, "asdf", "numerical", {})

    pytest.raises(ValueError, FactorInfo, "asdf", "numerical", {},
                  num_columns="asdf")
    pytest.raises(ValueError, FactorInfo, "asdf", "numerical", {},
                  num_columns=1, categories=1)

    pytest.raises(TypeError, FactorInfo, "asdf", "categorical", {})
    pytest.raises(ValueError, FactorInfo, "asdf", "categorical", {},
                  num_columns=1)
    pytest.raises(TypeError, FactorInfo, "asdf", "categorical", {},
                  categories=1)

    # Make sure longs are legal for num_columns
    # (Important on python2+win64, where array shapes are tuples-of-longs)
    if not six.PY3:
        fi_long = FactorInfo("asdf", "numerical", {"a": 1},
                             num_columns=long(10))
        assert fi_long.num_columns == 10

class SubtermInfo(object):
    """A SubtermInfo object is a simple metadata container describing a single
    primitive interaction and how it is coded in our design matrix. Our final
    design matrix is produced by coding each primitive interaction in order
    from left to right, and then stacking the resulting columns. For each
    :class:`Term`, we have one or more of these objects which describe how
    that term is encoded. :attr:`DesignInfo.term_codings` is a dictionary
    which maps term objects to lists of SubtermInfo objects.

    To code a primitive interaction, the following steps are performed:

    * Evaluate each factor on the provided data.
    * Encode each factor into one or more proto-columns. For numerical
      factors, these proto-columns are identical to whatever the factor
      evaluates to; for categorical factors, they are encoded using a
      specified contrast matrix.
    * Form all pairwise, elementwise products between proto-columns generated
      by different factors. (For example, if factor 1 generated proto-columns
      A and B, and factor 2 generated proto-columns C and D, then our final
      columns are ``A * C``, ``B * C``, ``A * D``, ``B * D``.)
    * The resulting columns are stored directly into the final design matrix.

    Sometimes multiple primitive interactions are needed to encode a single
    term; this occurs, for example, in the formula ``"1 + a:b"`` when ``a``
    and ``b`` are categorical. See :ref:`formulas-building` for full details.

    .. versionadded:: 0.4.0

    Attributes:

    .. attribute:: factors

       The factors which appear in this subterm's interaction.

    .. attribute:: contrast_matrices

       A dict mapping factor objects to :class:`ContrastMatrix` objects,
       describing how each categorical factor in this interaction is coded.

    .. attribute:: num_columns

       The number of design matrix columns which this interaction generates.

    """

    def __init__(self, factors, contrast_matrices, num_columns):
        self.factors = tuple(factors)
        factor_set = frozenset(factors)
        if not isinstance(contrast_matrices, dict):
            raise ValueError("contrast_matrices must be dict")
        for factor, contrast_matrix in six.iteritems(contrast_matrices):
            if factor not in factor_set:
                raise ValueError("Unexpected factor in contrast_matrices dict")
            if not isinstance(contrast_matrix, ContrastMatrix):
                raise ValueError("Expected a ContrastMatrix, not %r"
                                 % (contrast_matrix,))
        self.contrast_matrices = contrast_matrices
        if not isinstance(num_columns, six.integer_types):
            raise ValueError("num_columns must be an integer")
        self.num_columns = num_columns

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        repr_pretty_impl(p, self, [],
                         [("factors", self.factors),
                          ("contrast_matrices", self.contrast_matrices),
                          ("num_columns", self.num_columns)])

    __getstate__ = no_pickling

def test_SubtermInfo():
    cm = ContrastMatrix(np.ones((2, 2)), ["[1]", "[2]"])
    s = SubtermInfo(["a", "x"], {"a": cm}, 4)
    assert s.factors == ("a", "x")
    assert s.contrast_matrices == {"a": cm}
    assert s.num_columns == 4

    # Make sure longs are accepted for num_columns
    if not six.PY3:
        s = SubtermInfo(["a", "x"], {"a": cm}, long(4))
        assert s.num_columns == 4

    # smoke test
    repr(s)

    import pytest
    pytest.raises(TypeError, SubtermInfo, 1, {}, 1)
    pytest.raises(ValueError, SubtermInfo, ["a", "x"], 1, 1)
    pytest.raises(ValueError, SubtermInfo, ["a", "x"], {"z": cm}, 1)
    pytest.raises(ValueError, SubtermInfo, ["a", "x"], {"a": 1}, 1)
    pytest.raises(ValueError, SubtermInfo, ["a", "x"], {}, 1.5)

class DesignInfo(object):
    """A DesignInfo object holds metadata about a design matrix.

    This is the main object that Patsy uses to pass metadata about a design
    matrix to statistical libraries, in order to allow further downstream
    processing like intelligent tests, prediction on new data, etc. Usually
    encountered as the `.design_info` attribute on design matrices.

    """

    def __init__(self, column_names,
                 factor_infos=None, term_codings=None):
        self.column_name_indexes = OrderedDict(zip(column_names,
                                                   range(len(column_names))))

        if (factor_infos is None) != (term_codings is None):
            raise ValueError("Must specify either both or neither of "
                             "factor_infos= and term_codings=")

        self.factor_infos = factor_infos
        self.term_codings = term_codings

        # factor_infos is a dict containing one entry for every factor
        #    mentioned in our terms
        #    and mapping each to FactorInfo object
        if self.factor_infos is not None:
            if not isinstance(self.factor_infos, dict):
                raise ValueError("factor_infos should be a dict")

            if not isinstance(self.term_codings, OrderedDict):
                raise ValueError("term_codings must be an OrderedDict")
            for term, subterms in six.iteritems(self.term_codings):
                if not isinstance(term, Term):
                    raise ValueError("expected a Term, not %r" % (term,))
                if not isinstance(subterms, list):
                    raise ValueError("term_codings must contain lists")
                term_factors = set(term.factors)
                for subterm in subterms:
                    if not isinstance(subterm, SubtermInfo):
                        raise ValueError("expected SubtermInfo, "
                                         "not %r" % (subterm,))
                    if not term_factors.issuperset(subterm.factors):
                        raise ValueError("unexpected factors in subterm")

            all_factors = set()
            for term in self.term_codings:
                all_factors.update(term.factors)
            if all_factors != set(self.factor_infos):
                raise ValueError("Provided Term objects and factor_infos "
                                 "do not match")
            for factor, factor_info in six.iteritems(self.factor_infos):
                if not isinstance(factor_info, FactorInfo):
                    raise ValueError("expected FactorInfo object, not %r"
                                     % (factor_info,))
                if factor != factor_info.factor:
                    raise ValueError("mismatched factor_info.factor")

            for term, subterms in six.iteritems(self.term_codings):
                for subterm in subterms:
                    exp_cols = 1
                    cat_factors = set()
                    for factor in subterm.factors:
                        fi = self.factor_infos[factor]
                        if fi.type == "numerical":
                            exp_cols *= fi.num_columns
                        else:
                            assert fi.type == "categorical"
                            cm = subterm.contrast_matrices[factor].matrix
                            if cm.shape[0] != len(fi.categories):
                                raise ValueError("Mismatched contrast matrix "
                                                 "for factor %r" % (factor,))
                            cat_factors.add(factor)
                            exp_cols *= cm.shape[1]
                    if cat_factors != set(subterm.contrast_matrices):
                        raise ValueError("Mismatch between contrast_matrices "
                                         "and categorical factors")
                    if exp_cols != subterm.num_columns:
                        raise ValueError("Unexpected num_columns")

        if term_codings is None:
            # Need to invent term information
            self.term_slices = None
            # We invent one term per column, with the same name as the column
            term_names = column_names
            slices = [slice(i, i + 1) for i in range(len(column_names))]
            self.term_name_slices = OrderedDict(zip(term_names, slices))
        else:
            # Need to derive term information from term_codings
            self.term_slices = OrderedDict()
            idx = 0
            for term, subterm_infos in six.iteritems(self.term_codings):
                term_columns = 0
                for subterm_info in subterm_infos:
                    term_columns += subterm_info.num_columns
                self.term_slices[term] = slice(idx, idx + term_columns)
                idx += term_columns
            if idx != len(self.column_names):
                raise ValueError("mismatch between column_names and columns "
                                 "coded by given terms")
            self.term_name_slices = OrderedDict(
                [(term.name(), slice_)
                 for (term, slice_) in six.iteritems(self.term_slices)])

        # Guarantees:
        #   term_name_slices is never None
        #   The slices in term_name_slices are in order and exactly cover the
        #     whole range of columns.
        #   term_slices may be None
        #   If term_slices is not None, then its slices match the ones in
        #     term_name_slices.
        assert self.term_name_slices is not None
        if self.term_slices is not None:
            assert (list(self.term_slices.values())
                    == list(self.term_name_slices.values()))
        # These checks probably aren't necessary anymore now that we always
        # generate the slices ourselves, but we'll leave them in just to be
        # safe.
        covered = 0
        for slice_ in six.itervalues(self.term_name_slices):
            start, stop, step = slice_.indices(len(column_names))
            assert start == covered
            assert step == 1
            covered = stop
        assert covered == len(column_names)
        #   If there is any name overlap between terms and columns, they refer
        #     to the same columns.
        for column_name, index in six.iteritems(self.column_name_indexes):
            if column_name in self.term_name_slices:
                slice_ = self.term_name_slices[column_name]
                if slice_ != slice(index, index + 1):
                    raise ValueError("term/column name collision")

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        assert not cycle
        repr_pretty_impl(p, self,
                         [self.column_names],
                         [("factor_infos", self.factor_infos),
                          ("term_codings", self.term_codings)])

    @property
    def column_names(self):
        "A list of the column names, in order."
        return list(self.column_name_indexes)

    @property
    def terms(self):
        "A list of :class:`Terms`, in order, or else None."
        if self.term_slices is None:
            return None
        return list(self.term_slices)

    @property
    def term_names(self):
        "A list of terms, in order."
        return list(self.term_name_slices)

    @property
    def builder(self):
        ".. deprecated:: 0.4.0"
        warnings.warn(DeprecationWarning(
            "The DesignInfo.builder attribute is deprecated starting in "
            "patsy v0.4.0; distinct builder objects have been eliminated "
            "and design_info.builder is now just a long-winded way of "
            "writing 'design_info' (i.e. the .builder attribute just "
            "returns self)"), stacklevel=2)
        return self

    @property
    def design_info(self):
        ".. deprecated:: 0.4.0"
        warnings.warn(DeprecationWarning(
            "Starting in patsy v0.4.0, the DesignMatrixBuilder class has "
            "been merged into the DesignInfo class. So there's no need to "
            "use builder.design_info to access the DesignInfo; 'builder' "
            "already *is* a DesignInfo."), stacklevel=2)
        return self

    def slice(self, columns_specifier):
        """Locate a subset of design matrix columns, specified symbolically.

        A patsy design matrix has two levels of structure: the individual
        columns (which are named), and the :ref:`terms <formulas>` in
        the formula that generated those columns. This is a one-to-many
        relationship: a single term may span several columns. This method
        provides a user-friendly API for locating those columns.

        (While we talk about columns here, this is probably most useful for
        indexing into other arrays that are derived from the design matrix,
        such as regression coefficients or covariance matrices.)

        The `columns_specifier` argument can take a number of forms:

        * A term name
        * A column name
        * A :class:`Term` object
        * An integer giving a raw index
        * A raw slice object

        In all cases, a Python :func:`slice` object is returned, which can be
        used directly for indexing.

        Example::

          y, X = dmatrices("y ~ a", demo_data("y", "a", nlevels=3))
          betas = np.linalg.lstsq(X, y)[0]
          a_betas = betas[X.design_info.slice("a")]

        (If you want to look up a single individual column by name, use
        ``design_info.column_name_indexes[name]``.)
        """
        if isinstance(columns_specifier, slice):
            return columns_specifier
        if np.issubdtype(type(columns_specifier), np.integer):
            return slice(columns_specifier, columns_specifier + 1)
        if (self.term_slices is not None
            and columns_specifier in self.term_slices):
            return self.term_slices[columns_specifier]
        if columns_specifier in self.term_name_slices:
            return self.term_name_slices[columns_specifier]
        if columns_specifier in self.column_name_indexes:
            idx = self.column_name_indexes[columns_specifier]
            return slice(idx, idx + 1)
        raise PatsyError("unknown column specified '%s'"
                            % (columns_specifier,))

    def linear_constraint(self, constraint_likes):
        """Construct a linear constraint in matrix form from a (possibly
        symbolic) description.

        Possible inputs:

        * A dictionary which is taken as a set of equality constraint. Keys
          can be either string column names, or integer column indexes.
        * A string giving a arithmetic expression referring to the matrix
          columns by name.
        * A list of such strings which are ANDed together.
        * A tuple (A, b) where A and b are array_likes, and the constraint is
          Ax = b. If necessary, these will be coerced to the proper
          dimensionality by appending dimensions with size 1.

        The string-based language has the standard arithmetic operators, / * +
        - and parentheses, plus "=" is used for equality and "," is used to
        AND together multiple constraint equations within a string. You can
        If no = appears in some expression, then that expression is assumed to
        be equal to zero. Division is always float-based, even if
        ``__future__.true_division`` isn't in effect.

        Returns a :class:`LinearConstraint` object.

        Examples::

          di = DesignInfo(["x1", "x2", "x3"])

          # Equivalent ways to write x1 == 0:
          di.linear_constraint({"x1": 0})  # by name
          di.linear_constraint({0: 0})  # by index
          di.linear_constraint("x1 = 0")  # string based
          di.linear_constraint("x1")  # can leave out "= 0"
          di.linear_constraint("2 * x1 = (x1 + 2 * x1) / 3")
          di.linear_constraint(([1, 0, 0], 0))  # constraint matrices

          # Equivalent ways to write x1 == 0 and x3 == 10
          di.linear_constraint({"x1": 0, "x3": 10})
          di.linear_constraint({0: 0, 2: 10})
          di.linear_constraint({0: 0, "x3": 10})
          di.linear_constraint("x1 = 0, x3 = 10")
          di.linear_constraint("x1, x3 = 10")
          di.linear_constraint(["x1", "x3 = 0"])  # list of strings
          di.linear_constraint("x1 = 0, x3 - 10 = x1")
          di.linear_constraint([[1, 0, 0], [0, 0, 1]], [0, 10])

          # You can also chain together equalities, just like Python:
          di.linear_constraint("x1 = x2 = 3")
        """
        return linear_constraint(constraint_likes, self.column_names)

    def describe(self):
        """Returns a human-readable string describing this design info.

        Example:

        .. ipython::

          In [1]: y, X = dmatrices("y ~ x1 + x2", demo_data("y", "x1", "x2"))

          In [2]: y.design_info.describe()
          Out[2]: 'y'

          In [3]: X.design_info.describe()
          Out[3]: '1 + x1 + x2'

        .. warning::

           There is no guarantee that the strings returned by this function
           can be parsed as formulas, or that if they can be parsed as a
           formula that they will produce a model equivalent to the one you
           started with. This function produces a best-effort description
           intended for humans to read.

        """

        names = []
        for name in self.term_names:
            if name == "Intercept":
                names.append("1")
            else:
                names.append(name)
        return " + ".join(names)

    def subset(self, which_terms):
        """Create a new :class:`DesignInfo` for design matrices that contain a
        subset of the terms that the current :class:`DesignInfo` does.

        For example, if ``design_info`` has terms ``x``, ``y``, and ``z``,
        then::

          design_info2 = design_info.subset(["x", "z"])

        will return a new DesignInfo that can be used to construct design
        matrices with only the columns corresponding to the terms ``x`` and
        ``z``. After we do this, then in general these two expressions will
        return the same thing (here we assume that ``x``, ``y``, and ``z``
        each generate a single column of the output)::

          build_design_matrix([design_info], data)[0][:, [0, 2]]
          build_design_matrix([design_info2], data)[0]

        However, a critical difference is that in the second case, ``data``
        need not contain any values for ``y``. This is very useful when doing
        prediction using a subset of a model, in which situation R usually
        forces you to specify dummy values for ``y``.

        If using a formula to specify the terms to include, remember that like
        any formula, the intercept term will be included by default, so use
        ``0`` or ``-1`` in your formula if you want to avoid this.

        This method can also be used to reorder the terms in your design
        matrix, in case you want to do that for some reason. I can't think of
        any.

        Note that this method will generally *not* produce the same result as
        creating a new model directly. Consider these DesignInfo objects::

            design1 = dmatrix("1 + C(a)", data)
            design2 = design1.subset("0 + C(a)")
            design3 = dmatrix("0 + C(a)", data)

        Here ``design2`` and ``design3`` will both produce design matrices
        that contain an encoding of ``C(a)`` without any intercept term. But
        ``design3`` uses a full-rank encoding for the categorical term
        ``C(a)``, while ``design2`` uses the same reduced-rank encoding as
        ``design1``.

        :arg which_terms: The terms which should be kept in the new
          :class:`DesignMatrixBuilder`. If this is a string, then it is parsed
          as a formula, and then the names of the resulting terms are taken as
          the terms to keep. If it is a list, then it can contain a mixture of
          term names (as strings) and :class:`Term` objects.

        .. versionadded: 0.2.0
           New method on the class DesignMatrixBuilder.

        .. versionchanged: 0.4.0
           Moved from DesignMatrixBuilder to DesignInfo, as part of the
           removal of DesignMatrixBuilder.

        """
        if isinstance(which_terms, str):
            desc = ModelDesc.from_formula(which_terms)
            if desc.lhs_termlist:
                raise PatsyError("right-hand-side-only formula required")
            which_terms = [term.name() for term in desc.rhs_termlist]

        if self.term_codings is None:
            # This is a minimal DesignInfo
            # If the name is unknown we just let the KeyError escape
            new_names = []
            for t in which_terms:
                new_names += self.column_names[self.term_name_slices[t]]
            return DesignInfo(new_names)
        else:
            term_name_to_term = {}
            for term in self.term_codings:
                term_name_to_term[term.name()] = term

            new_column_names = []
            new_factor_infos = {}
            new_term_codings = OrderedDict()
            for name_or_term in which_terms:
                term = term_name_to_term.get(name_or_term, name_or_term)
                # If the name is unknown we just let the KeyError escape
                s = self.term_slices[term]
                new_column_names += self.column_names[s]
                for f in term.factors:
                    new_factor_infos[f] = self.factor_infos[f]
                new_term_codings[term] = self.term_codings[term]
            return DesignInfo(new_column_names,
                              factor_infos=new_factor_infos,
                              term_codings=new_term_codings)

    @classmethod
    def from_array(cls, array_like, default_column_prefix="column"):
        """Find or construct a DesignInfo appropriate for a given array_like.

        If the input `array_like` already has a ``.design_info``
        attribute, then it will be returned. Otherwise, a new DesignInfo
        object will be constructed, using names either taken from the
        `array_like` (e.g., for a pandas DataFrame with named columns), or
        constructed using `default_column_prefix`.

        This is how :func:`dmatrix` (for example) creates a DesignInfo object
        if an arbitrary matrix is passed in.

        :arg array_like: An ndarray or pandas container.
        :arg default_column_prefix: If it's necessary to invent column names,
          then this will be used to construct them.
        :returns: a DesignInfo object
        """
        if hasattr(array_like, "design_info") and isinstance(array_like.design_info, cls):
            return array_like.design_info
        arr = atleast_2d_column_default(array_like, preserve_pandas=True)
        if arr.ndim > 2:
            raise ValueError("design matrix can't have >2 dimensions")
        columns = getattr(arr, "columns", range(arr.shape[1]))
        if (hasattr(columns, "dtype")
            and not safe_issubdtype(columns.dtype, np.integer)):
            column_names = [str(obj) for obj in columns]
        else:
            column_names = ["%s%s" % (default_column_prefix, i)
                            for i in columns]
        return DesignInfo(column_names)

    __getstate__ = no_pickling

def test_DesignInfo():
    import pytest
    class _MockFactor(object):
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name
    f_x = _MockFactor("x")
    f_y = _MockFactor("y")
    t_x = Term([f_x])
    t_y = Term([f_y])
    factor_infos = {f_x:
                      FactorInfo(f_x, "numerical", {}, num_columns=3),
                    f_y:
                      FactorInfo(f_y, "numerical", {}, num_columns=1),
                   }
    term_codings = OrderedDict([(t_x, [SubtermInfo([f_x], {}, 3)]),
                                (t_y, [SubtermInfo([f_y], {}, 1)])])
    di = DesignInfo(["x1", "x2", "x3", "y"], factor_infos, term_codings)
    assert di.column_names == ["x1", "x2", "x3", "y"]
    assert di.term_names == ["x", "y"]
    assert di.terms == [t_x, t_y]
    assert di.column_name_indexes == {"x1": 0, "x2": 1, "x3": 2, "y": 3}
    assert di.term_name_slices == {"x": slice(0, 3), "y": slice(3, 4)}
    assert di.term_slices == {t_x: slice(0, 3), t_y: slice(3, 4)}
    assert di.describe() == "x + y"

    assert di.slice(1) == slice(1, 2)
    assert di.slice("x1") == slice(0, 1)
    assert di.slice("x2") == slice(1, 2)
    assert di.slice("x3") == slice(2, 3)
    assert di.slice("x") == slice(0, 3)
    assert di.slice(t_x) == slice(0, 3)
    assert di.slice("y") == slice(3, 4)
    assert di.slice(t_y) == slice(3, 4)
    assert di.slice(slice(2, 4)) == slice(2, 4)
    pytest.raises(PatsyError, di.slice, "asdf")

    # smoke test
    repr(di)

    assert_no_pickling(di)

    # One without term objects
    di = DesignInfo(["a1", "a2", "a3", "b"])
    assert di.column_names == ["a1", "a2", "a3", "b"]
    assert di.term_names == ["a1", "a2", "a3", "b"]
    assert di.terms is None
    assert di.column_name_indexes == {"a1": 0, "a2": 1, "a3": 2, "b": 3}
    assert di.term_name_slices == {"a1": slice(0, 1),
                                   "a2": slice(1, 2),
                                   "a3": slice(2, 3),
                                   "b": slice(3, 4)}
    assert di.term_slices is None
    assert di.describe() == "a1 + a2 + a3 + b"

    assert di.slice(1) == slice(1, 2)
    assert di.slice("a1") == slice(0, 1)
    assert di.slice("a2") == slice(1, 2)
    assert di.slice("a3") == slice(2, 3)
    assert di.slice("b") == slice(3, 4)

    # Check intercept handling in describe()
    assert DesignInfo(["Intercept", "a", "b"]).describe() == "1 + a + b"

    # Failure modes
    # must specify either both or neither of factor_infos and term_codings:
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], factor_infos=factor_infos)
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], term_codings=term_codings)
    # factor_infos must be a dict
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], list(factor_infos), term_codings)
    # wrong number of column names:
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y1", "y2"], factor_infos, term_codings)
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3"], factor_infos, term_codings)
    # name overlap problems
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "y", "y2"], factor_infos, term_codings)
    # duplicate name
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x1", "x1", "y"], factor_infos, term_codings)

    # f_y is in factor_infos, but not mentioned in any term
    term_codings_x_only = OrderedDict(term_codings)
    del term_codings_x_only[t_y]
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3"], factor_infos, term_codings_x_only)

    # f_a is in a term, but not in factor_infos
    f_a = _MockFactor("a")
    t_a = Term([f_a])
    term_codings_with_a = OrderedDict(term_codings)
    term_codings_with_a[t_a] = [SubtermInfo([f_a], {}, 1)]
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y", "a"],
                  factor_infos, term_codings_with_a)

    # bad factor_infos
    not_factor_infos = dict(factor_infos)
    not_factor_infos[f_x] = "what is this I don't even"
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], not_factor_infos, term_codings)

    mismatch_factor_infos = dict(factor_infos)
    mismatch_factor_infos[f_x] = FactorInfo(f_a, "numerical", {}, num_columns=3)
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], mismatch_factor_infos, term_codings)

    # bad term_codings
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], factor_infos, dict(term_codings))

    not_term_codings = OrderedDict(term_codings)
    not_term_codings["this is a string"] = term_codings[t_x]
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], factor_infos, not_term_codings)

    non_list_term_codings = OrderedDict(term_codings)
    non_list_term_codings[t_y] = tuple(term_codings[t_y])
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], factor_infos, non_list_term_codings)

    non_subterm_term_codings = OrderedDict(term_codings)
    non_subterm_term_codings[t_y][0] = "not a SubtermInfo"
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], factor_infos, non_subterm_term_codings)

    bad_subterm = OrderedDict(term_codings)
    # f_x is a factor in this model, but it is not a factor in t_y
    term_codings[t_y][0] = SubtermInfo([f_x], {}, 1)
    pytest.raises(ValueError, DesignInfo,
                  ["x1", "x2", "x3", "y"], factor_infos, bad_subterm)

    # contrast matrix has wrong number of rows
    factor_codings_a = {f_a:
                          FactorInfo(f_a, "categorical", {},
                                     categories=["a1", "a2"])}
    term_codings_a_bad_rows = OrderedDict([
        (t_a,
         [SubtermInfo([f_a],
                      {f_a: ContrastMatrix(np.ones((3, 2)),
                                           ["[1]", "[2]"])},
                      2)])])
    pytest.raises(ValueError, DesignInfo,
                  ["a[1]", "a[2]"],
                  factor_codings_a,
                  term_codings_a_bad_rows)

    # have a contrast matrix for a non-categorical factor
    t_ax = Term([f_a, f_x])
    factor_codings_ax = {f_a:
                           FactorInfo(f_a, "categorical", {},
                                      categories=["a1", "a2"]),
                         f_x:
                           FactorInfo(f_x, "numerical", {},
                                      num_columns=2)}
    term_codings_ax_extra_cm = OrderedDict([
        (t_ax,
         [SubtermInfo([f_a, f_x],
                      {f_a: ContrastMatrix(np.ones((2, 2)), ["[1]", "[2]"]),
                       f_x: ContrastMatrix(np.ones((2, 2)), ["[1]", "[2]"])},
                      4)])])
    pytest.raises(ValueError, DesignInfo,
                  ["a[1]:x[1]", "a[2]:x[1]", "a[1]:x[2]", "a[2]:x[2]"],
                  factor_codings_ax,
                  term_codings_ax_extra_cm)

    # no contrast matrix for a categorical factor
    term_codings_ax_missing_cm = OrderedDict([
        (t_ax,
         [SubtermInfo([f_a, f_x],
                      {},
                      4)])])
    # This actually fails before it hits the relevant check with a KeyError,
    # but that's okay... the previous test still exercises the check.
    pytest.raises((ValueError, KeyError), DesignInfo,
                  ["a[1]:x[1]", "a[2]:x[1]", "a[1]:x[2]", "a[2]:x[2]"],
                  factor_codings_ax,
                  term_codings_ax_missing_cm)

    # subterm num_columns doesn't match the value computed from the individual
    # factors
    term_codings_ax_wrong_subterm_columns = OrderedDict([
        (t_ax,
         [SubtermInfo([f_a, f_x],
                      {f_a: ContrastMatrix(np.ones((2, 3)),
                                           ["[1]", "[2]", "[3]"])},
                      # should be 2 * 3 = 6
                      5)])])
    pytest.raises(ValueError, DesignInfo,
                  ["a[1]:x[1]", "a[2]:x[1]", "a[3]:x[1]",
                   "a[1]:x[2]", "a[2]:x[2]", "a[3]:x[2]"],
                  factor_codings_ax,
                  term_codings_ax_wrong_subterm_columns)

def test_DesignInfo_from_array():
    di = DesignInfo.from_array([1, 2, 3])
    assert di.column_names == ["column0"]
    di2 = DesignInfo.from_array([[1, 2], [2, 3], [3, 4]])
    assert di2.column_names == ["column0", "column1"]
    di3 = DesignInfo.from_array([1, 2, 3], default_column_prefix="x")
    assert di3.column_names == ["x0"]
    di4 = DesignInfo.from_array([[1, 2], [2, 3], [3, 4]],
                                default_column_prefix="x")
    assert di4.column_names == ["x0", "x1"]
    m = DesignMatrix([1, 2, 3], di3)
    assert DesignInfo.from_array(m) is di3
    # But weird objects are ignored
    m.design_info = "asdf"
    di_weird = DesignInfo.from_array(m)
    assert di_weird.column_names == ["column0"]

    import pytest
    pytest.raises(ValueError, DesignInfo.from_array, np.ones((2, 2, 2)))

    from patsy.util import have_pandas
    if have_pandas:
        import pandas
        # with named columns
        di5 = DesignInfo.from_array(pandas.DataFrame([[1, 2]],
                                                     columns=["a", "b"]))
        assert di5.column_names == ["a", "b"]
        # with irregularly numbered columns
        di6 = DesignInfo.from_array(pandas.DataFrame([[1, 2]],
                                                     columns=[0, 10]))
        assert di6.column_names == ["column0", "column10"]
        # with .design_info attr
        df = pandas.DataFrame([[1, 2]])
        df.design_info = di6
        assert DesignInfo.from_array(df) is di6

def test_DesignInfo_linear_constraint():
    di = DesignInfo(["a1", "a2", "a3", "b"])
    con = di.linear_constraint(["2 * a1 = b + 1", "a3"])
    assert con.variable_names == ["a1", "a2", "a3", "b"]
    assert np.all(con.coefs == [[2, 0, 0, -1], [0, 0, 1, 0]])
    assert np.all(con.constants == [[1], [0]])

def test_DesignInfo_deprecated_attributes():
    d = DesignInfo(["a1", "a2"])
    def check(attr):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert getattr(d, attr) is d
        assert len(w) == 1
        assert w[0].category is DeprecationWarning
    check("builder")
    check("design_info")

# Idea: format with a reasonable amount of precision, then if that turns out
# to be higher than necessary, remove as many zeros as we can. But only do
# this while we can do it to *all* the ordinarily-formatted numbers, to keep
# decimal points aligned.
def _format_float_column(precision, col):
    format_str = "%." + str(precision) + "f"
    assert col.ndim == 1
    # We don't want to look at numbers like "1e-5" or "nan" when stripping.
    simple_float_chars = set("+-0123456789.")
    col_strs = np.array([format_str % (x,) for x in col], dtype=object)
    # Really every item should have a decimal, but just in case, we don't want
    # to strip zeros off the end of "10" or something like that.
    mask = np.array([simple_float_chars.issuperset(col_str) and "." in col_str
                     for col_str in col_strs])
    mask_idxes = np.nonzero(mask)[0]
    strip_char = "0"
    if np.any(mask):
        while True:
            if np.all([s.endswith(strip_char) for s in col_strs[mask]]):
                for idx in mask_idxes:
                    col_strs[idx] = col_strs[idx][:-1]
            else:
                if strip_char == "0":
                    strip_char = "."
                else:
                    break
    return col_strs

def test__format_float_column():
    def t(precision, numbers, expected):
        got = _format_float_column(precision, np.asarray(numbers))
        print(got, expected)
        assert np.array_equal(got, expected)
    # This acts weird on old python versions (e.g. it can be "-nan"), so don't
    # hardcode it:
    nan_string = "%.3f" % (np.nan,)
    t(3, [1, 2.1234, 2.1239, np.nan], ["1.000", "2.123", "2.124", nan_string])
    t(3, [1, 2, 3, np.nan], ["1", "2", "3", nan_string])
    t(3, [1.0001, 2, 3, np.nan], ["1", "2", "3", nan_string])
    t(4, [1.0001, 2, 3, np.nan], ["1.0001", "2.0000", "3.0000", nan_string])

# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array
class DesignMatrix(np.ndarray):
    """A simple numpy array subclass that carries design matrix metadata.

    .. attribute:: design_info

       A :class:`DesignInfo` object containing metadata about this design
       matrix.

    This class also defines a fancy __repr__ method with labeled
    columns. Otherwise it is identical to a regular numpy ndarray.

    .. warning::

       You should never check for this class using
       :func:`isinstance`. Limitations of the numpy API mean that it is
       impossible to prevent the creation of numpy arrays that have type
       DesignMatrix, but that are not actually design matrices (and such
       objects will behave like regular ndarrays in every way). Instead, check
       for the presence of a ``.design_info`` attribute -- this will be
       present only on "real" DesignMatrix objects.
    """

    def __new__(cls, input_array, design_info=None,
                default_column_prefix="column"):
        """Create a DesignMatrix, or cast an existing matrix to a DesignMatrix.

        A call like::

          DesignMatrix(my_array)

        will convert an arbitrary array_like object into a DesignMatrix.

        The return from this function is guaranteed to be a two-dimensional
        ndarray with a real-valued floating point dtype, and a
        ``.design_info`` attribute which matches its shape. If the
        `design_info` argument is not given, then one is created via
        :meth:`DesignInfo.from_array` using the given
        `default_column_prefix`.

        Depending on the input array, it is possible this will pass through
        its input unchanged, or create a view.
        """
        # Pass through existing DesignMatrixes. The design_info check is
        # necessary because numpy is sort of annoying and cannot be stopped
        # from turning non-design-matrix arrays into DesignMatrix
        # instances. (E.g., my_dm.diagonal() will return a DesignMatrix
        # object, but one without a design_info attribute.)
        if (isinstance(input_array, DesignMatrix)
            and hasattr(input_array, "design_info")):
            return input_array
        self = atleast_2d_column_default(input_array).view(cls)
        # Upcast integer to floating point
        if safe_issubdtype(self.dtype, np.integer):
            self = np.asarray(self, dtype=float).view(cls)
        if self.ndim > 2:
            raise ValueError("DesignMatrix must be 2d")
        assert self.ndim == 2
        if design_info is None:
            design_info = DesignInfo.from_array(self, default_column_prefix)
        if len(design_info.column_names) != self.shape[1]:
            raise ValueError("wrong number of column names for design matrix "
                             "(got %s, wanted %s)"
                             % (len(design_info.column_names), self.shape[1]))
        self.design_info = design_info
        if not safe_issubdtype(self.dtype, np.floating):
            raise ValueError("design matrix must be real-valued floating point")
        return self

    __repr__ = repr_pretty_delegate
    def _repr_pretty_(self, p, cycle):
        if not hasattr(self, "design_info"):
            # Not a real DesignMatrix
            p.pretty(np.asarray(self))
            return
        assert not cycle

        # XX: could try calculating width of the current terminal window:
        #   http://stackoverflow.com/questions/566746/how-to-get-console-window-width-in-python
        # sadly it looks like ipython does not actually pass this information
        # in, even if we use _repr_pretty_ -- the pretty-printer object has a
        # fixed width it always uses. (As of IPython 0.12.)
        MAX_TOTAL_WIDTH = 78
        SEP = 2
        INDENT = 2
        MAX_ROWS = 30
        PRECISION = 5

        names = self.design_info.column_names
        column_name_widths = [len(name) for name in names]
        min_total_width = (INDENT + SEP * (self.shape[1] - 1)
                           + np.sum(column_name_widths))
        if min_total_width <= MAX_TOTAL_WIDTH:
            printable_part = np.asarray(self)[:MAX_ROWS, :]
            formatted_cols = [_format_float_column(PRECISION,
                                                   printable_part[:, i])
                              for i in range(self.shape[1])]
            def max_width(col):
                assert col.ndim == 1
                if not col.shape[0]:
                    return 0
                else:
                    return max([len(s) for s in col])
            column_num_widths = [max_width(col) for col in formatted_cols]
            column_widths = [max(name_width, num_width)
                             for (name_width, num_width)
                             in zip(column_name_widths, column_num_widths)]
            total_width = (INDENT + SEP * (self.shape[1] - 1)
                           + np.sum(column_widths))
            print_numbers = (total_width < MAX_TOTAL_WIDTH)
        else:
            print_numbers = False

        p.begin_group(INDENT, "DesignMatrix with shape %s" % (self.shape,))
        p.breakable("\n" + " " * p.indentation)
        if print_numbers:
            # We can fit the numbers on the screen
            sep = " " * SEP
            # list() is for Py3 compatibility
            for row in [names] + list(zip(*formatted_cols)):
                cells = [cell.rjust(width)
                         for (width, cell) in zip(column_widths, row)]
                p.text(sep.join(cells))
                p.text("\n" + " " * p.indentation)
            if MAX_ROWS < self.shape[0]:
                p.text("[%s rows omitted]" % (self.shape[0] - MAX_ROWS,))
                p.text("\n" + " " * p.indentation)
        else:
            p.begin_group(2, "Columns:")
            p.breakable("\n" + " " * p.indentation)
            p.pretty(names)
            p.end_group(2, "")
            p.breakable("\n" + " " * p.indentation)

        p.begin_group(2, "Terms:")
        p.breakable("\n" + " " * p.indentation)
        for term_name, span in six.iteritems(self.design_info.term_name_slices):
            if span.start != 0:
                p.breakable(", ")
            p.pretty(term_name)
            if span.stop - span.start == 1:
                coltext = "column %s" % (span.start,)
            else:
                coltext = "columns %s:%s" % (span.start, span.stop)
            p.text(" (%s)" % (coltext,))
        p.end_group(2, "")

        if not print_numbers or self.shape[0] > MAX_ROWS:
            # some data was not shown
            p.breakable("\n" + " " * p.indentation)
            p.text("(to view full data, use np.asarray(this_obj))")

        p.end_group(INDENT, "")

    # No __array_finalize__ method, because we don't want slices of this
    # object to keep the design_info (they may have different columns!), or
    # anything fancy like that.

    __reduce__ = no_pickling

def test_design_matrix():
    import pytest

    di = DesignInfo(["a1", "a2", "a3", "b"])
    mm = DesignMatrix([[12, 14, 16, 18]], di)
    assert mm.design_info.column_names == ["a1", "a2", "a3", "b"]

    bad_di = DesignInfo(["a1"])
    pytest.raises(ValueError, DesignMatrix, [[12, 14, 16, 18]], bad_di)

    mm2 = DesignMatrix([[12, 14, 16, 18]])
    assert mm2.design_info.column_names == ["column0", "column1", "column2",
                                            "column3"]

    mm3 = DesignMatrix([12, 14, 16, 18])
    assert mm3.shape == (4, 1)

    # DesignMatrix always has exactly 2 dimensions
    pytest.raises(ValueError, DesignMatrix, [[[1]]])

    # DesignMatrix constructor passes through existing DesignMatrixes
    mm4 = DesignMatrix(mm)
    assert mm4 is mm
    # But not if they are really slices:
    mm5 = DesignMatrix(mm.diagonal())
    assert mm5 is not mm

    mm6 = DesignMatrix([[12, 14, 16, 18]], default_column_prefix="x")
    assert mm6.design_info.column_names == ["x0", "x1", "x2", "x3"]

    assert_no_pickling(mm6)

    # Only real-valued matrices can be DesignMatrixs
    pytest.raises(ValueError, DesignMatrix, [1, 2, 3j])
    pytest.raises(ValueError, DesignMatrix, ["a", "b", "c"])
    pytest.raises(ValueError, DesignMatrix, [1, 2, object()])

    # Just smoke tests
    repr(mm)
    repr(DesignMatrix(np.arange(100)))
    repr(DesignMatrix(np.arange(100) * 2.0))
    repr(mm[1:, :])
    repr(DesignMatrix(np.arange(100).reshape((1, 100))))
    repr(DesignMatrix([np.nan, np.inf]))
    repr(DesignMatrix([np.nan, 0, 1e20, 20.5]))
    # handling of zero-size matrices
    repr(DesignMatrix(np.zeros((1, 0))))
    repr(DesignMatrix(np.zeros((0, 1))))
    repr(DesignMatrix(np.zeros((0, 0))))
