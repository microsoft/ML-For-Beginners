# Natural Language Toolkit: Feature Structures
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>,
#         Rob Speer,
#         Steven Bird <stevenbird1@gmail.com>
# URL: <https://www.nltk.org>
# For license information, see LICENSE.TXT

"""
Basic data classes for representing feature structures, and for
performing basic operations on those feature structures.  A feature
structure is a mapping from feature identifiers to feature values,
where each feature value is either a basic value (such as a string or
an integer), or a nested feature structure.  There are two types of
feature structure, implemented by two subclasses of ``FeatStruct``:

    - feature dictionaries, implemented by ``FeatDict``, act like
      Python dictionaries.  Feature identifiers may be strings or
      instances of the ``Feature`` class.
    - feature lists, implemented by ``FeatList``, act like Python
      lists.  Feature identifiers are integers.

Feature structures are typically used to represent partial information
about objects.  A feature identifier that is not mapped to a value
stands for a feature whose value is unknown (*not* a feature without
a value).  Two feature structures that represent (potentially
overlapping) information about the same object can be combined by
unification.  When two inconsistent feature structures are unified,
the unification fails and returns None.

Features can be specified using "feature paths", or tuples of feature
identifiers that specify path through the nested feature structures to
a value.  Feature structures may contain reentrant feature values.  A
"reentrant feature value" is a single feature value that can be
accessed via multiple feature paths.  Unification preserves the
reentrance relations imposed by both of the unified feature
structures.  In the feature structure resulting from unification, any
modifications to a reentrant feature value will be visible using any
of its feature paths.

Feature structure variables are encoded using the ``nltk.sem.Variable``
class.  The variables' values are tracked using a bindings
dictionary, which maps variables to their values.  When two feature
structures are unified, a fresh bindings dictionary is created to
track their values; and before unification completes, all bound
variables are replaced by their values.  Thus, the bindings
dictionaries are usually strictly internal to the unification process.
However, it is possible to track the bindings of variables if you
choose to, by supplying your own initial bindings dictionary to the
``unify()`` function.

When unbound variables are unified with one another, they become
aliased.  This is encoded by binding one variable to the other.

Lightweight Feature Structures
==============================
Many of the functions defined by ``nltk.featstruct`` can be applied
directly to simple Python dictionaries and lists, rather than to
full-fledged ``FeatDict`` and ``FeatList`` objects.  In other words,
Python ``dicts`` and ``lists`` can be used as "light-weight" feature
structures.

    >>> from nltk.featstruct import unify
    >>> unify(dict(x=1, y=dict()), dict(a='a', y=dict(b='b')))  # doctest: +SKIP
    {'y': {'b': 'b'}, 'x': 1, 'a': 'a'}

However, you should keep in mind the following caveats:

  - Python dictionaries & lists ignore reentrance when checking for
    equality between values.  But two FeatStructs with different
    reentrances are considered nonequal, even if all their base
    values are equal.

  - FeatStructs can be easily frozen, allowing them to be used as
    keys in hash tables.  Python dictionaries and lists can not.

  - FeatStructs display reentrance in their string representations;
    Python dictionaries and lists do not.

  - FeatStructs may *not* be mixed with Python dictionaries and lists
    (e.g., when performing unification).

  - FeatStructs provide a number of useful methods, such as ``walk()``
    and ``cyclic()``, which are not available for Python dicts and lists.

In general, if your feature structures will contain any reentrances,
or if you plan to use them as dictionary keys, it is strongly
recommended that you use full-fledged ``FeatStruct`` objects.
"""

import copy
import re
from functools import total_ordering

from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
    Expression,
    LogicalExpressionException,
    LogicParser,
    SubstituteBindingsI,
    Variable,
)

######################################################################
# Feature Structure
######################################################################


@total_ordering
class FeatStruct(SubstituteBindingsI):
    """
    A mapping from feature identifiers to feature values, where each
    feature value is either a basic value (such as a string or an
    integer), or a nested feature structure.  There are two types of
    feature structure:

      - feature dictionaries, implemented by ``FeatDict``, act like
        Python dictionaries.  Feature identifiers may be strings or
        instances of the ``Feature`` class.
      - feature lists, implemented by ``FeatList``, act like Python
        lists.  Feature identifiers are integers.

    Feature structures may be indexed using either simple feature
    identifiers or 'feature paths.'  A feature path is a sequence
    of feature identifiers that stand for a corresponding sequence of
    indexing operations.  In particular, ``fstruct[(f1,f2,...,fn)]`` is
    equivalent to ``fstruct[f1][f2]...[fn]``.

    Feature structures may contain reentrant feature structures.  A
    "reentrant feature structure" is a single feature structure
    object that can be accessed via multiple feature paths.  Feature
    structures may also be cyclic.  A feature structure is "cyclic"
    if there is any feature path from the feature structure to itself.

    Two feature structures are considered equal if they assign the
    same values to all features, and have the same reentrancies.

    By default, feature structures are mutable.  They may be made
    immutable with the ``freeze()`` method.  Once they have been
    frozen, they may be hashed, and thus used as dictionary keys.
    """

    _frozen = False
    """:ivar: A flag indicating whether this feature structure is
       frozen or not.  Once this flag is set, it should never be
       un-set; and no further modification should be made to this
       feature structure."""

    ##////////////////////////////////////////////////////////////
    # { Constructor
    ##////////////////////////////////////////////////////////////

    def __new__(cls, features=None, **morefeatures):
        """
        Construct and return a new feature structure.  If this
        constructor is called directly, then the returned feature
        structure will be an instance of either the ``FeatDict`` class
        or the ``FeatList`` class.

        :param features: The initial feature values for this feature
            structure:

            - FeatStruct(string) -> FeatStructReader().read(string)
            - FeatStruct(mapping) -> FeatDict(mapping)
            - FeatStruct(sequence) -> FeatList(sequence)
            - FeatStruct() -> FeatDict()
        :param morefeatures: If ``features`` is a mapping or None,
            then ``morefeatures`` provides additional features for the
            ``FeatDict`` constructor.
        """
        # If the FeatStruct constructor is called directly, then decide
        # whether to create a FeatDict or a FeatList, based on the
        # contents of the `features` argument.
        if cls is FeatStruct:
            if features is None:
                return FeatDict.__new__(FeatDict, **morefeatures)
            elif _is_mapping(features):
                return FeatDict.__new__(FeatDict, features, **morefeatures)
            elif morefeatures:
                raise TypeError(
                    "Keyword arguments may only be specified "
                    "if features is None or is a mapping."
                )
            if isinstance(features, str):
                if FeatStructReader._START_FDICT_RE.match(features):
                    return FeatDict.__new__(FeatDict, features, **morefeatures)
                else:
                    return FeatList.__new__(FeatList, features, **morefeatures)
            elif _is_sequence(features):
                return FeatList.__new__(FeatList, features)
            else:
                raise TypeError("Expected string or mapping or sequence")

        # Otherwise, construct the object as normal.
        else:
            return super().__new__(cls, features, **morefeatures)

    ##////////////////////////////////////////////////////////////
    # { Uniform Accessor Methods
    ##////////////////////////////////////////////////////////////
    # These helper functions allow the methods defined by FeatStruct
    # to treat all feature structures as mappings, even if they're
    # really lists.  (Lists are treated as mappings from ints to vals)

    def _keys(self):
        """Return an iterable of the feature identifiers used by this
        FeatStruct."""
        raise NotImplementedError()  # Implemented by subclasses.

    def _values(self):
        """Return an iterable of the feature values directly defined
        by this FeatStruct."""
        raise NotImplementedError()  # Implemented by subclasses.

    def _items(self):
        """Return an iterable of (fid,fval) pairs, where fid is a
        feature identifier and fval is the corresponding feature
        value, for all features defined by this FeatStruct."""
        raise NotImplementedError()  # Implemented by subclasses.

    ##////////////////////////////////////////////////////////////
    # { Equality & Hashing
    ##////////////////////////////////////////////////////////////

    def equal_values(self, other, check_reentrance=False):
        """
        Return True if ``self`` and ``other`` assign the same value to
        to every feature.  In particular, return true if
        ``self[p]==other[p]`` for every feature path *p* such
        that ``self[p]`` or ``other[p]`` is a base value (i.e.,
        not a nested feature structure).

        :param check_reentrance: If True, then also return False if
            there is any difference between the reentrances of ``self``
            and ``other``.
        :note: the ``==`` is equivalent to ``equal_values()`` with
            ``check_reentrance=True``.
        """
        return self._equal(other, check_reentrance, set(), set(), set())

    def __eq__(self, other):
        """
        Return true if ``self`` and ``other`` are both feature structures,
        assign the same values to all features, and contain the same
        reentrances.  I.e., return
        ``self.equal_values(other, check_reentrance=True)``.

        :see: ``equal_values()``
        """
        return self._equal(other, True, set(), set(), set())

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, FeatStruct):
            # raise_unorderable_types("<", self, other)
            # Sometimes feature values can be pure strings,
            # so we need to be able to compare with non-featstructs:
            return self.__class__.__name__ < other.__class__.__name__
        else:
            return len(self) < len(other)

    def __hash__(self):
        """
        If this feature structure is frozen, return its hash value;
        otherwise, raise ``TypeError``.
        """
        if not self._frozen:
            raise TypeError("FeatStructs must be frozen before they " "can be hashed.")
        try:
            return self._hash
        except AttributeError:
            self._hash = self._calculate_hashvalue(set())
            return self._hash

    def _equal(
        self, other, check_reentrance, visited_self, visited_other, visited_pairs
    ):
        """
        Return True iff self and other have equal values.

        :param visited_self: A set containing the ids of all ``self``
            feature structures we've already visited.
        :param visited_other: A set containing the ids of all ``other``
            feature structures we've already visited.
        :param visited_pairs: A set containing ``(selfid, otherid)`` pairs
            for all pairs of feature structures we've already visited.
        """
        # If we're the same object, then we're equal.
        if self is other:
            return True

        # If we have different classes, we're definitely not equal.
        if self.__class__ != other.__class__:
            return False

        # If we define different features, we're definitely not equal.
        # (Perform len test first because it's faster -- we should
        # do profiling to see if this actually helps)
        if len(self) != len(other):
            return False
        if set(self._keys()) != set(other._keys()):
            return False

        # If we're checking reentrance, then any time we revisit a
        # structure, make sure that it was paired with the same
        # feature structure that it is now.  Note: if check_reentrance,
        # then visited_pairs will never contain two pairs whose first
        # values are equal, or two pairs whose second values are equal.
        if check_reentrance:
            if id(self) in visited_self or id(other) in visited_other:
                return (id(self), id(other)) in visited_pairs

        # If we're not checking reentrance, then we still need to deal
        # with cycles.  If we encounter the same (self, other) pair a
        # second time, then we won't learn anything more by examining
        # their children a second time, so just return true.
        else:
            if (id(self), id(other)) in visited_pairs:
                return True

        # Keep track of which nodes we've visited.
        visited_self.add(id(self))
        visited_other.add(id(other))
        visited_pairs.add((id(self), id(other)))

        # Now we have to check all values.  If any of them don't match,
        # then return false.
        for (fname, self_fval) in self._items():
            other_fval = other[fname]
            if isinstance(self_fval, FeatStruct):
                if not self_fval._equal(
                    other_fval,
                    check_reentrance,
                    visited_self,
                    visited_other,
                    visited_pairs,
                ):
                    return False
            else:
                if self_fval != other_fval:
                    return False

        # Everything matched up; return true.
        return True

    def _calculate_hashvalue(self, visited):
        """
        Return a hash value for this feature structure.

        :require: ``self`` must be frozen.
        :param visited: A set containing the ids of all feature
            structures we've already visited while hashing.
        """
        if id(self) in visited:
            return 1
        visited.add(id(self))

        hashval = 5831
        for (fname, fval) in sorted(self._items()):
            hashval *= 37
            hashval += hash(fname)
            hashval *= 37
            if isinstance(fval, FeatStruct):
                hashval += fval._calculate_hashvalue(visited)
            else:
                hashval += hash(fval)
            # Convert to a 32 bit int.
            hashval = int(hashval & 0x7FFFFFFF)
        return hashval

    ##////////////////////////////////////////////////////////////
    # { Freezing
    ##////////////////////////////////////////////////////////////

    #: Error message used by mutating methods when called on a frozen
    #: feature structure.
    _FROZEN_ERROR = "Frozen FeatStructs may not be modified."

    def freeze(self):
        """
        Make this feature structure, and any feature structures it
        contains, immutable.  Note: this method does not attempt to
        'freeze' any feature value that is not a ``FeatStruct``; it
        is recommended that you use only immutable feature values.
        """
        if self._frozen:
            return
        self._freeze(set())

    def frozen(self):
        """
        Return True if this feature structure is immutable.  Feature
        structures can be made immutable with the ``freeze()`` method.
        Immutable feature structures may not be made mutable again,
        but new mutable copies can be produced with the ``copy()`` method.
        """
        return self._frozen

    def _freeze(self, visited):
        """
        Make this feature structure, and any feature structure it
        contains, immutable.

        :param visited: A set containing the ids of all feature
            structures we've already visited while freezing.
        """
        if id(self) in visited:
            return
        visited.add(id(self))
        self._frozen = True
        for (fname, fval) in sorted(self._items()):
            if isinstance(fval, FeatStruct):
                fval._freeze(visited)

    ##////////////////////////////////////////////////////////////
    # { Copying
    ##////////////////////////////////////////////////////////////

    def copy(self, deep=True):
        """
        Return a new copy of ``self``.  The new copy will not be frozen.

        :param deep: If true, create a deep copy; if false, create
            a shallow copy.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return self.__class__(self)

    # Subclasses should define __deepcopy__ to ensure that the new
    # copy will not be frozen.
    def __deepcopy__(self, memo):
        raise NotImplementedError()  # Implemented by subclasses.

    ##////////////////////////////////////////////////////////////
    # { Structural Information
    ##////////////////////////////////////////////////////////////

    def cyclic(self):
        """
        Return True if this feature structure contains itself.
        """
        return self._find_reentrances({})[id(self)]

    def walk(self):
        """
        Return an iterator that generates this feature structure, and
        each feature structure it contains.  Each feature structure will
        be generated exactly once.
        """
        return self._walk(set())

    def _walk(self, visited):
        """
        Return an iterator that generates this feature structure, and
        each feature structure it contains.

        :param visited: A set containing the ids of all feature
            structures we've already visited while freezing.
        """
        raise NotImplementedError()  # Implemented by subclasses.

    def _walk(self, visited):
        if id(self) in visited:
            return
        visited.add(id(self))
        yield self
        for fval in self._values():
            if isinstance(fval, FeatStruct):
                yield from fval._walk(visited)

    # Walk through the feature tree.  The first time we see a feature
    # value, map it to False (not reentrant).  If we see a feature
    # value more than once, then map it to True (reentrant).
    def _find_reentrances(self, reentrances):
        """
        Return a dictionary that maps from the ``id`` of each feature
        structure contained in ``self`` (including ``self``) to a
        boolean value, indicating whether it is reentrant or not.
        """
        if id(self) in reentrances:
            # We've seen it more than once.
            reentrances[id(self)] = True
        else:
            # This is the first time we've seen it.
            reentrances[id(self)] = False

            # Recurse to contained feature structures.
            for fval in self._values():
                if isinstance(fval, FeatStruct):
                    fval._find_reentrances(reentrances)

        return reentrances

    ##////////////////////////////////////////////////////////////
    # { Variables & Bindings
    ##////////////////////////////////////////////////////////////

    def substitute_bindings(self, bindings):
        """:see: ``nltk.featstruct.substitute_bindings()``"""
        return substitute_bindings(self, bindings)

    def retract_bindings(self, bindings):
        """:see: ``nltk.featstruct.retract_bindings()``"""
        return retract_bindings(self, bindings)

    def variables(self):
        """:see: ``nltk.featstruct.find_variables()``"""
        return find_variables(self)

    def rename_variables(self, vars=None, used_vars=(), new_vars=None):
        """:see: ``nltk.featstruct.rename_variables()``"""
        return rename_variables(self, vars, used_vars, new_vars)

    def remove_variables(self):
        """
        Return the feature structure that is obtained by deleting
        any feature whose value is a ``Variable``.

        :rtype: FeatStruct
        """
        return remove_variables(self)

    ##////////////////////////////////////////////////////////////
    # { Unification
    ##////////////////////////////////////////////////////////////

    def unify(self, other, bindings=None, trace=False, fail=None, rename_vars=True):
        return unify(self, other, bindings, trace, fail, rename_vars)

    def subsumes(self, other):
        """
        Return True if ``self`` subsumes ``other``.  I.e., return true
        If unifying ``self`` with ``other`` would result in a feature
        structure equal to ``other``.
        """
        return subsumes(self, other)

    ##////////////////////////////////////////////////////////////
    # { String Representations
    ##////////////////////////////////////////////////////////////

    def __repr__(self):
        """
        Display a single-line representation of this feature structure,
        suitable for embedding in other representations.
        """
        return self._repr(self._find_reentrances({}), {})

    def _repr(self, reentrances, reentrance_ids):
        """
        Return a string representation of this feature structure.

        :param reentrances: A dictionary that maps from the ``id`` of
            each feature value in self, indicating whether that value
            is reentrant or not.
        :param reentrance_ids: A dictionary mapping from each ``id``
            of a feature value to a unique identifier.  This is modified
            by ``repr``: the first time a reentrant feature value is
            displayed, an identifier is added to ``reentrance_ids`` for it.
        """
        raise NotImplementedError()


# Mutation: disable if frozen.
_FROZEN_ERROR = "Frozen FeatStructs may not be modified."
_FROZEN_NOTICE = "\n%sIf self is frozen, raise ValueError."


def _check_frozen(method, indent=""):
    """
    Given a method function, return a new method function that first
    checks if ``self._frozen`` is true; and if so, raises ``ValueError``
    with an appropriate message.  Otherwise, call the method and return
    its result.
    """

    def wrapped(self, *args, **kwargs):
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        else:
            return method(self, *args, **kwargs)

    wrapped.__name__ = method.__name__
    wrapped.__doc__ = (method.__doc__ or "") + (_FROZEN_NOTICE % indent)
    return wrapped


######################################################################
# Feature Dictionary
######################################################################


class FeatDict(FeatStruct, dict):
    """
    A feature structure that acts like a Python dictionary.  I.e., a
    mapping from feature identifiers to feature values, where a feature
    identifier can be a string or a ``Feature``; and where a feature value
    can be either a basic value (such as a string or an integer), or a nested
    feature structure.  A feature identifiers for a ``FeatDict`` is
    sometimes called a "feature name".

    Two feature dicts are considered equal if they assign the same
    values to all features, and have the same reentrances.

    :see: ``FeatStruct`` for information about feature paths, reentrance,
        cyclic feature structures, mutability, freezing, and hashing.
    """

    def __init__(self, features=None, **morefeatures):
        """
        Create a new feature dictionary, with the specified features.

        :param features: The initial value for this feature
            dictionary.  If ``features`` is a ``FeatStruct``, then its
            features are copied (shallow copy).  If ``features`` is a
            dict, then a feature is created for each item, mapping its
            key to its value.  If ``features`` is a string, then it is
            processed using ``FeatStructReader``.  If ``features`` is a list of
            tuples ``(name, val)``, then a feature is created for each tuple.
        :param morefeatures: Additional features for the new feature
            dictionary.  If a feature is listed under both ``features`` and
            ``morefeatures``, then the value from ``morefeatures`` will be
            used.
        """
        if isinstance(features, str):
            FeatStructReader().fromstring(features, self)
            self.update(**morefeatures)
        else:
            # update() checks the types of features.
            self.update(features, **morefeatures)

    # ////////////////////////////////////////////////////////////
    # { Dict methods
    # ////////////////////////////////////////////////////////////
    _INDEX_ERROR = "Expected feature name or path.  Got %r."

    def __getitem__(self, name_or_path):
        """If the feature with the given name or path exists, return
        its value; otherwise, raise ``KeyError``."""
        if isinstance(name_or_path, (str, Feature)):
            return dict.__getitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            try:
                val = self
                for fid in name_or_path:
                    if not isinstance(val, FeatStruct):
                        raise KeyError  # path contains base value
                    val = val[fid]
                return val
            except (KeyError, IndexError) as e:
                raise KeyError(name_or_path) from e
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def get(self, name_or_path, default=None):
        """If the feature with the given name or path exists, return its
        value; otherwise, return ``default``."""
        try:
            return self[name_or_path]
        except KeyError:
            return default

    def __contains__(self, name_or_path):
        """Return true if a feature with the given name or path exists."""
        try:
            self[name_or_path]
            return True
        except KeyError:
            return False

    def has_key(self, name_or_path):
        """Return true if a feature with the given name or path exists."""
        return name_or_path in self

    def __delitem__(self, name_or_path):
        """If the feature with the given name or path exists, delete
        its value; otherwise, raise ``KeyError``."""
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (str, Feature)):
            return dict.__delitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError("The path () can not be set")
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)  # path contains base value
                del parent[name_or_path[-1]]
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def __setitem__(self, name_or_path, value):
        """Set the value for the feature with the given name or path
        to ``value``.  If ``name_or_path`` is an invalid path, raise
        ``KeyError``."""
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (str, Feature)):
            return dict.__setitem__(self, name_or_path, value)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError("The path () can not be set")
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)  # path contains base value
                parent[name_or_path[-1]] = value
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    clear = _check_frozen(dict.clear)
    pop = _check_frozen(dict.pop)
    popitem = _check_frozen(dict.popitem)
    setdefault = _check_frozen(dict.setdefault)

    def update(self, features=None, **morefeatures):
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if features is None:
            items = ()
        elif hasattr(features, "items") and callable(features.items):
            items = features.items()
        elif hasattr(features, "__iter__"):
            items = features
        else:
            raise ValueError("Expected mapping or list of tuples")

        for key, val in items:
            if not isinstance(key, (str, Feature)):
                raise TypeError("Feature names must be strings")
            self[key] = val
        for key, val in morefeatures.items():
            if not isinstance(key, (str, Feature)):
                raise TypeError("Feature names must be strings")
            self[key] = val

    ##////////////////////////////////////////////////////////////
    # { Copying
    ##////////////////////////////////////////////////////////////

    def __deepcopy__(self, memo):
        memo[id(self)] = selfcopy = self.__class__()
        for (key, val) in self._items():
            selfcopy[copy.deepcopy(key, memo)] = copy.deepcopy(val, memo)
        return selfcopy

    ##////////////////////////////////////////////////////////////
    # { Uniform Accessor Methods
    ##////////////////////////////////////////////////////////////

    def _keys(self):
        return self.keys()

    def _values(self):
        return self.values()

    def _items(self):
        return self.items()

    ##////////////////////////////////////////////////////////////
    # { String Representations
    ##////////////////////////////////////////////////////////////

    def __str__(self):
        """
        Display a multi-line representation of this feature dictionary
        as an FVM (feature value matrix).
        """
        return "\n".join(self._str(self._find_reentrances({}), {}))

    def _repr(self, reentrances, reentrance_ids):
        segments = []
        prefix = ""
        suffix = ""

        # If this is the first time we've seen a reentrant structure,
        # then assign it a unique identifier.
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)

        # sorting note: keys are unique strings, so we'll never fall
        # through to comparing values.
        for (fname, fval) in sorted(self.items()):
            display = getattr(fname, "display", None)
            if id(fval) in reentrance_ids:
                segments.append(f"{fname}->({reentrance_ids[id(fval)]})")
            elif (
                display == "prefix" and not prefix and isinstance(fval, (Variable, str))
            ):
                prefix = "%s" % fval
            elif display == "slash" and not suffix:
                if isinstance(fval, Variable):
                    suffix = "/%s" % fval.name
                else:
                    suffix = "/%s" % repr(fval)
            elif isinstance(fval, Variable):
                segments.append(f"{fname}={fval.name}")
            elif fval is True:
                segments.append("+%s" % fname)
            elif fval is False:
                segments.append("-%s" % fname)
            elif isinstance(fval, Expression):
                segments.append(f"{fname}=<{fval}>")
            elif not isinstance(fval, FeatStruct):
                segments.append(f"{fname}={repr(fval)}")
            else:
                fval_repr = fval._repr(reentrances, reentrance_ids)
                segments.append(f"{fname}={fval_repr}")
        # If it's reentrant, then add on an identifier tag.
        if reentrances[id(self)]:
            prefix = f"({reentrance_ids[id(self)]}){prefix}"
        return "{}[{}]{}".format(prefix, ", ".join(segments), suffix)

    def _str(self, reentrances, reentrance_ids):
        """
        :return: A list of lines composing a string representation of
            this feature dictionary.
        :param reentrances: A dictionary that maps from the ``id`` of
            each feature value in self, indicating whether that value
            is reentrant or not.
        :param reentrance_ids: A dictionary mapping from each ``id``
            of a feature value to a unique identifier.  This is modified
            by ``repr``: the first time a reentrant feature value is
            displayed, an identifier is added to ``reentrance_ids`` for
            it.
        """
        # If this is the first time we've seen a reentrant structure,
        # then tack on an id string.
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)

        # Special case: empty feature dict.
        if len(self) == 0:
            if reentrances[id(self)]:
                return ["(%s) []" % reentrance_ids[id(self)]]
            else:
                return ["[]"]

        # What's the longest feature name?  Use this to align names.
        maxfnamelen = max(len("%s" % k) for k in self.keys())

        lines = []
        # sorting note: keys are unique strings, so we'll never fall
        # through to comparing values.
        for (fname, fval) in sorted(self.items()):
            fname = ("%s" % fname).ljust(maxfnamelen)
            if isinstance(fval, Variable):
                lines.append(f"{fname} = {fval.name}")

            elif isinstance(fval, Expression):
                lines.append(f"{fname} = <{fval}>")

            elif isinstance(fval, FeatList):
                fval_repr = fval._repr(reentrances, reentrance_ids)
                lines.append(f"{fname} = {repr(fval_repr)}")

            elif not isinstance(fval, FeatDict):
                # It's not a nested feature structure -- just print it.
                lines.append(f"{fname} = {repr(fval)}")

            elif id(fval) in reentrance_ids:
                # It's a feature structure we've seen before -- print
                # the reentrance id.
                lines.append(f"{fname} -> ({reentrance_ids[id(fval)]})")

            else:
                # It's a new feature structure.  Separate it from
                # other values by a blank line.
                if lines and lines[-1] != "":
                    lines.append("")

                # Recursively print the feature's value (fval).
                fval_lines = fval._str(reentrances, reentrance_ids)

                # Indent each line to make room for fname.
                fval_lines = [(" " * (maxfnamelen + 3)) + l for l in fval_lines]

                # Pick which line we'll display fname on, & splice it in.
                nameline = (len(fval_lines) - 1) // 2
                fval_lines[nameline] = (
                    fname + " =" + fval_lines[nameline][maxfnamelen + 2 :]
                )

                # Add the feature structure to the output.
                lines += fval_lines

                # Separate FeatStructs by a blank line.
                lines.append("")

        # Get rid of any excess blank lines.
        if lines[-1] == "":
            lines.pop()

        # Add brackets around everything.
        maxlen = max(len(line) for line in lines)
        lines = ["[ {}{} ]".format(line, " " * (maxlen - len(line))) for line in lines]

        # If it's reentrant, then add on an identifier tag.
        if reentrances[id(self)]:
            idstr = "(%s) " % reentrance_ids[id(self)]
            lines = [(" " * len(idstr)) + l for l in lines]
            idline = (len(lines) - 1) // 2
            lines[idline] = idstr + lines[idline][len(idstr) :]

        return lines


######################################################################
# Feature List
######################################################################


class FeatList(FeatStruct, list):
    """
    A list of feature values, where each feature value is either a
    basic value (such as a string or an integer), or a nested feature
    structure.

    Feature lists may contain reentrant feature values.  A "reentrant
    feature value" is a single feature value that can be accessed via
    multiple feature paths.  Feature lists may also be cyclic.

    Two feature lists are considered equal if they assign the same
    values to all features, and have the same reentrances.

    :see: ``FeatStruct`` for information about feature paths, reentrance,
        cyclic feature structures, mutability, freezing, and hashing.
    """

    def __init__(self, features=()):
        """
        Create a new feature list, with the specified features.

        :param features: The initial list of features for this feature
            list.  If ``features`` is a string, then it is paresd using
            ``FeatStructReader``.  Otherwise, it should be a sequence
            of basic values and nested feature structures.
        """
        if isinstance(features, str):
            FeatStructReader().fromstring(features, self)
        else:
            list.__init__(self, features)

    # ////////////////////////////////////////////////////////////
    # { List methods
    # ////////////////////////////////////////////////////////////
    _INDEX_ERROR = "Expected int or feature path.  Got %r."

    def __getitem__(self, name_or_path):
        if isinstance(name_or_path, int):
            return list.__getitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            try:
                val = self
                for fid in name_or_path:
                    if not isinstance(val, FeatStruct):
                        raise KeyError  # path contains base value
                    val = val[fid]
                return val
            except (KeyError, IndexError) as e:
                raise KeyError(name_or_path) from e
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def __delitem__(self, name_or_path):
        """If the feature with the given name or path exists, delete
        its value; otherwise, raise ``KeyError``."""
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (int, slice)):
            return list.__delitem__(self, name_or_path)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError("The path () can not be set")
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)  # path contains base value
                del parent[name_or_path[-1]]
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    def __setitem__(self, name_or_path, value):
        """Set the value for the feature with the given name or path
        to ``value``.  If ``name_or_path`` is an invalid path, raise
        ``KeyError``."""
        if self._frozen:
            raise ValueError(_FROZEN_ERROR)
        if isinstance(name_or_path, (int, slice)):
            return list.__setitem__(self, name_or_path, value)
        elif isinstance(name_or_path, tuple):
            if len(name_or_path) == 0:
                raise ValueError("The path () can not be set")
            else:
                parent = self[name_or_path[:-1]]
                if not isinstance(parent, FeatStruct):
                    raise KeyError(name_or_path)  # path contains base value
                parent[name_or_path[-1]] = value
        else:
            raise TypeError(self._INDEX_ERROR % name_or_path)

    #    __delslice__ = _check_frozen(list.__delslice__, '               ')
    #    __setslice__ = _check_frozen(list.__setslice__, '               ')
    __iadd__ = _check_frozen(list.__iadd__)
    __imul__ = _check_frozen(list.__imul__)
    append = _check_frozen(list.append)
    extend = _check_frozen(list.extend)
    insert = _check_frozen(list.insert)
    pop = _check_frozen(list.pop)
    remove = _check_frozen(list.remove)
    reverse = _check_frozen(list.reverse)
    sort = _check_frozen(list.sort)

    ##////////////////////////////////////////////////////////////
    # { Copying
    ##////////////////////////////////////////////////////////////

    def __deepcopy__(self, memo):
        memo[id(self)] = selfcopy = self.__class__()
        selfcopy.extend(copy.deepcopy(fval, memo) for fval in self)
        return selfcopy

    ##////////////////////////////////////////////////////////////
    # { Uniform Accessor Methods
    ##////////////////////////////////////////////////////////////

    def _keys(self):
        return list(range(len(self)))

    def _values(self):
        return self

    def _items(self):
        return enumerate(self)

    ##////////////////////////////////////////////////////////////
    # { String Representations
    ##////////////////////////////////////////////////////////////

    # Special handling for: reentrances, variables, expressions.
    def _repr(self, reentrances, reentrance_ids):
        # If this is the first time we've seen a reentrant structure,
        # then assign it a unique identifier.
        if reentrances[id(self)]:
            assert id(self) not in reentrance_ids
            reentrance_ids[id(self)] = repr(len(reentrance_ids) + 1)
            prefix = "(%s)" % reentrance_ids[id(self)]
        else:
            prefix = ""

        segments = []
        for fval in self:
            if id(fval) in reentrance_ids:
                segments.append("->(%s)" % reentrance_ids[id(fval)])
            elif isinstance(fval, Variable):
                segments.append(fval.name)
            elif isinstance(fval, Expression):
                segments.append("%s" % fval)
            elif isinstance(fval, FeatStruct):
                segments.append(fval._repr(reentrances, reentrance_ids))
            else:
                segments.append("%s" % repr(fval))

        return "{}[{}]".format(prefix, ", ".join(segments))


######################################################################
# Variables & Bindings
######################################################################


def substitute_bindings(fstruct, bindings, fs_class="default"):
    """
    Return the feature structure that is obtained by replacing each
    variable bound by ``bindings`` with its binding.  If a variable is
    aliased to a bound variable, then it will be replaced by that
    variable's value.  If a variable is aliased to an unbound
    variable, then it will be replaced by that variable.

    :type bindings: dict(Variable -> any)
    :param bindings: A dictionary mapping from variables to values.
    """
    if fs_class == "default":
        fs_class = _default_fs_class(fstruct)
    fstruct = copy.deepcopy(fstruct)
    _substitute_bindings(fstruct, bindings, fs_class, set())
    return fstruct


def _substitute_bindings(fstruct, bindings, fs_class, visited):
    # Visit each node only once:
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))

    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError("Expected mapping or sequence")
    for (fname, fval) in items:
        while isinstance(fval, Variable) and fval in bindings:
            fval = fstruct[fname] = bindings[fval]
        if isinstance(fval, fs_class):
            _substitute_bindings(fval, bindings, fs_class, visited)
        elif isinstance(fval, SubstituteBindingsI):
            fstruct[fname] = fval.substitute_bindings(bindings)


def retract_bindings(fstruct, bindings, fs_class="default"):
    """
    Return the feature structure that is obtained by replacing each
    feature structure value that is bound by ``bindings`` with the
    variable that binds it.  A feature structure value must be
    identical to a bound value (i.e., have equal id) to be replaced.

    ``bindings`` is modified to point to this new feature structure,
    rather than the original feature structure.  Feature structure
    values in ``bindings`` may be modified if they are contained in
    ``fstruct``.
    """
    if fs_class == "default":
        fs_class = _default_fs_class(fstruct)
    (fstruct, new_bindings) = copy.deepcopy((fstruct, bindings))
    bindings.update(new_bindings)
    inv_bindings = {id(val): var for (var, val) in bindings.items()}
    _retract_bindings(fstruct, inv_bindings, fs_class, set())
    return fstruct


def _retract_bindings(fstruct, inv_bindings, fs_class, visited):
    # Visit each node only once:
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))

    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError("Expected mapping or sequence")
    for (fname, fval) in items:
        if isinstance(fval, fs_class):
            if id(fval) in inv_bindings:
                fstruct[fname] = inv_bindings[id(fval)]
            _retract_bindings(fval, inv_bindings, fs_class, visited)


def find_variables(fstruct, fs_class="default"):
    """
    :return: The set of variables used by this feature structure.
    :rtype: set(Variable)
    """
    if fs_class == "default":
        fs_class = _default_fs_class(fstruct)
    return _variables(fstruct, set(), fs_class, set())


def _variables(fstruct, vars, fs_class, visited):
    # Visit each node only once:
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError("Expected mapping or sequence")
    for (fname, fval) in items:
        if isinstance(fval, Variable):
            vars.add(fval)
        elif isinstance(fval, fs_class):
            _variables(fval, vars, fs_class, visited)
        elif isinstance(fval, SubstituteBindingsI):
            vars.update(fval.variables())
    return vars


def rename_variables(
    fstruct, vars=None, used_vars=(), new_vars=None, fs_class="default"
):
    """
    Return the feature structure that is obtained by replacing
    any of this feature structure's variables that are in ``vars``
    with new variables.  The names for these new variables will be
    names that are not used by any variable in ``vars``, or in
    ``used_vars``, or in this feature structure.

    :type vars: set
    :param vars: The set of variables that should be renamed.
        If not specified, ``find_variables(fstruct)`` is used; i.e., all
        variables will be given new names.
    :type used_vars: set
    :param used_vars: A set of variables whose names should not be
        used by the new variables.
    :type new_vars: dict(Variable -> Variable)
    :param new_vars: A dictionary that is used to hold the mapping
        from old variables to new variables.  For each variable *v*
        in this feature structure:

        - If ``new_vars`` maps *v* to *v'*, then *v* will be
          replaced by *v'*.
        - If ``new_vars`` does not contain *v*, but ``vars``
          does contain *v*, then a new entry will be added to
          ``new_vars``, mapping *v* to the new variable that is used
          to replace it.

    To consistently rename the variables in a set of feature
    structures, simply apply rename_variables to each one, using
    the same dictionary:

        >>> from nltk.featstruct import FeatStruct
        >>> fstruct1 = FeatStruct('[subj=[agr=[gender=?y]], obj=[agr=[gender=?y]]]')
        >>> fstruct2 = FeatStruct('[subj=[agr=[number=?z,gender=?y]], obj=[agr=[number=?z,gender=?y]]]')
        >>> new_vars = {}  # Maps old vars to alpha-renamed vars
        >>> fstruct1.rename_variables(new_vars=new_vars)
        [obj=[agr=[gender=?y2]], subj=[agr=[gender=?y2]]]
        >>> fstruct2.rename_variables(new_vars=new_vars)
        [obj=[agr=[gender=?y2, number=?z2]], subj=[agr=[gender=?y2, number=?z2]]]

    If new_vars is not specified, then an empty dictionary is used.
    """
    if fs_class == "default":
        fs_class = _default_fs_class(fstruct)

    # Default values:
    if new_vars is None:
        new_vars = {}
    if vars is None:
        vars = find_variables(fstruct, fs_class)
    else:
        vars = set(vars)

    # Add our own variables to used_vars.
    used_vars = find_variables(fstruct, fs_class).union(used_vars)

    # Copy ourselves, and rename variables in the copy.
    return _rename_variables(
        copy.deepcopy(fstruct), vars, used_vars, new_vars, fs_class, set()
    )


def _rename_variables(fstruct, vars, used_vars, new_vars, fs_class, visited):
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))
    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError("Expected mapping or sequence")
    for (fname, fval) in items:
        if isinstance(fval, Variable):
            # If it's in new_vars, then rebind it.
            if fval in new_vars:
                fstruct[fname] = new_vars[fval]
            # If it's in vars, pick a new name for it.
            elif fval in vars:
                new_vars[fval] = _rename_variable(fval, used_vars)
                fstruct[fname] = new_vars[fval]
                used_vars.add(new_vars[fval])
        elif isinstance(fval, fs_class):
            _rename_variables(fval, vars, used_vars, new_vars, fs_class, visited)
        elif isinstance(fval, SubstituteBindingsI):
            # Pick new names for any variables in `vars`
            for var in fval.variables():
                if var in vars and var not in new_vars:
                    new_vars[var] = _rename_variable(var, used_vars)
                    used_vars.add(new_vars[var])
            # Replace all variables in `new_vars`.
            fstruct[fname] = fval.substitute_bindings(new_vars)
    return fstruct


def _rename_variable(var, used_vars):
    name, n = re.sub(r"\d+$", "", var.name), 2
    if not name:
        name = "?"
    while Variable(f"{name}{n}") in used_vars:
        n += 1
    return Variable(f"{name}{n}")


def remove_variables(fstruct, fs_class="default"):
    """
    :rtype: FeatStruct
    :return: The feature structure that is obtained by deleting
        all features whose values are ``Variables``.
    """
    if fs_class == "default":
        fs_class = _default_fs_class(fstruct)
    return _remove_variables(copy.deepcopy(fstruct), fs_class, set())


def _remove_variables(fstruct, fs_class, visited):
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))

    if _is_mapping(fstruct):
        items = list(fstruct.items())
    elif _is_sequence(fstruct):
        items = list(enumerate(fstruct))
    else:
        raise ValueError("Expected mapping or sequence")

    for (fname, fval) in items:
        if isinstance(fval, Variable):
            del fstruct[fname]
        elif isinstance(fval, fs_class):
            _remove_variables(fval, fs_class, visited)
    return fstruct


######################################################################
# Unification
######################################################################


class _UnificationFailure:
    def __repr__(self):
        return "nltk.featstruct.UnificationFailure"


UnificationFailure = _UnificationFailure()
"""A unique value used to indicate unification failure.  It can be
   returned by ``Feature.unify_base_values()`` or by custom ``fail()``
   functions to indicate that unificaiton should fail."""


# The basic unification algorithm:
#   1. Make copies of self and other (preserving reentrance)
#   2. Destructively unify self and other
#   3. Apply forward pointers, to preserve reentrance.
#   4. Replace bound variables with their values.
def unify(
    fstruct1,
    fstruct2,
    bindings=None,
    trace=False,
    fail=None,
    rename_vars=True,
    fs_class="default",
):
    """
    Unify ``fstruct1`` with ``fstruct2``, and return the resulting feature
    structure.  This unified feature structure is the minimal
    feature structure that contains all feature value assignments from both
    ``fstruct1`` and ``fstruct2``, and that preserves all reentrancies.

    If no such feature structure exists (because ``fstruct1`` and
    ``fstruct2`` specify incompatible values for some feature), then
    unification fails, and ``unify`` returns None.

    Bound variables are replaced by their values.  Aliased
    variables are replaced by their representative variable
    (if unbound) or the value of their representative variable
    (if bound).  I.e., if variable *v* is in ``bindings``,
    then *v* is replaced by ``bindings[v]``.  This will
    be repeated until the variable is replaced by an unbound
    variable or a non-variable value.

    Unbound variables are bound when they are unified with
    values; and aliased when they are unified with variables.
    I.e., if variable *v* is not in ``bindings``, and is
    unified with a variable or value *x*, then
    ``bindings[v]`` is set to *x*.

    If ``bindings`` is unspecified, then all variables are
    assumed to be unbound.  I.e., ``bindings`` defaults to an
    empty dict.

        >>> from nltk.featstruct import FeatStruct
        >>> FeatStruct('[a=?x]').unify(FeatStruct('[b=?x]'))
        [a=?x, b=?x2]

    :type bindings: dict(Variable -> any)
    :param bindings: A set of variable bindings to be used and
        updated during unification.
    :type trace: bool
    :param trace: If true, generate trace output.
    :type rename_vars: bool
    :param rename_vars: If True, then rename any variables in
        ``fstruct2`` that are also used in ``fstruct1``, in order to
        avoid collisions on variable names.
    """
    # Decide which class(es) will be treated as feature structures,
    # for the purposes of unification.
    if fs_class == "default":
        fs_class = _default_fs_class(fstruct1)
        if _default_fs_class(fstruct2) != fs_class:
            raise ValueError(
                "Mixing FeatStruct objects with Python "
                "dicts and lists is not supported."
            )
    assert isinstance(fstruct1, fs_class)
    assert isinstance(fstruct2, fs_class)

    # If bindings are unspecified, use an empty set of bindings.
    user_bindings = bindings is not None
    if bindings is None:
        bindings = {}

    # Make copies of fstruct1 and fstruct2 (since the unification
    # algorithm is destructive). Do it all at once, to preserve
    # reentrance links between fstruct1 and fstruct2.  Copy bindings
    # as well, in case there are any bound vars that contain parts
    # of fstruct1 or fstruct2.
    (fstruct1copy, fstruct2copy, bindings_copy) = copy.deepcopy(
        (fstruct1, fstruct2, bindings)
    )

    # Copy the bindings back to the original bindings dict.
    bindings.update(bindings_copy)

    if rename_vars:
        vars1 = find_variables(fstruct1copy, fs_class)
        vars2 = find_variables(fstruct2copy, fs_class)
        _rename_variables(fstruct2copy, vars1, vars2, {}, fs_class, set())

    # Do the actual unification.  If it fails, return None.
    forward = {}
    if trace:
        _trace_unify_start((), fstruct1copy, fstruct2copy)
    try:
        result = _destructively_unify(
            fstruct1copy, fstruct2copy, bindings, forward, trace, fail, fs_class, ()
        )
    except _UnificationFailureError:
        return None

    # _destructively_unify might return UnificationFailure, e.g. if we
    # tried to unify a mapping with a sequence.
    if result is UnificationFailure:
        if fail is None:
            return None
        else:
            return fail(fstruct1copy, fstruct2copy, ())

    # Replace any feature structure that has a forward pointer
    # with the target of its forward pointer.
    result = _apply_forwards(result, forward, fs_class, set())
    if user_bindings:
        _apply_forwards_to_bindings(forward, bindings)

    # Replace bound vars with values.
    _resolve_aliases(bindings)
    _substitute_bindings(result, bindings, fs_class, set())

    # Return the result.
    if trace:
        _trace_unify_succeed((), result)
    if trace:
        _trace_bindings((), bindings)
    return result


class _UnificationFailureError(Exception):
    """An exception that is used by ``_destructively_unify`` to abort
    unification when a failure is encountered."""


def _destructively_unify(
    fstruct1, fstruct2, bindings, forward, trace, fail, fs_class, path
):
    """
    Attempt to unify ``fstruct1`` and ``fstruct2`` by modifying them
    in-place.  If the unification succeeds, then ``fstruct1`` will
    contain the unified value, the value of ``fstruct2`` is undefined,
    and forward[id(fstruct2)] is set to fstruct1.  If the unification
    fails, then a _UnificationFailureError is raised, and the
    values of ``fstruct1`` and ``fstruct2`` are undefined.

    :param bindings: A dictionary mapping variables to values.
    :param forward: A dictionary mapping feature structures ids
        to replacement structures.  When two feature structures
        are merged, a mapping from one to the other will be added
        to the forward dictionary; and changes will be made only
        to the target of the forward dictionary.
        ``_destructively_unify`` will always 'follow' any links
        in the forward dictionary for fstruct1 and fstruct2 before
        actually unifying them.
    :param trace: If true, generate trace output
    :param path: The feature path that led us to this unification
        step.  Used for trace output.
    """
    # If fstruct1 is already identical to fstruct2, we're done.
    # Note: this, together with the forward pointers, ensures
    # that unification will terminate even for cyclic structures.
    if fstruct1 is fstruct2:
        if trace:
            _trace_unify_identity(path, fstruct1)
        return fstruct1

    # Set fstruct2's forward pointer to point to fstruct1; this makes
    # fstruct1 the canonical copy for fstruct2.  Note that we need to
    # do this before we recurse into any child structures, in case
    # they're cyclic.
    forward[id(fstruct2)] = fstruct1

    # Unifying two mappings:
    if _is_mapping(fstruct1) and _is_mapping(fstruct2):
        for fname in fstruct1:
            if getattr(fname, "default", None) is not None:
                fstruct2.setdefault(fname, fname.default)
        for fname in fstruct2:
            if getattr(fname, "default", None) is not None:
                fstruct1.setdefault(fname, fname.default)

        # Unify any values that are defined in both fstruct1 and
        # fstruct2.  Copy any values that are defined in fstruct2 but
        # not in fstruct1 to fstruct1.  Note: sorting fstruct2's
        # features isn't actually necessary; but we do it to give
        # deterministic behavior, e.g. for tracing.
        for fname, fval2 in sorted(fstruct2.items()):
            if fname in fstruct1:
                fstruct1[fname] = _unify_feature_values(
                    fname,
                    fstruct1[fname],
                    fval2,
                    bindings,
                    forward,
                    trace,
                    fail,
                    fs_class,
                    path + (fname,),
                )
            else:
                fstruct1[fname] = fval2

        return fstruct1  # Contains the unified value.

    # Unifying two sequences:
    elif _is_sequence(fstruct1) and _is_sequence(fstruct2):
        # If the lengths don't match, fail.
        if len(fstruct1) != len(fstruct2):
            return UnificationFailure

        # Unify corresponding values in fstruct1 and fstruct2.
        for findex in range(len(fstruct1)):
            fstruct1[findex] = _unify_feature_values(
                findex,
                fstruct1[findex],
                fstruct2[findex],
                bindings,
                forward,
                trace,
                fail,
                fs_class,
                path + (findex,),
            )

        return fstruct1  # Contains the unified value.

    # Unifying sequence & mapping: fail.  The failure function
    # doesn't get a chance to recover in this case.
    elif (_is_sequence(fstruct1) or _is_mapping(fstruct1)) and (
        _is_sequence(fstruct2) or _is_mapping(fstruct2)
    ):
        return UnificationFailure

    # Unifying anything else: not allowed!
    raise TypeError("Expected mappings or sequences")


def _unify_feature_values(
    fname, fval1, fval2, bindings, forward, trace, fail, fs_class, fpath
):
    """
    Attempt to unify ``fval1`` and and ``fval2``, and return the
    resulting unified value.  The method of unification will depend on
    the types of ``fval1`` and ``fval2``:

      1. If they're both feature structures, then destructively
         unify them (see ``_destructively_unify()``.
      2. If they're both unbound variables, then alias one variable
         to the other (by setting bindings[v2]=v1).
      3. If one is an unbound variable, and the other is a value,
         then bind the unbound variable to the value.
      4. If one is a feature structure, and the other is a base value,
         then fail.
      5. If they're both base values, then unify them.  By default,
         this will succeed if they are equal, and fail otherwise.
    """
    if trace:
        _trace_unify_start(fpath, fval1, fval2)

    # Look up the "canonical" copy of fval1 and fval2
    while id(fval1) in forward:
        fval1 = forward[id(fval1)]
    while id(fval2) in forward:
        fval2 = forward[id(fval2)]

    # If fval1 or fval2 is a bound variable, then
    # replace it by the variable's bound value.  This
    # includes aliased variables, which are encoded as
    # variables bound to other variables.
    fvar1 = fvar2 = None
    while isinstance(fval1, Variable) and fval1 in bindings:
        fvar1 = fval1
        fval1 = bindings[fval1]
    while isinstance(fval2, Variable) and fval2 in bindings:
        fvar2 = fval2
        fval2 = bindings[fval2]

    # Case 1: Two feature structures (recursive case)
    if isinstance(fval1, fs_class) and isinstance(fval2, fs_class):
        result = _destructively_unify(
            fval1, fval2, bindings, forward, trace, fail, fs_class, fpath
        )

    # Case 2: Two unbound variables (create alias)
    elif isinstance(fval1, Variable) and isinstance(fval2, Variable):
        if fval1 != fval2:
            bindings[fval2] = fval1
        result = fval1

    # Case 3: An unbound variable and a value (bind)
    elif isinstance(fval1, Variable):
        bindings[fval1] = fval2
        result = fval1
    elif isinstance(fval2, Variable):
        bindings[fval2] = fval1
        result = fval2

    # Case 4: A feature structure & a base value (fail)
    elif isinstance(fval1, fs_class) or isinstance(fval2, fs_class):
        result = UnificationFailure

    # Case 5: Two base values
    else:
        # Case 5a: Feature defines a custom unification method for base values
        if isinstance(fname, Feature):
            result = fname.unify_base_values(fval1, fval2, bindings)
        # Case 5b: Feature value defines custom unification method
        elif isinstance(fval1, CustomFeatureValue):
            result = fval1.unify(fval2)
            # Sanity check: unify value should be symmetric
            if isinstance(fval2, CustomFeatureValue) and result != fval2.unify(fval1):
                raise AssertionError(
                    "CustomFeatureValue objects %r and %r disagree "
                    "about unification value: %r vs. %r"
                    % (fval1, fval2, result, fval2.unify(fval1))
                )
        elif isinstance(fval2, CustomFeatureValue):
            result = fval2.unify(fval1)
        # Case 5c: Simple values -- check if they're equal.
        else:
            if fval1 == fval2:
                result = fval1
            else:
                result = UnificationFailure

        # If either value was a bound variable, then update the
        # bindings.  (This is really only necessary if fname is a
        # Feature or if either value is a CustomFeatureValue.)
        if result is not UnificationFailure:
            if fvar1 is not None:
                bindings[fvar1] = result
                result = fvar1
            if fvar2 is not None and fvar2 != fvar1:
                bindings[fvar2] = result
                result = fvar2

    # If we unification failed, call the failure function; it
    # might decide to continue anyway.
    if result is UnificationFailure:
        if fail is not None:
            result = fail(fval1, fval2, fpath)
        if trace:
            _trace_unify_fail(fpath[:-1], result)
        if result is UnificationFailure:
            raise _UnificationFailureError

    # Normalize the result.
    if isinstance(result, fs_class):
        result = _apply_forwards(result, forward, fs_class, set())

    if trace:
        _trace_unify_succeed(fpath, result)
    if trace and isinstance(result, fs_class):
        _trace_bindings(fpath, bindings)

    return result


def _apply_forwards_to_bindings(forward, bindings):
    """
    Replace any feature structure that has a forward pointer with
    the target of its forward pointer (to preserve reentrancy).
    """
    for (var, value) in bindings.items():
        while id(value) in forward:
            value = forward[id(value)]
        bindings[var] = value


def _apply_forwards(fstruct, forward, fs_class, visited):
    """
    Replace any feature structure that has a forward pointer with
    the target of its forward pointer (to preserve reentrancy).
    """
    # Follow our own forwards pointers (if any)
    while id(fstruct) in forward:
        fstruct = forward[id(fstruct)]

    # Visit each node only once:
    if id(fstruct) in visited:
        return
    visited.add(id(fstruct))

    if _is_mapping(fstruct):
        items = fstruct.items()
    elif _is_sequence(fstruct):
        items = enumerate(fstruct)
    else:
        raise ValueError("Expected mapping or sequence")
    for fname, fval in items:
        if isinstance(fval, fs_class):
            # Replace w/ forwarded value.
            while id(fval) in forward:
                fval = forward[id(fval)]
            fstruct[fname] = fval
            # Recurse to child.
            _apply_forwards(fval, forward, fs_class, visited)

    return fstruct


def _resolve_aliases(bindings):
    """
    Replace any bound aliased vars with their binding; and replace
    any unbound aliased vars with their representative var.
    """
    for (var, value) in bindings.items():
        while isinstance(value, Variable) and value in bindings:
            value = bindings[var] = bindings[value]


def _trace_unify_start(path, fval1, fval2):
    if path == ():
        print("\nUnification trace:")
    else:
        fullname = ".".join("%s" % n for n in path)
        print("  " + "|   " * (len(path) - 1) + "|")
        print("  " + "|   " * (len(path) - 1) + "| Unify feature: %s" % fullname)
    print("  " + "|   " * len(path) + " / " + _trace_valrepr(fval1))
    print("  " + "|   " * len(path) + "|\\ " + _trace_valrepr(fval2))


def _trace_unify_identity(path, fval1):
    print("  " + "|   " * len(path) + "|")
    print("  " + "|   " * len(path) + "| (identical objects)")
    print("  " + "|   " * len(path) + "|")
    print("  " + "|   " * len(path) + "+-->" + repr(fval1))


def _trace_unify_fail(path, result):
    if result is UnificationFailure:
        resume = ""
    else:
        resume = " (nonfatal)"
    print("  " + "|   " * len(path) + "|   |")
    print("  " + "X   " * len(path) + "X   X <-- FAIL" + resume)


def _trace_unify_succeed(path, fval1):
    # Print the result.
    print("  " + "|   " * len(path) + "|")
    print("  " + "|   " * len(path) + "+-->" + repr(fval1))


def _trace_bindings(path, bindings):
    # Print the bindings (if any).
    if len(bindings) > 0:
        binditems = sorted(bindings.items(), key=lambda v: v[0].name)
        bindstr = "{%s}" % ", ".join(
            f"{var}: {_trace_valrepr(val)}" for (var, val) in binditems
        )
        print("  " + "|   " * len(path) + "    Bindings: " + bindstr)


def _trace_valrepr(val):
    if isinstance(val, Variable):
        return "%s" % val
    else:
        return "%s" % repr(val)


def subsumes(fstruct1, fstruct2):
    """
    Return True if ``fstruct1`` subsumes ``fstruct2``.  I.e., return
    true if unifying ``fstruct1`` with ``fstruct2`` would result in a
    feature structure equal to ``fstruct2.``

    :rtype: bool
    """
    return fstruct2 == unify(fstruct1, fstruct2)


def conflicts(fstruct1, fstruct2, trace=0):
    """
    Return a list of the feature paths of all features which are
    assigned incompatible values by ``fstruct1`` and ``fstruct2``.

    :rtype: list(tuple)
    """
    conflict_list = []

    def add_conflict(fval1, fval2, path):
        conflict_list.append(path)
        return fval1

    unify(fstruct1, fstruct2, fail=add_conflict, trace=trace)
    return conflict_list


######################################################################
# Helper Functions
######################################################################


def _is_mapping(v):
    return hasattr(v, "__contains__") and hasattr(v, "keys")


def _is_sequence(v):
    return hasattr(v, "__iter__") and hasattr(v, "__len__") and not isinstance(v, str)


def _default_fs_class(obj):
    if isinstance(obj, FeatStruct):
        return FeatStruct
    if isinstance(obj, (dict, list)):
        return (dict, list)
    else:
        raise ValueError(
            "To unify objects of type %s, you must specify "
            "fs_class explicitly." % obj.__class__.__name__
        )


######################################################################
# FeatureValueSet & FeatureValueTuple
######################################################################


class SubstituteBindingsSequence(SubstituteBindingsI):
    """
    A mixin class for sequence classes that distributes variables() and
    substitute_bindings() over the object's elements.
    """

    def variables(self):
        return [elt for elt in self if isinstance(elt, Variable)] + sum(
            (
                list(elt.variables())
                for elt in self
                if isinstance(elt, SubstituteBindingsI)
            ),
            [],
        )

    def substitute_bindings(self, bindings):
        return self.__class__([self.subst(v, bindings) for v in self])

    def subst(self, v, bindings):
        if isinstance(v, SubstituteBindingsI):
            return v.substitute_bindings(bindings)
        else:
            return bindings.get(v, v)


class FeatureValueTuple(SubstituteBindingsSequence, tuple):
    """
    A base feature value that is a tuple of other base feature values.
    FeatureValueTuple implements ``SubstituteBindingsI``, so it any
    variable substitutions will be propagated to the elements
    contained by the set.  A ``FeatureValueTuple`` is immutable.
    """

    def __repr__(self):  # [xx] really use %s here?
        if len(self) == 0:
            return "()"
        return "(%s)" % ", ".join(f"{b}" for b in self)


class FeatureValueSet(SubstituteBindingsSequence, frozenset):
    """
    A base feature value that is a set of other base feature values.
    FeatureValueSet implements ``SubstituteBindingsI``, so it any
    variable substitutions will be propagated to the elements
    contained by the set.  A ``FeatureValueSet`` is immutable.
    """

    def __repr__(self):  # [xx] really use %s here?
        if len(self) == 0:
            return "{/}"  # distinguish from dict.
        # n.b., we sort the string reprs of our elements, to ensure
        # that our own repr is deterministic.
        return "{%s}" % ", ".join(sorted(f"{b}" for b in self))

    __str__ = __repr__


class FeatureValueUnion(SubstituteBindingsSequence, frozenset):
    """
    A base feature value that represents the union of two or more
    ``FeatureValueSet`` or ``Variable``.
    """

    def __new__(cls, values):
        # If values contains FeatureValueUnions, then collapse them.
        values = _flatten(values, FeatureValueUnion)

        # If the resulting list contains no variables, then
        # use a simple FeatureValueSet instead.
        if sum(isinstance(v, Variable) for v in values) == 0:
            values = _flatten(values, FeatureValueSet)
            return FeatureValueSet(values)

        # If we contain a single variable, return that variable.
        if len(values) == 1:
            return list(values)[0]

        # Otherwise, build the FeatureValueUnion.
        return frozenset.__new__(cls, values)

    def __repr__(self):
        # n.b., we sort the string reprs of our elements, to ensure
        # that our own repr is deterministic.  also, note that len(self)
        # is guaranteed to be 2 or more.
        return "{%s}" % "+".join(sorted(f"{b}" for b in self))


class FeatureValueConcat(SubstituteBindingsSequence, tuple):
    """
    A base feature value that represents the concatenation of two or
    more ``FeatureValueTuple`` or ``Variable``.
    """

    def __new__(cls, values):
        # If values contains FeatureValueConcats, then collapse them.
        values = _flatten(values, FeatureValueConcat)

        # If the resulting list contains no variables, then
        # use a simple FeatureValueTuple instead.
        if sum(isinstance(v, Variable) for v in values) == 0:
            values = _flatten(values, FeatureValueTuple)
            return FeatureValueTuple(values)

        # If we contain a single variable, return that variable.
        if len(values) == 1:
            return list(values)[0]

        # Otherwise, build the FeatureValueConcat.
        return tuple.__new__(cls, values)

    def __repr__(self):
        # n.b.: len(self) is guaranteed to be 2 or more.
        return "(%s)" % "+".join(f"{b}" for b in self)


def _flatten(lst, cls):
    """
    Helper function -- return a copy of list, with all elements of
    type ``cls`` spliced in rather than appended in.
    """
    result = []
    for elt in lst:
        if isinstance(elt, cls):
            result.extend(elt)
        else:
            result.append(elt)
    return result


######################################################################
# Specialized Features
######################################################################


@total_ordering
class Feature:
    """
    A feature identifier that's specialized to put additional
    constraints, default values, etc.
    """

    def __init__(self, name, default=None, display=None):
        assert display in (None, "prefix", "slash")

        self._name = name  # [xx] rename to .identifier?
        self._default = default  # [xx] not implemented yet.
        self._display = display

        if self._display == "prefix":
            self._sortkey = (-1, self._name)
        elif self._display == "slash":
            self._sortkey = (1, self._name)
        else:
            self._sortkey = (0, self._name)

    @property
    def name(self):
        """The name of this feature."""
        return self._name

    @property
    def default(self):
        """Default value for this feature."""
        return self._default

    @property
    def display(self):
        """Custom display location: can be prefix, or slash."""
        return self._display

    def __repr__(self):
        return "*%s*" % self.name

    def __lt__(self, other):
        if isinstance(other, str):
            return True
        if not isinstance(other, Feature):
            raise_unorderable_types("<", self, other)
        return self._sortkey < other._sortkey

    def __eq__(self, other):
        return type(self) == type(other) and self._name == other._name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._name)

    # ////////////////////////////////////////////////////////////
    # These can be overridden by subclasses:
    # ////////////////////////////////////////////////////////////

    def read_value(self, s, position, reentrances, parser):
        return parser.read_value(s, position, reentrances)

    def unify_base_values(self, fval1, fval2, bindings):
        """
        If possible, return a single value..  If not, return
        the value ``UnificationFailure``.
        """
        if fval1 == fval2:
            return fval1
        else:
            return UnificationFailure


class SlashFeature(Feature):
    def read_value(self, s, position, reentrances, parser):
        return parser.read_partial(s, position, reentrances)


class RangeFeature(Feature):
    RANGE_RE = re.compile(r"(-?\d+):(-?\d+)")

    def read_value(self, s, position, reentrances, parser):
        m = self.RANGE_RE.match(s, position)
        if not m:
            raise ValueError("range", position)
        return (int(m.group(1)), int(m.group(2))), m.end()

    def unify_base_values(self, fval1, fval2, bindings):
        if fval1 is None:
            return fval2
        if fval2 is None:
            return fval1
        rng = max(fval1[0], fval2[0]), min(fval1[1], fval2[1])
        if rng[1] < rng[0]:
            return UnificationFailure
        return rng


SLASH = SlashFeature("slash", default=False, display="slash")
TYPE = Feature("type", display="prefix")


######################################################################
# Specialized Feature Values
######################################################################


@total_ordering
class CustomFeatureValue:
    """
    An abstract base class for base values that define a custom
    unification method.  The custom unification method of
    ``CustomFeatureValue`` will be used during unification if:

      - The ``CustomFeatureValue`` is unified with another base value.
      - The ``CustomFeatureValue`` is not the value of a customized
        ``Feature`` (which defines its own unification method).

    If two ``CustomFeatureValue`` objects are unified with one another
    during feature structure unification, then the unified base values
    they return *must* be equal; otherwise, an ``AssertionError`` will
    be raised.

    Subclasses must define ``unify()``, ``__eq__()`` and ``__lt__()``.
    Subclasses may also wish to define ``__hash__()``.
    """

    def unify(self, other):
        """
        If this base value unifies with ``other``, then return the
        unified value.  Otherwise, return ``UnificationFailure``.
        """
        raise NotImplementedError("abstract base class")

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        return NotImplemented

    def __hash__(self):
        raise TypeError("%s objects or unhashable" % self.__class__.__name__)


######################################################################
# Feature Structure Reader
######################################################################


class FeatStructReader:
    def __init__(
        self,
        features=(SLASH, TYPE),
        fdict_class=FeatStruct,
        flist_class=FeatList,
        logic_parser=None,
    ):
        self._features = {f.name: f for f in features}
        self._fdict_class = fdict_class
        self._flist_class = flist_class
        self._prefix_feature = None
        self._slash_feature = None
        for feature in features:
            if feature.display == "slash":
                if self._slash_feature:
                    raise ValueError("Multiple features w/ display=slash")
                self._slash_feature = feature
            if feature.display == "prefix":
                if self._prefix_feature:
                    raise ValueError("Multiple features w/ display=prefix")
                self._prefix_feature = feature
        self._features_with_defaults = [
            feature for feature in features if feature.default is not None
        ]
        if logic_parser is None:
            logic_parser = LogicParser()
        self._logic_parser = logic_parser

    def fromstring(self, s, fstruct=None):
        """
        Convert a string representation of a feature structure (as
        displayed by repr) into a ``FeatStruct``.  This process
        imposes the following restrictions on the string
        representation:

        - Feature names cannot contain any of the following:
          whitespace, parentheses, quote marks, equals signs,
          dashes, commas, and square brackets.  Feature names may
          not begin with plus signs or minus signs.
        - Only the following basic feature value are supported:
          strings, integers, variables, None, and unquoted
          alphanumeric strings.
        - For reentrant values, the first mention must specify
          a reentrance identifier and a value; and any subsequent
          mentions must use arrows (``'->'``) to reference the
          reentrance identifier.
        """
        s = s.strip()
        value, position = self.read_partial(s, 0, {}, fstruct)
        if position != len(s):
            self._error(s, "end of string", position)
        return value

    _START_FSTRUCT_RE = re.compile(r"\s*(?:\((\d+)\)\s*)?(\??[\w-]+)?(\[)")
    _END_FSTRUCT_RE = re.compile(r"\s*]\s*")
    _SLASH_RE = re.compile(r"/")
    _FEATURE_NAME_RE = re.compile(r'\s*([+-]?)([^\s\(\)<>"\'\-=\[\],]+)\s*')
    _REENTRANCE_RE = re.compile(r"\s*->\s*")
    _TARGET_RE = re.compile(r"\s*\((\d+)\)\s*")
    _ASSIGN_RE = re.compile(r"\s*=\s*")
    _COMMA_RE = re.compile(r"\s*,\s*")
    _BARE_PREFIX_RE = re.compile(r"\s*(?:\((\d+)\)\s*)?(\??[\w-]+\s*)()")
    # This one is used to distinguish fdicts from flists:
    _START_FDICT_RE = re.compile(
        r"(%s)|(%s\s*(%s\s*(=|->)|[+-]%s|\]))"
        % (
            _BARE_PREFIX_RE.pattern,
            _START_FSTRUCT_RE.pattern,
            _FEATURE_NAME_RE.pattern,
            _FEATURE_NAME_RE.pattern,
        )
    )

    def read_partial(self, s, position=0, reentrances=None, fstruct=None):
        """
        Helper function that reads in a feature structure.

        :param s: The string to read.
        :param position: The position in the string to start parsing.
        :param reentrances: A dictionary from reentrance ids to values.
            Defaults to an empty dictionary.
        :return: A tuple (val, pos) of the feature structure created by
            parsing and the position where the parsed feature structure ends.
        :rtype: bool
        """
        if reentrances is None:
            reentrances = {}
        try:
            return self._read_partial(s, position, reentrances, fstruct)
        except ValueError as e:
            if len(e.args) != 2:
                raise
            self._error(s, *e.args)

    def _read_partial(self, s, position, reentrances, fstruct=None):
        # Create the new feature structure
        if fstruct is None:
            if self._START_FDICT_RE.match(s, position):
                fstruct = self._fdict_class()
            else:
                fstruct = self._flist_class()

        # Read up to the open bracket.
        match = self._START_FSTRUCT_RE.match(s, position)
        if not match:
            match = self._BARE_PREFIX_RE.match(s, position)
            if not match:
                raise ValueError("open bracket or identifier", position)
        position = match.end()

        # If there as an identifier, record it.
        if match.group(1):
            identifier = match.group(1)
            if identifier in reentrances:
                raise ValueError("new identifier", match.start(1))
            reentrances[identifier] = fstruct

        if isinstance(fstruct, FeatDict):
            fstruct.clear()
            return self._read_partial_featdict(s, position, match, reentrances, fstruct)
        else:
            del fstruct[:]
            return self._read_partial_featlist(s, position, match, reentrances, fstruct)

    def _read_partial_featlist(self, s, position, match, reentrances, fstruct):
        # Prefix features are not allowed:
        if match.group(2):
            raise ValueError("open bracket")
        # Bare prefixes are not allowed:
        if not match.group(3):
            raise ValueError("open bracket")

        # Build a list of the features defined by the structure.
        while position < len(s):
            # Check for the close bracket.
            match = self._END_FSTRUCT_RE.match(s, position)
            if match is not None:
                return fstruct, match.end()

            # Reentances have the form "-> (target)"
            match = self._REENTRANCE_RE.match(s, position)
            if match:
                position = match.end()
                match = self._TARGET_RE.match(s, position)
                if not match:
                    raise ValueError("identifier", position)
                target = match.group(1)
                if target not in reentrances:
                    raise ValueError("bound identifier", position)
                position = match.end()
                fstruct.append(reentrances[target])

            # Anything else is a value.
            else:
                value, position = self._read_value(0, s, position, reentrances)
                fstruct.append(value)

            # If there's a close bracket, handle it at the top of the loop.
            if self._END_FSTRUCT_RE.match(s, position):
                continue

            # Otherwise, there should be a comma
            match = self._COMMA_RE.match(s, position)
            if match is None:
                raise ValueError("comma", position)
            position = match.end()

        # We never saw a close bracket.
        raise ValueError("close bracket", position)

    def _read_partial_featdict(self, s, position, match, reentrances, fstruct):
        # If there was a prefix feature, record it.
        if match.group(2):
            if self._prefix_feature is None:
                raise ValueError("open bracket or identifier", match.start(2))
            prefixval = match.group(2).strip()
            if prefixval.startswith("?"):
                prefixval = Variable(prefixval)
            fstruct[self._prefix_feature] = prefixval

        # If group 3 is empty, then we just have a bare prefix, so
        # we're done.
        if not match.group(3):
            return self._finalize(s, match.end(), reentrances, fstruct)

        # Build a list of the features defined by the structure.
        # Each feature has one of the three following forms:
        #     name = value
        #     name -> (target)
        #     +name
        #     -name
        while position < len(s):
            # Use these variables to hold info about each feature:
            name = value = None

            # Check for the close bracket.
            match = self._END_FSTRUCT_RE.match(s, position)
            if match is not None:
                return self._finalize(s, match.end(), reentrances, fstruct)

            # Get the feature name's name
            match = self._FEATURE_NAME_RE.match(s, position)
            if match is None:
                raise ValueError("feature name", position)
            name = match.group(2)
            position = match.end()

            # Check if it's a special feature.
            if name[0] == "*" and name[-1] == "*":
                name = self._features.get(name[1:-1])
                if name is None:
                    raise ValueError("known special feature", match.start(2))

            # Check if this feature has a value already.
            if name in fstruct:
                raise ValueError("new name", match.start(2))

            # Boolean value ("+name" or "-name")
            if match.group(1) == "+":
                value = True
            if match.group(1) == "-":
                value = False

            # Reentrance link ("-> (target)")
            if value is None:
                match = self._REENTRANCE_RE.match(s, position)
                if match is not None:
                    position = match.end()
                    match = self._TARGET_RE.match(s, position)
                    if not match:
                        raise ValueError("identifier", position)
                    target = match.group(1)
                    if target not in reentrances:
                        raise ValueError("bound identifier", position)
                    position = match.end()
                    value = reentrances[target]

            # Assignment ("= value").
            if value is None:
                match = self._ASSIGN_RE.match(s, position)
                if match:
                    position = match.end()
                    value, position = self._read_value(name, s, position, reentrances)
                # None of the above: error.
                else:
                    raise ValueError("equals sign", position)

            # Store the value.
            fstruct[name] = value

            # If there's a close bracket, handle it at the top of the loop.
            if self._END_FSTRUCT_RE.match(s, position):
                continue

            # Otherwise, there should be a comma
            match = self._COMMA_RE.match(s, position)
            if match is None:
                raise ValueError("comma", position)
            position = match.end()

        # We never saw a close bracket.
        raise ValueError("close bracket", position)

    def _finalize(self, s, pos, reentrances, fstruct):
        """
        Called when we see the close brace -- checks for a slash feature,
        and adds in default values.
        """
        # Add the slash feature (if any)
        match = self._SLASH_RE.match(s, pos)
        if match:
            name = self._slash_feature
            v, pos = self._read_value(name, s, match.end(), reentrances)
            fstruct[name] = v
        ## Add any default features.  -- handle in unficiation instead?
        # for feature in self._features_with_defaults:
        #    fstruct.setdefault(feature, feature.default)
        # Return the value.
        return fstruct, pos

    def _read_value(self, name, s, position, reentrances):
        if isinstance(name, Feature):
            return name.read_value(s, position, reentrances, self)
        else:
            return self.read_value(s, position, reentrances)

    def read_value(self, s, position, reentrances):
        for (handler, regexp) in self.VALUE_HANDLERS:
            match = regexp.match(s, position)
            if match:
                handler_func = getattr(self, handler)
                return handler_func(s, position, reentrances, match)
        raise ValueError("value", position)

    def _error(self, s, expected, position):
        lines = s.split("\n")
        while position > len(lines[0]):
            position -= len(lines.pop(0)) + 1  # +1 for the newline.
        estr = (
            "Error parsing feature structure\n    "
            + lines[0]
            + "\n    "
            + " " * position
            + "^ "
            + "Expected %s" % expected
        )
        raise ValueError(estr)

    # ////////////////////////////////////////////////////////////
    # { Value Readers
    # ////////////////////////////////////////////////////////////

    #: A table indicating how feature values should be processed.  Each
    #: entry in the table is a pair (handler, regexp).  The first entry
    #: with a matching regexp will have its handler called.  Handlers
    #: should have the following signature::
    #:
    #:    def handler(s, position, reentrances, match): ...
    #:
    #: and should return a tuple (value, position), where position is
    #: the string position where the value ended.  (n.b.: order is
    #: important here!)
    VALUE_HANDLERS = [
        ("read_fstruct_value", _START_FSTRUCT_RE),
        ("read_var_value", re.compile(r"\?[a-zA-Z_][a-zA-Z0-9_]*")),
        ("read_str_value", re.compile("[uU]?[rR]?(['\"])")),
        ("read_int_value", re.compile(r"-?\d+")),
        ("read_sym_value", re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")),
        (
            "read_app_value",
            re.compile(r"<(app)\((\?[a-z][a-z]*)\s*," r"\s*(\?[a-z][a-z]*)\)>"),
        ),
        #       ('read_logic_value', re.compile(r'<([^>]*)>')),
        # lazily match any character after '<' until we hit a '>' not preceded by '-'
        ("read_logic_value", re.compile(r"<(.*?)(?<!-)>")),
        ("read_set_value", re.compile(r"{")),
        ("read_tuple_value", re.compile(r"\(")),
    ]

    def read_fstruct_value(self, s, position, reentrances, match):
        return self.read_partial(s, position, reentrances)

    def read_str_value(self, s, position, reentrances, match):
        return read_str(s, position)

    def read_int_value(self, s, position, reentrances, match):
        return int(match.group()), match.end()

    # Note: the '?' is included in the variable name.
    def read_var_value(self, s, position, reentrances, match):
        return Variable(match.group()), match.end()

    _SYM_CONSTS = {"None": None, "True": True, "False": False}

    def read_sym_value(self, s, position, reentrances, match):
        val, end = match.group(), match.end()
        return self._SYM_CONSTS.get(val, val), end

    def read_app_value(self, s, position, reentrances, match):
        """Mainly included for backwards compat."""
        return self._logic_parser.parse("%s(%s)" % match.group(2, 3)), match.end()

    def read_logic_value(self, s, position, reentrances, match):
        try:
            try:
                expr = self._logic_parser.parse(match.group(1))
            except LogicalExpressionException as e:
                raise ValueError from e
            return expr, match.end()
        except ValueError as e:
            raise ValueError("logic expression", match.start(1)) from e

    def read_tuple_value(self, s, position, reentrances, match):
        return self._read_seq_value(
            s, position, reentrances, match, ")", FeatureValueTuple, FeatureValueConcat
        )

    def read_set_value(self, s, position, reentrances, match):
        return self._read_seq_value(
            s, position, reentrances, match, "}", FeatureValueSet, FeatureValueUnion
        )

    def _read_seq_value(
        self, s, position, reentrances, match, close_paren, seq_class, plus_class
    ):
        """
        Helper function used by read_tuple_value and read_set_value.
        """
        cp = re.escape(close_paren)
        position = match.end()
        # Special syntax of empty tuples:
        m = re.compile(r"\s*/?\s*%s" % cp).match(s, position)
        if m:
            return seq_class(), m.end()
        # Read values:
        values = []
        seen_plus = False
        while True:
            # Close paren: return value.
            m = re.compile(r"\s*%s" % cp).match(s, position)
            if m:
                if seen_plus:
                    return plus_class(values), m.end()
                else:
                    return seq_class(values), m.end()

            # Read the next value.
            val, position = self.read_value(s, position, reentrances)
            values.append(val)

            # Comma or looking at close paren
            m = re.compile(r"\s*(,|\+|(?=%s))\s*" % cp).match(s, position)
            if not m:
                raise ValueError("',' or '+' or '%s'" % cp, position)
            if m.group(1) == "+":
                seen_plus = True
            position = m.end()


######################################################################
# { Demo
######################################################################


def display_unification(fs1, fs2, indent="  "):
    # Print the two input feature structures, side by side.
    fs1_lines = ("%s" % fs1).split("\n")
    fs2_lines = ("%s" % fs2).split("\n")
    if len(fs1_lines) > len(fs2_lines):
        blankline = "[" + " " * (len(fs2_lines[0]) - 2) + "]"
        fs2_lines += [blankline] * len(fs1_lines)
    else:
        blankline = "[" + " " * (len(fs1_lines[0]) - 2) + "]"
        fs1_lines += [blankline] * len(fs2_lines)
    for (fs1_line, fs2_line) in zip(fs1_lines, fs2_lines):
        print(indent + fs1_line + "   " + fs2_line)
    print(indent + "-" * len(fs1_lines[0]) + "   " + "-" * len(fs2_lines[0]))

    linelen = len(fs1_lines[0]) * 2 + 3
    print(indent + "|               |".center(linelen))
    print(indent + "+-----UNIFY-----+".center(linelen))
    print(indent + "|".center(linelen))
    print(indent + "V".center(linelen))

    bindings = {}

    result = fs1.unify(fs2, bindings)
    if result is None:
        print(indent + "(FAILED)".center(linelen))
    else:
        print(
            "\n".join(indent + l.center(linelen) for l in ("%s" % result).split("\n"))
        )
        if bindings and len(bindings.bound_variables()) > 0:
            print(repr(bindings).center(linelen))
    return result


def interactive_demo(trace=False):
    import random
    import sys

    HELP = """
    1-%d: Select the corresponding feature structure
    q: Quit
    t: Turn tracing on or off
    l: List all feature structures
    ?: Help
    """

    print(
        """
    This demo will repeatedly present you with a list of feature
    structures, and ask you to choose two for unification.  Whenever a
    new feature structure is generated, it is added to the list of
    choices that you can pick from.  However, since this can be a
    large number of feature structures, the demo will only print out a
    random subset for you to choose between at a given time.  If you
    want to see the complete lists, type "l".  For a list of valid
    commands, type "?".
    """
    )
    print('Press "Enter" to continue...')
    sys.stdin.readline()

    fstruct_strings = [
        "[agr=[number=sing, gender=masc]]",
        "[agr=[gender=masc, person=3]]",
        "[agr=[gender=fem, person=3]]",
        "[subj=[agr=(1)[]], agr->(1)]",
        "[obj=?x]",
        "[subj=?x]",
        "[/=None]",
        "[/=NP]",
        "[cat=NP]",
        "[cat=VP]",
        "[cat=PP]",
        "[subj=[agr=[gender=?y]], obj=[agr=[gender=?y]]]",
        "[gender=masc, agr=?C]",
        "[gender=?S, agr=[gender=?S,person=3]]",
    ]

    all_fstructs = [
        (i, FeatStruct(fstruct_strings[i])) for i in range(len(fstruct_strings))
    ]

    def list_fstructs(fstructs):
        for i, fstruct in fstructs:
            print()
            lines = ("%s" % fstruct).split("\n")
            print("%3d: %s" % (i + 1, lines[0]))
            for line in lines[1:]:
                print("     " + line)
        print()

    while True:
        # Pick 5 feature structures at random from the master list.
        MAX_CHOICES = 5
        if len(all_fstructs) > MAX_CHOICES:
            fstructs = sorted(random.sample(all_fstructs, MAX_CHOICES))
        else:
            fstructs = all_fstructs

        print("_" * 75)

        print("Choose two feature structures to unify:")
        list_fstructs(fstructs)

        selected = [None, None]
        for (nth, i) in (("First", 0), ("Second", 1)):
            while selected[i] is None:
                print(
                    (
                        "%s feature structure (1-%d,q,t,l,?): "
                        % (nth, len(all_fstructs))
                    ),
                    end=" ",
                )
                try:
                    input = sys.stdin.readline().strip()
                    if input in ("q", "Q", "x", "X"):
                        return
                    if input in ("t", "T"):
                        trace = not trace
                        print("   Trace = %s" % trace)
                        continue
                    if input in ("h", "H", "?"):
                        print(HELP % len(fstructs))
                        continue
                    if input in ("l", "L"):
                        list_fstructs(all_fstructs)
                        continue
                    num = int(input) - 1
                    selected[i] = all_fstructs[num][1]
                    print()
                except:
                    print("Bad sentence number")
                    continue

        if trace:
            result = selected[0].unify(selected[1], trace=1)
        else:
            result = display_unification(selected[0], selected[1])
        if result is not None:
            for i, fstruct in all_fstructs:
                if repr(result) == repr(fstruct):
                    break
            else:
                all_fstructs.append((len(all_fstructs), result))

        print('\nType "Enter" to continue unifying; or "q" to quit.')
        input = sys.stdin.readline().strip()
        if input in ("q", "Q", "x", "X"):
            return


def demo(trace=False):
    """
    Just for testing
    """
    # import random

    # processor breaks with values like '3rd'
    fstruct_strings = [
        "[agr=[number=sing, gender=masc]]",
        "[agr=[gender=masc, person=3]]",
        "[agr=[gender=fem, person=3]]",
        "[subj=[agr=(1)[]], agr->(1)]",
        "[obj=?x]",
        "[subj=?x]",
        "[/=None]",
        "[/=NP]",
        "[cat=NP]",
        "[cat=VP]",
        "[cat=PP]",
        "[subj=[agr=[gender=?y]], obj=[agr=[gender=?y]]]",
        "[gender=masc, agr=?C]",
        "[gender=?S, agr=[gender=?S,person=3]]",
    ]
    all_fstructs = [FeatStruct(fss) for fss in fstruct_strings]
    # MAX_CHOICES = 5
    # if len(all_fstructs) > MAX_CHOICES:
    # fstructs = random.sample(all_fstructs, MAX_CHOICES)
    # fstructs.sort()
    # else:
    # fstructs = all_fstructs

    for fs1 in all_fstructs:
        for fs2 in all_fstructs:
            print(
                "\n*******************\nfs1 is:\n%s\n\nfs2 is:\n%s\n\nresult is:\n%s"
                % (fs1, fs2, unify(fs1, fs2))
            )


if __name__ == "__main__":
    demo()

__all__ = [
    "FeatStruct",
    "FeatDict",
    "FeatList",
    "unify",
    "subsumes",
    "conflicts",
    "Feature",
    "SlashFeature",
    "RangeFeature",
    "SLASH",
    "TYPE",
    "FeatStructReader",
]
