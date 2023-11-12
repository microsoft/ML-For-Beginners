# Natural Language Toolkit: Text Trees
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Peter Ljunglöf <peter.ljunglof@gu.se>
#         Tom Aarsen <>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

import warnings
from abc import ABCMeta, abstractmethod

from nltk.tree.tree import Tree
from nltk.util import slice_bounds


######################################################################
## Parented trees
######################################################################
class AbstractParentedTree(Tree, metaclass=ABCMeta):
    """
    An abstract base class for a ``Tree`` that automatically maintains
    pointers to parent nodes.  These parent pointers are updated
    whenever any change is made to a tree's structure.  Two subclasses
    are currently defined:

      - ``ParentedTree`` is used for tree structures where each subtree
        has at most one parent.  This class should be used in cases
        where there is no"sharing" of subtrees.

      - ``MultiParentedTree`` is used for tree structures where a
        subtree may have zero or more parents.  This class should be
        used in cases where subtrees may be shared.

    Subclassing
    ===========
    The ``AbstractParentedTree`` class redefines all operations that
    modify a tree's structure to call two methods, which are used by
    subclasses to update parent information:

      - ``_setparent()`` is called whenever a new child is added.
      - ``_delparent()`` is called whenever a child is removed.
    """

    def __init__(self, node, children=None):
        super().__init__(node, children)
        # If children is None, the tree is read from node, and
        # all parents will be set during parsing.
        if children is not None:
            # Otherwise we have to set the parent of the children.
            # Iterate over self, and *not* children, because children
            # might be an iterator.
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    self._setparent(child, i, dry_run=True)
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    self._setparent(child, i)

    # ////////////////////////////////////////////////////////////
    # Parent management
    # ////////////////////////////////////////////////////////////
    @abstractmethod
    def _setparent(self, child, index, dry_run=False):
        """
        Update the parent pointer of ``child`` to point to ``self``.  This
        method is only called if the type of ``child`` is ``Tree``;
        i.e., it is not called when adding a leaf to a tree.  This method
        is always called before the child is actually added to the
        child list of ``self``.

        :type child: Tree
        :type index: int
        :param index: The index of ``child`` in ``self``.
        :raise TypeError: If ``child`` is a tree with an impropriate
            type.  Typically, if ``child`` is a tree, then its type needs
            to match the type of ``self``.  This prevents mixing of
            different tree types (single-parented, multi-parented, and
            non-parented).
        :param dry_run: If true, the don't actually set the child's
            parent pointer; just check for any error conditions, and
            raise an exception if one is found.
        """

    @abstractmethod
    def _delparent(self, child, index):
        """
        Update the parent pointer of ``child`` to not point to self.  This
        method is only called if the type of ``child`` is ``Tree``; i.e., it
        is not called when removing a leaf from a tree.  This method
        is always called before the child is actually removed from the
        child list of ``self``.

        :type child: Tree
        :type index: int
        :param index: The index of ``child`` in ``self``.
        """

    # ////////////////////////////////////////////////////////////
    # Methods that add/remove children
    # ////////////////////////////////////////////////////////////
    # Every method that adds or removes a child must make
    # appropriate calls to _setparent() and _delparent().

    def __delitem__(self, index):
        # del ptree[start:stop]
        if isinstance(index, slice):
            start, stop, step = slice_bounds(self, index, allow_step=True)
            # Clear all the children pointers.
            for i in range(start, stop, step):
                if isinstance(self[i], Tree):
                    self._delparent(self[i], i)
            # Delete the children from our child list.
            super().__delitem__(index)

        # del ptree[i]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError("index out of range")
            # Clear the child's parent pointer.
            if isinstance(self[index], Tree):
                self._delparent(self[index], index)
            # Remove the child from our child list.
            super().__delitem__(index)

        elif isinstance(index, (list, tuple)):
            # del ptree[()]
            if len(index) == 0:
                raise IndexError("The tree position () may not be deleted.")
            # del ptree[(i,)]
            elif len(index) == 1:
                del self[index[0]]
            # del ptree[i1, i2, i3]
            else:
                del self[index[0]][index[1:]]

        else:
            raise TypeError(
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    def __setitem__(self, index, value):
        # ptree[start:stop] = value
        if isinstance(index, slice):
            start, stop, step = slice_bounds(self, index, allow_step=True)
            # make a copy of value, in case it's an iterator
            if not isinstance(value, (list, tuple)):
                value = list(value)
            # Check for any error conditions, so we can avoid ending
            # up in an inconsistent state if an error does occur.
            for i, child in enumerate(value):
                if isinstance(child, Tree):
                    self._setparent(child, start + i * step, dry_run=True)
            # clear the child pointers of all parents we're removing
            for i in range(start, stop, step):
                if isinstance(self[i], Tree):
                    self._delparent(self[i], i)
            # set the child pointers of the new children.  We do this
            # after clearing *all* child pointers, in case we're e.g.
            # reversing the elements in a tree.
            for i, child in enumerate(value):
                if isinstance(child, Tree):
                    self._setparent(child, start + i * step)
            # finally, update the content of the child list itself.
            super().__setitem__(index, value)

        # ptree[i] = value
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError("index out of range")
            # if the value is not changing, do nothing.
            if value is self[index]:
                return
            # Set the new child's parent pointer.
            if isinstance(value, Tree):
                self._setparent(value, index)
            # Remove the old child's parent pointer
            if isinstance(self[index], Tree):
                self._delparent(self[index], index)
            # Update our child list.
            super().__setitem__(index, value)

        elif isinstance(index, (list, tuple)):
            # ptree[()] = value
            if len(index) == 0:
                raise IndexError("The tree position () may not be assigned to.")
            # ptree[(i,)] = value
            elif len(index) == 1:
                self[index[0]] = value
            # ptree[i1, i2, i3] = value
            else:
                self[index[0]][index[1:]] = value

        else:
            raise TypeError(
                "%s indices must be integers, not %s"
                % (type(self).__name__, type(index).__name__)
            )

    def append(self, child):
        if isinstance(child, Tree):
            self._setparent(child, len(self))
        super().append(child)

    def extend(self, children):
        for child in children:
            if isinstance(child, Tree):
                self._setparent(child, len(self))
            super().append(child)

    def insert(self, index, child):
        # Handle negative indexes.  Note that if index < -len(self),
        # we do *not* raise an IndexError, unlike __getitem__.  This
        # is done for consistency with list.__getitem__ and list.index.
        if index < 0:
            index += len(self)
        if index < 0:
            index = 0
        # Set the child's parent, and update our child list.
        if isinstance(child, Tree):
            self._setparent(child, index)
        super().insert(index, child)

    def pop(self, index=-1):
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError("index out of range")
        if isinstance(self[index], Tree):
            self._delparent(self[index], index)
        return super().pop(index)

    # n.b.: like `list`, this is done by equality, not identity!
    # To remove a specific child, use del ptree[i].
    def remove(self, child):
        index = self.index(child)
        if isinstance(self[index], Tree):
            self._delparent(self[index], index)
        super().remove(child)

    # We need to implement __getslice__ and friends, even though
    # they're deprecated, because otherwise list.__getslice__ will get
    # called (since we're subclassing from list).  Just delegate to
    # __getitem__ etc., but use max(0, start) and max(0, stop) because
    # because negative indices are already handled *before*
    # __getslice__ is called; and we don't want to double-count them.
    if hasattr(list, "__getslice__"):

        def __getslice__(self, start, stop):
            return self.__getitem__(slice(max(0, start), max(0, stop)))

        def __delslice__(self, start, stop):
            return self.__delitem__(slice(max(0, start), max(0, stop)))

        def __setslice__(self, start, stop, value):
            return self.__setitem__(slice(max(0, start), max(0, stop)), value)

    def __getnewargs__(self):
        """Method used by the pickle module when un-pickling.
        This method provides the arguments passed to ``__new__``
        upon un-pickling. Without this method, ParentedTree instances
        cannot be pickled and unpickled in Python 3.7+ onwards.

        :return: Tuple of arguments for ``__new__``, i.e. the label
            and the children of this node.
        :rtype: Tuple[Any, List[AbstractParentedTree]]
        """
        return (self._label, list(self))


class ParentedTree(AbstractParentedTree):
    """
    A ``Tree`` that automatically maintains parent pointers for
    single-parented trees.  The following are methods for querying
    the structure of a parented tree: ``parent``, ``parent_index``,
    ``left_sibling``, ``right_sibling``, ``root``, ``treeposition``.

    Each ``ParentedTree`` may have at most one parent.  In
    particular, subtrees may not be shared.  Any attempt to reuse a
    single ``ParentedTree`` as a child of more than one parent (or
    as multiple children of the same parent) will cause a
    ``ValueError`` exception to be raised.

    ``ParentedTrees`` should never be used in the same tree as ``Trees``
    or ``MultiParentedTrees``.  Mixing tree implementations may result
    in incorrect parent pointers and in ``TypeError`` exceptions.
    """

    def __init__(self, node, children=None):
        self._parent = None
        """The parent of this Tree, or None if it has no parent."""
        super().__init__(node, children)
        if children is None:
            # If children is None, the tree is read from node.
            # After parsing, the parent of the immediate children
            # will point to an intermediate tree, not self.
            # We fix this by brute force:
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    child._parent = None
                    self._setparent(child, i)

    def _frozen_class(self):
        from nltk.tree.immutable import ImmutableParentedTree

        return ImmutableParentedTree

    def copy(self, deep=False):
        if not deep:
            warnings.warn(
                f"{self.__class__.__name__} objects do not support shallow copies. Defaulting to a deep copy."
            )
        return super().copy(deep=True)

    # /////////////////////////////////////////////////////////////////
    # Methods
    # /////////////////////////////////////////////////////////////////

    def parent(self):
        """The parent of this tree, or None if it has no parent."""
        return self._parent

    def parent_index(self):
        """
        The index of this tree in its parent.  I.e.,
        ``ptree.parent()[ptree.parent_index()] is ptree``.  Note that
        ``ptree.parent_index()`` is not necessarily equal to
        ``ptree.parent.index(ptree)``, since the ``index()`` method
        returns the first child that is equal to its argument.
        """
        if self._parent is None:
            return None
        for i, child in enumerate(self._parent):
            if child is self:
                return i
        assert False, "expected to find self in self._parent!"

    def left_sibling(self):
        """The left sibling of this tree, or None if it has none."""
        parent_index = self.parent_index()
        if self._parent and parent_index > 0:
            return self._parent[parent_index - 1]
        return None  # no left sibling

    def right_sibling(self):
        """The right sibling of this tree, or None if it has none."""
        parent_index = self.parent_index()
        if self._parent and parent_index < (len(self._parent) - 1):
            return self._parent[parent_index + 1]
        return None  # no right sibling

    def root(self):
        """
        The root of this tree.  I.e., the unique ancestor of this tree
        whose parent is None.  If ``ptree.parent()`` is None, then
        ``ptree`` is its own root.
        """
        root = self
        while root.parent() is not None:
            root = root.parent()
        return root

    def treeposition(self):
        """
        The tree position of this tree, relative to the root of the
        tree.  I.e., ``ptree.root[ptree.treeposition] is ptree``.
        """
        if self.parent() is None:
            return ()
        else:
            return self.parent().treeposition() + (self.parent_index(),)

    # /////////////////////////////////////////////////////////////////
    # Parent Management
    # /////////////////////////////////////////////////////////////////

    def _delparent(self, child, index):
        # Sanity checks
        assert isinstance(child, ParentedTree)
        assert self[index] is child
        assert child._parent is self

        # Delete child's parent pointer.
        child._parent = None

    def _setparent(self, child, index, dry_run=False):
        # If the child's type is incorrect, then complain.
        if not isinstance(child, ParentedTree):
            raise TypeError("Can not insert a non-ParentedTree into a ParentedTree")

        # If child already has a parent, then complain.
        if hasattr(child, "_parent") and child._parent is not None:
            raise ValueError("Can not insert a subtree that already has a parent.")

        # Set child's parent pointer & index.
        if not dry_run:
            child._parent = self


class MultiParentedTree(AbstractParentedTree):
    """
    A ``Tree`` that automatically maintains parent pointers for
    multi-parented trees.  The following are methods for querying the
    structure of a multi-parented tree: ``parents()``, ``parent_indices()``,
    ``left_siblings()``, ``right_siblings()``, ``roots``, ``treepositions``.

    Each ``MultiParentedTree`` may have zero or more parents.  In
    particular, subtrees may be shared.  If a single
    ``MultiParentedTree`` is used as multiple children of the same
    parent, then that parent will appear multiple times in its
    ``parents()`` method.

    ``MultiParentedTrees`` should never be used in the same tree as
    ``Trees`` or ``ParentedTrees``.  Mixing tree implementations may
    result in incorrect parent pointers and in ``TypeError`` exceptions.
    """

    def __init__(self, node, children=None):
        self._parents = []
        """A list of this tree's parents.  This list should not
           contain duplicates, even if a parent contains this tree
           multiple times."""
        super().__init__(node, children)
        if children is None:
            # If children is None, the tree is read from node.
            # After parsing, the parent(s) of the immediate children
            # will point to an intermediate tree, not self.
            # We fix this by brute force:
            for i, child in enumerate(self):
                if isinstance(child, Tree):
                    child._parents = []
                    self._setparent(child, i)

    def _frozen_class(self):
        from nltk.tree.immutable import ImmutableMultiParentedTree

        return ImmutableMultiParentedTree

    # /////////////////////////////////////////////////////////////////
    # Methods
    # /////////////////////////////////////////////////////////////////

    def parents(self):
        """
        The set of parents of this tree.  If this tree has no parents,
        then ``parents`` is the empty set.  To check if a tree is used
        as multiple children of the same parent, use the
        ``parent_indices()`` method.

        :type: list(MultiParentedTree)
        """
        return list(self._parents)

    def left_siblings(self):
        """
        A list of all left siblings of this tree, in any of its parent
        trees.  A tree may be its own left sibling if it is used as
        multiple contiguous children of the same parent.  A tree may
        appear multiple times in this list if it is the left sibling
        of this tree with respect to multiple parents.

        :type: list(MultiParentedTree)
        """
        return [
            parent[index - 1]
            for (parent, index) in self._get_parent_indices()
            if index > 0
        ]

    def right_siblings(self):
        """
        A list of all right siblings of this tree, in any of its parent
        trees.  A tree may be its own right sibling if it is used as
        multiple contiguous children of the same parent.  A tree may
        appear multiple times in this list if it is the right sibling
        of this tree with respect to multiple parents.

        :type: list(MultiParentedTree)
        """
        return [
            parent[index + 1]
            for (parent, index) in self._get_parent_indices()
            if index < (len(parent) - 1)
        ]

    def _get_parent_indices(self):
        return [
            (parent, index)
            for parent in self._parents
            for index, child in enumerate(parent)
            if child is self
        ]

    def roots(self):
        """
        The set of all roots of this tree.  This set is formed by
        tracing all possible parent paths until trees with no parents
        are found.

        :type: list(MultiParentedTree)
        """
        return list(self._get_roots_helper({}).values())

    def _get_roots_helper(self, result):
        if self._parents:
            for parent in self._parents:
                parent._get_roots_helper(result)
        else:
            result[id(self)] = self
        return result

    def parent_indices(self, parent):
        """
        Return a list of the indices where this tree occurs as a child
        of ``parent``.  If this child does not occur as a child of
        ``parent``, then the empty list is returned.  The following is
        always true::

          for parent_index in ptree.parent_indices(parent):
              parent[parent_index] is ptree
        """
        if parent not in self._parents:
            return []
        else:
            return [index for (index, child) in enumerate(parent) if child is self]

    def treepositions(self, root):
        """
        Return a list of all tree positions that can be used to reach
        this multi-parented tree starting from ``root``.  I.e., the
        following is always true::

          for treepos in ptree.treepositions(root):
              root[treepos] is ptree
        """
        if self is root:
            return [()]
        else:
            return [
                treepos + (index,)
                for parent in self._parents
                for treepos in parent.treepositions(root)
                for (index, child) in enumerate(parent)
                if child is self
            ]

    # /////////////////////////////////////////////////////////////////
    # Parent Management
    # /////////////////////////////////////////////////////////////////

    def _delparent(self, child, index):
        # Sanity checks
        assert isinstance(child, MultiParentedTree)
        assert self[index] is child
        assert len([p for p in child._parents if p is self]) == 1

        # If the only copy of child in self is at index, then delete
        # self from child's parent list.
        for i, c in enumerate(self):
            if c is child and i != index:
                break
        else:
            child._parents.remove(self)

    def _setparent(self, child, index, dry_run=False):
        # If the child's type is incorrect, then complain.
        if not isinstance(child, MultiParentedTree):
            raise TypeError(
                "Can not insert a non-MultiParentedTree into a MultiParentedTree"
            )

        # Add self as a parent pointer if it's not already listed.
        if not dry_run:
            for parent in child._parents:
                if parent is self:
                    break
            else:
                child._parents.append(self)


__all__ = [
    "ParentedTree",
    "MultiParentedTree",
]
