# encoding: utf-8
"""A dict subclass that supports attribute style access.

Authors:

* Fernando Perez (original)
* Brian Granger (refactoring to a dict subclass)
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

__all__ = ['Struct']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class Struct(dict):
    """A dict subclass with attribute style access.

    This dict subclass has a a few extra features:

    * Attribute style access.
    * Protection of class members (like keys, items) when using attribute
      style access.
    * The ability to restrict assignment to only existing keys.
    * Intelligent merging.
    * Overloaded operators.
    """
    _allownew = True
    def __init__(self, *args, **kw):
        """Initialize with a dictionary, another Struct, or data.

        Parameters
        ----------
        *args : dict, Struct
            Initialize with one dict or Struct
        **kw : dict
            Initialize with key, value pairs.

        Examples
        --------
        >>> s = Struct(a=10,b=30)
        >>> s.a
        10
        >>> s.b
        30
        >>> s2 = Struct(s,c=30)
        >>> sorted(s2.keys())
        ['a', 'b', 'c']
        """
        object.__setattr__(self, '_allownew', True)
        dict.__init__(self, *args, **kw)

    def __setitem__(self, key, value):
        """Set an item with check for allownew.

        Examples
        --------
        >>> s = Struct()
        >>> s['a'] = 10
        >>> s.allow_new_attr(False)
        >>> s['a'] = 10
        >>> s['a']
        10
        >>> try:
        ...     s['b'] = 20
        ... except KeyError:
        ...     print('this is not allowed')
        ...
        this is not allowed
        """
        if not self._allownew and key not in self:
            raise KeyError(
                "can't create new attribute %s when allow_new_attr(False)" % key)
        dict.__setitem__(self, key, value)

    def __setattr__(self, key, value):
        """Set an attr with protection of class members.

        This calls :meth:`self.__setitem__` but convert :exc:`KeyError` to
        :exc:`AttributeError`.

        Examples
        --------
        >>> s = Struct()
        >>> s.a = 10
        >>> s.a
        10
        >>> try:
        ...     s.get = 10
        ... except AttributeError:
        ...     print("you can't set a class member")
        ...
        you can't set a class member
        """
        # If key is an str it might be a class member or instance var
        if isinstance(key, str):
            # I can't simply call hasattr here because it calls getattr, which
            # calls self.__getattr__, which returns True for keys in
            # self._data.  But I only want keys in the class and in
            # self.__dict__
            if key in self.__dict__ or hasattr(Struct, key):
                raise AttributeError(
                    'attr %s is a protected member of class Struct.' % key
                )
        try:
            self.__setitem__(key, value)
        except KeyError as e:
            raise AttributeError(e) from e

    def __getattr__(self, key):
        """Get an attr by calling :meth:`dict.__getitem__`.

        Like :meth:`__setattr__`, this method converts :exc:`KeyError` to
        :exc:`AttributeError`.

        Examples
        --------
        >>> s = Struct(a=10)
        >>> s.a
        10
        >>> type(s.get)
        <...method'>
        >>> try:
        ...     s.b
        ... except AttributeError:
        ...     print("I don't have that key")
        ...
        I don't have that key
        """
        try:
            result = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        else:
            return result

    def __iadd__(self, other):
        """s += s2 is a shorthand for s.merge(s2).

        Examples
        --------
        >>> s = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,c=40)
        >>> s += s2
        >>> sorted(s.keys())
        ['a', 'b', 'c']
        """
        self.merge(other)
        return self

    def __add__(self,other):
        """s + s2 -> New Struct made from s.merge(s2).

        Examples
        --------
        >>> s1 = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,c=40)
        >>> s = s1 + s2
        >>> sorted(s.keys())
        ['a', 'b', 'c']
        """
        sout = self.copy()
        sout.merge(other)
        return sout

    def __sub__(self,other):
        """s1 - s2 -> remove keys in s2 from s1.

        Examples
        --------
        >>> s1 = Struct(a=10,b=30)
        >>> s2 = Struct(a=40)
        >>> s = s1 - s2
        >>> s
        {'b': 30}
        """
        sout = self.copy()
        sout -= other
        return sout

    def __isub__(self,other):
        """Inplace remove keys from self that are in other.

        Examples
        --------
        >>> s1 = Struct(a=10,b=30)
        >>> s2 = Struct(a=40)
        >>> s1 -= s2
        >>> s1
        {'b': 30}
        """
        for k in other.keys():
            if k in self:
                del self[k]
        return self

    def __dict_invert(self, data):
        """Helper function for merge.

        Takes a dictionary whose values are lists and returns a dict with
        the elements of each list as keys and the original keys as values.
        """
        outdict = {}
        for k,lst in data.items():
            if isinstance(lst, str):
                lst = lst.split()
            for entry in lst:
                outdict[entry] = k
        return outdict

    def dict(self):
        return self

    def copy(self):
        """Return a copy as a Struct.

        Examples
        --------
        >>> s = Struct(a=10,b=30)
        >>> s2 = s.copy()
        >>> type(s2) is Struct
        True
        """
        return Struct(dict.copy(self))

    def hasattr(self, key):
        """hasattr function available as a method.

        Implemented like has_key.

        Examples
        --------
        >>> s = Struct(a=10)
        >>> s.hasattr('a')
        True
        >>> s.hasattr('b')
        False
        >>> s.hasattr('get')
        False
        """
        return key in self

    def allow_new_attr(self, allow = True):
        """Set whether new attributes can be created in this Struct.

        This can be used to catch typos by verifying that the attribute user
        tries to change already exists in this Struct.
        """
        object.__setattr__(self, '_allownew', allow)

    def merge(self, __loc_data__=None, __conflict_solve=None, **kw):
        """Merge two Structs with customizable conflict resolution.

        This is similar to :meth:`update`, but much more flexible. First, a
        dict is made from data+key=value pairs. When merging this dict with
        the Struct S, the optional dictionary 'conflict' is used to decide
        what to do.

        If conflict is not given, the default behavior is to preserve any keys
        with their current value (the opposite of the :meth:`update` method's
        behavior).

        Parameters
        ----------
        __loc_data__ : dict, Struct
            The data to merge into self
        __conflict_solve : dict
            The conflict policy dict.  The keys are binary functions used to
            resolve the conflict and the values are lists of strings naming
            the keys the conflict resolution function applies to.  Instead of
            a list of strings a space separated string can be used, like
            'a b c'.
        **kw : dict
            Additional key, value pairs to merge in

        Notes
        -----
        The `__conflict_solve` dict is a dictionary of binary functions which will be used to
        solve key conflicts.  Here is an example::

            __conflict_solve = dict(
                func1=['a','b','c'],
                func2=['d','e']
            )

        In this case, the function :func:`func1` will be used to resolve
        keys 'a', 'b' and 'c' and the function :func:`func2` will be used for
        keys 'd' and 'e'.  This could also be written as::

            __conflict_solve = dict(func1='a b c',func2='d e')

        These functions will be called for each key they apply to with the
        form::

            func1(self['a'], other['a'])

        The return value is used as the final merged value.

        As a convenience, merge() provides five (the most commonly needed)
        pre-defined policies: preserve, update, add, add_flip and add_s. The
        easiest explanation is their implementation::

            preserve = lambda old,new: old
            update   = lambda old,new: new
            add      = lambda old,new: old + new
            add_flip = lambda old,new: new + old  # note change of order!
            add_s    = lambda old,new: old + ' ' + new  # only for str!

        You can use those four words (as strings) as keys instead
        of defining them as functions, and the merge method will substitute
        the appropriate functions for you.

        For more complicated conflict resolution policies, you still need to
        construct your own functions.

        Examples
        --------
        This show the default policy:

        >>> s = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,c=40)
        >>> s.merge(s2)
        >>> sorted(s.items())
        [('a', 10), ('b', 30), ('c', 40)]

        Now, show how to specify a conflict dict:

        >>> s = Struct(a=10,b=30)
        >>> s2 = Struct(a=20,b=40)
        >>> conflict = {'update':'a','add':'b'}
        >>> s.merge(s2,conflict)
        >>> sorted(s.items())
        [('a', 20), ('b', 70)]
        """

        data_dict = dict(__loc_data__,**kw)

        # policies for conflict resolution: two argument functions which return
        # the value that will go in the new struct
        preserve = lambda old,new: old
        update   = lambda old,new: new
        add      = lambda old,new: old + new
        add_flip = lambda old,new: new + old  # note change of order!
        add_s    = lambda old,new: old + ' ' + new

        # default policy is to keep current keys when there's a conflict
        conflict_solve = dict.fromkeys(self, preserve)

        # the conflict_solve dictionary is given by the user 'inverted': we
        # need a name-function mapping, it comes as a function -> names
        # dict. Make a local copy (b/c we'll make changes), replace user
        # strings for the three builtin policies and invert it.
        if __conflict_solve:
            inv_conflict_solve_user = __conflict_solve.copy()
            for name, func in [('preserve',preserve), ('update',update),
                               ('add',add), ('add_flip',add_flip),
                               ('add_s',add_s)]:
                if name in inv_conflict_solve_user.keys():
                    inv_conflict_solve_user[func] = inv_conflict_solve_user[name]
                    del inv_conflict_solve_user[name]
            conflict_solve.update(self.__dict_invert(inv_conflict_solve_user))
        for key in data_dict:
            if key not in self:
                self[key] = data_dict[key]
            else:
                self[key] = conflict_solve[key](self[key],data_dict[key])

