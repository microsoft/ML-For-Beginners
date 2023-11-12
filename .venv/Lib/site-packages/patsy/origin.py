# This file is part of Patsy
# Copyright (C) 2011-2012 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# The core 'origin' tracking system. This point of this is to have machinery
# so if some object is ultimately derived from some portion of a string (e.g.,
# a formula), then we can keep track of that, and use it to give proper error
# messages.

# These are made available in the patsy.* namespace
__all__ = ["Origin"]

class Origin(object):
    """This represents the origin of some object in some string.

    For example, if we have an object ``x1_obj`` that was produced by parsing
    the ``x1`` in the formula ``"y ~ x1:x2"``, then we conventionally keep
    track of that relationship by doing::

      x1_obj.origin = Origin("y ~ x1:x2", 4, 6)

    Then later if we run into a problem, we can do::

      raise PatsyError("invalid factor", x1_obj)

    and we'll produce a nice error message like::

      PatsyError: invalid factor
          y ~ x1:x2
              ^^

    Origins are compared by value, and hashable.
    """

    def __init__(self, code, start, end):
        self.code = code
        self.start = start
        self.end = end

    @classmethod
    def combine(cls, origin_objs):
        """Class method for combining a set of Origins into one large Origin
        that spans them.

        Example usage: if we wanted to represent the origin of the "x1:x2"
        term, we could do ``Origin.combine([x1_obj, x2_obj])``.

        Single argument is an iterable, and each element in the iterable
        should be either:

        * An Origin object
        * ``None``
        * An object that has a ``.origin`` attribute which fulfills the above
          criteria.
          
        Returns either an Origin object, or None.
        """
        origins = []
        for obj in origin_objs:
            if obj is not None and not isinstance(obj, Origin):
                obj = obj.origin
            if obj is None:
                continue
            origins.append(obj)
        if not origins:
            return None
        codes = set([o.code for o in origins])
        assert len(codes) == 1
        start = min([o.start for o in origins])
        end = max([o.end for o in origins])
        return cls(codes.pop(), start, end)

    def relevant_code(self):
        """Extracts and returns the span of the original code represented by
        this Origin. Example: ``x1``."""
        return self.code[self.start:self.end]

    def __eq__(self, other):
        return (isinstance(other, Origin)
                and self.code == other.code
                and self.start == other.start
                and self.end == other.end)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((Origin, self.code, self.start, self.end))

    def caretize(self, indent=0):
        """Produces a user-readable two line string indicating the origin of
        some code. Example::

          y ~ x1:x2
              ^^

        If optional argument 'indent' is given, then both lines will be
        indented by this much. The returned string does not have a trailing
        newline.
        """
        return ("%s%s\n%s%s%s" 
                % (" " * indent,
                   self.code,
                   " " * indent,
                   " " * self.start,
                   "^" * (self.end - self.start)))

    def __repr__(self):
        return "<Origin %s->%s<-%s (%s-%s)>" % (
            self.code[:self.start],
            self.code[self.start:self.end],
            self.code[self.end:],
            self.start, self.end)

    # We reimplement patsy.util.no_pickling, to avoid circular import issues
    def __getstate__(self):
        raise NotImplementedError

def test_Origin():
    o1 = Origin("012345", 2, 4)
    o2 = Origin("012345", 4, 5)
    assert o1.caretize() == "012345\n  ^^"
    assert o2.caretize() == "012345\n    ^"
    o3 = Origin.combine([o1, o2])
    assert o3.code == "012345"
    assert o3.start == 2
    assert o3.end == 5
    assert o3.caretize(indent=2) == "  012345\n    ^^^"
    assert o3 == Origin("012345", 2, 5)

    class ObjWithOrigin(object):
        def __init__(self, origin=None):
            self.origin = origin
    o4 = Origin.combine([ObjWithOrigin(o1), ObjWithOrigin(), None])
    assert o4 == o1
    o5 = Origin.combine([ObjWithOrigin(o1), o2])
    assert o5 == o3

    assert Origin.combine([ObjWithOrigin(), ObjWithOrigin()]) is None

    from patsy.util import assert_no_pickling
    assert_no_pickling(Origin("", 0, 0))
