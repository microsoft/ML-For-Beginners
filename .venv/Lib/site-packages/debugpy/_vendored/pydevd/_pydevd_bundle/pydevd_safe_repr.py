# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

# Gotten from ptvsd for supporting the format expected there.
import sys
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER
import locale
from _pydev_bundle import pydev_log


class SafeRepr(object):
    # Can be used to override the encoding from locale.getpreferredencoding()
    locale_preferred_encoding = None

    # Can be used to override the encoding used for sys.stdout.encoding
    sys_stdout_encoding = None

    # String types are truncated to maxstring_outer when at the outer-
    # most level, and truncated to maxstring_inner characters inside
    # collections.
    maxstring_outer = 2 ** 16
    maxstring_inner = 30
    string_types = (str, bytes)
    bytes = bytes
    set_info = (set, '{', '}', False)
    frozenset_info = (frozenset, 'frozenset({', '})', False)
    int_types = (int,)
    long_iter_types = (list, tuple, bytearray, range,
                       dict, set, frozenset)

    # Collection types are recursively iterated for each limit in
    # maxcollection.
    maxcollection = (15, 10)

    # Specifies type, prefix string, suffix string, and whether to include a
    # comma if there is only one element. (Using a sequence rather than a
    # mapping because we use isinstance() to determine the matching type.)
    collection_types = [
        (tuple, '(', ')', True),
        (list, '[', ']', False),
        frozenset_info,
        set_info,
    ]
    try:
        from collections import deque
        collection_types.append((deque, 'deque([', '])', False))
    except Exception:
        pass

    # type, prefix string, suffix string, item prefix string,
    # item key/value separator, item suffix string
    dict_types = [(dict, '{', '}', '', ': ', '')]
    try:
        from collections import OrderedDict
        dict_types.append((OrderedDict, 'OrderedDict([', '])', '(', ', ', ')'))
    except Exception:
        pass

    # All other types are treated identically to strings, but using
    # different limits.
    maxother_outer = 2 ** 16
    maxother_inner = 30

    convert_to_hex = False
    raw_value = False

    def __call__(self, obj):
        '''
        :param object obj:
            The object for which we want a representation.

        :return str:
            Returns bytes encoded as utf-8 on py2 and str on py3.
        '''
        try:
            return ''.join(self._repr(obj, 0))
        except Exception:
            try:
                return 'An exception was raised: %r' % sys.exc_info()[1]
            except Exception:
                return 'An exception was raised'

    def _repr(self, obj, level):
        '''Returns an iterable of the parts in the final repr string.'''

        try:
            obj_repr = type(obj).__repr__
        except Exception:
            obj_repr = None

        def has_obj_repr(t):
            r = t.__repr__
            try:
                return obj_repr == r
            except Exception:
                return obj_repr is r

        for t, prefix, suffix, comma in self.collection_types:
            if isinstance(obj, t) and has_obj_repr(t):
                return self._repr_iter(obj, level, prefix, suffix, comma)

        for t, prefix, suffix, item_prefix, item_sep, item_suffix in self.dict_types:  # noqa
            if isinstance(obj, t) and has_obj_repr(t):
                return self._repr_dict(obj, level, prefix, suffix,
                                       item_prefix, item_sep, item_suffix)

        for t in self.string_types:
            if isinstance(obj, t) and has_obj_repr(t):
                return self._repr_str(obj, level)

        if self._is_long_iter(obj):
            return self._repr_long_iter(obj)

        return self._repr_other(obj, level)

    # Determines whether an iterable exceeds the limits set in
    # maxlimits, and is therefore unsafe to repr().
    def _is_long_iter(self, obj, level=0):
        try:
            # Strings have their own limits (and do not nest). Because
            # they don't have __iter__ in 2.x, this check goes before
            # the next one.
            if isinstance(obj, self.string_types):
                return len(obj) > self.maxstring_inner

            # If it's not an iterable (and not a string), it's fine.
            if not hasattr(obj, '__iter__'):
                return False

            # If it's not an instance of these collection types then it
            # is fine. Note: this is a fix for
            # https://github.com/Microsoft/ptvsd/issues/406
            if not isinstance(obj, self.long_iter_types):
                return False

            # Iterable is its own iterator - this is a one-off iterable
            # like generator or enumerate(). We can't really count that,
            # but repr() for these should not include any elements anyway,
            # so we can treat it the same as non-iterables.
            if obj is iter(obj):
                return False

            # range reprs fine regardless of length.
            if isinstance(obj, range):
                return False

            # numpy and scipy collections (ndarray etc) have
            # self-truncating repr, so they're always safe.
            try:
                module = type(obj).__module__.partition('.')[0]
                if module in ('numpy', 'scipy'):
                    return False
            except Exception:
                pass

            # Iterables that nest too deep are considered long.
            if level >= len(self.maxcollection):
                return True

            # It is too long if the length exceeds the limit, or any
            # of its elements are long iterables.
            if hasattr(obj, '__len__'):
                try:
                    size = len(obj)
                except Exception:
                    size = None
                if size is not None and size > self.maxcollection[level]:
                    return True
                return any((self._is_long_iter(item, level + 1) for item in obj))  # noqa
            return any(i > self.maxcollection[level] or self._is_long_iter(item, level + 1) for i, item in enumerate(obj))  # noqa

        except Exception:
            # If anything breaks, assume the worst case.
            return True

    def _repr_iter(self, obj, level, prefix, suffix,
                   comma_after_single_element=False):
        yield prefix

        if level >= len(self.maxcollection):
            yield '...'
        else:
            count = self.maxcollection[level]
            yield_comma = False
            for item in obj:
                if yield_comma:
                    yield ', '
                yield_comma = True

                count -= 1
                if count <= 0:
                    yield '...'
                    break

                for p in self._repr(item, 100 if item is obj else level + 1):
                    yield p
            else:
                if comma_after_single_element:
                    if count == self.maxcollection[level] - 1:
                        yield ','
        yield suffix

    def _repr_long_iter(self, obj):
        try:
            length = hex(len(obj)) if self.convert_to_hex else len(obj)
            obj_repr = '<%s, len() = %s>' % (type(obj).__name__, length)
        except Exception:
            try:
                obj_repr = '<' + type(obj).__name__ + '>'
            except Exception:
                obj_repr = '<no repr available for object>'
        yield obj_repr

    def _repr_dict(self, obj, level, prefix, suffix,
                   item_prefix, item_sep, item_suffix):
        if not obj:
            yield prefix + suffix
            return
        if level >= len(self.maxcollection):
            yield prefix + '...' + suffix
            return

        yield prefix

        count = self.maxcollection[level]
        yield_comma = False

        if IS_PY36_OR_GREATER:
            # On Python 3.6 (onwards) dictionaries now keep
            # insertion order.
            sorted_keys = list(obj)
        else:
            try:
                sorted_keys = sorted(obj)
            except Exception:
                sorted_keys = list(obj)

        for key in sorted_keys:
            if yield_comma:
                yield ', '
            yield_comma = True

            count -= 1
            if count <= 0:
                yield '...'
                break

            yield item_prefix
            for p in self._repr(key, level + 1):
                yield p

            yield item_sep

            try:
                item = obj[key]
            except Exception:
                yield '<?>'
            else:
                for p in self._repr(item, 100 if item is obj else level + 1):
                    yield p
            yield item_suffix

        yield suffix

    def _repr_str(self, obj, level):
        try:
            if self.raw_value:
                # For raw value retrieval, ignore all limits.
                if isinstance(obj, bytes):
                    yield obj.decode('latin-1')
                else:
                    yield obj
                return

            limit_inner = self.maxother_inner
            limit_outer = self.maxother_outer
            limit = limit_inner if level > 0 else limit_outer
            if len(obj) <= limit:
                # Note that we check the limit before doing the repr (so, the final string
                # may actually be considerably bigger on some cases, as besides
                # the additional u, b, ' chars, some chars may be escaped in repr, so
                # even a single char such as \U0010ffff may end up adding more
                # chars than expected).
                yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                return

            # Slightly imprecise calculations - we may end up with a string that is
            # up to 6 characters longer than limit. If you need precise formatting,
            # you are using the wrong class.
            left_count, right_count = max(1, int(2 * limit / 3)), max(1, int(limit / 3))  # noqa

            # Important: only do repr after slicing to avoid duplicating a byte array that could be
            # huge.

            # Note: we don't deal with high surrogates here because we're not dealing with the
            # repr() of a random object.
            # i.e.: A high surrogate unicode char may be splitted on Py2, but as we do a `repr`
            # afterwards, that's ok.

            # Also, we just show the unicode/string/bytes repr() directly to make clear what the
            # input type was (so, on py2 a unicode would start with u' and on py3 a bytes would
            # start with b').

            part1 = obj[:left_count]
            part1 = repr(part1)
            part1 = part1[:part1.rindex("'")]  # Remove the last '

            part2 = obj[-right_count:]
            part2 = repr(part2)
            part2 = part2[part2.index("'") + 1:]  # Remove the first ' (and possibly u or b).

            yield part1
            yield '...'
            yield part2
        except:
            # This shouldn't really happen, but let's play it safe.
            pydev_log.exception('Error getting string representation to show.')
            for part in self._repr_obj(obj, level,
                                  self.maxother_inner, self.maxother_outer):
                yield part

    def _repr_other(self, obj, level):
        return self._repr_obj(obj, level,
                              self.maxother_inner, self.maxother_outer)

    def _repr_obj(self, obj, level, limit_inner, limit_outer):
        try:
            if self.raw_value:
                # For raw value retrieval, ignore all limits.
                if isinstance(obj, bytes):
                    yield obj.decode('latin-1')
                    return

                try:
                    mv = memoryview(obj)
                except Exception:
                    yield self._convert_to_unicode_or_bytes_repr(repr(obj))
                    return
                else:
                    # Map bytes to Unicode codepoints with same values.
                    yield mv.tobytes().decode('latin-1')
                    return
            elif self.convert_to_hex and isinstance(obj, self.int_types):
                obj_repr = hex(obj)
            else:
                obj_repr = repr(obj)
        except Exception:
            try:
                obj_repr = object.__repr__(obj)
            except Exception:
                try:
                    obj_repr = '<no repr available for ' + type(obj).__name__ + '>'  # noqa
                except Exception:
                    obj_repr = '<no repr available for object>'

        limit = limit_inner if level > 0 else limit_outer

        if limit >= len(obj_repr):
            yield self._convert_to_unicode_or_bytes_repr(obj_repr)
            return

        # Slightly imprecise calculations - we may end up with a string that is
        # up to 3 characters longer than limit. If you need precise formatting,
        # you are using the wrong class.
        left_count, right_count = max(1, int(2 * limit / 3)), max(1, int(limit / 3))  # noqa

        yield obj_repr[:left_count]
        yield '...'
        yield obj_repr[-right_count:]

    def _convert_to_unicode_or_bytes_repr(self, obj_repr):
        return obj_repr

    def _bytes_as_unicode_if_possible(self, obj_repr):
        # We try to decode with 3 possible encoding (sys.stdout.encoding,
        # locale.getpreferredencoding() and 'utf-8). If no encoding can decode
        # the input, we return the original bytes.
        try_encodings = []
        encoding = self.sys_stdout_encoding or getattr(sys.stdout, 'encoding', '')
        if encoding:
            try_encodings.append(encoding.lower())

        preferred_encoding = self.locale_preferred_encoding or locale.getpreferredencoding()
        if preferred_encoding:
            preferred_encoding = preferred_encoding.lower()
            if preferred_encoding not in try_encodings:
                try_encodings.append(preferred_encoding)

        if 'utf-8' not in try_encodings:
            try_encodings.append('utf-8')

        for encoding in try_encodings:
            try:
                return obj_repr.decode(encoding)
            except UnicodeDecodeError:
                pass

        return obj_repr  # Return the original version (in bytes)
