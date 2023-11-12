# -*- coding: utf-8 -*-
import pytest
from numpy.testing import assert_equal

from statsmodels.tools.decorators import (cache_readonly, deprecated_alias)


def test_cache_readonly():

    class Example:
        def __init__(self):
            self._cache = {}
            self.a = 0

        @cache_readonly
        def b(self):
            return 1

    ex = Example()

    # Try accessing/setting a readonly attribute
    assert_equal(ex.__dict__, dict(a=0, _cache={}))

    b = ex.b
    assert_equal(b, 1)
    assert_equal(ex.__dict__, dict(a=0, _cache=dict(b=1,)))
    # assert_equal(ex.__dict__, dict(a=0, b=1, _cache=dict(b=1)))

    with pytest.raises(AttributeError):
        ex.b = -1

    assert_equal(ex._cache, dict(b=1,))


def dummy_factory(msg, remove_version, warning):
    class Dummy:
        y = deprecated_alias('y', 'x',
                             remove_version=remove_version,
                             msg=msg,
                             warning=warning)

        def __init__(self, y):
            self.x = y

    return Dummy(1)


@pytest.mark.parametrize('warning', [FutureWarning, UserWarning])
@pytest.mark.parametrize('remove_version', [None, '0.11'])
@pytest.mark.parametrize('msg', ['test message', None])
def test_deprecated_alias(msg, remove_version, warning):
    dummy_set = dummy_factory(msg, remove_version, warning)
    with pytest.warns(warning) as w:
        dummy_set.y = 2
        assert dummy_set.x == 2

    assert warning.__class__ is w[0].category.__class__

    dummy_get = dummy_factory(msg, remove_version, warning)
    with pytest.warns(warning) as w:
        x = dummy_get.y
        assert x == 1

    assert warning.__class__ is w[0].category.__class__
    message = str(w[0].message)
    if not msg:
        if remove_version:
            assert 'will be removed' in message
        else:
            assert 'will be removed' not in message
    else:
        assert msg in message
