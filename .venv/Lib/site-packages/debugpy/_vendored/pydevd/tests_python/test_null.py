def test_null():
    from _pydevd_bundle.pydevd_constants import Null
    null = Null()
    assert not null
    assert len(null) == 0
    
    with null as n:
        n.write('foo')