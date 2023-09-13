import pytest


def some_call():
    assert 0  # raise here


def test_example():
    some_call()  # stop here


if __name__ == '__main__':
    pytest.main([__file__, '--capture=no', '--noconftest'])
    print('TEST SUCEEDED!')
