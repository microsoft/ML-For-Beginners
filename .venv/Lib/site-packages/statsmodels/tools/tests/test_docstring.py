import pytest

from statsmodels.tools.docstring import Docstring, remove_parameters, Parameter

good = """
This is the summary.

This is the extended summary.

Parameters
----------
x : int
    The first parameter.
y : float
    The second parameter.
z : {int, float, None}
    The final parameter.

Returns
-------
float
    Some floating point value.

See Also
--------
statsmodels.api
    The main API location.

Notes
-----
This is where the notes go.

.. index: default
   :refguide: something, else, and more

References
----------
.. [*] Reference 1 here

Examples
--------
Using the API is simple

>>> import statsmodels.api
"""

bad = """
Returns
-------
float
    Some floating point value.

Unknown
-------
I don't know what this section does.
"""

repeat = """
Returns
-------
float
    Some floating point value.

Returns
-------
float
    Some floating point value.
"""

bad_yields = """
Returns
-------
float
    The return.

Yields
------
float
    Can't also yield.
"""

with_sig = """
func(x)
func(x, y)
func(x, y, z=1)
""" + good


def test_remove_parameter():
    ds = Docstring(good)
    ds.remove_parameters('x')
    assert 'x : int' not in str(ds)

    ds = Docstring(good)
    ds.remove_parameters(['x', 'y'])
    assert 'x : int' not in str(ds)
    assert 'y : float' not in str(ds)

    with pytest.raises(ValueError):
        Docstring(good).remove_parameters(['w'])

    ds = remove_parameters(good, 'x')
    assert 'x : int' not in ds
    assert isinstance(ds, str)


def test_insert_parameters():
    new = Parameter('w', 'ndarray', ['An array input.'])
    ds = Docstring(good)
    ds.insert_parameters('y', new)
    assert 'w : ndarray' in str(ds)
    assert 'An array input.' in str(ds)

    other = Parameter('q', 'DataFrame', ['A pandas dataframe.'])
    ds = Docstring(good)
    ds.insert_parameters(None, [new, other])
    assert 'w : ndarray' in str(ds)
    assert 'An array input.' in str(ds)
    assert 'q : DataFrame' in str(ds)
    assert 'A pandas dataframe.' in str(ds)
    assert '---\nw : ndarray' in str(ds)

    ds = Docstring(good)
    with pytest.raises(ValueError):
        ds.insert_parameters('unknown', new)


def test_set_unknown():
    ds = Docstring(good)
    with pytest.raises(ValueError):
        ds._ds['Unknown'] = ['unknown']


def test_replace_block():
    ds = Docstring(good)
    ds.replace_block('summary', ['The is the new summary.'])
    assert 'The is the new summary.' in str(ds)

    ds = Docstring(good)
    ds.replace_block('summary', 'The is the new summary.')
    assert 'The is the new summary.' in str(ds)

    with pytest.raises(ValueError):
        ds.replace_block('unknown', ['The is the new summary.'])


def test_repeat():
    with pytest.raises(ValueError):
        Docstring(repeat)


def test_bad():
    with pytest.raises(ValueError):
        Docstring(bad)


def test_empty_ds():
    ds = Docstring(None)
    ds.replace_block('summary', ['The is the new summary.'])

    ds.remove_parameters('x')

    new = Parameter('w', 'ndarray', ['An array input.'])
    ds.insert_parameters('y', new)
    assert str(ds) == 'None'


def test_yield_return():
    with pytest.raises(ValueError):
        Docstring(bad_yields)


def test_multiple_sig():
    Docstring(with_sig)
