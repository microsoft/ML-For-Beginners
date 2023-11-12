"""
The :mod:`imblearn.exceptions` module includes all custom warnings and error
classes and functions used across imbalanced-learn.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT


def raise_isinstance_error(variable_name, possible_type, variable):
    """Raise consistent error message for isinstance() function.

    Parameters
    ----------
    variable_name : str
        The name of the variable.

    possible_type : type
        The possible type of the variable.

    variable : object
        The variable to check.

    Raises
    ------
    ValueError
        If the instance is not of the possible type.
    """
    raise ValueError(
        f"{variable_name} has to be one of {possible_type}. "
        f"Got {type(variable)} instead."
    )
