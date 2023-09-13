"""
Handlers for IPythonDirective's @doctest pseudo-decorator.

The Sphinx extension that provides support for embedded IPython code provides
a pseudo-decorator @doctest, which treats the input/output block as a
doctest, raising a RuntimeError during doc generation if the actual output
(after running the input) does not match the expected output.

An example usage is:

.. code-block:: rst

   .. ipython::

        In [1]: x = 1

        @doctest
        In [2]: x + 2
        Out[3]: 3

One can also provide arguments to the decorator. The first argument should be
the name of a custom handler. The specification of any other arguments is
determined by the handler. For example,

.. code-block:: rst

      .. ipython::

         @doctest float
         In [154]: 0.1 + 0.2
         Out[154]: 0.3

allows the actual output ``0.30000000000000004`` to match the expected output
due to a comparison with `np.allclose`.

This module contains handlers for the @doctest pseudo-decorator. Handlers
should have the following function signature::

    handler(sphinx_shell, args, input_lines, found, submitted)

where `sphinx_shell` is the embedded Sphinx shell, `args` contains the list
of arguments that follow: '@doctest handler_name', `input_lines` contains
a list of the lines relevant to the current doctest, `found` is a string
containing the output from the IPython shell, and `submitted` is a string
containing the expected output from the IPython shell.

Handlers must be registered in the `doctests` dict at the end of this module.

"""

def str_to_array(s):
    """
    Simplistic converter of strings from repr to float NumPy arrays.

    If the repr representation has ellipsis in it, then this will fail.

    Parameters
    ----------
    s : str
        The repr version of a NumPy array.

    Examples
    --------
    >>> s = "array([ 0.3,  inf,  nan])"
    >>> a = str_to_array(s)

    """
    import numpy as np

    # Need to make sure eval() knows about inf and nan.
    # This also assumes default printoptions for NumPy.
    from numpy import inf, nan

    if s.startswith(u'array'):
        # Remove array( and )
        s = s[6:-1]

    if s.startswith(u'['):
        a = np.array(eval(s), dtype=float)
    else:
        # Assume its a regular float. Force 1D so we can index into it.
        a = np.atleast_1d(float(s))
    return a

def float_doctest(sphinx_shell, args, input_lines, found, submitted):
    """
    Doctest which allow the submitted output to vary slightly from the input.

    Here is how it might appear in an rst file:

    .. code-block:: rst

       .. ipython::

          @doctest float
          In [1]: 0.1 + 0.2
          Out[1]: 0.3

    """
    import numpy as np

    if len(args) == 2:
        rtol = 1e-05
        atol = 1e-08
    else:
        # Both must be specified if any are specified.
        try:
            rtol = float(args[2])
            atol = float(args[3])
        except IndexError as e:
            e = ("Both `rtol` and `atol` must be specified "
                 "if either are specified: {0}".format(args))
            raise IndexError(e) from e

    try:
        submitted = str_to_array(submitted)
        found = str_to_array(found)
    except:
        # For example, if the array is huge and there are ellipsis in it.
        error = True
    else:
        found_isnan = np.isnan(found)
        submitted_isnan = np.isnan(submitted)
        error = not np.allclose(found_isnan, submitted_isnan)
        error |= not np.allclose(found[~found_isnan],
                                 submitted[~submitted_isnan],
                                 rtol=rtol, atol=atol)

    TAB = ' ' * 4
    directive = sphinx_shell.directive
    if directive is None:
        source = 'Unavailable'
        content = 'Unavailable'
    else:
        source = directive.state.document.current_source
        # Add tabs and make into a single string.
        content = '\n'.join([TAB + line for line in directive.content])

    if error:

        e = ('doctest float comparison failure\n\n'
             'Document source: {0}\n\n'
             'Raw content: \n{1}\n\n'
             'On input line(s):\n{TAB}{2}\n\n'
             'we found output:\n{TAB}{3}\n\n'
             'instead of the expected:\n{TAB}{4}\n\n')
        e = e.format(source, content, '\n'.join(input_lines), repr(found),
                     repr(submitted), TAB=TAB)
        raise RuntimeError(e)

# dict of allowable doctest handlers. The key represents the first argument
# that must be given to @doctest in order to activate the handler.
doctests = {
    'float': float_doctest,
}
