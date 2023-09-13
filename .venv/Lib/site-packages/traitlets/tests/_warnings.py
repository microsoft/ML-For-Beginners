# From scikit-image: https://github.com/scikit-image/scikit-image/blob/c2f8c4ab123ebe5f7b827bc495625a32bb225c10/skimage/_shared/_warnings.py
# Licensed under modified BSD license

__all__ = ["all_warnings", "expected_warnings"]

import inspect
import os
import re
import sys
import warnings
from contextlib import contextmanager
from unittest import mock


@contextmanager
def all_warnings():
    """
    Context for use in testing to ensure that all warnings are raised.
    Examples
    --------
    >>> import warnings
    >>> def foo():
    ...     warnings.warn(RuntimeWarning("bar"))

    We raise the warning once, while the warning filter is set to "once".
    Hereafter, the warning is invisible, even with custom filters:
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('once')
    ...     foo()

    We can now run ``foo()`` without a warning being raised:
    >>> from numpy.testing import assert_warns  # doctest: +SKIP
    >>> foo()  # doctest: +SKIP

    To catch the warning, we call in the help of ``all_warnings``:
    >>> with all_warnings():  # doctest: +SKIP
    ...     assert_warns(RuntimeWarning, foo)
    """

    # Whenever a warning is triggered, Python adds a __warningregistry__
    # member to the *calling* module.  The exercize here is to find
    # and eradicate all those breadcrumbs that were left lying around.
    #
    # We proceed by first searching all parent calling frames and explicitly
    # clearing their warning registries (necessary for the doctests above to
    # pass).  Then, we search for all submodules of skimage and clear theirs
    # as well (necessary for the skimage test suite to pass).

    frame = inspect.currentframe()
    if frame:
        for f in inspect.getouterframes(frame):
            f[0].f_locals["__warningregistry__"] = {}
    del frame

    for _, mod in list(sys.modules.items()):
        try:
            mod.__warningregistry__.clear()
        except AttributeError:
            pass

    with warnings.catch_warnings(record=True) as w, mock.patch.dict(
        os.environ, {"TRAITLETS_ALL_DEPRECATIONS": "1"}
    ):
        warnings.simplefilter("always")
        yield w


@contextmanager
def expected_warnings(matching):
    r"""Context for use in testing to catch known warnings matching regexes

    Parameters
    ----------
    matching : list of strings or compiled regexes
        Regexes for the desired warning to catch

    Examples
    --------
    >>> from skimage import data, img_as_ubyte, img_as_float  # doctest: +SKIP
    >>> with expected_warnings(["precision loss"]):           # doctest: +SKIP
    ...     d = img_as_ubyte(img_as_float(data.coins()))      # doctest: +SKIP

    Notes
    -----
    Uses `all_warnings` to ensure all warnings are raised.
    Upon exiting, it checks the recorded warnings for the desired matching
    pattern(s).
    Raises a ValueError if any match was not found or an unexpected
    warning was raised.
    Allows for three types of behaviors: "and", "or", and "optional" matches.
    This is done to accomodate different build enviroments or loop conditions
    that may produce different warnings.  The behaviors can be combined.
    If you pass multiple patterns, you get an orderless "and", where all of the
    warnings must be raised.
    If you use the "|" operator in a pattern, you can catch one of several warnings.
    Finally, you can use "|\A\Z" in a pattern to signify it as optional.
    """
    with all_warnings() as w:
        # enter context
        yield w
        # exited user context, check the recorded warnings
        remaining = [m for m in matching if r"\A\Z" not in m.split("|")]
        for warn in w:
            found = False
            for match in matching:
                if re.search(match, str(warn.message)) is not None:
                    found = True
                    if match in remaining:
                        remaining.remove(match)
            if not found:
                raise ValueError("Unexpected warning: %s" % str(warn.message))
        if len(remaining) > 0:
            msg = "No warning raised matching:\n%s" % "\n".join(remaining)
            raise ValueError(msg)
