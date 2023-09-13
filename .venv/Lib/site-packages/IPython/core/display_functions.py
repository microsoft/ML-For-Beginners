# -*- coding: utf-8 -*-
"""Top-level display functions for displaying object in different formats."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


from binascii import b2a_hex
import os
import sys
import warnings

__all__ = ['display', 'clear_output', 'publish_display_data', 'update_display', 'DisplayHandle']

#-----------------------------------------------------------------------------
# utility functions
#-----------------------------------------------------------------------------


def _merge(d1, d2):
    """Like update, but merges sub-dicts instead of clobbering at the top level.

    Updates d1 in-place
    """

    if not isinstance(d2, dict) or not isinstance(d1, dict):
        return d2
    for key, value in d2.items():
        d1[key] = _merge(d1.get(key), value)
    return d1


#-----------------------------------------------------------------------------
# Main functions
#-----------------------------------------------------------------------------

class _Sentinel:
    def __repr__(self):
        return "<deprecated>"


_sentinel = _Sentinel()

# use * to indicate transient is keyword-only
def publish_display_data(
    data, metadata=None, source=_sentinel, *, transient=None, **kwargs
):
    """Publish data and metadata to all frontends.

    See the ``display_data`` message in the messaging documentation for
    more details about this message type.

    Keys of data and metadata can be any mime-type.

    Parameters
    ----------
    data : dict
        A dictionary having keys that are valid MIME types (like
        'text/plain' or 'image/svg+xml') and values that are the data for
        that MIME type. The data itself must be a JSON'able data
        structure. Minimally all data should have the 'text/plain' data,
        which can be displayed by all frontends. If more than the plain
        text is given, it is up to the frontend to decide which
        representation to use.
    metadata : dict
        A dictionary for metadata related to the data. This can contain
        arbitrary key, value pairs that frontends can use to interpret
        the data. mime-type keys matching those in data can be used
        to specify metadata about particular representations.
    source : str, deprecated
        Unused.
    transient : dict, keyword-only
        A dictionary of transient data, such as display_id.
    """
    from IPython.core.interactiveshell import InteractiveShell

    if source is not _sentinel:
        warnings.warn(
            "The `source` parameter emit a  deprecation warning since"
            " IPython 8.0, it had no effects for a long time and will "
            " be removed in future versions.",
            DeprecationWarning,
            stacklevel=2,
        )
    display_pub = InteractiveShell.instance().display_pub

    # only pass transient if supplied,
    # to avoid errors with older ipykernel.
    # TODO: We could check for ipykernel version and provide a detailed upgrade message.
    if transient:
        kwargs['transient'] = transient

    display_pub.publish(
        data=data,
        metadata=metadata,
        **kwargs
    )


def _new_id():
    """Generate a new random text id with urandom"""
    return b2a_hex(os.urandom(16)).decode('ascii')


def display(
    *objs,
    include=None,
    exclude=None,
    metadata=None,
    transient=None,
    display_id=None,
    raw=False,
    clear=False,
    **kwargs
):
    """Display a Python object in all frontends.

    By default all representations will be computed and sent to the frontends.
    Frontends can decide which representation is used and how.

    In terminal IPython this will be similar to using :func:`print`, for use in richer
    frontends see Jupyter notebook examples with rich display logic.

    Parameters
    ----------
    *objs : object
        The Python objects to display.
    raw : bool, optional
        Are the objects to be displayed already mimetype-keyed dicts of raw display data,
        or Python objects that need to be formatted before display? [default: False]
    include : list, tuple or set, optional
        A list of format type strings (MIME types) to include in the
        format data dict. If this is set *only* the format types included
        in this list will be computed.
    exclude : list, tuple or set, optional
        A list of format type strings (MIME types) to exclude in the format
        data dict. If this is set all format types will be computed,
        except for those included in this argument.
    metadata : dict, optional
        A dictionary of metadata to associate with the output.
        mime-type keys in this dictionary will be associated with the individual
        representation formats, if they exist.
    transient : dict, optional
        A dictionary of transient data to associate with the output.
        Data in this dict should not be persisted to files (e.g. notebooks).
    display_id : str, bool optional
        Set an id for the display.
        This id can be used for updating this display area later via update_display.
        If given as `True`, generate a new `display_id`
    clear : bool, optional
        Should the output area be cleared before displaying anything? If True,
        this will wait for additional output before clearing. [default: False]
    **kwargs : additional keyword-args, optional
        Additional keyword-arguments are passed through to the display publisher.

    Returns
    -------
    handle: DisplayHandle
        Returns a handle on updatable displays for use with :func:`update_display`,
        if `display_id` is given. Returns :any:`None` if no `display_id` is given
        (default).

    Examples
    --------
    >>> class Json(object):
    ...     def __init__(self, json):
    ...         self.json = json
    ...     def _repr_pretty_(self, pp, cycle):
    ...         import json
    ...         pp.text(json.dumps(self.json, indent=2))
    ...     def __repr__(self):
    ...         return str(self.json)
    ...

    >>> d = Json({1:2, 3: {4:5}})

    >>> print(d)
    {1: 2, 3: {4: 5}}

    >>> display(d)
    {
      "1": 2,
      "3": {
        "4": 5
      }
    }

    >>> def int_formatter(integer, pp, cycle):
    ...     pp.text('I'*integer)

    >>> plain = get_ipython().display_formatter.formatters['text/plain']
    >>> plain.for_type(int, int_formatter)
    <function _repr_pprint at 0x...>
    >>> display(7-5)
    II

    >>> del plain.type_printers[int]
    >>> display(7-5)
    2

    See Also
    --------
    :func:`update_display`

    Notes
    -----
    In Python, objects can declare their textual representation using the
    `__repr__` method. IPython expands on this idea and allows objects to declare
    other, rich representations including:

      - HTML
      - JSON
      - PNG
      - JPEG
      - SVG
      - LaTeX

    A single object can declare some or all of these representations; all are
    handled by IPython's display system.

    The main idea of the first approach is that you have to implement special
    display methods when you define your class, one for each representation you
    want to use. Here is a list of the names of the special methods and the
    values they must return:

      - `_repr_html_`: return raw HTML as a string, or a tuple (see below).
      - `_repr_json_`: return a JSONable dict, or a tuple (see below).
      - `_repr_jpeg_`: return raw JPEG data, or a tuple (see below).
      - `_repr_png_`: return raw PNG data, or a tuple (see below).
      - `_repr_svg_`: return raw SVG data as a string, or a tuple (see below).
      - `_repr_latex_`: return LaTeX commands in a string surrounded by "$",
                        or a tuple (see below).
      - `_repr_mimebundle_`: return a full mimebundle containing the mapping
                             from all mimetypes to data.
                             Use this for any mime-type not listed above.

    The above functions may also return the object's metadata alonside the
    data.  If the metadata is available, the functions will return a tuple
    containing the data and metadata, in that order.  If there is no metadata
    available, then the functions will return the data only.

    When you are directly writing your own classes, you can adapt them for
    display in IPython by following the above approach. But in practice, you
    often need to work with existing classes that you can't easily modify.

    You can refer to the documentation on integrating with the display system in
    order to register custom formatters for already existing types
    (:ref:`integrating_rich_display`).

    .. versionadded:: 5.4 display available without import
    .. versionadded:: 6.1 display available without import

    Since IPython 5.4 and 6.1 :func:`display` is automatically made available to
    the user without import. If you are using display in a document that might
    be used in a pure python context or with older version of IPython, use the
    following import at the top of your file::

        from IPython.display import display

    """
    from IPython.core.interactiveshell import InteractiveShell

    if not InteractiveShell.initialized():
        # Directly print objects.
        print(*objs)
        return

    if transient is None:
        transient = {}
    if metadata is None:
        metadata={}
    if display_id:
        if display_id is True:
            display_id = _new_id()
        transient['display_id'] = display_id
    if kwargs.get('update') and 'display_id' not in transient:
        raise TypeError('display_id required for update_display')
    if transient:
        kwargs['transient'] = transient

    if not objs and display_id:
        # if given no objects, but still a request for a display_id,
        # we assume the user wants to insert an empty output that
        # can be updated later
        objs = [{}]
        raw = True

    if not raw:
        format = InteractiveShell.instance().display_formatter.format

    if clear:
        clear_output(wait=True)

    for obj in objs:
        if raw:
            publish_display_data(data=obj, metadata=metadata, **kwargs)
        else:
            format_dict, md_dict = format(obj, include=include, exclude=exclude)
            if not format_dict:
                # nothing to display (e.g. _ipython_display_ took over)
                continue
            if metadata:
                # kwarg-specified metadata gets precedence
                _merge(md_dict, metadata)
            publish_display_data(data=format_dict, metadata=md_dict, **kwargs)
    if display_id:
        return DisplayHandle(display_id)


# use * for keyword-only display_id arg
def update_display(obj, *, display_id, **kwargs):
    """Update an existing display by id

    Parameters
    ----------
    obj
        The object with which to update the display
    display_id : keyword-only
        The id of the display to update

    See Also
    --------
    :func:`display`
    """
    kwargs['update'] = True
    display(obj, display_id=display_id, **kwargs)


class DisplayHandle(object):
    """A handle on an updatable display

    Call `.update(obj)` to display a new object.

    Call `.display(obj`) to add a new instance of this display,
    and update existing instances.

    See Also
    --------

        :func:`display`, :func:`update_display`

    """

    def __init__(self, display_id=None):
        if display_id is None:
            display_id = _new_id()
        self.display_id = display_id

    def __repr__(self):
        return "<%s display_id=%s>" % (self.__class__.__name__, self.display_id)

    def display(self, obj, **kwargs):
        """Make a new display with my id, updating existing instances.

        Parameters
        ----------
        obj
            object to display
        **kwargs
            additional keyword arguments passed to display
        """
        display(obj, display_id=self.display_id, **kwargs)

    def update(self, obj, **kwargs):
        """Update existing displays with my id

        Parameters
        ----------
        obj
            object to display
        **kwargs
            additional keyword arguments passed to update_display
        """
        update_display(obj, display_id=self.display_id, **kwargs)


def clear_output(wait=False):
    """Clear the output of the current cell receiving output.

    Parameters
    ----------
    wait : bool [default: false]
        Wait to clear the output until new output is available to replace it."""
    from IPython.core.interactiveshell import InteractiveShell
    if InteractiveShell.initialized():
        InteractiveShell.instance().display_pub.clear_output(wait)
    else:
        print('\033[2K\r', end='')
        sys.stdout.flush()
        print('\033[2K\r', end='')
        sys.stderr.flush()
