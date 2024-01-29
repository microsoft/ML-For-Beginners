# -*- coding: utf-8 -*-
"""Top-level display functions for displaying object in different formats."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


from binascii import b2a_base64, hexlify
import html
import json
import mimetypes
import os
import struct
import warnings
from copy import deepcopy
from os.path import splitext
from pathlib import Path, PurePath

from IPython.utils.py3compat import cast_unicode
from IPython.testing.skipdoctest import skip_doctest
from . import display_functions


__all__ = ['display_pretty', 'display_html', 'display_markdown',
           'display_svg', 'display_png', 'display_jpeg', 'display_latex', 'display_json',
           'display_javascript', 'display_pdf', 'DisplayObject', 'TextDisplayObject',
           'Pretty', 'HTML', 'Markdown', 'Math', 'Latex', 'SVG', 'ProgressBar', 'JSON',
           'GeoJSON', 'Javascript', 'Image', 'set_matplotlib_formats',
           'set_matplotlib_close',
           'Video']

_deprecated_names = ["display", "clear_output", "publish_display_data", "update_display", "DisplayHandle"]

__all__ = __all__ + _deprecated_names


# ----- warn to import from IPython.display -----

from warnings import warn


def __getattr__(name):
    if name in _deprecated_names:
        warn(f"Importing {name} from IPython.core.display is deprecated since IPython 7.14, please import from IPython display", DeprecationWarning, stacklevel=2)
        return getattr(display_functions, name)

    if name in globals().keys():
        return globals()[name]
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


#-----------------------------------------------------------------------------
# utility functions
#-----------------------------------------------------------------------------

def _safe_exists(path):
    """Check path, but don't let exceptions raise"""
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _display_mimetype(mimetype, objs, raw=False, metadata=None):
    """internal implementation of all display_foo methods

    Parameters
    ----------
    mimetype : str
        The mimetype to be published (e.g. 'image/png')
    *objs : object
        The Python objects to display, or if raw=True raw text data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    if metadata:
        metadata = {mimetype: metadata}
    if raw:
        # turn list of pngdata into list of { 'image/png': pngdata }
        objs = [ {mimetype: obj} for obj in objs ]
    display_functions.display(*objs, raw=raw, metadata=metadata, include=[mimetype])

#-----------------------------------------------------------------------------
# Main functions
#-----------------------------------------------------------------------------


def display_pretty(*objs, **kwargs):
    """Display the pretty (default) representation of an object.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw text data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('text/plain', objs, **kwargs)


def display_html(*objs, **kwargs):
    """Display the HTML representation of an object.

    Note: If raw=False and the object does not have a HTML
    representation, no HTML will be shown.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw HTML data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('text/html', objs, **kwargs)


def display_markdown(*objs, **kwargs):
    """Displays the Markdown representation of an object.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw markdown data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """

    _display_mimetype('text/markdown', objs, **kwargs)


def display_svg(*objs, **kwargs):
    """Display the SVG representation of an object.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw svg data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('image/svg+xml', objs, **kwargs)


def display_png(*objs, **kwargs):
    """Display the PNG representation of an object.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw png data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('image/png', objs, **kwargs)


def display_jpeg(*objs, **kwargs):
    """Display the JPEG representation of an object.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw JPEG data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('image/jpeg', objs, **kwargs)


def display_latex(*objs, **kwargs):
    """Display the LaTeX representation of an object.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw latex data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('text/latex', objs, **kwargs)


def display_json(*objs, **kwargs):
    """Display the JSON representation of an object.

    Note that not many frontends support displaying JSON.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw json data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('application/json', objs, **kwargs)


def display_javascript(*objs, **kwargs):
    """Display the Javascript representation of an object.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw javascript data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('application/javascript', objs, **kwargs)


def display_pdf(*objs, **kwargs):
    """Display the PDF representation of an object.

    Parameters
    ----------
    *objs : object
        The Python objects to display, or if raw=True raw javascript data to
        display.
    raw : bool
        Are the data objects raw data or Python objects that need to be
        formatted before display? [default: False]
    metadata : dict (optional)
        Metadata to be associated with the specific mimetype output.
    """
    _display_mimetype('application/pdf', objs, **kwargs)


#-----------------------------------------------------------------------------
# Smart classes
#-----------------------------------------------------------------------------


class DisplayObject(object):
    """An object that wraps data to be displayed."""

    _read_flags = 'r'
    _show_mem_addr = False
    metadata = None

    def __init__(self, data=None, url=None, filename=None, metadata=None):
        """Create a display object given raw data.

        When this object is returned by an expression or passed to the
        display function, it will result in the data being displayed
        in the frontend. The MIME type of the data should match the
        subclasses used, so the Png subclass should be used for 'image/png'
        data. If the data is a URL, the data will first be downloaded
        and then displayed.

        Parameters
        ----------
        data : unicode, str or bytes
            The raw data or a URL or file to load the data from
        url : unicode
            A URL to download the data from.
        filename : unicode
            Path to a local file to load the data from.
        metadata : dict
            Dict of metadata associated to be the object when displayed
        """
        if isinstance(data, (Path, PurePath)):
            data = str(data)

        if data is not None and isinstance(data, str):
            if data.startswith('http') and url is None:
                url = data
                filename = None
                data = None
            elif _safe_exists(data) and filename is None:
                url = None
                filename = data
                data = None

        self.url = url
        self.filename = filename
        # because of @data.setter methods in
        # subclasses ensure url and filename are set
        # before assigning to self.data
        self.data = data

        if metadata is not None:
            self.metadata = metadata
        elif self.metadata is None:
            self.metadata = {}

        self.reload()
        self._check_data()

    def __repr__(self):
        if not self._show_mem_addr:
            cls = self.__class__
            r = "<%s.%s object>" % (cls.__module__, cls.__name__)
        else:
            r = super(DisplayObject, self).__repr__()
        return r

    def _check_data(self):
        """Override in subclasses if there's something to check."""
        pass

    def _data_and_metadata(self):
        """shortcut for returning metadata with shape information, if defined"""
        if self.metadata:
            return self.data, deepcopy(self.metadata)
        else:
            return self.data

    def reload(self):
        """Reload the raw data from file or URL."""
        if self.filename is not None:
            encoding = None if "b" in self._read_flags else "utf-8"
            with open(self.filename, self._read_flags, encoding=encoding) as f:
                self.data = f.read()
        elif self.url is not None:
            # Deferred import
            from urllib.request import urlopen
            response = urlopen(self.url)
            data = response.read()
            # extract encoding from header, if there is one:
            encoding = None
            if 'content-type' in response.headers:
                for sub in response.headers['content-type'].split(';'):
                    sub = sub.strip()
                    if sub.startswith('charset'):
                        encoding = sub.split('=')[-1].strip()
                        break
            if 'content-encoding' in response.headers:
                # TODO: do deflate?
                if 'gzip' in response.headers['content-encoding']:
                    import gzip
                    from io import BytesIO

                    # assume utf-8 if encoding is not specified
                    with gzip.open(
                        BytesIO(data), "rt", encoding=encoding or "utf-8"
                    ) as fp:
                        encoding = None
                        data = fp.read()

            # decode data, if an encoding was specified
            # We only touch self.data once since
            # subclasses such as SVG have @data.setter methods
            # that transform self.data into ... well svg.
            if encoding:
                self.data = data.decode(encoding, 'replace')
            else:
                self.data = data


class TextDisplayObject(DisplayObject):
    """Create a text display object given raw data.

    Parameters
    ----------
    data : str or unicode
        The raw data or a URL or file to load the data from.
    url : unicode
        A URL to download the data from.
    filename : unicode
        Path to a local file to load the data from.
    metadata : dict
        Dict of metadata associated to be the object when displayed
    """
    def _check_data(self):
        if self.data is not None and not isinstance(self.data, str):
            raise TypeError("%s expects text, not %r" % (self.__class__.__name__, self.data))

class Pretty(TextDisplayObject):

    def _repr_pretty_(self, pp, cycle):
        return pp.text(self.data)


class HTML(TextDisplayObject):

    def __init__(self, data=None, url=None, filename=None, metadata=None):
        def warn():
            if not data:
                return False

            #
            # Avoid calling lower() on the entire data, because it could be a
            # long string and we're only interested in its beginning and end.
            #
            prefix = data[:10].lower()
            suffix = data[-10:].lower()
            return prefix.startswith("<iframe ") and suffix.endswith("</iframe>")

        if warn():
            warnings.warn("Consider using IPython.display.IFrame instead")
        super(HTML, self).__init__(data=data, url=url, filename=filename, metadata=metadata)

    def _repr_html_(self):
        return self._data_and_metadata()

    def __html__(self):
        """
        This method exists to inform other HTML-using modules (e.g. Markupsafe,
        htmltag, etc) that this object is HTML and does not need things like
        special characters (<>&) escaped.
        """
        return self._repr_html_()


class Markdown(TextDisplayObject):

    def _repr_markdown_(self):
        return self._data_and_metadata()


class Math(TextDisplayObject):

    def _repr_latex_(self):
        s = r"$\displaystyle %s$" % self.data.strip('$')
        if self.metadata:
            return s, deepcopy(self.metadata)
        else:
            return s


class Latex(TextDisplayObject):

    def _repr_latex_(self):
        return self._data_and_metadata()


class SVG(DisplayObject):
    """Embed an SVG into the display.

    Note if you just want to view a svg image via a URL use `:class:Image` with
    a url=URL keyword argument.
    """

    _read_flags = 'rb'
    # wrap data in a property, which extracts the <svg> tag, discarding
    # document headers
    _data = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, svg):
        if svg is None:
            self._data = None
            return
        # parse into dom object
        from xml.dom import minidom
        x = minidom.parseString(svg)
        # get svg tag (should be 1)
        found_svg = x.getElementsByTagName('svg')
        if found_svg:
            svg = found_svg[0].toxml()
        else:
            # fallback on the input, trust the user
            # but this is probably an error.
            pass
        svg = cast_unicode(svg)
        self._data = svg

    def _repr_svg_(self):
        return self._data_and_metadata()

class ProgressBar(DisplayObject):
    """Progressbar supports displaying a progressbar like element
    """
    def __init__(self, total):
        """Creates a new progressbar

        Parameters
        ----------
        total : int
            maximum size of the progressbar
        """
        self.total = total
        self._progress = 0
        self.html_width = '60ex'
        self.text_width = 60
        self._display_id = hexlify(os.urandom(8)).decode('ascii')

    def __repr__(self):
        fraction = self.progress / self.total
        filled = '=' * int(fraction * self.text_width)
        rest = ' ' * (self.text_width - len(filled))
        return '[{}{}] {}/{}'.format(
            filled, rest,
            self.progress, self.total,
        )

    def _repr_html_(self):
        return "<progress style='width:{}' max='{}' value='{}'></progress>".format(
            self.html_width, self.total, self.progress)

    def display(self):
        display_functions.display(self, display_id=self._display_id)

    def update(self):
        display_functions.display(self, display_id=self._display_id, update=True)

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value):
        self._progress = value
        self.update()

    def __iter__(self):
        self.display()
        self._progress = -1 # First iteration is 0
        return self

    def __next__(self):
        """Returns current value and increments display by one."""
        self.progress += 1
        if self.progress < self.total:
            return self.progress
        else:
            raise StopIteration()

class JSON(DisplayObject):
    """JSON expects a JSON-able dict or list

    not an already-serialized JSON string.

    Scalar types (None, number, string) are not allowed, only dict or list containers.
    """
    # wrap data in a property, which warns about passing already-serialized JSON
    _data = None
    def __init__(self, data=None, url=None, filename=None, expanded=False, metadata=None, root='root', **kwargs):
        """Create a JSON display object given raw data.

        Parameters
        ----------
        data : dict or list
            JSON data to display. Not an already-serialized JSON string.
            Scalar types (None, number, string) are not allowed, only dict
            or list containers.
        url : unicode
            A URL to download the data from.
        filename : unicode
            Path to a local file to load the data from.
        expanded : boolean
            Metadata to control whether a JSON display component is expanded.
        metadata : dict
            Specify extra metadata to attach to the json display object.
        root : str
            The name of the root element of the JSON tree
        """
        self.metadata = {
            'expanded': expanded,
            'root': root,
        }
        if metadata:
            self.metadata.update(metadata)
        if kwargs:
            self.metadata.update(kwargs)
        super(JSON, self).__init__(data=data, url=url, filename=filename)

    def _check_data(self):
        if self.data is not None and not isinstance(self.data, (dict, list)):
            raise TypeError("%s expects JSONable dict or list, not %r" % (self.__class__.__name__, self.data))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, (Path, PurePath)):
            data = str(data)

        if isinstance(data, str):
            if self.filename is None and self.url is None:
                warnings.warn("JSON expects JSONable dict or list, not JSON strings")
            data = json.loads(data)
        self._data = data

    def _data_and_metadata(self):
        return self.data, self.metadata

    def _repr_json_(self):
        return self._data_and_metadata()


_css_t = """var link = document.createElement("link");
	link.rel = "stylesheet";
	link.type = "text/css";
	link.href = "%s";
	document.head.appendChild(link);
"""

_lib_t1 = """new Promise(function(resolve, reject) {
	var script = document.createElement("script");
	script.onload = resolve;
	script.onerror = reject;
	script.src = "%s";
	document.head.appendChild(script);
}).then(() => {
"""

_lib_t2 = """
});"""

class GeoJSON(JSON):
    """GeoJSON expects JSON-able dict

    not an already-serialized JSON string.

    Scalar types (None, number, string) are not allowed, only dict containers.
    """

    def __init__(self, *args, **kwargs):
        """Create a GeoJSON display object given raw data.

        Parameters
        ----------
        data : dict or list
            VegaLite data. Not an already-serialized JSON string.
            Scalar types (None, number, string) are not allowed, only dict
            or list containers.
        url_template : string
            Leaflet TileLayer URL template: http://leafletjs.com/reference.html#url-template
        layer_options : dict
            Leaflet TileLayer options: http://leafletjs.com/reference.html#tilelayer-options
        url : unicode
            A URL to download the data from.
        filename : unicode
            Path to a local file to load the data from.
        metadata : dict
            Specify extra metadata to attach to the json display object.

        Examples
        --------
        The following will display an interactive map of Mars with a point of
        interest on frontend that do support GeoJSON display.

            >>> from IPython.display import GeoJSON

            >>> GeoJSON(data={
            ...     "type": "Feature",
            ...     "geometry": {
            ...         "type": "Point",
            ...         "coordinates": [-81.327, 296.038]
            ...     }
            ... },
            ... url_template="http://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/{basemap_id}/{z}/{x}/{y}.png",
            ... layer_options={
            ...     "basemap_id": "celestia_mars-shaded-16k_global",
            ...     "attribution" : "Celestia/praesepe",
            ...     "minZoom" : 0,
            ...     "maxZoom" : 18,
            ... })
            <IPython.core.display.GeoJSON object>

        In the terminal IPython, you will only see the text representation of
        the GeoJSON object.

        """

        super(GeoJSON, self).__init__(*args, **kwargs)


    def _ipython_display_(self):
        bundle = {
            'application/geo+json': self.data,
            'text/plain': '<IPython.display.GeoJSON object>'
        }
        metadata = {
            'application/geo+json': self.metadata
        }
        display_functions.display(bundle, metadata=metadata, raw=True)

class Javascript(TextDisplayObject):

    def __init__(self, data=None, url=None, filename=None, lib=None, css=None):
        """Create a Javascript display object given raw data.

        When this object is returned by an expression or passed to the
        display function, it will result in the data being displayed
        in the frontend. If the data is a URL, the data will first be
        downloaded and then displayed.

        In the Notebook, the containing element will be available as `element`,
        and jQuery will be available.  Content appended to `element` will be
        visible in the output area.

        Parameters
        ----------
        data : unicode, str or bytes
            The Javascript source code or a URL to download it from.
        url : unicode
            A URL to download the data from.
        filename : unicode
            Path to a local file to load the data from.
        lib : list or str
            A sequence of Javascript library URLs to load asynchronously before
            running the source code. The full URLs of the libraries should
            be given. A single Javascript library URL can also be given as a
            string.
        css : list or str
            A sequence of css files to load before running the source code.
            The full URLs of the css files should be given. A single css URL
            can also be given as a string.
        """
        if isinstance(lib, str):
            lib = [lib]
        elif lib is None:
            lib = []
        if isinstance(css, str):
            css = [css]
        elif css is None:
            css = []
        if not isinstance(lib, (list,tuple)):
            raise TypeError('expected sequence, got: %r' % lib)
        if not isinstance(css, (list,tuple)):
            raise TypeError('expected sequence, got: %r' % css)
        self.lib = lib
        self.css = css
        super(Javascript, self).__init__(data=data, url=url, filename=filename)

    def _repr_javascript_(self):
        r = ''
        for c in self.css:
            r += _css_t % c
        for l in self.lib:
            r += _lib_t1 % l
        r += self.data
        r += _lib_t2*len(self.lib)
        return r

# constants for identifying png/jpeg data
_PNG = b'\x89PNG\r\n\x1a\n'
_JPEG = b'\xff\xd8'

def _pngxy(data):
    """read the (width, height) from a PNG header"""
    ihdr = data.index(b'IHDR')
    # next 8 bytes are width/height
    return struct.unpack('>ii', data[ihdr+4:ihdr+12])

def _jpegxy(data):
    """read the (width, height) from a JPEG header"""
    # adapted from http://www.64lines.com/jpeg-width-height

    idx = 4
    while True:
        block_size = struct.unpack('>H', data[idx:idx+2])[0]
        idx = idx + block_size
        if data[idx:idx+2] == b'\xFF\xC0':
            # found Start of Frame
            iSOF = idx
            break
        else:
            # read another block
            idx += 2

    h, w = struct.unpack('>HH', data[iSOF+5:iSOF+9])
    return w, h

def _gifxy(data):
    """read the (width, height) from a GIF header"""
    return struct.unpack('<HH', data[6:10])


class Image(DisplayObject):

    _read_flags = 'rb'
    _FMT_JPEG = u'jpeg'
    _FMT_PNG = u'png'
    _FMT_GIF = u'gif'
    _ACCEPTABLE_EMBEDDINGS = [_FMT_JPEG, _FMT_PNG, _FMT_GIF]
    _MIMETYPES = {
        _FMT_PNG: 'image/png',
        _FMT_JPEG: 'image/jpeg',
        _FMT_GIF: 'image/gif',
    }

    def __init__(
        self,
        data=None,
        url=None,
        filename=None,
        format=None,
        embed=None,
        width=None,
        height=None,
        retina=False,
        unconfined=False,
        metadata=None,
        alt=None,
    ):
        """Create a PNG/JPEG/GIF image object given raw data.

        When this object is returned by an input cell or passed to the
        display function, it will result in the image being displayed
        in the frontend.

        Parameters
        ----------
        data : unicode, str or bytes
            The raw image data or a URL or filename to load the data from.
            This always results in embedded image data.

        url : unicode
            A URL to download the data from. If you specify `url=`,
            the image data will not be embedded unless you also specify `embed=True`.

        filename : unicode
            Path to a local file to load the data from.
            Images from a file are always embedded.

        format : unicode
            The format of the image data (png/jpeg/jpg/gif). If a filename or URL is given
            for format will be inferred from the filename extension.

        embed : bool
            Should the image data be embedded using a data URI (True) or be
            loaded using an <img> tag. Set this to True if you want the image
            to be viewable later with no internet connection in the notebook.

            Default is `True`, unless the keyword argument `url` is set, then
            default value is `False`.

            Note that QtConsole is not able to display images if `embed` is set to `False`

        width : int
            Width in pixels to which to constrain the image in html

        height : int
            Height in pixels to which to constrain the image in html

        retina : bool
            Automatically set the width and height to half of the measured
            width and height.
            This only works for embedded images because it reads the width/height
            from image data.
            For non-embedded images, you can just set the desired display width
            and height directly.

        unconfined : bool
            Set unconfined=True to disable max-width confinement of the image.

        metadata : dict
            Specify extra metadata to attach to the image.

        alt : unicode
            Alternative text for the image, for use by screen readers.

        Examples
        --------
        embedded image data, works in qtconsole and notebook
        when passed positionally, the first arg can be any of raw image data,
        a URL, or a filename from which to load image data.
        The result is always embedding image data for inline images.

        >>> Image('https://www.google.fr/images/srpr/logo3w.png') # doctest: +SKIP
        <IPython.core.display.Image object>

        >>> Image('/path/to/image.jpg')
        <IPython.core.display.Image object>

        >>> Image(b'RAW_PNG_DATA...')
        <IPython.core.display.Image object>

        Specifying Image(url=...) does not embed the image data,
        it only generates ``<img>`` tag with a link to the source.
        This will not work in the qtconsole or offline.

        >>> Image(url='https://www.google.fr/images/srpr/logo3w.png')
        <IPython.core.display.Image object>

        """
        if isinstance(data, (Path, PurePath)):
            data = str(data)

        if filename is not None:
            ext = self._find_ext(filename)
        elif url is not None:
            ext = self._find_ext(url)
        elif data is None:
            raise ValueError("No image data found. Expecting filename, url, or data.")
        elif isinstance(data, str) and (
            data.startswith('http') or _safe_exists(data)
        ):
            ext = self._find_ext(data)
        else:
            ext = None

        if format is None:
            if ext is not None:
                if ext == u'jpg' or ext == u'jpeg':
                    format = self._FMT_JPEG
                elif ext == u'png':
                    format = self._FMT_PNG
                elif ext == u'gif':
                    format = self._FMT_GIF
                else:
                    format = ext.lower()
            elif isinstance(data, bytes):
                # infer image type from image data header,
                # only if format has not been specified.
                if data[:2] == _JPEG:
                    format = self._FMT_JPEG

        # failed to detect format, default png
        if format is None:
            format = self._FMT_PNG

        if format.lower() == 'jpg':
            # jpg->jpeg
            format = self._FMT_JPEG

        self.format = format.lower()
        self.embed = embed if embed is not None else (url is None)

        if self.embed and self.format not in self._ACCEPTABLE_EMBEDDINGS:
            raise ValueError("Cannot embed the '%s' image format" % (self.format))
        if self.embed:
            self._mimetype = self._MIMETYPES.get(self.format)

        self.width = width
        self.height = height
        self.retina = retina
        self.unconfined = unconfined
        self.alt = alt
        super(Image, self).__init__(data=data, url=url, filename=filename,
                metadata=metadata)

        if self.width is None and self.metadata.get('width', {}):
            self.width = metadata['width']

        if self.height is None and self.metadata.get('height', {}):
            self.height = metadata['height']

        if self.alt is None and self.metadata.get("alt", {}):
            self.alt = metadata["alt"]

        if retina:
            self._retina_shape()


    def _retina_shape(self):
        """load pixel-doubled width and height from image data"""
        if not self.embed:
            return
        if self.format == self._FMT_PNG:
            w, h = _pngxy(self.data)
        elif self.format == self._FMT_JPEG:
            w, h = _jpegxy(self.data)
        elif self.format == self._FMT_GIF:
            w, h = _gifxy(self.data)
        else:
            # retina only supports png
            return
        self.width = w // 2
        self.height = h // 2

    def reload(self):
        """Reload the raw data from file or URL."""
        if self.embed:
            super(Image,self).reload()
            if self.retina:
                self._retina_shape()

    def _repr_html_(self):
        if not self.embed:
            width = height = klass = alt = ""
            if self.width:
                width = ' width="%d"' % self.width
            if self.height:
                height = ' height="%d"' % self.height
            if self.unconfined:
                klass = ' class="unconfined"'
            if self.alt:
                alt = ' alt="%s"' % html.escape(self.alt)
            return '<img src="{url}"{width}{height}{klass}{alt}/>'.format(
                url=self.url,
                width=width,
                height=height,
                klass=klass,
                alt=alt,
            )

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Return the image as a mimebundle

        Any new mimetype support should be implemented here.
        """
        if self.embed:
            mimetype = self._mimetype
            data, metadata = self._data_and_metadata(always_both=True)
            if metadata:
                metadata = {mimetype: metadata}
            return {mimetype: data}, metadata
        else:
            return {'text/html': self._repr_html_()}

    def _data_and_metadata(self, always_both=False):
        """shortcut for returning metadata with shape information, if defined"""
        try:
            b64_data = b2a_base64(self.data, newline=False).decode("ascii")
        except TypeError as e:
            raise FileNotFoundError(
                "No such file or directory: '%s'" % (self.data)) from e
        md = {}
        if self.metadata:
            md.update(self.metadata)
        if self.width:
            md['width'] = self.width
        if self.height:
            md['height'] = self.height
        if self.unconfined:
            md['unconfined'] = self.unconfined
        if self.alt:
            md["alt"] = self.alt
        if md or always_both:
            return b64_data, md
        else:
            return b64_data

    def _repr_png_(self):
        if self.embed and self.format == self._FMT_PNG:
            return self._data_and_metadata()

    def _repr_jpeg_(self):
        if self.embed and self.format == self._FMT_JPEG:
            return self._data_and_metadata()

    def _find_ext(self, s):
        base, ext = splitext(s)

        if not ext:
            return base

        # `splitext` includes leading period, so we skip it
        return ext[1:].lower()


class Video(DisplayObject):

    def __init__(self, data=None, url=None, filename=None, embed=False,
                 mimetype=None, width=None, height=None, html_attributes="controls"):
        """Create a video object given raw data or an URL.

        When this object is returned by an input cell or passed to the
        display function, it will result in the video being displayed
        in the frontend.

        Parameters
        ----------
        data : unicode, str or bytes
            The raw video data or a URL or filename to load the data from.
            Raw data will require passing ``embed=True``.

        url : unicode
            A URL for the video. If you specify ``url=``,
            the image data will not be embedded.

        filename : unicode
            Path to a local file containing the video.
            Will be interpreted as a local URL unless ``embed=True``.

        embed : bool
            Should the video be embedded using a data URI (True) or be
            loaded using a <video> tag (False).

            Since videos are large, embedding them should be avoided, if possible.
            You must confirm embedding as your intention by passing ``embed=True``.

            Local files can be displayed with URLs without embedding the content, via::

                Video('./video.mp4')

        mimetype : unicode
            Specify the mimetype for embedded videos.
            Default will be guessed from file extension, if available.

        width : int
            Width in pixels to which to constrain the video in HTML.
            If not supplied, defaults to the width of the video.

        height : int
            Height in pixels to which to constrain the video in html.
            If not supplied, defaults to the height of the video.

        html_attributes : str
            Attributes for the HTML ``<video>`` block.
            Default: ``"controls"`` to get video controls.
            Other examples: ``"controls muted"`` for muted video with controls,
            ``"loop autoplay"`` for looping autoplaying video without controls.

        Examples
        --------
        ::

            Video('https://archive.org/download/Sita_Sings_the_Blues/Sita_Sings_the_Blues_small.mp4')
            Video('path/to/video.mp4')
            Video('path/to/video.mp4', embed=True)
            Video('path/to/video.mp4', embed=True, html_attributes="controls muted autoplay")
            Video(b'raw-videodata', embed=True)
        """
        if isinstance(data, (Path, PurePath)):
            data = str(data)

        if url is None and isinstance(data, str) and data.startswith(('http:', 'https:')):
            url = data
            data = None
        elif data is not None and os.path.exists(data):
            filename = data
            data = None

        if data and not embed:
            msg = ''.join([
                "To embed videos, you must pass embed=True ",
                "(this may make your notebook files huge)\n",
                "Consider passing Video(url='...')",
            ])
            raise ValueError(msg)

        self.mimetype = mimetype
        self.embed = embed
        self.width = width
        self.height = height
        self.html_attributes = html_attributes
        super(Video, self).__init__(data=data, url=url, filename=filename)

    def _repr_html_(self):
        width = height = ''
        if self.width:
            width = ' width="%d"' % self.width
        if self.height:
            height = ' height="%d"' % self.height

        # External URLs and potentially local files are not embedded into the
        # notebook output.
        if not self.embed:
            url = self.url if self.url is not None else self.filename
            output = """<video src="{0}" {1} {2} {3}>
      Your browser does not support the <code>video</code> element.
    </video>""".format(url, self.html_attributes, width, height)
            return output

        # Embedded videos are base64-encoded.
        mimetype = self.mimetype
        if self.filename is not None:
            if not mimetype:
                mimetype, _ = mimetypes.guess_type(self.filename)

            with open(self.filename, 'rb') as f:
                video = f.read()
        else:
            video = self.data
        if isinstance(video, str):
            # unicode input is already b64-encoded
            b64_video = video
        else:
            b64_video = b2a_base64(video, newline=False).decode("ascii").rstrip()

        output = """<video {0} {1} {2}>
 <source src="data:{3};base64,{4}" type="{3}">
 Your browser does not support the video tag.
 </video>""".format(self.html_attributes, width, height, mimetype, b64_video)
        return output

    def reload(self):
        # TODO
        pass


@skip_doctest
def set_matplotlib_formats(*formats, **kwargs):
    """
    .. deprecated:: 7.23

       use `matplotlib_inline.backend_inline.set_matplotlib_formats()`

    Select figure formats for the inline backend. Optionally pass quality for JPEG.

    For example, this enables PNG and JPEG output with a JPEG quality of 90%::

        In [1]: set_matplotlib_formats('png', 'jpeg', quality=90)

    To set this in your config files use the following::

        c.InlineBackend.figure_formats = {'png', 'jpeg'}
        c.InlineBackend.print_figure_kwargs.update({'quality' : 90})

    Parameters
    ----------
    *formats : strs
        One or more figure formats to enable: 'png', 'retina', 'jpeg', 'svg', 'pdf'.
    **kwargs
        Keyword args will be relayed to ``figure.canvas.print_figure``.
    """
    warnings.warn(
        "`set_matplotlib_formats` is deprecated since IPython 7.23, directly "
        "use `matplotlib_inline.backend_inline.set_matplotlib_formats()`",
        DeprecationWarning,
        stacklevel=2,
    )

    from matplotlib_inline.backend_inline import (
        set_matplotlib_formats as set_matplotlib_formats_orig,
    )

    set_matplotlib_formats_orig(*formats, **kwargs)

@skip_doctest
def set_matplotlib_close(close=True):
    """
    .. deprecated:: 7.23

        use `matplotlib_inline.backend_inline.set_matplotlib_close()`

    Set whether the inline backend closes all figures automatically or not.

    By default, the inline backend used in the IPython Notebook will close all
    matplotlib figures automatically after each cell is run. This means that
    plots in different cells won't interfere. Sometimes, you may want to make
    a plot in one cell and then refine it in later cells. This can be accomplished
    by::

        In [1]: set_matplotlib_close(False)

    To set this in your config files use the following::

        c.InlineBackend.close_figures = False

    Parameters
    ----------
    close : bool
        Should all matplotlib figures be automatically closed after each cell is
        run?
    """
    warnings.warn(
        "`set_matplotlib_close` is deprecated since IPython 7.23, directly "
        "use `matplotlib_inline.backend_inline.set_matplotlib_close()`",
        DeprecationWarning,
        stacklevel=2,
    )

    from matplotlib_inline.backend_inline import (
        set_matplotlib_close as set_matplotlib_close_orig,
    )

    set_matplotlib_close_orig(close)
