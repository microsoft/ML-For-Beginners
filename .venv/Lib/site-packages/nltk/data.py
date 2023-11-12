# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Functions to find and load NLTK resource files, such as corpora,
grammars, and saved processing objects.  Resource files are identified
using URLs, such as ``nltk:corpora/abc/rural.txt`` or
``https://raw.githubusercontent.com/nltk/nltk/develop/nltk/test/toy.cfg``.
The following URL protocols are supported:

  - ``file:path``: Specifies the file whose path is *path*.
    Both relative and absolute paths may be used.

  - ``https://host/path``: Specifies the file stored on the web
    server *host* at path *path*.

  - ``nltk:path``: Specifies the file stored in the NLTK data
    package at *path*.  NLTK will search for these files in the
    directories specified by ``nltk.data.path``.

If no protocol is specified, then the default protocol ``nltk:`` will
be used.

This module provides to functions that can be used to access a
resource file, given its URL: ``load()`` loads a given resource, and
adds it to a resource cache; and ``retrieve()`` copies a given resource
to a local file.
"""

import codecs
import functools
import os
import pickle
import re
import sys
import textwrap
import zipfile
from abc import ABCMeta, abstractmethod
from gzip import WRITE as GZ_WRITE
from gzip import GzipFile
from io import BytesIO, TextIOWrapper
from urllib.request import url2pathname, urlopen

try:
    from zlib import Z_SYNC_FLUSH as FLUSH
except ImportError:
    from zlib import Z_FINISH as FLUSH

from nltk import grammar, sem
from nltk.compat import add_py3_data, py3_data
from nltk.internals import deprecated

textwrap_indent = functools.partial(textwrap.indent, prefix="  ")

######################################################################
# Search Path
######################################################################

path = []
"""A list of directories where the NLTK data package might reside.
   These directories will be checked in order when looking for a
   resource in the data package.  Note that this allows users to
   substitute in their own versions of resources, if they have them
   (e.g., in their home directory under ~/nltk_data)."""

# User-specified locations:
_paths_from_env = os.environ.get("NLTK_DATA", "").split(os.pathsep)
path += [d for d in _paths_from_env if d]
if "APPENGINE_RUNTIME" not in os.environ and os.path.expanduser("~/") != "~/":
    path.append(os.path.expanduser("~/nltk_data"))

if sys.platform.startswith("win"):
    # Common locations on Windows:
    path += [
        os.path.join(sys.prefix, "nltk_data"),
        os.path.join(sys.prefix, "share", "nltk_data"),
        os.path.join(sys.prefix, "lib", "nltk_data"),
        os.path.join(os.environ.get("APPDATA", "C:\\"), "nltk_data"),
        r"C:\nltk_data",
        r"D:\nltk_data",
        r"E:\nltk_data",
    ]
else:
    # Common locations on UNIX & OS X:
    path += [
        os.path.join(sys.prefix, "nltk_data"),
        os.path.join(sys.prefix, "share", "nltk_data"),
        os.path.join(sys.prefix, "lib", "nltk_data"),
        "/usr/share/nltk_data",
        "/usr/local/share/nltk_data",
        "/usr/lib/nltk_data",
        "/usr/local/lib/nltk_data",
    ]


######################################################################
# Util Functions
######################################################################


def gzip_open_unicode(
    filename,
    mode="rb",
    compresslevel=9,
    encoding="utf-8",
    fileobj=None,
    errors=None,
    newline=None,
):
    if fileobj is None:
        fileobj = GzipFile(filename, mode, compresslevel, fileobj)
    return TextIOWrapper(fileobj, encoding, errors, newline)


def split_resource_url(resource_url):
    """
    Splits a resource url into "<protocol>:<path>".

    >>> windows = sys.platform.startswith('win')
    >>> split_resource_url('nltk:home/nltk')
    ('nltk', 'home/nltk')
    >>> split_resource_url('nltk:/home/nltk')
    ('nltk', '/home/nltk')
    >>> split_resource_url('file:/home/nltk')
    ('file', '/home/nltk')
    >>> split_resource_url('file:///home/nltk')
    ('file', '/home/nltk')
    >>> split_resource_url('file:///C:/home/nltk')
    ('file', '/C:/home/nltk')
    """
    protocol, path_ = resource_url.split(":", 1)
    if protocol == "nltk":
        pass
    elif protocol == "file":
        if path_.startswith("/"):
            path_ = "/" + path_.lstrip("/")
    else:
        path_ = re.sub(r"^/{0,2}", "", path_)
    return protocol, path_


def normalize_resource_url(resource_url):
    r"""
    Normalizes a resource url

    >>> windows = sys.platform.startswith('win')
    >>> os.path.normpath(split_resource_url(normalize_resource_url('file:grammar.fcfg'))[1]) == \
    ... ('\\' if windows else '') + os.path.abspath(os.path.join(os.curdir, 'grammar.fcfg'))
    True
    >>> not windows or normalize_resource_url('file:C:/dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('file:C:\\dir\\file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('file:C:\\dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('file://C:/dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('file:////C:/dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('nltk:C:/dir/file') == 'file:///C:/dir/file'
    True
    >>> not windows or normalize_resource_url('nltk:C:\\dir\\file') == 'file:///C:/dir/file'
    True
    >>> windows or normalize_resource_url('file:/dir/file/toy.cfg') == 'file:///dir/file/toy.cfg'
    True
    >>> normalize_resource_url('nltk:home/nltk')
    'nltk:home/nltk'
    >>> windows or normalize_resource_url('nltk:/home/nltk') == 'file:///home/nltk'
    True
    >>> normalize_resource_url('https://example.com/dir/file')
    'https://example.com/dir/file'
    >>> normalize_resource_url('dir/file')
    'nltk:dir/file'
    """
    try:
        protocol, name = split_resource_url(resource_url)
    except ValueError:
        # the resource url has no protocol, use the nltk protocol by default
        protocol = "nltk"
        name = resource_url
    # use file protocol if the path is an absolute path
    if protocol == "nltk" and os.path.isabs(name):
        protocol = "file://"
        name = normalize_resource_name(name, False, None)
    elif protocol == "file":
        protocol = "file://"
        # name is absolute
        name = normalize_resource_name(name, False, None)
    elif protocol == "nltk":
        protocol = "nltk:"
        name = normalize_resource_name(name, True)
    else:
        # handled by urllib
        protocol += "://"
    return "".join([protocol, name])


def normalize_resource_name(resource_name, allow_relative=True, relative_path=None):
    """
    :type resource_name: str or unicode
    :param resource_name: The name of the resource to search for.
        Resource names are posix-style relative path names, such as
        ``corpora/brown``.  Directory names will automatically
        be converted to a platform-appropriate path separator.
        Directory trailing slashes are preserved

    >>> windows = sys.platform.startswith('win')
    >>> normalize_resource_name('.', True)
    './'
    >>> normalize_resource_name('./', True)
    './'
    >>> windows or normalize_resource_name('dir/file', False, '/') == '/dir/file'
    True
    >>> not windows or normalize_resource_name('C:/file', False, '/') == '/C:/file'
    True
    >>> windows or normalize_resource_name('/dir/file', False, '/') == '/dir/file'
    True
    >>> windows or normalize_resource_name('../dir/file', False, '/') == '/dir/file'
    True
    >>> not windows or normalize_resource_name('/dir/file', True, '/') == 'dir/file'
    True
    >>> windows or normalize_resource_name('/dir/file', True, '/') == '/dir/file'
    True
    """
    is_dir = bool(re.search(r"[\\/.]$", resource_name)) or resource_name.endswith(
        os.path.sep
    )
    if sys.platform.startswith("win"):
        resource_name = resource_name.lstrip("/")
    else:
        resource_name = re.sub(r"^/+", "/", resource_name)
    if allow_relative:
        resource_name = os.path.normpath(resource_name)
    else:
        if relative_path is None:
            relative_path = os.curdir
        resource_name = os.path.abspath(os.path.join(relative_path, resource_name))
    resource_name = resource_name.replace("\\", "/").replace(os.path.sep, "/")
    if sys.platform.startswith("win") and os.path.isabs(resource_name):
        resource_name = "/" + resource_name
    if is_dir and not resource_name.endswith("/"):
        resource_name += "/"
    return resource_name


######################################################################
# Path Pointers
######################################################################


class PathPointer(metaclass=ABCMeta):
    """
    An abstract base class for 'path pointers,' used by NLTK's data
    package to identify specific paths.  Two subclasses exist:
    ``FileSystemPathPointer`` identifies a file that can be accessed
    directly via a given absolute path.  ``ZipFilePathPointer``
    identifies a file contained within a zipfile, that can be accessed
    by reading that zipfile.
    """

    @abstractmethod
    def open(self, encoding=None):
        """
        Return a seekable read-only stream that can be used to read
        the contents of the file identified by this path pointer.

        :raise IOError: If the path specified by this pointer does
            not contain a readable file.
        """

    @abstractmethod
    def file_size(self):
        """
        Return the size of the file pointed to by this path pointer,
        in bytes.

        :raise IOError: If the path specified by this pointer does
            not contain a readable file.
        """

    @abstractmethod
    def join(self, fileid):
        """
        Return a new path pointer formed by starting at the path
        identified by this pointer, and then following the relative
        path given by ``fileid``.  The path components of ``fileid``
        should be separated by forward slashes, regardless of
        the underlying file system's path separator character.
        """


class FileSystemPathPointer(PathPointer, str):
    """
    A path pointer that identifies a file which can be accessed
    directly via a given absolute path.
    """

    @py3_data
    def __init__(self, _path):
        """
        Create a new path pointer for the given absolute path.

        :raise IOError: If the given path does not exist.
        """

        _path = os.path.abspath(_path)
        if not os.path.exists(_path):
            raise OSError("No such file or directory: %r" % _path)
        self._path = _path

        # There's no need to call str.__init__(), since it's a no-op;
        # str does all of its setup work in __new__.

    @property
    def path(self):
        """The absolute path identified by this path pointer."""
        return self._path

    def open(self, encoding=None):
        stream = open(self._path, "rb")
        if encoding is not None:
            stream = SeekableUnicodeStreamReader(stream, encoding)
        return stream

    def file_size(self):
        return os.stat(self._path).st_size

    def join(self, fileid):
        _path = os.path.join(self._path, fileid)
        return FileSystemPathPointer(_path)

    def __repr__(self):
        return "FileSystemPathPointer(%r)" % self._path

    def __str__(self):
        return self._path


@deprecated("Use gzip.GzipFile instead as it also uses a buffer.")
class BufferedGzipFile(GzipFile):
    """A ``GzipFile`` subclass for compatibility with older nltk releases.

    Use ``GzipFile`` directly as it also buffers in all supported
    Python versions.
    """

    @py3_data
    def __init__(
        self, filename=None, mode=None, compresslevel=9, fileobj=None, **kwargs
    ):
        """Return a buffered gzip file object."""
        GzipFile.__init__(self, filename, mode, compresslevel, fileobj)

    def write(self, data):
        # This is identical to GzipFile.write but does not return
        # the bytes written to retain compatibility.
        super().write(data)


class GzipFileSystemPathPointer(FileSystemPathPointer):
    """
    A subclass of ``FileSystemPathPointer`` that identifies a gzip-compressed
    file located at a given absolute path.  ``GzipFileSystemPathPointer`` is
    appropriate for loading large gzip-compressed pickle objects efficiently.
    """

    def open(self, encoding=None):
        stream = GzipFile(self._path, "rb")
        if encoding:
            stream = SeekableUnicodeStreamReader(stream, encoding)
        return stream


class ZipFilePathPointer(PathPointer):
    """
    A path pointer that identifies a file contained within a zipfile,
    which can be accessed by reading that zipfile.
    """

    @py3_data
    def __init__(self, zipfile, entry=""):
        """
        Create a new path pointer pointing at the specified entry
        in the given zipfile.

        :raise IOError: If the given zipfile does not exist, or if it
        does not contain the specified entry.
        """
        if isinstance(zipfile, str):
            zipfile = OpenOnDemandZipFile(os.path.abspath(zipfile))

        # Check that the entry exists:
        if entry:

            # Normalize the entry string, it should be relative:
            entry = normalize_resource_name(entry, True, "/").lstrip("/")

            try:
                zipfile.getinfo(entry)
            except Exception as e:
                # Sometimes directories aren't explicitly listed in
                # the zip file.  So if `entry` is a directory name,
                # then check if the zipfile contains any files that
                # are under the given directory.
                if entry.endswith("/") and [
                    n for n in zipfile.namelist() if n.startswith(entry)
                ]:
                    pass  # zipfile contains a file in that directory.
                else:
                    # Otherwise, complain.
                    raise OSError(
                        f"Zipfile {zipfile.filename!r} does not contain {entry!r}"
                    ) from e
        self._zipfile = zipfile
        self._entry = entry

    @property
    def zipfile(self):
        """
        The zipfile.ZipFile object used to access the zip file
        containing the entry identified by this path pointer.
        """
        return self._zipfile

    @property
    def entry(self):
        """
        The name of the file within zipfile that this path
        pointer points to.
        """
        return self._entry

    def open(self, encoding=None):
        data = self._zipfile.read(self._entry)
        stream = BytesIO(data)
        if self._entry.endswith(".gz"):
            stream = GzipFile(self._entry, fileobj=stream)
        elif encoding is not None:
            stream = SeekableUnicodeStreamReader(stream, encoding)
        return stream

    def file_size(self):
        return self._zipfile.getinfo(self._entry).file_size

    def join(self, fileid):
        entry = f"{self._entry}/{fileid}"
        return ZipFilePathPointer(self._zipfile, entry)

    def __repr__(self):
        return f"ZipFilePathPointer({self._zipfile.filename!r}, {self._entry!r})"

    def __str__(self):
        return os.path.normpath(os.path.join(self._zipfile.filename, self._entry))


######################################################################
# Access Functions
######################################################################

# Don't use a weak dictionary, because in the common case this
# causes a lot more reloading that necessary.
_resource_cache = {}
"""A dictionary used to cache resources so that they won't
   need to be loaded more than once."""


def find(resource_name, paths=None):
    """
    Find the given resource by searching through the directories and
    zip files in paths, where a None or empty string specifies an absolute path.
    Returns a corresponding path name.  If the given resource is not
    found, raise a ``LookupError``, whose message gives a pointer to
    the installation instructions for the NLTK downloader.

    Zip File Handling:

      - If ``resource_name`` contains a component with a ``.zip``
        extension, then it is assumed to be a zipfile; and the
        remaining path components are used to look inside the zipfile.

      - If any element of ``nltk.data.path`` has a ``.zip`` extension,
        then it is assumed to be a zipfile.

      - If a given resource name that does not contain any zipfile
        component is not found initially, then ``find()`` will make a
        second attempt to find that resource, by replacing each
        component *p* in the path with *p.zip/p*.  For example, this
        allows ``find()`` to map the resource name
        ``corpora/chat80/cities.pl`` to a zip file path pointer to
        ``corpora/chat80.zip/chat80/cities.pl``.

      - When using ``find()`` to locate a directory contained in a
        zipfile, the resource name must end with the forward slash
        character.  Otherwise, ``find()`` will not locate the
        directory.

    :type resource_name: str or unicode
    :param resource_name: The name of the resource to search for.
        Resource names are posix-style relative path names, such as
        ``corpora/brown``.  Directory names will be
        automatically converted to a platform-appropriate path separator.
    :rtype: str
    """
    resource_name = normalize_resource_name(resource_name, True)

    # Resolve default paths at runtime in-case the user overrides
    # nltk.data.path
    if paths is None:
        paths = path

    # Check if the resource name includes a zipfile name
    m = re.match(r"(.*\.zip)/?(.*)$|", resource_name)
    zipfile, zipentry = m.groups()

    # Check each item in our path
    for path_ in paths:
        # Is the path item a zipfile?
        if path_ and (os.path.isfile(path_) and path_.endswith(".zip")):
            try:
                return ZipFilePathPointer(path_, resource_name)
            except OSError:
                # resource not in zipfile
                continue

        # Is the path item a directory or is resource_name an absolute path?
        elif not path_ or os.path.isdir(path_):
            if zipfile is None:
                p = os.path.join(path_, url2pathname(resource_name))
                if os.path.exists(p):
                    if p.endswith(".gz"):
                        return GzipFileSystemPathPointer(p)
                    else:
                        return FileSystemPathPointer(p)
            else:
                p = os.path.join(path_, url2pathname(zipfile))
                if os.path.exists(p):
                    try:
                        return ZipFilePathPointer(p, zipentry)
                    except OSError:
                        # resource not in zipfile
                        continue

    # Fallback: if the path doesn't include a zip file, then try
    # again, assuming that one of the path components is inside a
    # zipfile of the same name.
    if zipfile is None:
        pieces = resource_name.split("/")
        for i in range(len(pieces)):
            modified_name = "/".join(pieces[:i] + [pieces[i] + ".zip"] + pieces[i:])
            try:
                return find(modified_name, paths)
            except LookupError:
                pass

    # Identify the package (i.e. the .zip file) to download.
    resource_zipname = resource_name.split("/")[1]
    if resource_zipname.endswith(".zip"):
        resource_zipname = resource_zipname.rpartition(".")[0]
    # Display a friendly error message if the resource wasn't found:
    msg = str(
        "Resource \33[93m{resource}\033[0m not found.\n"
        "Please use the NLTK Downloader to obtain the resource:\n\n"
        "\33[31m"  # To display red text in terminal.
        ">>> import nltk\n"
        ">>> nltk.download('{resource}')\n"
        "\033[0m"
    ).format(resource=resource_zipname)
    msg = textwrap_indent(msg)

    msg += "\n  For more information see: https://www.nltk.org/data.html\n"

    msg += "\n  Attempted to load \33[93m{resource_name}\033[0m\n".format(
        resource_name=resource_name
    )

    msg += "\n  Searched in:" + "".join("\n    - %r" % d for d in paths)
    sep = "*" * 70
    resource_not_found = f"\n{sep}\n{msg}\n{sep}\n"
    raise LookupError(resource_not_found)


def retrieve(resource_url, filename=None, verbose=True):
    """
    Copy the given resource to a local file.  If no filename is
    specified, then use the URL's filename.  If there is already a
    file named ``filename``, then raise a ``ValueError``.

    :type resource_url: str
    :param resource_url: A URL specifying where the resource should be
        loaded from.  The default protocol is "nltk:", which searches
        for the file in the the NLTK data package.
    """
    resource_url = normalize_resource_url(resource_url)
    if filename is None:
        if resource_url.startswith("file:"):
            filename = os.path.split(resource_url)[-1]
        else:
            filename = re.sub(r"(^\w+:)?.*/", "", resource_url)
    if os.path.exists(filename):
        filename = os.path.abspath(filename)
        raise ValueError("File %r already exists!" % filename)

    if verbose:
        print(f"Retrieving {resource_url!r}, saving to {filename!r}")

    # Open the input & output streams.
    infile = _open(resource_url)

    # Copy infile -> outfile, using 64k blocks.
    with open(filename, "wb") as outfile:
        while True:
            s = infile.read(1024 * 64)  # 64k blocks.
            outfile.write(s)
            if not s:
                break

    infile.close()


#: A dictionary describing the formats that are supported by NLTK's
#: load() method.  Keys are format names, and values are format
#: descriptions.
FORMATS = {
    "pickle": "A serialized python object, stored using the pickle module.",
    "json": "A serialized python object, stored using the json module.",
    "yaml": "A serialized python object, stored using the yaml module.",
    "cfg": "A context free grammar.",
    "pcfg": "A probabilistic CFG.",
    "fcfg": "A feature CFG.",
    "fol": "A list of first order logic expressions, parsed with "
    "nltk.sem.logic.Expression.fromstring.",
    "logic": "A list of first order logic expressions, parsed with "
    "nltk.sem.logic.LogicParser.  Requires an additional logic_parser "
    "parameter",
    "val": "A semantic valuation, parsed by nltk.sem.Valuation.fromstring.",
    "raw": "The raw (byte string) contents of a file.",
    "text": "The raw (unicode string) contents of a file. ",
}

#: A dictionary mapping from file extensions to format names, used
#: by load() when format="auto" to decide the format for a
#: given resource url.
AUTO_FORMATS = {
    "pickle": "pickle",
    "json": "json",
    "yaml": "yaml",
    "cfg": "cfg",
    "pcfg": "pcfg",
    "fcfg": "fcfg",
    "fol": "fol",
    "logic": "logic",
    "val": "val",
    "txt": "text",
    "text": "text",
}


def load(
    resource_url,
    format="auto",
    cache=True,
    verbose=False,
    logic_parser=None,
    fstruct_reader=None,
    encoding=None,
):
    """
    Load a given resource from the NLTK data package.  The following
    resource formats are currently supported:

      - ``pickle``
      - ``json``
      - ``yaml``
      - ``cfg`` (context free grammars)
      - ``pcfg`` (probabilistic CFGs)
      - ``fcfg`` (feature-based CFGs)
      - ``fol`` (formulas of First Order Logic)
      - ``logic`` (Logical formulas to be parsed by the given logic_parser)
      - ``val`` (valuation of First Order Logic model)
      - ``text`` (the file contents as a unicode string)
      - ``raw`` (the raw file contents as a byte string)

    If no format is specified, ``load()`` will attempt to determine a
    format based on the resource name's file extension.  If that
    fails, ``load()`` will raise a ``ValueError`` exception.

    For all text formats (everything except ``pickle``, ``json``, ``yaml`` and ``raw``),
    it tries to decode the raw contents using UTF-8, and if that doesn't
    work, it tries with ISO-8859-1 (Latin-1), unless the ``encoding``
    is specified.

    :type resource_url: str
    :param resource_url: A URL specifying where the resource should be
        loaded from.  The default protocol is "nltk:", which searches
        for the file in the the NLTK data package.
    :type cache: bool
    :param cache: If true, add this resource to a cache.  If load()
        finds a resource in its cache, then it will return it from the
        cache rather than loading it.
    :type verbose: bool
    :param verbose: If true, print a message when loading a resource.
        Messages are not displayed when a resource is retrieved from
        the cache.
    :type logic_parser: LogicParser
    :param logic_parser: The parser that will be used to parse logical
        expressions.
    :type fstruct_reader: FeatStructReader
    :param fstruct_reader: The parser that will be used to parse the
        feature structure of an fcfg.
    :type encoding: str
    :param encoding: the encoding of the input; only used for text formats.
    """
    resource_url = normalize_resource_url(resource_url)
    resource_url = add_py3_data(resource_url)

    # Determine the format of the resource.
    if format == "auto":
        resource_url_parts = resource_url.split(".")
        ext = resource_url_parts[-1]
        if ext == "gz":
            ext = resource_url_parts[-2]
        format = AUTO_FORMATS.get(ext)
        if format is None:
            raise ValueError(
                "Could not determine format for %s based "
                'on its file\nextension; use the "format" '
                "argument to specify the format explicitly." % resource_url
            )

    if format not in FORMATS:
        raise ValueError(f"Unknown format type: {format}!")

    # If we've cached the resource, then just return it.
    if cache:
        resource_val = _resource_cache.get((resource_url, format))
        if resource_val is not None:
            if verbose:
                print(f"<<Using cached copy of {resource_url}>>")
            return resource_val

    # Let the user know what's going on.
    if verbose:
        print(f"<<Loading {resource_url}>>")

    # Load the resource.
    opened_resource = _open(resource_url)

    if format == "raw":
        resource_val = opened_resource.read()
    elif format == "pickle":
        resource_val = pickle.load(opened_resource)
    elif format == "json":
        import json

        from nltk.jsontags import json_tags

        resource_val = json.load(opened_resource)
        tag = None
        if len(resource_val) != 1:
            tag = next(resource_val.keys())
        if tag not in json_tags:
            raise ValueError("Unknown json tag.")
    elif format == "yaml":
        import yaml

        resource_val = yaml.safe_load(opened_resource)
    else:
        # The resource is a text format.
        binary_data = opened_resource.read()
        if encoding is not None:
            string_data = binary_data.decode(encoding)
        else:
            try:
                string_data = binary_data.decode("utf-8")
            except UnicodeDecodeError:
                string_data = binary_data.decode("latin-1")
        if format == "text":
            resource_val = string_data
        elif format == "cfg":
            resource_val = grammar.CFG.fromstring(string_data, encoding=encoding)
        elif format == "pcfg":
            resource_val = grammar.PCFG.fromstring(string_data, encoding=encoding)
        elif format == "fcfg":
            resource_val = grammar.FeatureGrammar.fromstring(
                string_data,
                logic_parser=logic_parser,
                fstruct_reader=fstruct_reader,
                encoding=encoding,
            )
        elif format == "fol":
            resource_val = sem.read_logic(
                string_data,
                logic_parser=sem.logic.LogicParser(),
                encoding=encoding,
            )
        elif format == "logic":
            resource_val = sem.read_logic(
                string_data, logic_parser=logic_parser, encoding=encoding
            )
        elif format == "val":
            resource_val = sem.read_valuation(string_data, encoding=encoding)
        else:
            raise AssertionError(
                "Internal NLTK error: Format %s isn't "
                "handled by nltk.data.load()" % (format,)
            )

    opened_resource.close()

    # If requested, add it to the cache.
    if cache:
        try:
            _resource_cache[(resource_url, format)] = resource_val
            # TODO: add this line
            # print('<<Caching a copy of %s>>' % (resource_url,))
        except TypeError:
            # We can't create weak references to some object types, like
            # strings and tuples.  For now, just don't cache them.
            pass

    return resource_val


def show_cfg(resource_url, escape="##"):
    """
    Write out a grammar file, ignoring escaped and empty lines.

    :type resource_url: str
    :param resource_url: A URL specifying where the resource should be
        loaded from.  The default protocol is "nltk:", which searches
        for the file in the the NLTK data package.
    :type escape: str
    :param escape: Prepended string that signals lines to be ignored
    """
    resource_url = normalize_resource_url(resource_url)
    resource_val = load(resource_url, format="text", cache=False)
    lines = resource_val.splitlines()
    for l in lines:
        if l.startswith(escape):
            continue
        if re.match("^$", l):
            continue
        print(l)


def clear_cache():
    """
    Remove all objects from the resource cache.
    :see: load()
    """
    _resource_cache.clear()


def _open(resource_url):
    """
    Helper function that returns an open file object for a resource,
    given its resource URL.  If the given resource URL uses the "nltk:"
    protocol, or uses no protocol, then use ``nltk.data.find`` to find
    its path, and open it with the given mode; if the resource URL
    uses the 'file' protocol, then open the file with the given mode;
    otherwise, delegate to ``urllib2.urlopen``.

    :type resource_url: str
    :param resource_url: A URL specifying where the resource should be
        loaded from.  The default protocol is "nltk:", which searches
        for the file in the the NLTK data package.
    """
    resource_url = normalize_resource_url(resource_url)
    protocol, path_ = split_resource_url(resource_url)

    if protocol is None or protocol.lower() == "nltk":
        return find(path_, path + [""]).open()
    elif protocol.lower() == "file":
        # urllib might not use mode='rb', so handle this one ourselves:
        return find(path_, [""]).open()
    else:
        return urlopen(resource_url)


######################################################################
# Lazy Resource Loader
######################################################################


class LazyLoader:
    @py3_data
    def __init__(self, _path):
        self._path = _path

    def __load(self):
        resource = load(self._path)
        # This is where the magic happens!  Transform ourselves into
        # the object by modifying our own __dict__ and __class__ to
        # match that of `resource`.
        self.__dict__ = resource.__dict__
        self.__class__ = resource.__class__

    def __getattr__(self, attr):
        self.__load()
        # This looks circular, but its not, since __load() changes our
        # __class__ to something new:
        return getattr(self, attr)

    def __repr__(self):
        self.__load()
        # This looks circular, but its not, since __load() changes our
        # __class__ to something new:
        return repr(self)


######################################################################
# Open-On-Demand ZipFile
######################################################################


class OpenOnDemandZipFile(zipfile.ZipFile):
    """
    A subclass of ``zipfile.ZipFile`` that closes its file pointer
    whenever it is not using it; and re-opens it when it needs to read
    data from the zipfile.  This is useful for reducing the number of
    open file handles when many zip files are being accessed at once.
    ``OpenOnDemandZipFile`` must be constructed from a filename, not a
    file-like object (to allow re-opening).  ``OpenOnDemandZipFile`` is
    read-only (i.e. ``write()`` and ``writestr()`` are disabled.
    """

    @py3_data
    def __init__(self, filename):
        if not isinstance(filename, str):
            raise TypeError("ReopenableZipFile filename must be a string")
        zipfile.ZipFile.__init__(self, filename)
        assert self.filename == filename
        self.close()
        # After closing a ZipFile object, the _fileRefCnt needs to be cleared
        # for Python2and3 compatible code.
        self._fileRefCnt = 0

    def read(self, name):
        assert self.fp is None
        self.fp = open(self.filename, "rb")
        value = zipfile.ZipFile.read(self, name)
        # Ensure that _fileRefCnt needs to be set for Python2and3 compatible code.
        # Since we only opened one file here, we add 1.
        self._fileRefCnt += 1
        self.close()
        return value

    def write(self, *args, **kwargs):
        """:raise NotImplementedError: OpenOnDemandZipfile is read-only"""
        raise NotImplementedError("OpenOnDemandZipfile is read-only")

    def writestr(self, *args, **kwargs):
        """:raise NotImplementedError: OpenOnDemandZipfile is read-only"""
        raise NotImplementedError("OpenOnDemandZipfile is read-only")

    def __repr__(self):
        return repr("OpenOnDemandZipFile(%r)" % self.filename)


######################################################################
# Seekable Unicode Stream Reader
######################################################################


class SeekableUnicodeStreamReader:
    """
    A stream reader that automatically encodes the source byte stream
    into unicode (like ``codecs.StreamReader``); but still supports the
    ``seek()`` and ``tell()`` operations correctly.  This is in contrast
    to ``codecs.StreamReader``, which provide *broken* ``seek()`` and
    ``tell()`` methods.

    This class was motivated by ``StreamBackedCorpusView``, which
    makes extensive use of ``seek()`` and ``tell()``, and needs to be
    able to handle unicode-encoded files.

    Note: this class requires stateless decoders.  To my knowledge,
    this shouldn't cause a problem with any of python's builtin
    unicode encodings.
    """

    DEBUG = True  # : If true, then perform extra sanity checks.

    @py3_data
    def __init__(self, stream, encoding, errors="strict"):
        # Rewind the stream to its beginning.
        stream.seek(0)

        self.stream = stream
        """The underlying stream."""

        self.encoding = encoding
        """The name of the encoding that should be used to encode the
           underlying stream."""

        self.errors = errors
        """The error mode that should be used when decoding data from
           the underlying stream.  Can be 'strict', 'ignore', or
           'replace'."""

        self.decode = codecs.getdecoder(encoding)
        """The function that is used to decode byte strings into
           unicode strings."""

        self.bytebuffer = b""
        """A buffer to use bytes that have been read but have not yet
           been decoded.  This is only used when the final bytes from
           a read do not form a complete encoding for a character."""

        self.linebuffer = None
        """A buffer used by ``readline()`` to hold characters that have
           been read, but have not yet been returned by ``read()`` or
           ``readline()``.  This buffer consists of a list of unicode
           strings, where each string corresponds to a single line.
           The final element of the list may or may not be a complete
           line.  Note that the existence of a linebuffer makes the
           ``tell()`` operation more complex, because it must backtrack
           to the beginning of the buffer to determine the correct
           file position in the underlying byte stream."""

        self._rewind_checkpoint = 0
        """The file position at which the most recent read on the
           underlying stream began.  This is used, together with
           ``_rewind_numchars``, to backtrack to the beginning of
           ``linebuffer`` (which is required by ``tell()``)."""

        self._rewind_numchars = None
        """The number of characters that have been returned since the
           read that started at ``_rewind_checkpoint``.  This is used,
           together with ``_rewind_checkpoint``, to backtrack to the
           beginning of ``linebuffer`` (which is required by ``tell()``)."""

        self._bom = self._check_bom()
        """The length of the byte order marker at the beginning of
           the stream (or None for no byte order marker)."""

    # /////////////////////////////////////////////////////////////////
    # Read methods
    # /////////////////////////////////////////////////////////////////

    def read(self, size=None):
        """
        Read up to ``size`` bytes, decode them using this reader's
        encoding, and return the resulting unicode string.

        :param size: The maximum number of bytes to read.  If not
            specified, then read as many bytes as possible.
        :type size: int
        :rtype: unicode
        """
        chars = self._read(size)

        # If linebuffer is not empty, then include it in the result
        if self.linebuffer:
            chars = "".join(self.linebuffer) + chars
            self.linebuffer = None
            self._rewind_numchars = None

        return chars

    def discard_line(self):
        if self.linebuffer and len(self.linebuffer) > 1:
            line = self.linebuffer.pop(0)
            self._rewind_numchars += len(line)
        else:
            self.stream.readline()

    def readline(self, size=None):
        """
        Read a line of text, decode it using this reader's encoding,
        and return the resulting unicode string.

        :param size: The maximum number of bytes to read.  If no
            newline is encountered before ``size`` bytes have been read,
            then the returned value may not be a complete line of text.
        :type size: int
        """
        # If we have a non-empty linebuffer, then return the first
        # line from it.  (Note that the last element of linebuffer may
        # not be a complete line; so let _read() deal with it.)
        if self.linebuffer and len(self.linebuffer) > 1:
            line = self.linebuffer.pop(0)
            self._rewind_numchars += len(line)
            return line

        readsize = size or 72
        chars = ""

        # If there's a remaining incomplete line in the buffer, add it.
        if self.linebuffer:
            chars += self.linebuffer.pop()
            self.linebuffer = None

        while True:
            startpos = self.stream.tell() - len(self.bytebuffer)
            new_chars = self._read(readsize)

            # If we're at a '\r', then read one extra character, since
            # it might be a '\n', to get the proper line ending.
            if new_chars and new_chars.endswith("\r"):
                new_chars += self._read(1)

            chars += new_chars
            lines = chars.splitlines(True)
            if len(lines) > 1:
                line = lines[0]
                self.linebuffer = lines[1:]
                self._rewind_numchars = len(new_chars) - (len(chars) - len(line))
                self._rewind_checkpoint = startpos
                break
            elif len(lines) == 1:
                line0withend = lines[0]
                line0withoutend = lines[0].splitlines(False)[0]
                if line0withend != line0withoutend:  # complete line
                    line = line0withend
                    break

            if not new_chars or size is not None:
                line = chars
                break

            # Read successively larger blocks of text.
            if readsize < 8000:
                readsize *= 2

        return line

    def readlines(self, sizehint=None, keepends=True):
        """
        Read this file's contents, decode them using this reader's
        encoding, and return it as a list of unicode lines.

        :rtype: list(unicode)
        :param sizehint: Ignored.
        :param keepends: If false, then strip newlines.
        """
        return self.read().splitlines(keepends)

    def next(self):
        """Return the next decoded line from the underlying stream."""
        line = self.readline()
        if line:
            return line
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def __iter__(self):
        """Return self"""
        return self

    def __del__(self):
        # let garbage collector deal with still opened streams
        if not self.closed:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def xreadlines(self):
        """Return self"""
        return self

    # /////////////////////////////////////////////////////////////////
    # Pass-through methods & properties
    # /////////////////////////////////////////////////////////////////

    @property
    def closed(self):
        """True if the underlying stream is closed."""
        return self.stream.closed

    @property
    def name(self):
        """The name of the underlying stream."""
        return self.stream.name

    @property
    def mode(self):
        """The mode of the underlying stream."""
        return self.stream.mode

    def close(self):
        """
        Close the underlying stream.
        """
        self.stream.close()

    # /////////////////////////////////////////////////////////////////
    # Seek and tell
    # /////////////////////////////////////////////////////////////////

    def seek(self, offset, whence=0):
        """
        Move the stream to a new file position.  If the reader is
        maintaining any buffers, then they will be cleared.

        :param offset: A byte count offset.
        :param whence: If 0, then the offset is from the start of the file
            (offset should be positive), if 1, then the offset is from the
            current position (offset may be positive or negative); and if 2,
            then the offset is from the end of the file (offset should
            typically be negative).
        """
        if whence == 1:
            raise ValueError(
                "Relative seek is not supported for "
                "SeekableUnicodeStreamReader -- consider "
                "using char_seek_forward() instead."
            )
        self.stream.seek(offset, whence)
        self.linebuffer = None
        self.bytebuffer = b""
        self._rewind_numchars = None
        self._rewind_checkpoint = self.stream.tell()

    def char_seek_forward(self, offset):
        """
        Move the read pointer forward by ``offset`` characters.
        """
        if offset < 0:
            raise ValueError("Negative offsets are not supported")
        # Clear all buffers.
        self.seek(self.tell())
        # Perform the seek operation.
        self._char_seek_forward(offset)

    def _char_seek_forward(self, offset, est_bytes=None):
        """
        Move the file position forward by ``offset`` characters,
        ignoring all buffers.

        :param est_bytes: A hint, giving an estimate of the number of
            bytes that will be needed to move forward by ``offset`` chars.
            Defaults to ``offset``.
        """
        if est_bytes is None:
            est_bytes = offset
        bytes = b""

        while True:
            # Read in a block of bytes.
            newbytes = self.stream.read(est_bytes - len(bytes))
            bytes += newbytes

            # Decode the bytes to characters.
            chars, bytes_decoded = self._incr_decode(bytes)

            # If we got the right number of characters, then seek
            # backwards over any truncated characters, and return.
            if len(chars) == offset:
                self.stream.seek(-len(bytes) + bytes_decoded, 1)
                return

            # If we went too far, then we can back-up until we get it
            # right, using the bytes we've already read.
            if len(chars) > offset:
                while len(chars) > offset:
                    # Assume at least one byte/char.
                    est_bytes += offset - len(chars)
                    chars, bytes_decoded = self._incr_decode(bytes[:est_bytes])
                self.stream.seek(-len(bytes) + bytes_decoded, 1)
                return

            # Otherwise, we haven't read enough bytes yet; loop again.
            est_bytes += offset - len(chars)

    def tell(self):
        """
        Return the current file position on the underlying byte
        stream.  If this reader is maintaining any buffers, then the
        returned file position will be the position of the beginning
        of those buffers.
        """
        # If nothing's buffered, then just return our current filepos:
        if self.linebuffer is None:
            return self.stream.tell() - len(self.bytebuffer)

        # Otherwise, we'll need to backtrack the filepos until we
        # reach the beginning of the buffer.

        # Store our original file position, so we can return here.
        orig_filepos = self.stream.tell()

        # Calculate an estimate of where we think the newline is.
        bytes_read = (orig_filepos - len(self.bytebuffer)) - self._rewind_checkpoint
        buf_size = sum(len(line) for line in self.linebuffer)
        est_bytes = int(
            bytes_read * self._rewind_numchars / (self._rewind_numchars + buf_size)
        )

        self.stream.seek(self._rewind_checkpoint)
        self._char_seek_forward(self._rewind_numchars, est_bytes)
        filepos = self.stream.tell()

        # Sanity check
        if self.DEBUG:
            self.stream.seek(filepos)
            check1 = self._incr_decode(self.stream.read(50))[0]
            check2 = "".join(self.linebuffer)
            assert check1.startswith(check2) or check2.startswith(check1)

        # Return to our original filepos (so we don't have to throw
        # out our buffer.)
        self.stream.seek(orig_filepos)

        # Return the calculated filepos
        return filepos

    # /////////////////////////////////////////////////////////////////
    # Helper methods
    # /////////////////////////////////////////////////////////////////

    def _read(self, size=None):
        """
        Read up to ``size`` bytes from the underlying stream, decode
        them using this reader's encoding, and return the resulting
        unicode string.  ``linebuffer`` is not included in the result.
        """
        if size == 0:
            return ""

        # Skip past the byte order marker, if present.
        if self._bom and self.stream.tell() == 0:
            self.stream.read(self._bom)

        # Read the requested number of bytes.
        if size is None:
            new_bytes = self.stream.read()
        else:
            new_bytes = self.stream.read(size)
        bytes = self.bytebuffer + new_bytes

        # Decode the bytes into unicode characters
        chars, bytes_decoded = self._incr_decode(bytes)

        # If we got bytes but couldn't decode any, then read further.
        if (size is not None) and (not chars) and (len(new_bytes) > 0):
            while not chars:
                new_bytes = self.stream.read(1)
                if not new_bytes:
                    break  # end of file.
                bytes += new_bytes
                chars, bytes_decoded = self._incr_decode(bytes)

        # Record any bytes we didn't consume.
        self.bytebuffer = bytes[bytes_decoded:]

        # Return the result
        return chars

    def _incr_decode(self, bytes):
        """
        Decode the given byte string into a unicode string, using this
        reader's encoding.  If an exception is encountered that
        appears to be caused by a truncation error, then just decode
        the byte string without the bytes that cause the trunctaion
        error.

        Return a tuple ``(chars, num_consumed)``, where ``chars`` is
        the decoded unicode string, and ``num_consumed`` is the
        number of bytes that were consumed.
        """
        while True:
            try:
                return self.decode(bytes, "strict")
            except UnicodeDecodeError as exc:
                # If the exception occurs at the end of the string,
                # then assume that it's a truncation error.
                if exc.end == len(bytes):
                    return self.decode(bytes[: exc.start], self.errors)

                # Otherwise, if we're being strict, then raise it.
                elif self.errors == "strict":
                    raise

                # If we're not strict, then re-process it with our
                # errors setting.  This *may* raise an exception.
                else:
                    return self.decode(bytes, self.errors)

    _BOM_TABLE = {
        "utf8": [(codecs.BOM_UTF8, None)],
        "utf16": [(codecs.BOM_UTF16_LE, "utf16-le"), (codecs.BOM_UTF16_BE, "utf16-be")],
        "utf16le": [(codecs.BOM_UTF16_LE, None)],
        "utf16be": [(codecs.BOM_UTF16_BE, None)],
        "utf32": [(codecs.BOM_UTF32_LE, "utf32-le"), (codecs.BOM_UTF32_BE, "utf32-be")],
        "utf32le": [(codecs.BOM_UTF32_LE, None)],
        "utf32be": [(codecs.BOM_UTF32_BE, None)],
    }

    def _check_bom(self):
        # Normalize our encoding name
        enc = re.sub("[ -]", "", self.encoding.lower())

        # Look up our encoding in the BOM table.
        bom_info = self._BOM_TABLE.get(enc)

        if bom_info:
            # Read a prefix, to check against the BOM(s)
            bytes = self.stream.read(16)
            self.stream.seek(0)

            # Check for each possible BOM.
            for (bom, new_encoding) in bom_info:
                if bytes.startswith(bom):
                    if new_encoding:
                        self.encoding = new_encoding
                    return len(bom)

        return None


__all__ = [
    "path",
    "PathPointer",
    "FileSystemPathPointer",
    "BufferedGzipFile",
    "GzipFileSystemPathPointer",
    "GzipFileSystemPathPointer",
    "find",
    "retrieve",
    "FORMATS",
    "AUTO_FORMATS",
    "load",
    "show_cfg",
    "clear_cache",
    "LazyLoader",
    "OpenOnDemandZipFile",
    "GzipFileSystemPathPointer",
    "SeekableUnicodeStreamReader",
]
