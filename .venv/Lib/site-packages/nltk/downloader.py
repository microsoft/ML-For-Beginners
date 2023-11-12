# Natural Language Toolkit: Corpus & Model Downloader
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
The NLTK corpus and module downloader.  This module defines several
interfaces which can be used to download corpora, models, and other
data packages that can be used with NLTK.

Downloading Packages
====================
If called with no arguments, ``download()`` will display an interactive
interface which can be used to download and install new packages.
If Tkinter is available, then a graphical interface will be shown,
otherwise a simple text interface will be provided.

Individual packages can be downloaded by calling the ``download()``
function with a single argument, giving the package identifier for the
package that should be downloaded:

    >>> download('treebank') # doctest: +SKIP
    [nltk_data] Downloading package 'treebank'...
    [nltk_data]   Unzipping corpora/treebank.zip.

NLTK also provides a number of \"package collections\", consisting of
a group of related packages.  To download all packages in a
colleciton, simply call ``download()`` with the collection's
identifier:

    >>> download('all-corpora') # doctest: +SKIP
    [nltk_data] Downloading package 'abc'...
    [nltk_data]   Unzipping corpora/abc.zip.
    [nltk_data] Downloading package 'alpino'...
    [nltk_data]   Unzipping corpora/alpino.zip.
      ...
    [nltk_data] Downloading package 'words'...
    [nltk_data]   Unzipping corpora/words.zip.

Download Directory
==================
By default, packages are installed in either a system-wide directory
(if Python has sufficient access to write to it); or in the current
user's home directory.  However, the ``download_dir`` argument may be
used to specify a different installation target, if desired.

See ``Downloader.default_download_dir()`` for more a detailed
description of how the default download directory is chosen.

NLTK Download Server
====================
Before downloading any packages, the corpus and module downloader
contacts the NLTK download server, to retrieve an index file
describing the available packages.  By default, this index file is
loaded from ``https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml``.
If necessary, it is possible to create a new ``Downloader`` object,
specifying a different URL for the package index file.

Usage::

    python nltk/downloader.py [-d DATADIR] [-q] [-f] [-k] PACKAGE_IDS

or::

    python -m nltk.downloader [-d DATADIR] [-q] [-f] [-k] PACKAGE_IDS
"""
# ----------------------------------------------------------------------

"""

  0     1  2    3
[label][----][label][----]
[column  ][column     ]

Notes
=====
Handling data files..  Some questions:

* Should the data files be kept zipped or unzipped?  I say zipped.

* Should the data files be kept in svn at all?  Advantages: history;
  automatic version numbers; 'svn up' could be used rather than the
  downloader to update the corpora.  Disadvantages: they're big,
  which makes working from svn a bit of a pain.  And we're planning
  to potentially make them much bigger.  I don't think we want
  people to have to download 400MB corpora just to use nltk from svn.

* Compromise: keep the data files in trunk/data rather than in
  trunk/nltk.  That way you can check them out in svn if you want
  to; but you don't need to, and you can use the downloader instead.

* Also: keep models in mind.  When we change the code, we'd
  potentially like the models to get updated.  This could require a
  little thought.

* So.. let's assume we have a trunk/data directory, containing a bunch
  of packages.  The packages should be kept as zip files, because we
  really shouldn't be editing them much (well -- we may edit models
  more, but they tend to be binary-ish files anyway, where diffs
  aren't that helpful).  So we'll have trunk/data, with a bunch of
  files like abc.zip and treebank.zip and propbank.zip.  For each
  package we could also have eg treebank.xml and propbank.xml,
  describing the contents of the package (name, copyright, license,
  etc).  Collections would also have .xml files.  Finally, we would
  pull all these together to form a single index.xml file.  Some
  directory structure wouldn't hurt.  So how about::

    /trunk/data/ ....................... root of data svn
      index.xml ........................ main index file
      src/ ............................. python scripts
      packages/ ........................ dir for packages
        corpora/ ....................... zip & xml files for corpora
        grammars/ ...................... zip & xml files for grammars
        taggers/ ....................... zip & xml files for taggers
        tokenizers/ .................... zip & xml files for tokenizers
        etc.
      collections/ ..................... xml files for collections

  Where the root (/trunk/data) would contain a makefile; and src/
  would contain a script to update the info.xml file.  It could also
  contain scripts to rebuild some of the various model files.  The
  script that builds index.xml should probably check that each zip
  file expands entirely into a single subdir, whose name matches the
  package's uid.

Changes I need to make:
  - in index: change "size" to "filesize" or "compressed-size"
  - in index: add "unzipped-size"
  - when checking status: check both compressed & uncompressed size.
    uncompressed size is important to make sure we detect a problem
    if something got partially unzipped.  define new status values
    to differentiate stale vs corrupt vs corruptly-uncompressed??
    (we shouldn't need to re-download the file if the zip file is ok
    but it didn't get uncompressed fully.)
  - add other fields to the index: author, license, copyright, contact,
    etc.

the current grammars/ package would become a single new package (eg
toy-grammars or book-grammars).

xml file should have:
  - authorship info
  - license info
  - copyright info
  - contact info
  - info about what type of data/annotation it contains?
  - recommended corpus reader?

collections can contain other collections.  they can also contain
multiple package types (corpora & models).  Have a single 'basics'
package that includes everything we talk about in the book?

n.b.: there will have to be a fallback to the punkt tokenizer, in case
they didn't download that model.

default: unzip or not?

"""
import functools
import itertools
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import warnings
import zipfile
from hashlib import md5
from xml.etree import ElementTree

try:
    TKINTER = True
    from tkinter import Button, Canvas, Entry, Frame, IntVar, Label, Menu, TclError, Tk
    from tkinter.messagebox import showerror

    from nltk.draw.table import Table
    from nltk.draw.util import ShowText
except ImportError:
    TKINTER = False
    TclError = ValueError

from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import nltk

# urllib2 = nltk.internals.import_from_stdlib('urllib2')


######################################################################
# Directory entry objects (from the data server's index file)
######################################################################


class Package:
    """
    A directory entry for a downloadable package.  These entries are
    extracted from the XML index file that is downloaded by
    ``Downloader``.  Each package consists of a single file; but if
    that file is a zip file, then it can be automatically decompressed
    when the package is installed.
    """

    def __init__(
        self,
        id,
        url,
        name=None,
        subdir="",
        size=None,
        unzipped_size=None,
        checksum=None,
        svn_revision=None,
        copyright="Unknown",
        contact="Unknown",
        license="Unknown",
        author="Unknown",
        unzip=True,
        **kw,
    ):
        self.id = id
        """A unique identifier for this package."""

        self.name = name or id
        """A string name for this package."""

        self.subdir = subdir
        """The subdirectory where this package should be installed.
           E.g., ``'corpora'`` or ``'taggers'``."""

        self.url = url
        """A URL that can be used to download this package's file."""

        self.size = int(size)
        """The filesize (in bytes) of the package file."""

        self.unzipped_size = int(unzipped_size)
        """The total filesize of the files contained in the package's
           zipfile."""

        self.checksum = checksum
        """The MD-5 checksum of the package file."""

        self.svn_revision = svn_revision
        """A subversion revision number for this package."""

        self.copyright = copyright
        """Copyright holder for this package."""

        self.contact = contact
        """Name & email of the person who should be contacted with
           questions about this package."""

        self.license = license
        """License information for this package."""

        self.author = author
        """Author of this package."""

        ext = os.path.splitext(url.split("/")[-1])[1]
        self.filename = os.path.join(subdir, id + ext)
        """The filename that should be used for this package's file.  It
           is formed by joining ``self.subdir`` with ``self.id``, and
           using the same extension as ``url``."""

        self.unzip = bool(int(unzip))  # '0' or '1'
        """A flag indicating whether this corpus should be unzipped by
           default."""

        # Include any other attributes provided by the XML file.
        self.__dict__.update(kw)

    @staticmethod
    def fromxml(xml):
        if isinstance(xml, str):
            xml = ElementTree.parse(xml)
        for key in xml.attrib:
            xml.attrib[key] = str(xml.attrib[key])
        return Package(**xml.attrib)

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return "<Package %s>" % self.id


class Collection:
    """
    A directory entry for a collection of downloadable packages.
    These entries are extracted from the XML index file that is
    downloaded by ``Downloader``.
    """

    def __init__(self, id, children, name=None, **kw):
        self.id = id
        """A unique identifier for this collection."""

        self.name = name or id
        """A string name for this collection."""

        self.children = children
        """A list of the ``Collections`` or ``Packages`` directly
           contained by this collection."""

        self.packages = None
        """A list of ``Packages`` contained by this collection or any
           collections it recursively contains."""

        # Include any other attributes provided by the XML file.
        self.__dict__.update(kw)

    @staticmethod
    def fromxml(xml):
        if isinstance(xml, str):
            xml = ElementTree.parse(xml)
        for key in xml.attrib:
            xml.attrib[key] = str(xml.attrib[key])
        children = [child.get("ref") for child in xml.findall("item")]
        return Collection(children=children, **xml.attrib)

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return "<Collection %s>" % self.id


######################################################################
# Message Passing Objects
######################################################################


class DownloaderMessage:
    """A status message object, used by ``incr_download`` to
    communicate its progress."""


class StartCollectionMessage(DownloaderMessage):
    """Data server has started working on a collection of packages."""

    def __init__(self, collection):
        self.collection = collection


class FinishCollectionMessage(DownloaderMessage):
    """Data server has finished working on a collection of packages."""

    def __init__(self, collection):
        self.collection = collection


class StartPackageMessage(DownloaderMessage):
    """Data server has started working on a package."""

    def __init__(self, package):
        self.package = package


class FinishPackageMessage(DownloaderMessage):
    """Data server has finished working on a package."""

    def __init__(self, package):
        self.package = package


class StartDownloadMessage(DownloaderMessage):
    """Data server has started downloading a package."""

    def __init__(self, package):
        self.package = package


class FinishDownloadMessage(DownloaderMessage):
    """Data server has finished downloading a package."""

    def __init__(self, package):
        self.package = package


class StartUnzipMessage(DownloaderMessage):
    """Data server has started unzipping a package."""

    def __init__(self, package):
        self.package = package


class FinishUnzipMessage(DownloaderMessage):
    """Data server has finished unzipping a package."""

    def __init__(self, package):
        self.package = package


class UpToDateMessage(DownloaderMessage):
    """The package download file is already up-to-date"""

    def __init__(self, package):
        self.package = package


class StaleMessage(DownloaderMessage):
    """The package download file is out-of-date or corrupt"""

    def __init__(self, package):
        self.package = package


class ErrorMessage(DownloaderMessage):
    """Data server encountered an error"""

    def __init__(self, package, message):
        self.package = package
        if isinstance(message, Exception):
            self.message = str(message)
        else:
            self.message = message


class ProgressMessage(DownloaderMessage):
    """Indicates how much progress the data server has made"""

    def __init__(self, progress):
        self.progress = progress


class SelectDownloadDirMessage(DownloaderMessage):
    """Indicates what download directory the data server is using"""

    def __init__(self, download_dir):
        self.download_dir = download_dir


######################################################################
# NLTK Data Server
######################################################################


class Downloader:
    """
    A class used to access the NLTK data server, which can be used to
    download corpora and other data packages.
    """

    # /////////////////////////////////////////////////////////////////
    # Configuration
    # /////////////////////////////////////////////////////////////////

    INDEX_TIMEOUT = 60 * 60  # 1 hour
    """The amount of time after which the cached copy of the data
       server index will be considered 'stale,' and will be
       re-downloaded."""

    DEFAULT_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml"
    """The default URL for the NLTK data server's index.  An
       alternative URL can be specified when creating a new
       ``Downloader`` object."""

    # /////////////////////////////////////////////////////////////////
    # Status Constants
    # /////////////////////////////////////////////////////////////////

    INSTALLED = "installed"
    """A status string indicating that a package or collection is
       installed and up-to-date."""
    NOT_INSTALLED = "not installed"
    """A status string indicating that a package or collection is
       not installed."""
    STALE = "out of date"
    """A status string indicating that a package or collection is
       corrupt or out-of-date."""
    PARTIAL = "partial"
    """A status string indicating that a collection is partially
       installed (i.e., only some of its packages are installed.)"""

    # /////////////////////////////////////////////////////////////////
    # Constructor
    # /////////////////////////////////////////////////////////////////

    def __init__(self, server_index_url=None, download_dir=None):
        self._url = server_index_url or self.DEFAULT_URL
        """The URL for the data server's index file."""

        self._collections = {}
        """Dictionary from collection identifier to ``Collection``"""

        self._packages = {}
        """Dictionary from package identifier to ``Package``"""

        self._download_dir = download_dir
        """The default directory to which packages will be downloaded."""

        self._index = None
        """The XML index file downloaded from the data server"""

        self._index_timestamp = None
        """Time at which ``self._index`` was downloaded.  If it is more
           than ``INDEX_TIMEOUT`` seconds old, it will be re-downloaded."""

        self._status_cache = {}
        """Dictionary from package/collection identifier to status
           string (``INSTALLED``, ``NOT_INSTALLED``, ``STALE``, or
           ``PARTIAL``).  Cache is used for packages only, not
           collections."""

        self._errors = None
        """Flag for telling if all packages got successfully downloaded or not."""

        # decide where we're going to save things to.
        if self._download_dir is None:
            self._download_dir = self.default_download_dir()

    # /////////////////////////////////////////////////////////////////
    # Information
    # /////////////////////////////////////////////////////////////////

    def list(
        self,
        download_dir=None,
        show_packages=True,
        show_collections=True,
        header=True,
        more_prompt=False,
        skip_installed=False,
    ):
        lines = 0  # for more_prompt
        if download_dir is None:
            download_dir = self._download_dir
            print("Using default data directory (%s)" % download_dir)
        if header:
            print("=" * (26 + len(self._url)))
            print(" Data server index for <%s>" % self._url)
            print("=" * (26 + len(self._url)))
            lines += 3  # for more_prompt
        stale = partial = False

        categories = []
        if show_packages:
            categories.append("packages")
        if show_collections:
            categories.append("collections")
        for category in categories:
            print("%s:" % category.capitalize())
            lines += 1  # for more_prompt
            for info in sorted(getattr(self, category)(), key=str):
                status = self.status(info, download_dir)
                if status == self.INSTALLED and skip_installed:
                    continue
                if status == self.STALE:
                    stale = True
                if status == self.PARTIAL:
                    partial = True
                prefix = {
                    self.INSTALLED: "*",
                    self.STALE: "-",
                    self.PARTIAL: "P",
                    self.NOT_INSTALLED: " ",
                }[status]
                name = textwrap.fill(
                    "-" * 27 + (info.name or info.id), 75, subsequent_indent=27 * " "
                )[27:]
                print("  [{}] {} {}".format(prefix, info.id.ljust(20, "."), name))
                lines += len(name.split("\n"))  # for more_prompt
                if more_prompt and lines > 20:
                    user_input = input("Hit Enter to continue: ")
                    if user_input.lower() in ("x", "q"):
                        return
                    lines = 0
            print()
        msg = "([*] marks installed packages"
        if stale:
            msg += "; [-] marks out-of-date or corrupt packages"
        if partial:
            msg += "; [P] marks partially installed collections"
        print(textwrap.fill(msg + ")", subsequent_indent=" ", width=76))

    def packages(self):
        self._update_index()
        return self._packages.values()

    def corpora(self):
        self._update_index()
        return [pkg for (id, pkg) in self._packages.items() if pkg.subdir == "corpora"]

    def models(self):
        self._update_index()
        return [pkg for (id, pkg) in self._packages.items() if pkg.subdir != "corpora"]

    def collections(self):
        self._update_index()
        return self._collections.values()

    # /////////////////////////////////////////////////////////////////
    # Downloading
    # /////////////////////////////////////////////////////////////////

    def _info_or_id(self, info_or_id):
        if isinstance(info_or_id, str):
            return self.info(info_or_id)
        else:
            return info_or_id

    # [xx] When during downloading is it 'safe' to abort?  Only unsafe
    # time is *during* an unzip -- we don't want to leave a
    # partially-unzipped corpus in place because we wouldn't notice
    # it.  But if we had the exact total size of the unzipped corpus,
    # then that would be fine.  Then we could abort anytime we want!
    # So this is really what we should do.  That way the threaded
    # downloader in the gui can just kill the download thread anytime
    # it wants.

    def incr_download(self, info_or_id, download_dir=None, force=False):
        # If they didn't specify a download_dir, then use the default one.
        if download_dir is None:
            download_dir = self._download_dir
            yield SelectDownloadDirMessage(download_dir)

        # If they gave us a list of ids, then download each one.
        if isinstance(info_or_id, (list, tuple)):
            yield from self._download_list(info_or_id, download_dir, force)
            return

        # Look up the requested collection or package.
        try:
            info = self._info_or_id(info_or_id)
        except (OSError, ValueError) as e:
            yield ErrorMessage(None, f"Error loading {info_or_id}: {e}")
            return

        # Handle collections.
        if isinstance(info, Collection):
            yield StartCollectionMessage(info)
            yield from self.incr_download(info.children, download_dir, force)
            yield FinishCollectionMessage(info)

        # Handle Packages (delegate to a helper function).
        else:
            yield from self._download_package(info, download_dir, force)

    def _num_packages(self, item):
        if isinstance(item, Package):
            return 1
        else:
            return len(item.packages)

    def _download_list(self, items, download_dir, force):
        # Look up the requested items.
        for i in range(len(items)):
            try:
                items[i] = self._info_or_id(items[i])
            except (OSError, ValueError) as e:
                yield ErrorMessage(items[i], e)
                return

        # Download each item, re-scaling their progress.
        num_packages = sum(self._num_packages(item) for item in items)
        progress = 0
        for i, item in enumerate(items):
            if isinstance(item, Package):
                delta = 1.0 / num_packages
            else:
                delta = len(item.packages) / num_packages
            for msg in self.incr_download(item, download_dir, force):
                if isinstance(msg, ProgressMessage):
                    yield ProgressMessage(progress + msg.progress * delta)
                else:
                    yield msg

            progress += 100 * delta

    def _download_package(self, info, download_dir, force):
        yield StartPackageMessage(info)
        yield ProgressMessage(0)

        # Do we already have the current version?
        status = self.status(info, download_dir)
        if not force and status == self.INSTALLED:
            yield UpToDateMessage(info)
            yield ProgressMessage(100)
            yield FinishPackageMessage(info)
            return

        # Remove the package from our status cache
        self._status_cache.pop(info.id, None)

        # Check for (and remove) any old/stale version.
        filepath = os.path.join(download_dir, info.filename)
        if os.path.exists(filepath):
            if status == self.STALE:
                yield StaleMessage(info)
            os.remove(filepath)

        # Ensure the download_dir exists
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        if not os.path.exists(os.path.join(download_dir, info.subdir)):
            os.makedirs(os.path.join(download_dir, info.subdir))

        # Download the file.  This will raise an IOError if the url
        # is not found.
        yield StartDownloadMessage(info)
        yield ProgressMessage(5)
        try:
            infile = urlopen(info.url)
            with open(filepath, "wb") as outfile:
                num_blocks = max(1, info.size / (1024 * 16))
                for block in itertools.count():
                    s = infile.read(1024 * 16)  # 16k blocks.
                    outfile.write(s)
                    if not s:
                        break
                    if block % 2 == 0:  # how often?
                        yield ProgressMessage(min(80, 5 + 75 * (block / num_blocks)))
            infile.close()
        except OSError as e:
            yield ErrorMessage(
                info,
                "Error downloading %r from <%s>:" "\n  %s" % (info.id, info.url, e),
            )
            return
        yield FinishDownloadMessage(info)
        yield ProgressMessage(80)

        # If it's a zipfile, uncompress it.
        if info.filename.endswith(".zip"):
            zipdir = os.path.join(download_dir, info.subdir)
            # Unzip if we're unzipping by default; *or* if it's already
            # been unzipped (presumably a previous version).
            if info.unzip or os.path.exists(os.path.join(zipdir, info.id)):
                yield StartUnzipMessage(info)
                for msg in _unzip_iter(filepath, zipdir, verbose=False):
                    # Somewhat of a hack, but we need a proper package reference
                    msg.package = info
                    yield msg
                yield FinishUnzipMessage(info)

        yield FinishPackageMessage(info)

    def download(
        self,
        info_or_id=None,
        download_dir=None,
        quiet=False,
        force=False,
        prefix="[nltk_data] ",
        halt_on_error=True,
        raise_on_error=False,
        print_error_to=sys.stderr,
    ):

        print_to = functools.partial(print, file=print_error_to)
        # If no info or id is given, then use the interactive shell.
        if info_or_id is None:
            # [xx] hmm -- changing self._download_dir here seems like
            # the wrong thing to do.  Maybe the _interactive_download
            # function should make a new copy of self to use?
            if download_dir is not None:
                self._download_dir = download_dir
            self._interactive_download()
            return True

        else:
            # Define a helper function for displaying output:
            def show(s, prefix2=""):
                print_to(
                    textwrap.fill(
                        s,
                        initial_indent=prefix + prefix2,
                        subsequent_indent=prefix + prefix2 + " " * 4,
                    )
                )

            for msg in self.incr_download(info_or_id, download_dir, force):
                # Error messages
                if isinstance(msg, ErrorMessage):
                    show(msg.message)
                    if raise_on_error:
                        raise ValueError(msg.message)
                    if halt_on_error:
                        return False
                    self._errors = True
                    if not quiet:
                        print_to("Error installing package. Retry? [n/y/e]")
                        choice = input().strip()
                        if choice in ["y", "Y"]:
                            if not self.download(
                                msg.package.id,
                                download_dir,
                                quiet,
                                force,
                                prefix,
                                halt_on_error,
                                raise_on_error,
                            ):
                                return False
                        elif choice in ["e", "E"]:
                            return False

                # All other messages
                if not quiet:
                    # Collection downloading messages:
                    if isinstance(msg, StartCollectionMessage):
                        show("Downloading collection %r" % msg.collection.id)
                        prefix += "   | "
                        print_to(prefix)
                    elif isinstance(msg, FinishCollectionMessage):
                        print_to(prefix)
                        prefix = prefix[:-4]
                        if self._errors:
                            show(
                                "Downloaded collection %r with errors"
                                % msg.collection.id
                            )
                        else:
                            show("Done downloading collection %s" % msg.collection.id)

                    # Package downloading messages:
                    elif isinstance(msg, StartPackageMessage):
                        show(
                            "Downloading package %s to %s..."
                            % (msg.package.id, download_dir)
                        )
                    elif isinstance(msg, UpToDateMessage):
                        show("Package %s is already up-to-date!" % msg.package.id, "  ")
                    # elif isinstance(msg, StaleMessage):
                    #    show('Package %s is out-of-date or corrupt' %
                    #         msg.package.id, '  ')
                    elif isinstance(msg, StartUnzipMessage):
                        show("Unzipping %s." % msg.package.filename, "  ")

                    # Data directory message:
                    elif isinstance(msg, SelectDownloadDirMessage):
                        download_dir = msg.download_dir
        return True

    def is_stale(self, info_or_id, download_dir=None):
        return self.status(info_or_id, download_dir) == self.STALE

    def is_installed(self, info_or_id, download_dir=None):
        return self.status(info_or_id, download_dir) == self.INSTALLED

    def clear_status_cache(self, id=None):
        if id is None:
            self._status_cache.clear()
        else:
            self._status_cache.pop(id, None)

    def status(self, info_or_id, download_dir=None):
        """
        Return a constant describing the status of the given package
        or collection.  Status can be one of ``INSTALLED``,
        ``NOT_INSTALLED``, ``STALE``, or ``PARTIAL``.
        """
        if download_dir is None:
            download_dir = self._download_dir
        info = self._info_or_id(info_or_id)

        # Handle collections:
        if isinstance(info, Collection):
            pkg_status = [self.status(pkg.id) for pkg in info.packages]
            if self.STALE in pkg_status:
                return self.STALE
            elif self.PARTIAL in pkg_status:
                return self.PARTIAL
            elif self.INSTALLED in pkg_status and self.NOT_INSTALLED in pkg_status:
                return self.PARTIAL
            elif self.NOT_INSTALLED in pkg_status:
                return self.NOT_INSTALLED
            else:
                return self.INSTALLED

        # Handle packages:
        else:
            filepath = os.path.join(download_dir, info.filename)
            if download_dir != self._download_dir:
                return self._pkg_status(info, filepath)
            else:
                if info.id not in self._status_cache:
                    self._status_cache[info.id] = self._pkg_status(info, filepath)
                return self._status_cache[info.id]

    def _pkg_status(self, info, filepath):
        if not os.path.exists(filepath):
            return self.NOT_INSTALLED

        # Check if the file has the correct size.
        try:
            filestat = os.stat(filepath)
        except OSError:
            return self.NOT_INSTALLED
        if filestat.st_size != int(info.size):
            return self.STALE

        # Check if the file's checksum matches
        if md5_hexdigest(filepath) != info.checksum:
            return self.STALE

        # If it's a zipfile, and it's been at least partially
        # unzipped, then check if it's been fully unzipped.
        if filepath.endswith(".zip"):
            unzipdir = filepath[:-4]
            if not os.path.exists(unzipdir):
                return self.INSTALLED  # but not unzipped -- ok!
            if not os.path.isdir(unzipdir):
                return self.STALE

            unzipped_size = sum(
                os.stat(os.path.join(d, f)).st_size
                for d, _, files in os.walk(unzipdir)
                for f in files
            )
            if unzipped_size != info.unzipped_size:
                return self.STALE

        # Otherwise, everything looks good.
        return self.INSTALLED

    def update(self, quiet=False, prefix="[nltk_data] "):
        """
        Re-download any packages whose status is STALE.
        """
        self.clear_status_cache()
        for pkg in self.packages():
            if self.status(pkg) == self.STALE:
                self.download(pkg, quiet=quiet, prefix=prefix)

    # /////////////////////////////////////////////////////////////////
    # Index
    # /////////////////////////////////////////////////////////////////

    def _update_index(self, url=None):
        """A helper function that ensures that self._index is
        up-to-date.  If the index is older than self.INDEX_TIMEOUT,
        then download it again."""
        # Check if the index is already up-to-date.  If so, do nothing.
        if not (
            self._index is None
            or url is not None
            or time.time() - self._index_timestamp > self.INDEX_TIMEOUT
        ):
            return

        # If a URL was specified, then update our URL.
        self._url = url or self._url

        # Download the index file.
        self._index = nltk.internals.ElementWrapper(
            ElementTree.parse(urlopen(self._url)).getroot()
        )
        self._index_timestamp = time.time()

        # Build a dictionary of packages.
        packages = [Package.fromxml(p) for p in self._index.findall("packages/package")]
        self._packages = {p.id: p for p in packages}

        # Build a dictionary of collections.
        collections = [
            Collection.fromxml(c) for c in self._index.findall("collections/collection")
        ]
        self._collections = {c.id: c for c in collections}

        # Replace identifiers with actual children in collection.children.
        for collection in self._collections.values():
            for i, child_id in enumerate(collection.children):
                if child_id in self._packages:
                    collection.children[i] = self._packages[child_id]
                elif child_id in self._collections:
                    collection.children[i] = self._collections[child_id]
                else:
                    print(
                        "removing collection member with no package: {}".format(
                            child_id
                        )
                    )
                    del collection.children[i]

        # Fill in collection.packages for each collection.
        for collection in self._collections.values():
            packages = {}
            queue = [collection]
            for child in queue:
                if isinstance(child, Collection):
                    queue.extend(child.children)
                elif isinstance(child, Package):
                    packages[child.id] = child
                else:
                    pass
            collection.packages = packages.values()

        # Flush the status cache
        self._status_cache.clear()

    def index(self):
        """
        Return the XML index describing the packages available from
        the data server.  If necessary, this index will be downloaded
        from the data server.
        """
        self._update_index()
        return self._index

    def info(self, id):
        """Return the ``Package`` or ``Collection`` record for the
        given item."""
        self._update_index()
        if id in self._packages:
            return self._packages[id]
        if id in self._collections:
            return self._collections[id]
        raise ValueError("Package %r not found in index" % id)

    def xmlinfo(self, id):
        """Return the XML info record for the given item"""
        self._update_index()
        for package in self._index.findall("packages/package"):
            if package.get("id") == id:
                return package
        for collection in self._index.findall("collections/collection"):
            if collection.get("id") == id:
                return collection
        raise ValueError("Package %r not found in index" % id)

    # /////////////////////////////////////////////////////////////////
    # URL & Data Directory
    # /////////////////////////////////////////////////////////////////

    def _get_url(self):
        """The URL for the data server's index file."""
        return self._url

    def _set_url(self, url):
        """
        Set a new URL for the data server. If we're unable to contact
        the given url, then the original url is kept.
        """
        original_url = self._url
        try:
            self._update_index(url)
        except:
            self._url = original_url
            raise

    url = property(_get_url, _set_url)

    def default_download_dir(self):
        """
        Return the directory to which packages will be downloaded by
        default.  This value can be overridden using the constructor,
        or on a case-by-case basis using the ``download_dir`` argument when
        calling ``download()``.

        On Windows, the default download directory is
        ``PYTHONHOME/lib/nltk``, where *PYTHONHOME* is the
        directory containing Python, e.g. ``C:\\Python25``.

        On all other platforms, the default directory is the first of
        the following which exists or which can be created with write
        permission: ``/usr/share/nltk_data``, ``/usr/local/share/nltk_data``,
        ``/usr/lib/nltk_data``, ``/usr/local/lib/nltk_data``, ``~/nltk_data``.
        """
        # Check if we are on GAE where we cannot write into filesystem.
        if "APPENGINE_RUNTIME" in os.environ:
            return

        # Check if we have sufficient permissions to install in a
        # variety of system-wide locations.
        for nltkdir in nltk.data.path:
            if os.path.exists(nltkdir) and nltk.internals.is_writable(nltkdir):
                return nltkdir

        # On Windows, use %APPDATA%
        if sys.platform == "win32" and "APPDATA" in os.environ:
            homedir = os.environ["APPDATA"]

        # Otherwise, install in the user's home directory.
        else:
            homedir = os.path.expanduser("~/")
            if homedir == "~/":
                raise ValueError("Could not find a default download directory")

        # append "nltk_data" to the home directory
        return os.path.join(homedir, "nltk_data")

    def _get_download_dir(self):
        """
        The default directory to which packages will be downloaded.
        This defaults to the value returned by ``default_download_dir()``.
        To override this default on a case-by-case basis, use the
        ``download_dir`` argument when calling ``download()``.
        """
        return self._download_dir

    def _set_download_dir(self, download_dir):
        self._download_dir = download_dir
        # Clear the status cache.
        self._status_cache.clear()

    download_dir = property(_get_download_dir, _set_download_dir)

    # /////////////////////////////////////////////////////////////////
    # Interactive Shell
    # /////////////////////////////////////////////////////////////////

    def _interactive_download(self):
        # Try the GUI first; if that doesn't work, try the simple
        # interactive shell.
        if TKINTER:
            try:
                DownloaderGUI(self).mainloop()
            except TclError:
                DownloaderShell(self).run()
        else:
            DownloaderShell(self).run()


class DownloaderShell:
    def __init__(self, dataserver):
        self._ds = dataserver

    def _simple_interactive_menu(self, *options):
        print("-" * 75)
        spc = (68 - sum(len(o) for o in options)) // (len(options) - 1) * " "
        print("    " + spc.join(options))
        print("-" * 75)

    def run(self):
        print("NLTK Downloader")
        while True:
            self._simple_interactive_menu(
                "d) Download",
                "l) List",
                " u) Update",
                "c) Config",
                "h) Help",
                "q) Quit",
            )
            user_input = input("Downloader> ").strip()
            if not user_input:
                print()
                continue
            command = user_input.lower().split()[0]
            args = user_input.split()[1:]
            try:
                if command == "l":
                    print()
                    self._ds.list(self._ds.download_dir, header=False, more_prompt=True)
                elif command == "h":
                    self._simple_interactive_help()
                elif command == "c":
                    self._simple_interactive_config()
                elif command in ("q", "x"):
                    return
                elif command == "d":
                    self._simple_interactive_download(args)
                elif command == "u":
                    self._simple_interactive_update()
                else:
                    print("Command %r unrecognized" % user_input)
            except HTTPError as e:
                print("Error reading from server: %s" % e)
            except URLError as e:
                print("Error connecting to server: %s" % e.reason)
            # try checking if user_input is a package name, &
            # downloading it?
            print()

    def _simple_interactive_download(self, args):
        if args:
            for arg in args:
                try:
                    self._ds.download(arg, prefix="    ")
                except (OSError, ValueError) as e:
                    print(e)
        else:
            while True:
                print()
                print("Download which package (l=list; x=cancel)?")
                user_input = input("  Identifier> ")
                if user_input.lower() == "l":
                    self._ds.list(
                        self._ds.download_dir,
                        header=False,
                        more_prompt=True,
                        skip_installed=True,
                    )
                    continue
                elif user_input.lower() in ("x", "q", ""):
                    return
                elif user_input:
                    for id in user_input.split():
                        try:
                            self._ds.download(id, prefix="    ")
                        except (OSError, ValueError) as e:
                            print(e)
                    break

    def _simple_interactive_update(self):
        while True:
            stale_packages = []
            stale = partial = False
            for info in sorted(getattr(self._ds, "packages")(), key=str):
                if self._ds.status(info) == self._ds.STALE:
                    stale_packages.append((info.id, info.name))

            print()
            if stale_packages:
                print("Will update following packages (o=ok; x=cancel)")
                for pid, pname in stale_packages:
                    name = textwrap.fill(
                        "-" * 27 + (pname), 75, subsequent_indent=27 * " "
                    )[27:]
                    print("  [ ] {} {}".format(pid.ljust(20, "."), name))
                print()

                user_input = input("  Identifier> ")
                if user_input.lower() == "o":
                    for pid, pname in stale_packages:
                        try:
                            self._ds.download(pid, prefix="    ")
                        except (OSError, ValueError) as e:
                            print(e)
                    break
                elif user_input.lower() in ("x", "q", ""):
                    return
            else:
                print("Nothing to update.")
                return

    def _simple_interactive_help(self):
        print()
        print("Commands:")
        print(
            "  d) Download a package or collection     u) Update out of date packages"
        )
        print("  l) List packages & collections          h) Help")
        print("  c) View & Modify Configuration          q) Quit")

    def _show_config(self):
        print()
        print("Data Server:")
        print("  - URL: <%s>" % self._ds.url)
        print("  - %d Package Collections Available" % len(self._ds.collections()))
        print("  - %d Individual Packages Available" % len(self._ds.packages()))
        print()
        print("Local Machine:")
        print("  - Data directory: %s" % self._ds.download_dir)

    def _simple_interactive_config(self):
        self._show_config()
        while True:
            print()
            self._simple_interactive_menu(
                "s) Show Config", "u) Set Server URL", "d) Set Data Dir", "m) Main Menu"
            )
            user_input = input("Config> ").strip().lower()
            if user_input == "s":
                self._show_config()
            elif user_input == "d":
                new_dl_dir = input("  New Directory> ").strip()
                if new_dl_dir in ("", "x", "q", "X", "Q"):
                    print("  Cancelled!")
                elif os.path.isdir(new_dl_dir):
                    self._ds.download_dir = new_dl_dir
                else:
                    print("Directory %r not found!  Create it first." % new_dl_dir)
            elif user_input == "u":
                new_url = input("  New URL> ").strip()
                if new_url in ("", "x", "q", "X", "Q"):
                    print("  Cancelled!")
                else:
                    if not new_url.startswith(("http://", "https://")):
                        new_url = "http://" + new_url
                    try:
                        self._ds.url = new_url
                    except Exception as e:
                        print(f"Error reading <{new_url!r}>:\n  {e}")
            elif user_input == "m":
                break


class DownloaderGUI:
    """
    Graphical interface for downloading packages from the NLTK data
    server.
    """

    # /////////////////////////////////////////////////////////////////
    # Column Configuration
    # /////////////////////////////////////////////////////////////////

    COLUMNS = [
        "",
        "Identifier",
        "Name",
        "Size",
        "Status",
        "Unzipped Size",
        "Copyright",
        "Contact",
        "License",
        "Author",
        "Subdir",
        "Checksum",
    ]
    """A list of the names of columns.  This controls the order in
       which the columns will appear.  If this is edited, then
       ``_package_to_columns()`` may need to be edited to match."""

    COLUMN_WEIGHTS = {"": 0, "Name": 5, "Size": 0, "Status": 0}
    """A dictionary specifying how columns should be resized when the
       table is resized.  Columns with weight 0 will not be resized at
       all; and columns with high weight will be resized more.
       Default weight (for columns not explicitly listed) is 1."""

    COLUMN_WIDTHS = {
        "": 1,
        "Identifier": 20,
        "Name": 45,
        "Size": 10,
        "Unzipped Size": 10,
        "Status": 12,
    }
    """A dictionary specifying how wide each column should be, in
       characters.  The default width (for columns not explicitly
       listed) is specified by ``DEFAULT_COLUMN_WIDTH``."""

    DEFAULT_COLUMN_WIDTH = 30
    """The default width for columns that are not explicitly listed
       in ``COLUMN_WIDTHS``."""

    INITIAL_COLUMNS = ["", "Identifier", "Name", "Size", "Status"]
    """The set of columns that should be displayed by default."""

    # Perform a few import-time sanity checks to make sure that the
    # column configuration variables are defined consistently:
    for c in COLUMN_WEIGHTS:
        assert c in COLUMNS
    for c in COLUMN_WIDTHS:
        assert c in COLUMNS
    for c in INITIAL_COLUMNS:
        assert c in COLUMNS

    # /////////////////////////////////////////////////////////////////
    # Color Configuration
    # /////////////////////////////////////////////////////////////////

    _BACKDROP_COLOR = ("#000", "#ccc")

    _ROW_COLOR = {
        Downloader.INSTALLED: ("#afa", "#080"),
        Downloader.PARTIAL: ("#ffa", "#880"),
        Downloader.STALE: ("#faa", "#800"),
        Downloader.NOT_INSTALLED: ("#fff", "#888"),
    }

    _MARK_COLOR = ("#000", "#ccc")

    # _FRONT_TAB_COLOR = ('#ccf', '#008')
    # _BACK_TAB_COLOR = ('#88a', '#448')
    _FRONT_TAB_COLOR = ("#fff", "#45c")
    _BACK_TAB_COLOR = ("#aaa", "#67a")

    _PROGRESS_COLOR = ("#f00", "#aaa")

    _TAB_FONT = "helvetica -16 bold"

    # /////////////////////////////////////////////////////////////////
    # Constructor
    # /////////////////////////////////////////////////////////////////

    def __init__(self, dataserver, use_threads=True):
        self._ds = dataserver
        self._use_threads = use_threads

        # For the threaded downloader:
        self._download_lock = threading.Lock()
        self._download_msg_queue = []
        self._download_abort_queue = []
        self._downloading = False

        # For tkinter after callbacks:
        self._afterid = {}

        # A message log.
        self._log_messages = []
        self._log_indent = 0
        self._log("NLTK Downloader Started!")

        # Create the main window.
        top = self.top = Tk()
        top.geometry("+50+50")
        top.title("NLTK Downloader")
        top.configure(background=self._BACKDROP_COLOR[1])

        # Set up some bindings now, in case anything goes wrong.
        top.bind("<Control-q>", self.destroy)
        top.bind("<Control-x>", self.destroy)
        self._destroyed = False

        self._column_vars = {}

        # Initialize the GUI.
        self._init_widgets()
        self._init_menu()
        try:
            self._fill_table()
        except HTTPError as e:
            showerror("Error reading from server", e)
        except URLError as e:
            showerror("Error connecting to server", e.reason)

        self._show_info()
        self._select_columns()
        self._table.select(0)

        # Make sure we get notified when we're destroyed, so we can
        # cancel any download in progress.
        self._table.bind("<Destroy>", self._destroy)

    def _log(self, msg):
        self._log_messages.append(
            "{} {}{}".format(time.ctime(), " | " * self._log_indent, msg)
        )

    # /////////////////////////////////////////////////////////////////
    # Internals
    # /////////////////////////////////////////////////////////////////

    def _init_widgets(self):
        # Create the top-level frame structures
        f1 = Frame(self.top, relief="raised", border=2, padx=8, pady=0)
        f1.pack(sid="top", expand=True, fill="both")
        f1.grid_rowconfigure(2, weight=1)
        f1.grid_columnconfigure(0, weight=1)
        Frame(f1, height=8).grid(column=0, row=0)  # spacer
        tabframe = Frame(f1)
        tabframe.grid(column=0, row=1, sticky="news")
        tableframe = Frame(f1)
        tableframe.grid(column=0, row=2, sticky="news")
        buttonframe = Frame(f1)
        buttonframe.grid(column=0, row=3, sticky="news")
        Frame(f1, height=8).grid(column=0, row=4)  # spacer
        infoframe = Frame(f1)
        infoframe.grid(column=0, row=5, sticky="news")
        Frame(f1, height=8).grid(column=0, row=6)  # spacer
        progressframe = Frame(
            self.top, padx=3, pady=3, background=self._BACKDROP_COLOR[1]
        )
        progressframe.pack(side="bottom", fill="x")
        self.top["border"] = 0
        self.top["highlightthickness"] = 0

        # Create the tabs
        self._tab_names = ["Collections", "Corpora", "Models", "All Packages"]
        self._tabs = {}
        for i, tab in enumerate(self._tab_names):
            label = Label(tabframe, text=tab, font=self._TAB_FONT)
            label.pack(side="left", padx=((i + 1) % 2) * 10)
            label.bind("<Button-1>", self._select_tab)
            self._tabs[tab.lower()] = label

        # Create the table.
        column_weights = [self.COLUMN_WEIGHTS.get(column, 1) for column in self.COLUMNS]
        self._table = Table(
            tableframe,
            self.COLUMNS,
            column_weights=column_weights,
            highlightthickness=0,
            listbox_height=16,
            reprfunc=self._table_reprfunc,
        )
        self._table.columnconfig(0, foreground=self._MARK_COLOR[0])  # marked
        for i, column in enumerate(self.COLUMNS):
            width = self.COLUMN_WIDTHS.get(column, self.DEFAULT_COLUMN_WIDTH)
            self._table.columnconfig(i, width=width)
        self._table.pack(expand=True, fill="both")
        self._table.focus()
        self._table.bind_to_listboxes("<Double-Button-1>", self._download)
        self._table.bind("<space>", self._table_mark)
        self._table.bind("<Return>", self._download)
        self._table.bind("<Left>", self._prev_tab)
        self._table.bind("<Right>", self._next_tab)
        self._table.bind("<Control-a>", self._mark_all)

        # Create entry boxes for URL & download_dir
        infoframe.grid_columnconfigure(1, weight=1)

        info = [
            ("url", "Server Index:", self._set_url),
            ("download_dir", "Download Directory:", self._set_download_dir),
        ]
        self._info = {}
        for (i, (key, label, callback)) in enumerate(info):
            Label(infoframe, text=label).grid(column=0, row=i, sticky="e")
            entry = Entry(
                infoframe,
                font="courier",
                relief="groove",
                disabledforeground="#007aff",
                foreground="#007aff",
            )
            self._info[key] = (entry, callback)
            entry.bind("<Return>", self._info_save)
            entry.bind("<Button-1>", lambda e, key=key: self._info_edit(key))
            entry.grid(column=1, row=i, sticky="ew")

        # If the user edits url or download_dir, and then clicks outside
        # the entry box, then save their results.
        self.top.bind("<Button-1>", self._info_save)

        # Create Download & Refresh buttons.
        self._download_button = Button(
            buttonframe, text="Download", command=self._download, width=8
        )
        self._download_button.pack(side="left")
        self._refresh_button = Button(
            buttonframe, text="Refresh", command=self._refresh, width=8
        )
        self._refresh_button.pack(side="right")

        # Create Progress bar
        self._progresslabel = Label(
            progressframe,
            text="",
            foreground=self._BACKDROP_COLOR[0],
            background=self._BACKDROP_COLOR[1],
        )
        self._progressbar = Canvas(
            progressframe,
            width=200,
            height=16,
            background=self._PROGRESS_COLOR[1],
            relief="sunken",
            border=1,
        )
        self._init_progressbar()
        self._progressbar.pack(side="right")
        self._progresslabel.pack(side="left")

    def _init_menu(self):
        menubar = Menu(self.top)

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(
            label="Download", underline=0, command=self._download, accelerator="Return"
        )
        filemenu.add_separator()
        filemenu.add_command(
            label="Change Server Index",
            underline=7,
            command=lambda: self._info_edit("url"),
        )
        filemenu.add_command(
            label="Change Download Directory",
            underline=0,
            command=lambda: self._info_edit("download_dir"),
        )
        filemenu.add_separator()
        filemenu.add_command(label="Show Log", underline=5, command=self._show_log)
        filemenu.add_separator()
        filemenu.add_command(
            label="Exit", underline=1, command=self.destroy, accelerator="Ctrl-x"
        )
        menubar.add_cascade(label="File", underline=0, menu=filemenu)

        # Create a menu to control which columns of the table are
        # shown.  n.b.: we never hide the first two columns (mark and
        # identifier).
        viewmenu = Menu(menubar, tearoff=0)
        for column in self._table.column_names[2:]:
            var = IntVar(self.top)
            assert column not in self._column_vars
            self._column_vars[column] = var
            if column in self.INITIAL_COLUMNS:
                var.set(1)
            viewmenu.add_checkbutton(
                label=column, underline=0, variable=var, command=self._select_columns
            )
        menubar.add_cascade(label="View", underline=0, menu=viewmenu)

        # Create a sort menu
        # [xx] this should be selectbuttons; and it should include
        # reversed sorts as options.
        sortmenu = Menu(menubar, tearoff=0)
        for column in self._table.column_names[1:]:
            sortmenu.add_command(
                label="Sort by %s" % column,
                command=(lambda c=column: self._table.sort_by(c, "ascending")),
            )
        sortmenu.add_separator()
        # sortmenu.add_command(label='Descending Sort:')
        for column in self._table.column_names[1:]:
            sortmenu.add_command(
                label="Reverse sort by %s" % column,
                command=(lambda c=column: self._table.sort_by(c, "descending")),
            )
        menubar.add_cascade(label="Sort", underline=0, menu=sortmenu)

        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", underline=0, command=self.about)
        helpmenu.add_command(
            label="Instructions", underline=0, command=self.help, accelerator="F1"
        )
        menubar.add_cascade(label="Help", underline=0, menu=helpmenu)
        self.top.bind("<F1>", self.help)

        self.top.config(menu=menubar)

    def _select_columns(self):
        for (column, var) in self._column_vars.items():
            if var.get():
                self._table.show_column(column)
            else:
                self._table.hide_column(column)

    def _refresh(self):
        self._ds.clear_status_cache()
        try:
            self._fill_table()
        except HTTPError as e:
            showerror("Error reading from server", e)
        except URLError as e:
            showerror("Error connecting to server", e.reason)
        self._table.select(0)

    def _info_edit(self, info_key):
        self._info_save()  # just in case.
        (entry, callback) = self._info[info_key]
        entry["state"] = "normal"
        entry["relief"] = "sunken"
        entry.focus()

    def _info_save(self, e=None):
        focus = self._table
        for entry, callback in self._info.values():
            if entry["state"] == "disabled":
                continue
            if e is not None and e.widget is entry and e.keysym != "Return":
                focus = entry
            else:
                entry["state"] = "disabled"
                entry["relief"] = "groove"
                callback(entry.get())
        focus.focus()

    def _table_reprfunc(self, row, col, val):
        if self._table.column_names[col].endswith("Size"):
            if isinstance(val, str):
                return "  %s" % val
            elif val < 1024**2:
                return "  %.1f KB" % (val / 1024.0**1)
            elif val < 1024**3:
                return "  %.1f MB" % (val / 1024.0**2)
            else:
                return "  %.1f GB" % (val / 1024.0**3)

        if col in (0, ""):
            return str(val)
        else:
            return "  %s" % val

    def _set_url(self, url):
        if url == self._ds.url:
            return
        try:
            self._ds.url = url
            self._fill_table()
        except OSError as e:
            showerror("Error Setting Server Index", str(e))
        self._show_info()

    def _set_download_dir(self, download_dir):
        if self._ds.download_dir == download_dir:
            return
        # check if the dir exists, and if not, ask if we should create it?

        # Clear our status cache, & re-check what's installed
        self._ds.download_dir = download_dir
        try:
            self._fill_table()
        except HTTPError as e:
            showerror("Error reading from server", e)
        except URLError as e:
            showerror("Error connecting to server", e.reason)
        self._show_info()

    def _show_info(self):
        print("showing info", self._ds.url)
        for entry, cb in self._info.values():
            entry["state"] = "normal"
            entry.delete(0, "end")
        self._info["url"][0].insert(0, self._ds.url)
        self._info["download_dir"][0].insert(0, self._ds.download_dir)
        for entry, cb in self._info.values():
            entry["state"] = "disabled"

    def _prev_tab(self, *e):
        for i, tab in enumerate(self._tab_names):
            if tab.lower() == self._tab and i > 0:
                self._tab = self._tab_names[i - 1].lower()
                try:
                    return self._fill_table()
                except HTTPError as e:
                    showerror("Error reading from server", e)
                except URLError as e:
                    showerror("Error connecting to server", e.reason)

    def _next_tab(self, *e):
        for i, tab in enumerate(self._tab_names):
            if tab.lower() == self._tab and i < (len(self._tabs) - 1):
                self._tab = self._tab_names[i + 1].lower()
                try:
                    return self._fill_table()
                except HTTPError as e:
                    showerror("Error reading from server", e)
                except URLError as e:
                    showerror("Error connecting to server", e.reason)

    def _select_tab(self, event):
        self._tab = event.widget["text"].lower()
        try:
            self._fill_table()
        except HTTPError as e:
            showerror("Error reading from server", e)
        except URLError as e:
            showerror("Error connecting to server", e.reason)

    _tab = "collections"
    # _tab = 'corpora'
    _rows = None

    def _fill_table(self):
        selected_row = self._table.selected_row()
        self._table.clear()
        if self._tab == "all packages":
            items = self._ds.packages()
        elif self._tab == "corpora":
            items = self._ds.corpora()
        elif self._tab == "models":
            items = self._ds.models()
        elif self._tab == "collections":
            items = self._ds.collections()
        else:
            assert 0, "bad tab value %r" % self._tab
        rows = [self._package_to_columns(item) for item in items]
        self._table.extend(rows)

        # Highlight the active tab.
        for tab, label in self._tabs.items():
            if tab == self._tab:
                label.configure(
                    foreground=self._FRONT_TAB_COLOR[0],
                    background=self._FRONT_TAB_COLOR[1],
                )
            else:
                label.configure(
                    foreground=self._BACK_TAB_COLOR[0],
                    background=self._BACK_TAB_COLOR[1],
                )

        self._table.sort_by("Identifier", order="ascending")
        self._color_table()
        self._table.select(selected_row)

        # This is a hack, because the scrollbar isn't updating its
        # position right -- I'm not sure what the underlying cause is
        # though.  (This is on OS X w/ python 2.5)  The length of
        # delay that's necessary seems to depend on how fast the
        # comptuer is. :-/
        self.top.after(150, self._table._scrollbar.set, *self._table._mlb.yview())
        self.top.after(300, self._table._scrollbar.set, *self._table._mlb.yview())

    def _update_table_status(self):
        for row_num in range(len(self._table)):
            status = self._ds.status(self._table[row_num, "Identifier"])
            self._table[row_num, "Status"] = status
        self._color_table()

    def _download(self, *e):
        # If we're using threads, then delegate to the threaded
        # downloader instead.
        if self._use_threads:
            return self._download_threaded(*e)

        marked = [
            self._table[row, "Identifier"]
            for row in range(len(self._table))
            if self._table[row, 0] != ""
        ]
        selection = self._table.selected_row()
        if not marked and selection is not None:
            marked = [self._table[selection, "Identifier"]]

        download_iter = self._ds.incr_download(marked, self._ds.download_dir)
        self._log_indent = 0
        self._download_cb(download_iter, marked)

    _DL_DELAY = 10

    def _download_cb(self, download_iter, ids):
        try:
            msg = next(download_iter)
        except StopIteration:
            # self._fill_table(sort=False)
            self._update_table_status()
            afterid = self.top.after(10, self._show_progress, 0)
            self._afterid["_download_cb"] = afterid
            return

        def show(s):
            self._progresslabel["text"] = s
            self._log(s)

        if isinstance(msg, ProgressMessage):
            self._show_progress(msg.progress)
        elif isinstance(msg, ErrorMessage):
            show(msg.message)
            if msg.package is not None:
                self._select(msg.package.id)
            self._show_progress(None)
            return  # halt progress.
        elif isinstance(msg, StartCollectionMessage):
            show("Downloading collection %s" % msg.collection.id)
            self._log_indent += 1
        elif isinstance(msg, StartPackageMessage):
            show("Downloading package %s" % msg.package.id)
        elif isinstance(msg, UpToDateMessage):
            show("Package %s is up-to-date!" % msg.package.id)
        # elif isinstance(msg, StaleMessage):
        #    show('Package %s is out-of-date or corrupt' % msg.package.id)
        elif isinstance(msg, FinishDownloadMessage):
            show("Finished downloading %r." % msg.package.id)
        elif isinstance(msg, StartUnzipMessage):
            show("Unzipping %s" % msg.package.filename)
        elif isinstance(msg, FinishCollectionMessage):
            self._log_indent -= 1
            show("Finished downloading collection %r." % msg.collection.id)
            self._clear_mark(msg.collection.id)
        elif isinstance(msg, FinishPackageMessage):
            self._clear_mark(msg.package.id)
        afterid = self.top.after(self._DL_DELAY, self._download_cb, download_iter, ids)
        self._afterid["_download_cb"] = afterid

    def _select(self, id):
        for row in range(len(self._table)):
            if self._table[row, "Identifier"] == id:
                self._table.select(row)
                return

    def _color_table(self):
        # Color rows according to status.
        for row in range(len(self._table)):
            bg, sbg = self._ROW_COLOR[self._table[row, "Status"]]
            fg, sfg = ("black", "white")
            self._table.rowconfig(
                row,
                foreground=fg,
                selectforeground=sfg,
                background=bg,
                selectbackground=sbg,
            )
            # Color the marked column
            self._table.itemconfigure(
                row, 0, foreground=self._MARK_COLOR[0], background=self._MARK_COLOR[1]
            )

    def _clear_mark(self, id):
        for row in range(len(self._table)):
            if self._table[row, "Identifier"] == id:
                self._table[row, 0] = ""

    def _mark_all(self, *e):
        for row in range(len(self._table)):
            self._table[row, 0] = "X"

    def _table_mark(self, *e):
        selection = self._table.selected_row()
        if selection >= 0:
            if self._table[selection][0] != "":
                self._table[selection, 0] = ""
            else:
                self._table[selection, 0] = "X"
        self._table.select(delta=1)

    def _show_log(self):
        text = "\n".join(self._log_messages)
        ShowText(self.top, "NLTK Downloader Log", text)

    def _package_to_columns(self, pkg):
        """
        Given a package, return a list of values describing that
        package, one for each column in ``self.COLUMNS``.
        """
        row = []
        for column_index, column_name in enumerate(self.COLUMNS):
            if column_index == 0:  # Mark:
                row.append("")
            elif column_name == "Identifier":
                row.append(pkg.id)
            elif column_name == "Status":
                row.append(self._ds.status(pkg))
            else:
                attr = column_name.lower().replace(" ", "_")
                row.append(getattr(pkg, attr, "n/a"))
        return row

    # /////////////////////////////////////////////////////////////////
    # External Interface
    # /////////////////////////////////////////////////////////////////

    def destroy(self, *e):
        if self._destroyed:
            return
        self.top.destroy()
        self._destroyed = True

    def _destroy(self, *e):
        if self.top is not None:
            for afterid in self._afterid.values():
                self.top.after_cancel(afterid)

        # Abort any download in progress.
        if self._downloading and self._use_threads:
            self._abort_download()

        # Make sure the garbage collector destroys these now;
        # otherwise, they may get destroyed when we're not in the main
        # thread, which would make Tkinter unhappy.
        self._column_vars.clear()

    def mainloop(self, *args, **kwargs):
        self.top.mainloop(*args, **kwargs)

    # /////////////////////////////////////////////////////////////////
    # HELP
    # /////////////////////////////////////////////////////////////////

    HELP = textwrap.dedent(
        """\
    This tool can be used to download a variety of corpora and models
    that can be used with NLTK.  Each corpus or model is distributed
    in a single zip file, known as a \"package file.\"  You can
    download packages individually, or you can download pre-defined
    collections of packages.

    When you download a package, it will be saved to the \"download
    directory.\"  A default download directory is chosen when you run

    the downloader; but you may also select a different download
    directory.  On Windows, the default download directory is


    \"package.\"

    The NLTK downloader can be used to download a variety of corpora,
    models, and other data packages.

    Keyboard shortcuts::
      [return]\t Download
      [up]\t Select previous package
      [down]\t Select next package
      [left]\t Select previous tab
      [right]\t Select next tab
    """
    )

    def help(self, *e):
        # The default font's not very legible; try using 'fixed' instead.
        try:
            ShowText(
                self.top,
                "Help: NLTK Downloader",
                self.HELP.strip(),
                width=75,
                font="fixed",
            )
        except:
            ShowText(self.top, "Help: NLTK Downloader", self.HELP.strip(), width=75)

    def about(self, *e):
        ABOUT = "NLTK Downloader\n" + "Written by Edward Loper"
        TITLE = "About: NLTK Downloader"
        try:
            from tkinter.messagebox import Message

            Message(message=ABOUT, title=TITLE).show()
        except ImportError:
            ShowText(self.top, TITLE, ABOUT)

    # /////////////////////////////////////////////////////////////////
    # Progress Bar
    # /////////////////////////////////////////////////////////////////

    _gradient_width = 5

    def _init_progressbar(self):
        c = self._progressbar
        width, height = int(c["width"]), int(c["height"])
        for i in range(0, (int(c["width"]) * 2) // self._gradient_width):
            c.create_line(
                i * self._gradient_width + 20,
                -20,
                i * self._gradient_width - height - 20,
                height + 20,
                width=self._gradient_width,
                fill="#%02x0000" % (80 + abs(i % 6 - 3) * 12),
            )
        c.addtag_all("gradient")
        c.itemconfig("gradient", state="hidden")

        # This is used to display progress
        c.addtag_withtag(
            "redbox", c.create_rectangle(0, 0, 0, 0, fill=self._PROGRESS_COLOR[0])
        )

    def _show_progress(self, percent):
        c = self._progressbar
        if percent is None:
            c.coords("redbox", 0, 0, 0, 0)
            c.itemconfig("gradient", state="hidden")
        else:
            width, height = int(c["width"]), int(c["height"])
            x = percent * int(width) // 100 + 1
            c.coords("redbox", 0, 0, x, height + 1)

    def _progress_alive(self):
        c = self._progressbar
        if not self._downloading:
            c.itemconfig("gradient", state="hidden")
        else:
            c.itemconfig("gradient", state="normal")
            x1, y1, x2, y2 = c.bbox("gradient")
            if x1 <= -100:
                c.move("gradient", (self._gradient_width * 6) - 4, 0)
            else:
                c.move("gradient", -4, 0)
            afterid = self.top.after(200, self._progress_alive)
            self._afterid["_progress_alive"] = afterid

    # /////////////////////////////////////////////////////////////////
    # Threaded downloader
    # /////////////////////////////////////////////////////////////////

    def _download_threaded(self, *e):
        # If the user tries to start a new download while we're already
        # downloading something, then abort the current download instead.
        if self._downloading:
            self._abort_download()
            return

        # Change the 'download' button to an 'abort' button.
        self._download_button["text"] = "Cancel"

        marked = [
            self._table[row, "Identifier"]
            for row in range(len(self._table))
            if self._table[row, 0] != ""
        ]
        selection = self._table.selected_row()
        if not marked and selection is not None:
            marked = [self._table[selection, "Identifier"]]

        # Create a new data server object for the download operation,
        # just in case the user modifies our data server during the
        # download (e.g., clicking 'refresh' or editing the index url).
        ds = Downloader(self._ds.url, self._ds.download_dir)

        # Start downloading in a separate thread.
        assert self._download_msg_queue == []
        assert self._download_abort_queue == []
        self._DownloadThread(
            ds,
            marked,
            self._download_lock,
            self._download_msg_queue,
            self._download_abort_queue,
        ).start()

        # Monitor the download message queue & display its progress.
        self._log_indent = 0
        self._downloading = True
        self._monitor_message_queue()

        # Display an indication that we're still alive and well by
        # cycling the progress bar.
        self._progress_alive()

    def _abort_download(self):
        if self._downloading:
            self._download_lock.acquire()
            self._download_abort_queue.append("abort")
            self._download_lock.release()

    class _DownloadThread(threading.Thread):
        def __init__(self, data_server, items, lock, message_queue, abort):
            self.data_server = data_server
            self.items = items
            self.lock = lock
            self.message_queue = message_queue
            self.abort = abort
            threading.Thread.__init__(self)

        def run(self):
            for msg in self.data_server.incr_download(self.items):
                self.lock.acquire()
                self.message_queue.append(msg)
                # Check if we've been told to kill ourselves:
                if self.abort:
                    self.message_queue.append("aborted")
                    self.lock.release()
                    return
                self.lock.release()
            self.lock.acquire()
            self.message_queue.append("finished")
            self.lock.release()

    _MONITOR_QUEUE_DELAY = 100

    def _monitor_message_queue(self):
        def show(s):
            self._progresslabel["text"] = s
            self._log(s)

        # Try to acquire the lock; if it's busy, then just try again later.
        if not self._download_lock.acquire():
            return
        for msg in self._download_msg_queue:

            # Done downloading?
            if msg == "finished" or msg == "aborted":
                # self._fill_table(sort=False)
                self._update_table_status()
                self._downloading = False
                self._download_button["text"] = "Download"
                del self._download_msg_queue[:]
                del self._download_abort_queue[:]
                self._download_lock.release()
                if msg == "aborted":
                    show("Download aborted!")
                    self._show_progress(None)
                else:
                    afterid = self.top.after(100, self._show_progress, None)
                    self._afterid["_monitor_message_queue"] = afterid
                return

            # All other messages
            elif isinstance(msg, ProgressMessage):
                self._show_progress(msg.progress)
            elif isinstance(msg, ErrorMessage):
                show(msg.message)
                if msg.package is not None:
                    self._select(msg.package.id)
                self._show_progress(None)
                self._downloading = False
                return  # halt progress.
            elif isinstance(msg, StartCollectionMessage):
                show("Downloading collection %r" % msg.collection.id)
                self._log_indent += 1
            elif isinstance(msg, StartPackageMessage):
                self._ds.clear_status_cache(msg.package.id)
                show("Downloading package %r" % msg.package.id)
            elif isinstance(msg, UpToDateMessage):
                show("Package %s is up-to-date!" % msg.package.id)
            # elif isinstance(msg, StaleMessage):
            #    show('Package %s is out-of-date or corrupt; updating it' %
            #         msg.package.id)
            elif isinstance(msg, FinishDownloadMessage):
                show("Finished downloading %r." % msg.package.id)
            elif isinstance(msg, StartUnzipMessage):
                show("Unzipping %s" % msg.package.filename)
            elif isinstance(msg, FinishUnzipMessage):
                show("Finished installing %s" % msg.package.id)
            elif isinstance(msg, FinishCollectionMessage):
                self._log_indent -= 1
                show("Finished downloading collection %r." % msg.collection.id)
                self._clear_mark(msg.collection.id)
            elif isinstance(msg, FinishPackageMessage):
                self._update_table_status()
                self._clear_mark(msg.package.id)

        # Let the user know when we're aborting a download (but
        # waiting for a good point to abort it, so we don't end up
        # with a partially unzipped package or anything like that).
        if self._download_abort_queue:
            self._progresslabel["text"] = "Aborting download..."

        # Clear the message queue and then release the lock
        del self._download_msg_queue[:]
        self._download_lock.release()

        # Check the queue again after MONITOR_QUEUE_DELAY msec.
        afterid = self.top.after(self._MONITOR_QUEUE_DELAY, self._monitor_message_queue)
        self._afterid["_monitor_message_queue"] = afterid


######################################################################
# Helper Functions
######################################################################
# [xx] It may make sense to move these to nltk.internals.


def md5_hexdigest(file):
    """
    Calculate and return the MD5 checksum for a given file.
    ``file`` may either be a filename or an open stream.
    """
    if isinstance(file, str):
        with open(file, "rb") as infile:
            return _md5_hexdigest(infile)
    return _md5_hexdigest(file)


def _md5_hexdigest(fp):
    md5_digest = md5()
    while True:
        block = fp.read(1024 * 16)  # 16k blocks
        if not block:
            break
        md5_digest.update(block)
    return md5_digest.hexdigest()


# change this to periodically yield progress messages?
# [xx] get rid of topdir parameter -- we should be checking
# this when we build the index, anyway.
def unzip(filename, root, verbose=True):
    """
    Extract the contents of the zip file ``filename`` into the
    directory ``root``.
    """
    for message in _unzip_iter(filename, root, verbose):
        if isinstance(message, ErrorMessage):
            raise Exception(message)


def _unzip_iter(filename, root, verbose=True):
    if verbose:
        sys.stdout.write("Unzipping %s" % os.path.split(filename)[1])
        sys.stdout.flush()

    try:
        zf = zipfile.ZipFile(filename)
    except zipfile.error as e:
        yield ErrorMessage(filename, "Error with downloaded zip file")
        return
    except Exception as e:
        yield ErrorMessage(filename, e)
        return

    zf.extractall(root)

    if verbose:
        print()


######################################################################
# Index Builder
######################################################################
# This may move to a different file sometime.


def build_index(root, base_url):
    """
    Create a new data.xml index file, by combining the xml description
    files for various packages and collections.  ``root`` should be the
    path to a directory containing the package xml and zip files; and
    the collection xml files.  The ``root`` directory is expected to
    have the following subdirectories::

      root/
        packages/ .................. subdirectory for packages
          corpora/ ................. zip & xml files for corpora
          grammars/ ................ zip & xml files for grammars
          taggers/ ................. zip & xml files for taggers
          tokenizers/ .............. zip & xml files for tokenizers
          etc.
        collections/ ............... xml files for collections

    For each package, there should be two files: ``package.zip``
    (where *package* is the package name)
    which contains the package itself as a compressed zip file; and
    ``package.xml``, which is an xml description of the package.  The
    zipfile ``package.zip`` should expand to a single subdirectory
    named ``package/``.  The base filename ``package`` must match
    the identifier given in the package's xml file.

    For each collection, there should be a single file ``collection.zip``
    describing the collection, where *collection* is the name of the collection.

    All identifiers (for both packages and collections) must be unique.
    """
    # Find all packages.
    packages = []
    for pkg_xml, zf, subdir in _find_packages(os.path.join(root, "packages")):
        zipstat = os.stat(zf.filename)
        url = f"{base_url}/{subdir}/{os.path.split(zf.filename)[1]}"
        unzipped_size = sum(zf_info.file_size for zf_info in zf.infolist())

        # Fill in several fields of the package xml with calculated values.
        pkg_xml.set("unzipped_size", "%s" % unzipped_size)
        pkg_xml.set("size", "%s" % zipstat.st_size)
        pkg_xml.set("checksum", "%s" % md5_hexdigest(zf.filename))
        pkg_xml.set("subdir", subdir)
        # pkg_xml.set('svn_revision', _svn_revision(zf.filename))
        if not pkg_xml.get("url"):
            pkg_xml.set("url", url)

        # Record the package.
        packages.append(pkg_xml)

    # Find all collections
    collections = list(_find_collections(os.path.join(root, "collections")))

    # Check that all UIDs are unique
    uids = set()
    for item in packages + collections:
        if item.get("id") in uids:
            raise ValueError("Duplicate UID: %s" % item.get("id"))
        uids.add(item.get("id"))

    # Put it all together
    top_elt = ElementTree.Element("nltk_data")
    top_elt.append(ElementTree.Element("packages"))
    top_elt[0].extend(sorted(packages, key=lambda package: package.get("id")))
    top_elt.append(ElementTree.Element("collections"))
    top_elt[1].extend(sorted(collections, key=lambda collection: collection.get("id")))

    _indent_xml(top_elt)
    return top_elt


def _indent_xml(xml, prefix=""):
    """
    Helper for ``build_index()``: Given an XML ``ElementTree``, modify it
    (and its descendents) ``text`` and ``tail`` attributes to generate
    an indented tree, where each nested element is indented by 2
    spaces with respect to its parent.
    """
    if len(xml) > 0:
        xml.text = (xml.text or "").strip() + "\n" + prefix + "  "
        for child in xml:
            _indent_xml(child, prefix + "  ")
        for child in xml[:-1]:
            child.tail = (child.tail or "").strip() + "\n" + prefix + "  "
        xml[-1].tail = (xml[-1].tail or "").strip() + "\n" + prefix


def _check_package(pkg_xml, zipfilename, zf):
    """
    Helper for ``build_index()``: Perform some checks to make sure that
    the given package is consistent.
    """
    # The filename must patch the id given in the XML file.
    uid = os.path.splitext(os.path.split(zipfilename)[1])[0]
    if pkg_xml.get("id") != uid:
        raise ValueError(
            "package identifier mismatch ({} vs {})".format(pkg_xml.get("id"), uid)
        )

    # Zip file must expand to a subdir whose name matches uid.
    if sum((name != uid and not name.startswith(uid + "/")) for name in zf.namelist()):
        raise ValueError(
            "Zipfile %s.zip does not expand to a single "
            "subdirectory %s/" % (uid, uid)
        )


# update for git?
def _svn_revision(filename):
    """
    Helper for ``build_index()``: Calculate the subversion revision
    number for a given file (by using ``subprocess`` to run ``svn``).
    """
    p = subprocess.Popen(
        ["svn", "status", "-v", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (stdout, stderr) = p.communicate()
    if p.returncode != 0 or stderr or not stdout:
        raise ValueError(
            "Error determining svn_revision for %s: %s"
            % (os.path.split(filename)[1], textwrap.fill(stderr))
        )
    return stdout.split()[2]


def _find_collections(root):
    """
    Helper for ``build_index()``: Yield a list of ElementTree.Element
    objects, each holding the xml for a single package collection.
    """
    for dirname, _subdirs, files in os.walk(root):
        for filename in files:
            if filename.endswith(".xml"):
                xmlfile = os.path.join(dirname, filename)
                yield ElementTree.parse(xmlfile).getroot()


def _find_packages(root):
    """
    Helper for ``build_index()``: Yield a list of tuples
    ``(pkg_xml, zf, subdir)``, where:
      - ``pkg_xml`` is an ``ElementTree.Element`` holding the xml for a
        package
      - ``zf`` is a ``zipfile.ZipFile`` for the package's contents.
      - ``subdir`` is the subdirectory (relative to ``root``) where
        the package was found (e.g. 'corpora' or 'grammars').
    """
    from nltk.corpus.reader.util import _path_from

    # Find all packages.
    packages = []
    for dirname, subdirs, files in os.walk(root):
        relpath = "/".join(_path_from(root, dirname))
        for filename in files:
            if filename.endswith(".xml"):
                xmlfilename = os.path.join(dirname, filename)
                zipfilename = xmlfilename[:-4] + ".zip"
                try:
                    zf = zipfile.ZipFile(zipfilename)
                except Exception as e:
                    raise ValueError(f"Error reading file {zipfilename!r}!\n{e}") from e
                try:
                    pkg_xml = ElementTree.parse(xmlfilename).getroot()
                except Exception as e:
                    raise ValueError(f"Error reading file {xmlfilename!r}!\n{e}") from e

                # Check that the UID matches the filename
                uid = os.path.split(xmlfilename[:-4])[1]
                if pkg_xml.get("id") != uid:
                    raise ValueError(
                        "package identifier mismatch (%s "
                        "vs %s)" % (pkg_xml.get("id"), uid)
                    )

                # Check that the zipfile expands to a subdir whose
                # name matches the uid.
                if sum(
                    (name != uid and not name.startswith(uid + "/"))
                    for name in zf.namelist()
                ):
                    raise ValueError(
                        "Zipfile %s.zip does not expand to a "
                        "single subdirectory %s/" % (uid, uid)
                    )

                yield pkg_xml, zf, relpath

            elif filename.endswith(".zip"):
                # Warn user in case a .xml does not exist for a .zip
                resourcename = os.path.splitext(filename)[0]
                xmlfilename = os.path.join(dirname, resourcename + ".xml")
                if not os.path.exists(xmlfilename):
                    warnings.warn(
                        f"{filename} exists, but {resourcename + '.xml'} cannot be found! "
                        f"This could mean that {resourcename} can not be downloaded.",
                        stacklevel=2,
                    )

        # Don't recurse into svn subdirectories:
        try:
            subdirs.remove(".svn")
        except ValueError:
            pass


######################################################################
# Main:
######################################################################

# There should be a command-line interface

# Aliases
_downloader = Downloader()
download = _downloader.download


def download_shell():
    DownloaderShell(_downloader).run()


def download_gui():
    DownloaderGUI(_downloader).mainloop()


def update():
    _downloader.update()


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option(
        "-d",
        "--dir",
        dest="dir",
        help="download package to directory DIR",
        metavar="DIR",
    )
    parser.add_option(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="work quietly",
    )
    parser.add_option(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        default=False,
        help="download even if already installed",
    )
    parser.add_option(
        "-e",
        "--exit-on-error",
        dest="halt_on_error",
        action="store_true",
        default=False,
        help="exit if an error occurs",
    )
    parser.add_option(
        "-u",
        "--url",
        dest="server_index_url",
        default=os.environ.get("NLTK_DOWNLOAD_URL"),
        help="download server index url",
    )

    (options, args) = parser.parse_args()

    downloader = Downloader(server_index_url=options.server_index_url)

    if args:
        for pkg_id in args:
            rv = downloader.download(
                info_or_id=pkg_id,
                download_dir=options.dir,
                quiet=options.quiet,
                force=options.force,
                halt_on_error=options.halt_on_error,
            )
            if rv == False and options.halt_on_error:
                break
    else:
        downloader.download(
            download_dir=options.dir,
            quiet=options.quiet,
            force=options.force,
            halt_on_error=options.halt_on_error,
        )
