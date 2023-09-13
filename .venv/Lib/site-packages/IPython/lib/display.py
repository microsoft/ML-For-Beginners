"""Various display related classes.

Authors : MinRK, gregcaporaso, dannystaple
"""
from html import escape as html_escape
from os.path import exists, isfile, splitext, abspath, join, isdir
from os import walk, sep, fsdecode

from IPython.core.display import DisplayObject, TextDisplayObject

from typing import Tuple, Iterable, Optional

__all__ = ['Audio', 'IFrame', 'YouTubeVideo', 'VimeoVideo', 'ScribdDocument',
           'FileLink', 'FileLinks', 'Code']


class Audio(DisplayObject):
    """Create an audio object.

    When this object is returned by an input cell or passed to the
    display function, it will result in Audio controls being displayed
    in the frontend (only works in the notebook).

    Parameters
    ----------
    data : numpy array, list, unicode, str or bytes
        Can be one of

          * Numpy 1d array containing the desired waveform (mono)
          * Numpy 2d array containing waveforms for each channel.
            Shape=(NCHAN, NSAMPLES). For the standard channel order, see
            http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
          * List of float or integer representing the waveform (mono)
          * String containing the filename
          * Bytestring containing raw PCM data or
          * URL pointing to a file on the web.

        If the array option is used, the waveform will be normalized.

        If a filename or url is used, the format support will be browser
        dependent.
    url : unicode
        A URL to download the data from.
    filename : unicode
        Path to a local file to load the data from.
    embed : boolean
        Should the audio data be embedded using a data URI (True) or should
        the original source be referenced. Set this to True if you want the
        audio to playable later with no internet connection in the notebook.

        Default is `True`, unless the keyword argument `url` is set, then
        default value is `False`.
    rate : integer
        The sampling rate of the raw data.
        Only required when data parameter is being used as an array
    autoplay : bool
        Set to True if the audio should immediately start playing.
        Default is `False`.
    normalize : bool
        Whether audio should be normalized (rescaled) to the maximum possible
        range. Default is `True`. When set to `False`, `data` must be between
        -1 and 1 (inclusive), otherwise an error is raised.
        Applies only when `data` is a list or array of samples; other types of
        audio are never normalized.

    Examples
    --------

    >>> import pytest
    >>> np = pytest.importorskip("numpy")

    Generate a sound

    >>> import numpy as np
    >>> framerate = 44100
    >>> t = np.linspace(0,5,framerate*5)
    >>> data = np.sin(2*np.pi*220*t) + np.sin(2*np.pi*224*t)
    >>> Audio(data, rate=framerate)
    <IPython.lib.display.Audio object>

    Can also do stereo or more channels

    >>> dataleft = np.sin(2*np.pi*220*t)
    >>> dataright = np.sin(2*np.pi*224*t)
    >>> Audio([dataleft, dataright], rate=framerate)
    <IPython.lib.display.Audio object>

    From URL:

    >>> Audio("http://www.nch.com.au/acm/8k16bitpcm.wav")  # doctest: +SKIP
    >>> Audio(url="http://www.w3schools.com/html/horse.ogg")  # doctest: +SKIP

    From a File:

    >>> Audio('IPython/lib/tests/test.wav')  # doctest: +SKIP
    >>> Audio(filename='IPython/lib/tests/test.wav')  # doctest: +SKIP

    From Bytes:

    >>> Audio(b'RAW_WAV_DATA..')  # doctest: +SKIP
    >>> Audio(data=b'RAW_WAV_DATA..')  # doctest: +SKIP

    See Also
    --------
    ipywidgets.Audio

         Audio widget with more more flexibility and options.

    """
    _read_flags = 'rb'

    def __init__(self, data=None, filename=None, url=None, embed=None, rate=None, autoplay=False, normalize=True, *,
                 element_id=None):
        if filename is None and url is None and data is None:
            raise ValueError("No audio data found. Expecting filename, url, or data.")
        if embed is False and url is None:
            raise ValueError("No url found. Expecting url when embed=False")

        if url is not None and embed is not True:
            self.embed = False
        else:
            self.embed = True
        self.autoplay = autoplay
        self.element_id = element_id
        super(Audio, self).__init__(data=data, url=url, filename=filename)

        if self.data is not None and not isinstance(self.data, bytes):
            if rate is None:
                raise ValueError("rate must be specified when data is a numpy array or list of audio samples.")
            self.data = Audio._make_wav(data, rate, normalize)

    def reload(self):
        """Reload the raw data from file or URL."""
        import mimetypes
        if self.embed:
            super(Audio, self).reload()

        if self.filename is not None:
            self.mimetype = mimetypes.guess_type(self.filename)[0]
        elif self.url is not None:
            self.mimetype = mimetypes.guess_type(self.url)[0]
        else:
            self.mimetype = "audio/wav"

    @staticmethod
    def _make_wav(data, rate, normalize):
        """ Transform a numpy array to a PCM bytestring """
        from io import BytesIO
        import wave

        try:
            scaled, nchan = Audio._validate_and_normalize_with_numpy(data, normalize)
        except ImportError:
            scaled, nchan = Audio._validate_and_normalize_without_numpy(data, normalize)

        fp = BytesIO()
        waveobj = wave.open(fp,mode='wb')
        waveobj.setnchannels(nchan)
        waveobj.setframerate(rate)
        waveobj.setsampwidth(2)
        waveobj.setcomptype('NONE','NONE')
        waveobj.writeframes(scaled)
        val = fp.getvalue()
        waveobj.close()

        return val

    @staticmethod
    def _validate_and_normalize_with_numpy(data, normalize) -> Tuple[bytes, int]:
        import numpy as np

        data = np.array(data, dtype=float)
        if len(data.shape) == 1:
            nchan = 1
        elif len(data.shape) == 2:
            # In wave files,channels are interleaved. E.g.,
            # "L1R1L2R2..." for stereo. See
            # http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
            # for channel ordering
            nchan = data.shape[0]
            data = data.T.ravel()
        else:
            raise ValueError('Array audio input must be a 1D or 2D array')
        
        max_abs_value = np.max(np.abs(data))
        normalization_factor = Audio._get_normalization_factor(max_abs_value, normalize)
        scaled = data / normalization_factor * 32767
        return scaled.astype("<h").tobytes(), nchan

    @staticmethod
    def _validate_and_normalize_without_numpy(data, normalize):
        import array
        import sys

        data = array.array('f', data)

        try:
            max_abs_value = float(max([abs(x) for x in data]))
        except TypeError as e:
            raise TypeError('Only lists of mono audio are '
                'supported if numpy is not installed') from e

        normalization_factor = Audio._get_normalization_factor(max_abs_value, normalize)
        scaled = array.array('h', [int(x / normalization_factor * 32767) for x in data])
        if sys.byteorder == 'big':
            scaled.byteswap()
        nchan = 1
        return scaled.tobytes(), nchan

    @staticmethod
    def _get_normalization_factor(max_abs_value, normalize):
        if not normalize and max_abs_value > 1:
            raise ValueError('Audio data must be between -1 and 1 when normalize=False.')
        return max_abs_value if normalize else 1

    def _data_and_metadata(self):
        """shortcut for returning metadata with url information, if defined"""
        md = {}
        if self.url:
            md['url'] = self.url
        if md:
            return self.data, md
        else:
            return self.data

    def _repr_html_(self):
        src = """
                <audio {element_id} controls="controls" {autoplay}>
                    <source src="{src}" type="{type}" />
                    Your browser does not support the audio element.
                </audio>
              """
        return src.format(src=self.src_attr(), type=self.mimetype, autoplay=self.autoplay_attr(),
                          element_id=self.element_id_attr())

    def src_attr(self):
        import base64
        if self.embed and (self.data is not None):
            data = base64=base64.b64encode(self.data).decode('ascii')
            return """data:{type};base64,{base64}""".format(type=self.mimetype,
                                                            base64=data)
        elif self.url is not None:
            return self.url
        else:
            return ""

    def autoplay_attr(self):
        if(self.autoplay):
            return 'autoplay="autoplay"'
        else:
            return ''
    
    def element_id_attr(self):
        if (self.element_id):
            return 'id="{element_id}"'.format(element_id=self.element_id)
        else:
            return ''

class IFrame(object):
    """
    Generic class to embed an iframe in an IPython notebook
    """

    iframe = """
        <iframe
            width="{width}"
            height="{height}"
            src="{src}{params}"
            frameborder="0"
            allowfullscreen
            {extras}
        ></iframe>
        """

    def __init__(
        self, src, width, height, extras: Optional[Iterable[str]] = None, **kwargs
    ):
        if extras is None:
            extras = []

        self.src = src
        self.width = width
        self.height = height
        self.extras = extras
        self.params = kwargs

    def _repr_html_(self):
        """return the embed iframe"""
        if self.params:
            from urllib.parse import urlencode
            params = "?" + urlencode(self.params)
        else:
            params = ""
        return self.iframe.format(
            src=self.src,
            width=self.width,
            height=self.height,
            params=params,
            extras=" ".join(self.extras),
        )


class YouTubeVideo(IFrame):
    """Class for embedding a YouTube Video in an IPython session, based on its video id.

    e.g. to embed the video from https://www.youtube.com/watch?v=foo , you would
    do::

        vid = YouTubeVideo("foo")
        display(vid)

    To start from 30 seconds::

        vid = YouTubeVideo("abc", start=30)
        display(vid)

    To calculate seconds from time as hours, minutes, seconds use
    :class:`datetime.timedelta`::

        start=int(timedelta(hours=1, minutes=46, seconds=40).total_seconds())

    Other parameters can be provided as documented at
    https://developers.google.com/youtube/player_parameters#Parameters
    
    When converting the notebook using nbconvert, a jpeg representation of the video
    will be inserted in the document.
    """

    def __init__(self, id, width=400, height=300, allow_autoplay=False, **kwargs):
        self.id=id
        src = "https://www.youtube.com/embed/{0}".format(id)
        if allow_autoplay:
            extras = list(kwargs.get("extras", [])) + ['allow="autoplay"']
            kwargs.update(autoplay=1, extras=extras)
        super(YouTubeVideo, self).__init__(src, width, height, **kwargs)

    def _repr_jpeg_(self):
        # Deferred import
        from urllib.request import urlopen

        try:
            return urlopen("https://img.youtube.com/vi/{id}/hqdefault.jpg".format(id=self.id)).read()
        except IOError:
            return None

class VimeoVideo(IFrame):
    """
    Class for embedding a Vimeo video in an IPython session, based on its video id.
    """

    def __init__(self, id, width=400, height=300, **kwargs):
        src="https://player.vimeo.com/video/{0}".format(id)
        super(VimeoVideo, self).__init__(src, width, height, **kwargs)

class ScribdDocument(IFrame):
    """
    Class for embedding a Scribd document in an IPython session

    Use the start_page params to specify a starting point in the document
    Use the view_mode params to specify display type one off scroll | slideshow | book

    e.g to Display Wes' foundational paper about PANDAS in book mode from page 3

    ScribdDocument(71048089, width=800, height=400, start_page=3, view_mode="book")
    """

    def __init__(self, id, width=400, height=300, **kwargs):
        src="https://www.scribd.com/embeds/{0}/content".format(id)
        super(ScribdDocument, self).__init__(src, width, height, **kwargs)

class FileLink(object):
    """Class for embedding a local file link in an IPython session, based on path

    e.g. to embed a link that was generated in the IPython notebook as my/data.txt

    you would do::

        local_file = FileLink("my/data.txt")
        display(local_file)

    or in the HTML notebook, just::

        FileLink("my/data.txt")
    """

    html_link_str = "<a href='%s' target='_blank'>%s</a>"

    def __init__(self,
                 path,
                 url_prefix='',
                 result_html_prefix='',
                 result_html_suffix='<br>'):
        """
        Parameters
        ----------
        path : str
            path to the file or directory that should be formatted
        url_prefix : str
            prefix to be prepended to all files to form a working link [default:
            '']
        result_html_prefix : str
            text to append to beginning to link [default: '']
        result_html_suffix : str
            text to append at the end of link [default: '<br>']
        """
        if isdir(path):
            raise ValueError("Cannot display a directory using FileLink. "
              "Use FileLinks to display '%s'." % path)
        self.path = fsdecode(path)
        self.url_prefix = url_prefix
        self.result_html_prefix = result_html_prefix
        self.result_html_suffix = result_html_suffix

    def _format_path(self):
        fp = ''.join([self.url_prefix, html_escape(self.path)])
        return ''.join([self.result_html_prefix,
                        self.html_link_str % \
                            (fp, html_escape(self.path, quote=False)),
                        self.result_html_suffix])

    def _repr_html_(self):
        """return html link to file
        """
        if not exists(self.path):
            return ("Path (<tt>%s</tt>) doesn't exist. "
                    "It may still be in the process of "
                    "being generated, or you may have the "
                    "incorrect path." % self.path)

        return self._format_path()

    def __repr__(self):
        """return absolute path to file
        """
        return abspath(self.path)

class FileLinks(FileLink):
    """Class for embedding local file links in an IPython session, based on path

    e.g. to embed links to files that were generated in the IPython notebook
    under ``my/data``, you would do::

        local_files = FileLinks("my/data")
        display(local_files)

    or in the HTML notebook, just::

        FileLinks("my/data")
    """
    def __init__(self,
                 path,
                 url_prefix='',
                 included_suffixes=None,
                 result_html_prefix='',
                 result_html_suffix='<br>',
                 notebook_display_formatter=None,
                 terminal_display_formatter=None,
                 recursive=True):
        """
        See :class:`FileLink` for the ``path``, ``url_prefix``,
        ``result_html_prefix`` and ``result_html_suffix`` parameters.

        included_suffixes : list
          Filename suffixes to include when formatting output [default: include
          all files]

        notebook_display_formatter : function
          Used to format links for display in the notebook. See discussion of
          formatter functions below.

        terminal_display_formatter : function
          Used to format links for display in the terminal. See discussion of
          formatter functions below.

        Formatter functions must be of the form::

            f(dirname, fnames, included_suffixes)

        dirname : str
          The name of a directory
        fnames : list
          The files in that directory
        included_suffixes : list
          The file suffixes that should be included in the output (passing None
          meansto include all suffixes in the output in the built-in formatters)
        recursive : boolean
          Whether to recurse into subdirectories. Default is True.

        The function should return a list of lines that will be printed in the
        notebook (if passing notebook_display_formatter) or the terminal (if
        passing terminal_display_formatter). This function is iterated over for
        each directory in self.path. Default formatters are in place, can be
        passed here to support alternative formatting.

        """
        if isfile(path):
            raise ValueError("Cannot display a file using FileLinks. "
              "Use FileLink to display '%s'." % path)
        self.included_suffixes = included_suffixes
        # remove trailing slashes for more consistent output formatting
        path = path.rstrip('/')

        self.path = path
        self.url_prefix = url_prefix
        self.result_html_prefix = result_html_prefix
        self.result_html_suffix = result_html_suffix

        self.notebook_display_formatter = \
             notebook_display_formatter or self._get_notebook_display_formatter()
        self.terminal_display_formatter = \
             terminal_display_formatter or self._get_terminal_display_formatter()

        self.recursive = recursive

    def _get_display_formatter(
        self, dirname_output_format, fname_output_format, fp_format, fp_cleaner=None
    ):
        """generate built-in formatter function

        this is used to define both the notebook and terminal built-in
         formatters as they only differ by some wrapper text for each entry

        dirname_output_format: string to use for formatting directory
         names, dirname will be substituted for a single "%s" which
         must appear in this string
        fname_output_format: string to use for formatting file names,
         if a single "%s" appears in the string, fname will be substituted
         if two "%s" appear in the string, the path to fname will be
          substituted for the first and fname will be substituted for the
          second
        fp_format: string to use for formatting filepaths, must contain
         exactly two "%s" and the dirname will be substituted for the first
         and fname will be substituted for the second
        """
        def f(dirname, fnames, included_suffixes=None):
            result = []
            # begin by figuring out which filenames, if any,
            # are going to be displayed
            display_fnames = []
            for fname in fnames:
                if (isfile(join(dirname,fname)) and
                       (included_suffixes is None or
                        splitext(fname)[1] in included_suffixes)):
                      display_fnames.append(fname)

            if len(display_fnames) == 0:
                # if there are no filenames to display, don't print anything
                # (not even the directory name)
                pass
            else:
                # otherwise print the formatted directory name followed by
                # the formatted filenames
                dirname_output_line = dirname_output_format % dirname
                result.append(dirname_output_line)
                for fname in display_fnames:
                    fp = fp_format % (dirname,fname)
                    if fp_cleaner is not None:
                        fp = fp_cleaner(fp)
                    try:
                        # output can include both a filepath and a filename...
                        fname_output_line = fname_output_format % (fp, fname)
                    except TypeError:
                        # ... or just a single filepath
                        fname_output_line = fname_output_format % fname
                    result.append(fname_output_line)
            return result
        return f

    def _get_notebook_display_formatter(self,
                                        spacer="&nbsp;&nbsp;"):
        """ generate function to use for notebook formatting
        """
        dirname_output_format = \
         self.result_html_prefix + "%s/" + self.result_html_suffix
        fname_output_format = \
         self.result_html_prefix + spacer + self.html_link_str + self.result_html_suffix
        fp_format = self.url_prefix + '%s/%s'
        if sep == "\\":
            # Working on a platform where the path separator is "\", so
            # must convert these to "/" for generating a URI
            def fp_cleaner(fp):
                # Replace all occurrences of backslash ("\") with a forward
                # slash ("/") - this is necessary on windows when a path is
                # provided as input, but we must link to a URI
                return fp.replace('\\','/')
        else:
            fp_cleaner = None

        return self._get_display_formatter(dirname_output_format,
                                           fname_output_format,
                                           fp_format,
                                           fp_cleaner)

    def _get_terminal_display_formatter(self,
                                        spacer="  "):
        """ generate function to use for terminal formatting
        """
        dirname_output_format = "%s/"
        fname_output_format = spacer + "%s"
        fp_format = '%s/%s'

        return self._get_display_formatter(dirname_output_format,
                                           fname_output_format,
                                           fp_format)

    def _format_path(self):
        result_lines = []
        if self.recursive:
            walked_dir = list(walk(self.path))
        else:
            walked_dir = [next(walk(self.path))]
        walked_dir.sort()
        for dirname, subdirs, fnames in walked_dir:
            result_lines += self.notebook_display_formatter(dirname, fnames, self.included_suffixes)
        return '\n'.join(result_lines)

    def __repr__(self):
        """return newline-separated absolute paths
        """
        result_lines = []
        if self.recursive:
            walked_dir = list(walk(self.path))
        else:
            walked_dir = [next(walk(self.path))]
        walked_dir.sort()
        for dirname, subdirs, fnames in walked_dir:
            result_lines += self.terminal_display_formatter(dirname, fnames, self.included_suffixes)
        return '\n'.join(result_lines)


class Code(TextDisplayObject):
    """Display syntax-highlighted source code.

    This uses Pygments to highlight the code for HTML and Latex output.

    Parameters
    ----------
    data : str
        The code as a string
    url : str
        A URL to fetch the code from
    filename : str
        A local filename to load the code from
    language : str
        The short name of a Pygments lexer to use for highlighting.
        If not specified, it will guess the lexer based on the filename
        or the code. Available lexers: http://pygments.org/docs/lexers/
    """
    def __init__(self, data=None, url=None, filename=None, language=None):
        self.language = language
        super().__init__(data=data, url=url, filename=filename)

    def _get_lexer(self):
        if self.language:
            from pygments.lexers import get_lexer_by_name
            return get_lexer_by_name(self.language)
        elif self.filename:
            from pygments.lexers import get_lexer_for_filename
            return get_lexer_for_filename(self.filename)
        else:
            from pygments.lexers import guess_lexer
            return guess_lexer(self.data)

    def __repr__(self):
        return self.data

    def _repr_html_(self):
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        fmt = HtmlFormatter()
        style = '<style>{}</style>'.format(fmt.get_style_defs('.output_html'))
        return style + highlight(self.data, self._get_lexer(), fmt)

    def _repr_latex_(self):
        from pygments import highlight
        from pygments.formatters import LatexFormatter
        return highlight(self.data, self._get_lexer(), LatexFormatter())
