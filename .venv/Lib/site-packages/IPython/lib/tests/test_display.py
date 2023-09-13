"""Tests for IPython.lib.display.

"""
#-----------------------------------------------------------------------------
# Copyright (c) 2012, the IPython Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from tempfile import NamedTemporaryFile, mkdtemp
from os.path import split, join as pjoin, dirname
import pathlib
from unittest import TestCase, mock
import struct
import wave
from io import BytesIO

# Third-party imports
import pytest

try:
    import numpy
except ImportError:
    pass

# Our own imports
from IPython.lib import display

from IPython.testing.decorators import skipif_not_numpy

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

#--------------------------
# FileLink tests
#--------------------------

def test_instantiation_FileLink():
    """FileLink: Test class can be instantiated"""
    fl = display.FileLink('example.txt')
    # TODO: remove if when only Python >= 3.6 is supported
    fl = display.FileLink(pathlib.PurePath('example.txt'))

def test_warning_on_non_existent_path_FileLink():
    """FileLink: Calling _repr_html_ on non-existent files returns a warning"""
    fl = display.FileLink("example.txt")
    assert fl._repr_html_().startswith("Path (<tt>example.txt</tt>)")


def test_existing_path_FileLink():
    """FileLink: Calling _repr_html_ functions as expected on existing filepath
    """
    tf = NamedTemporaryFile()
    fl = display.FileLink(tf.name)
    actual = fl._repr_html_()
    expected = "<a href='%s' target='_blank'>%s</a><br>" % (tf.name, tf.name)
    assert actual == expected


def test_existing_path_FileLink_repr():
    """FileLink: Calling repr() functions as expected on existing filepath
    """
    tf = NamedTemporaryFile()
    fl = display.FileLink(tf.name)
    actual = repr(fl)
    expected = tf.name
    assert actual == expected


def test_error_on_directory_to_FileLink():
    """FileLink: Raises error when passed directory
    """
    td = mkdtemp()
    pytest.raises(ValueError, display.FileLink, td)

#--------------------------
# FileLinks tests
#--------------------------

def test_instantiation_FileLinks():
    """FileLinks: Test class can be instantiated
    """
    fls = display.FileLinks('example')

def test_warning_on_non_existent_path_FileLinks():
    """FileLinks: Calling _repr_html_ on non-existent files returns a warning"""
    fls = display.FileLinks("example")
    assert fls._repr_html_().startswith("Path (<tt>example</tt>)")


def test_existing_path_FileLinks():
    """FileLinks: Calling _repr_html_ functions as expected on existing dir
    """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    tf2 = NamedTemporaryFile(dir=td)
    fl = display.FileLinks(td)
    actual = fl._repr_html_()
    actual = actual.split('\n')
    actual.sort()
    # the links should always have forward slashes, even on windows, so replace
    # backslashes with forward slashes here
    expected = ["%s/<br>" % td,
                "&nbsp;&nbsp;<a href='%s' target='_blank'>%s</a><br>" %\
                 (tf2.name.replace("\\","/"),split(tf2.name)[1]),
                "&nbsp;&nbsp;<a href='%s' target='_blank'>%s</a><br>" %\
                 (tf1.name.replace("\\","/"),split(tf1.name)[1])]
    expected.sort()
    # We compare the sorted list of links here as that's more reliable
    assert actual == expected


def test_existing_path_FileLinks_alt_formatter():
    """FileLinks: Calling _repr_html_ functions as expected w/ an alt formatter
    """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    tf2 = NamedTemporaryFile(dir=td)
    def fake_formatter(dirname,fnames,included_suffixes):
        return ["hello","world"]
    fl = display.FileLinks(td,notebook_display_formatter=fake_formatter)
    actual = fl._repr_html_()
    actual = actual.split('\n')
    actual.sort()
    expected = ["hello","world"]
    expected.sort()
    # We compare the sorted list of links here as that's more reliable
    assert actual == expected


def test_existing_path_FileLinks_repr():
    """FileLinks: Calling repr() functions as expected on existing directory """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    tf2 = NamedTemporaryFile(dir=td)
    fl = display.FileLinks(td)
    actual = repr(fl)
    actual = actual.split('\n')
    actual.sort()
    expected = ['%s/' % td, '  %s' % split(tf1.name)[1],'  %s' % split(tf2.name)[1]]
    expected.sort()
    # We compare the sorted list of links here as that's more reliable
    assert actual == expected


def test_existing_path_FileLinks_repr_alt_formatter():
    """FileLinks: Calling repr() functions as expected w/ alt formatter
    """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    tf2 = NamedTemporaryFile(dir=td)
    def fake_formatter(dirname,fnames,included_suffixes):
        return ["hello","world"]
    fl = display.FileLinks(td,terminal_display_formatter=fake_formatter)
    actual = repr(fl)
    actual = actual.split('\n')
    actual.sort()
    expected = ["hello","world"]
    expected.sort()
    # We compare the sorted list of links here as that's more reliable
    assert actual == expected


def test_error_on_file_to_FileLinks():
    """FileLinks: Raises error when passed file
    """
    td = mkdtemp()
    tf1 = NamedTemporaryFile(dir=td)
    pytest.raises(ValueError, display.FileLinks, tf1.name)


def test_recursive_FileLinks():
    """FileLinks: Does not recurse when recursive=False
    """
    td = mkdtemp()
    tf = NamedTemporaryFile(dir=td)
    subtd = mkdtemp(dir=td)
    subtf = NamedTemporaryFile(dir=subtd)
    fl = display.FileLinks(td)
    actual = str(fl)
    actual = actual.split('\n')
    assert len(actual) == 4, actual
    fl = display.FileLinks(td, recursive=False)
    actual = str(fl)
    actual = actual.split('\n')
    assert len(actual) == 2, actual

def test_audio_from_file():
    path = pjoin(dirname(__file__), 'test.wav')
    display.Audio(filename=path)

class TestAudioDataWithNumpy(TestCase):

    @skipif_not_numpy
    def test_audio_from_numpy_array(self):
        test_tone = get_test_tone()
        audio = display.Audio(test_tone, rate=44100)
        assert len(read_wav(audio.data)) == len(test_tone)

    @skipif_not_numpy
    def test_audio_from_list(self):
        test_tone = get_test_tone()
        audio = display.Audio(list(test_tone), rate=44100)
        assert len(read_wav(audio.data)) == len(test_tone)

    @skipif_not_numpy
    def test_audio_from_numpy_array_without_rate_raises(self):
        self.assertRaises(ValueError, display.Audio, get_test_tone())

    @skipif_not_numpy
    def test_audio_data_normalization(self):
        expected_max_value = numpy.iinfo(numpy.int16).max
        for scale in [1, 0.5, 2]:
            audio = display.Audio(get_test_tone(scale), rate=44100)
            actual_max_value = numpy.max(numpy.abs(read_wav(audio.data)))
            assert actual_max_value == expected_max_value

    @skipif_not_numpy
    def test_audio_data_without_normalization(self):
        max_int16 = numpy.iinfo(numpy.int16).max
        for scale in [1, 0.5, 0.2]:
            test_tone = get_test_tone(scale)
            test_tone_max_abs = numpy.max(numpy.abs(test_tone))
            expected_max_value = int(max_int16 * test_tone_max_abs)
            audio = display.Audio(test_tone, rate=44100, normalize=False)
            actual_max_value = numpy.max(numpy.abs(read_wav(audio.data)))
            assert actual_max_value == expected_max_value

    def test_audio_data_without_normalization_raises_for_invalid_data(self):
        self.assertRaises(
            ValueError,
            lambda: display.Audio([1.001], rate=44100, normalize=False))
        self.assertRaises(
            ValueError,
            lambda: display.Audio([-1.001], rate=44100, normalize=False))

def simulate_numpy_not_installed():
    try:
        import numpy
        return mock.patch('numpy.array', mock.MagicMock(side_effect=ImportError))
    except ModuleNotFoundError:
        return lambda x:x

@simulate_numpy_not_installed()
class TestAudioDataWithoutNumpy(TestAudioDataWithNumpy):
    # All tests from `TestAudioDataWithNumpy` are inherited.

    @skipif_not_numpy
    def test_audio_raises_for_nested_list(self):
        stereo_signal = [list(get_test_tone())] * 2
        self.assertRaises(TypeError, lambda: display.Audio(stereo_signal, rate=44100))


@skipif_not_numpy
def get_test_tone(scale=1):
    return numpy.sin(2 * numpy.pi * 440 * numpy.linspace(0, 1, 44100)) * scale

def read_wav(data):
    with wave.open(BytesIO(data)) as wave_file:
        wave_data = wave_file.readframes(wave_file.getnframes())
        num_samples = wave_file.getnframes() * wave_file.getnchannels()
        return struct.unpack('<%sh' % num_samples, wave_data)

def test_code_from_file():
    c = display.Code(filename=__file__)
    assert c._repr_html_().startswith('<style>')
