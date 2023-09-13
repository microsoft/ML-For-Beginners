from numpy.testing import assert_equal, assert_almost_equal, suppress_warnings

from scipy.misc import face, ascent, electrocardiogram


def test_face():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        assert_equal(face().shape, (768, 1024, 3))


def test_ascent():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        assert_equal(ascent().shape, (512, 512))


def test_electrocardiogram():
    with suppress_warnings() as sup:
        sup.filter(category=DeprecationWarning)
        # Test shape, dtype and stats of signal
        ecg = electrocardiogram()
        assert ecg.dtype == float
        assert_equal(ecg.shape, (108000,))
        assert_almost_equal(ecg.mean(), -0.16510875)
        assert_almost_equal(ecg.std(), 0.5992473991177294)
