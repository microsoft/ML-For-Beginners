from scipy.datasets._registry import registry
from scipy.datasets._fetchers import data_fetcher
from scipy.datasets._utils import _clear_cache
from scipy.datasets import ascent, face, electrocardiogram, download_all
from numpy.testing import assert_equal, assert_almost_equal
import os
import pytest

try:
    import pooch
except ImportError:
    raise ImportError("Missing optional dependency 'pooch' required "
                      "for scipy.datasets module. Please use pip or "
                      "conda to install 'pooch'.")


data_dir = data_fetcher.path  # type: ignore


def _has_hash(path, expected_hash):
    """Check if the provided path has the expected hash."""
    if not os.path.exists(path):
        return False
    return pooch.file_hash(path) == expected_hash


class TestDatasets:

    @pytest.fixture(scope='module', autouse=True)
    def test_download_all(self):
        # This fixture requires INTERNET CONNECTION

        # test_setup phase
        download_all()

        yield

    def test_existence_all(self):
        assert len(os.listdir(data_dir)) >= len(registry)

    def test_ascent(self):
        assert_equal(ascent().shape, (512, 512))

        # hash check
        assert _has_hash(os.path.join(data_dir, "ascent.dat"),
                         registry["ascent.dat"])

    def test_face(self):
        assert_equal(face().shape, (768, 1024, 3))

        # hash check
        assert _has_hash(os.path.join(data_dir, "face.dat"),
                         registry["face.dat"])

    def test_electrocardiogram(self):
        # Test shape, dtype and stats of signal
        ecg = electrocardiogram()
        assert_equal(ecg.dtype, float)
        assert_equal(ecg.shape, (108000,))
        assert_almost_equal(ecg.mean(), -0.16510875)
        assert_almost_equal(ecg.std(), 0.5992473991177294)

        # hash check
        assert _has_hash(os.path.join(data_dir, "ecg.dat"),
                         registry["ecg.dat"])


def test_clear_cache(tmp_path):
    # Note: `tmp_path` is a pytest fixture, it handles cleanup
    dummy_basepath = tmp_path / "dummy_cache_dir"
    dummy_basepath.mkdir()

    # Create three dummy dataset files for dummy dataset methods
    dummy_method_map = {}
    for i in range(4):
        dummy_method_map[f"data{i}"] = [f"data{i}.dat"]
        data_filepath = dummy_basepath / f"data{i}.dat"
        data_filepath.write_text("")

    # clear files associated to single dataset method data0
    # also test callable argument instead of list of callables
    def data0():
        pass
    _clear_cache(datasets=data0, cache_dir=dummy_basepath,
                 method_map=dummy_method_map)
    assert not os.path.exists(dummy_basepath/"data0.dat")

    # clear files associated to multiple dataset methods "data3" and "data4"
    def data1():
        pass

    def data2():
        pass
    _clear_cache(datasets=[data1, data2], cache_dir=dummy_basepath,
                 method_map=dummy_method_map)
    assert not os.path.exists(dummy_basepath/"data1.dat")
    assert not os.path.exists(dummy_basepath/"data2.dat")

    # clear multiple dataset files "data3_0.dat" and "data3_1.dat"
    # associated with dataset method "data3"
    def data4():
        pass
    # create files
    (dummy_basepath / "data4_0.dat").write_text("")
    (dummy_basepath / "data4_1.dat").write_text("")

    dummy_method_map["data4"] = ["data4_0.dat", "data4_1.dat"]
    _clear_cache(datasets=[data4], cache_dir=dummy_basepath,
                 method_map=dummy_method_map)
    assert not os.path.exists(dummy_basepath/"data4_0.dat")
    assert not os.path.exists(dummy_basepath/"data4_1.dat")

    # wrong dataset method should raise ValueError since it
    # doesn't exist in the dummy_method_map
    def data5():
        pass
    with pytest.raises(ValueError):
        _clear_cache(datasets=[data5], cache_dir=dummy_basepath,
                     method_map=dummy_method_map)

    # remove all dataset cache
    _clear_cache(datasets=None, cache_dir=dummy_basepath)
    assert not os.path.exists(dummy_basepath)
