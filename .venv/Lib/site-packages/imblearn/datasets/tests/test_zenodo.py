"""Test the datasets loader.

Skipped if datasets is not already downloaded to data_home.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
from sklearn.utils._testing import SkipTest

from imblearn.datasets import fetch_datasets

DATASET_SHAPE = {
    "ecoli": (336, 7),
    "optical_digits": (5620, 64),
    "satimage": (6435, 36),
    "pen_digits": (10992, 16),
    "abalone": (4177, 10),
    "sick_euthyroid": (3163, 42),
    "spectrometer": (531, 93),
    "car_eval_34": (1728, 21),
    "isolet": (7797, 617),
    "us_crime": (1994, 100),
    "yeast_ml8": (2417, 103),
    "scene": (2407, 294),
    "libras_move": (360, 90),
    "thyroid_sick": (3772, 52),
    "coil_2000": (9822, 85),
    "arrhythmia": (452, 278),
    "solar_flare_m0": (1389, 32),
    "oil": (937, 49),
    "car_eval_4": (1728, 21),
    "wine_quality": (4898, 11),
    "letter_img": (20000, 16),
    "yeast_me2": (1484, 8),
    "webpage": (34780, 300),
    "ozone_level": (2536, 72),
    "mammography": (11183, 6),
    "protein_homo": (145751, 74),
    "abalone_19": (4177, 10),
}


def fetch(*args, **kwargs):
    return fetch_datasets(*args, download_if_missing=True, **kwargs)


@pytest.mark.xfail
def test_fetch():
    try:
        datasets1 = fetch(shuffle=True, random_state=42)
    except IOError:
        raise SkipTest("Zenodo dataset can not be loaded.")

    datasets2 = fetch(shuffle=True, random_state=37)

    for k in DATASET_SHAPE.keys():
        X1, X2 = datasets1[k].data, datasets2[k].data
        assert DATASET_SHAPE[k] == X1.shape
        assert X1.shape == X2.shape

        y1, y2 = datasets1[k].target, datasets2[k].target
        assert (X1.shape[0],) == y1.shape
        assert (X1.shape[0],) == y2.shape


def test_fetch_filter():
    try:
        datasets1 = fetch(filter_data=tuple([1]), shuffle=True, random_state=42)
    except IOError:
        raise SkipTest("Zenodo dataset can not be loaded.")

    datasets2 = fetch(filter_data=tuple(["ecoli"]), shuffle=True, random_state=37)

    X1, X2 = datasets1["ecoli"].data, datasets2["ecoli"].data
    assert DATASET_SHAPE["ecoli"] == X1.shape
    assert X1.shape == X2.shape

    assert X1.sum() == pytest.approx(X2.sum())

    y1, y2 = datasets1["ecoli"].target, datasets2["ecoli"].target
    assert (X1.shape[0],) == y1.shape
    assert (X1.shape[0],) == y2.shape


@pytest.mark.parametrize(
    "filter_data, err_msg",
    [
        (("rnf",), "is not a dataset available"),
        ((-1,), "dataset with the ID="),
        ((100,), "dataset with the ID="),
        ((1.00,), "value in the tuple"),
    ],
)
def test_fetch_error(filter_data, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        fetch_datasets(filter_data=filter_data)
