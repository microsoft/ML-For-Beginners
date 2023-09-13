"""
The :mod:`sklearn.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets. It also
features some artificial data generators.
"""
import textwrap

from ._base import (
    clear_data_home,
    get_data_home,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_files,
    load_iris,
    load_linnerud,
    load_sample_image,
    load_sample_images,
    load_wine,
)
from ._california_housing import fetch_california_housing
from ._covtype import fetch_covtype
from ._kddcup99 import fetch_kddcup99
from ._lfw import fetch_lfw_pairs, fetch_lfw_people
from ._olivetti_faces import fetch_olivetti_faces
from ._openml import fetch_openml
from ._rcv1 import fetch_rcv1
from ._samples_generator import (
    make_biclusters,
    make_blobs,
    make_checkerboard,
    make_circles,
    make_classification,
    make_friedman1,
    make_friedman2,
    make_friedman3,
    make_gaussian_quantiles,
    make_hastie_10_2,
    make_low_rank_matrix,
    make_moons,
    make_multilabel_classification,
    make_regression,
    make_s_curve,
    make_sparse_coded_signal,
    make_sparse_spd_matrix,
    make_sparse_uncorrelated,
    make_spd_matrix,
    make_swiss_roll,
)
from ._species_distributions import fetch_species_distributions
from ._svmlight_format_io import (
    dump_svmlight_file,
    load_svmlight_file,
    load_svmlight_files,
)
from ._twenty_newsgroups import fetch_20newsgroups, fetch_20newsgroups_vectorized

__all__ = [
    "clear_data_home",
    "dump_svmlight_file",
    "fetch_20newsgroups",
    "fetch_20newsgroups_vectorized",
    "fetch_lfw_pairs",
    "fetch_lfw_people",
    "fetch_olivetti_faces",
    "fetch_species_distributions",
    "fetch_california_housing",
    "fetch_covtype",
    "fetch_rcv1",
    "fetch_kddcup99",
    "fetch_openml",
    "get_data_home",
    "load_diabetes",
    "load_digits",
    "load_files",
    "load_iris",
    "load_breast_cancer",
    "load_linnerud",
    "load_sample_image",
    "load_sample_images",
    "load_svmlight_file",
    "load_svmlight_files",
    "load_wine",
    "make_biclusters",
    "make_blobs",
    "make_circles",
    "make_classification",
    "make_checkerboard",
    "make_friedman1",
    "make_friedman2",
    "make_friedman3",
    "make_gaussian_quantiles",
    "make_hastie_10_2",
    "make_low_rank_matrix",
    "make_moons",
    "make_multilabel_classification",
    "make_regression",
    "make_s_curve",
    "make_sparse_coded_signal",
    "make_sparse_spd_matrix",
    "make_sparse_uncorrelated",
    "make_spd_matrix",
    "make_swiss_roll",
]


def __getattr__(name):
    if name == "load_boston":
        msg = textwrap.dedent("""
            `load_boston` has been removed from scikit-learn since version 1.2.

            The Boston housing prices dataset has an ethical problem: as
            investigated in [1], the authors of this dataset engineered a
            non-invertible variable "B" assuming that racial self-segregation had a
            positive impact on house prices [2]. Furthermore the goal of the
            research that led to the creation of this dataset was to study the
            impact of air quality but it did not give adequate demonstration of the
            validity of this assumption.

            The scikit-learn maintainers therefore strongly discourage the use of
            this dataset unless the purpose of the code is to study and educate
            about ethical issues in data science and machine learning.

            In this special case, you can fetch the dataset from the original
            source::

                import pandas as pd
                import numpy as np

                data_url = "http://lib.stat.cmu.edu/datasets/boston"
                raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
                data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
                target = raw_df.values[1::2, 2]

            Alternative datasets include the California housing dataset and the
            Ames housing dataset. You can load the datasets as follows::

                from sklearn.datasets import fetch_california_housing
                housing = fetch_california_housing()

            for the California housing dataset and::

                from sklearn.datasets import fetch_openml
                housing = fetch_openml(name="house_prices", as_frame=True)

            for the Ames housing dataset.

            [1] M Carlisle.
            "Racist data destruction?"
            <https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>

            [2] Harrison Jr, David, and Daniel L. Rubinfeld.
            "Hedonic housing prices and the demand for clean air."
            Journal of environmental economics and management 5.1 (1978): 81-102.
            <https://www.researchgate.net/publication/4974606_Hedonic_housing_prices_and_the_demand_for_clean_air>
            """)
        raise ImportError(msg)
    try:
        return globals()[name]
    except KeyError:
        # This is turned into the appropriate ImportError
        raise AttributeError
