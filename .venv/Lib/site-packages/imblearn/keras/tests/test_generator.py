import numpy as np
import pytest
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer

keras = pytest.importorskip("keras")
from keras.layers import Dense  # noqa: E402
from keras.models import Sequential  # noqa: E402

from imblearn.datasets import make_imbalance  # noqa: E402
from imblearn.keras import (
    BalancedBatchGenerator,  # noqa: E402
    balanced_batch_generator,  # noqa: E402
)
from imblearn.over_sampling import RandomOverSampler  # noqa: E402
from imblearn.under_sampling import (
    ClusterCentroids,  # noqa: E402
    NearMiss,  # noqa: E402
)

3


@pytest.fixture
def data():
    iris = load_iris()
    X, y = make_imbalance(
        iris.data, iris.target, sampling_strategy={0: 30, 1: 50, 2: 40}
    )
    y = LabelBinarizer().fit_transform(y)
    return X, y


def _build_keras_model(n_classes, n_features):
    model = Sequential()
    model.add(Dense(n_classes, input_dim=n_features, activation="softmax"))
    model.compile(
        optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def test_balanced_batch_generator_class_no_return_indices(data):
    with pytest.raises(ValueError, match="needs to have an attribute"):
        BalancedBatchGenerator(
            *data, sampler=ClusterCentroids(estimator=KMeans(n_init=1)), batch_size=10
        )


@pytest.mark.filterwarnings("ignore:`wait_time` is not used")  # keras 2.2.4
@pytest.mark.parametrize(
    "sampler, sample_weight",
    [
        (None, None),
        (RandomOverSampler(), None),
        (NearMiss(), None),
        (None, np.random.uniform(size=120)),
    ],
)
def test_balanced_batch_generator_class(data, sampler, sample_weight):
    X, y = data
    model = _build_keras_model(y.shape[1], X.shape[1])
    training_generator = BalancedBatchGenerator(
        X,
        y,
        sample_weight=sample_weight,
        sampler=sampler,
        batch_size=10,
        random_state=42,
    )
    model.fit_generator(generator=training_generator, epochs=10)


@pytest.mark.parametrize("keep_sparse", [True, False])
def test_balanced_batch_generator_class_sparse(data, keep_sparse):
    X, y = data
    training_generator = BalancedBatchGenerator(
        sparse.csr_matrix(X),
        y,
        batch_size=10,
        keep_sparse=keep_sparse,
        random_state=42,
    )
    for idx in range(len(training_generator)):
        X_batch, _ = training_generator.__getitem__(idx)
        if keep_sparse:
            assert sparse.issparse(X_batch)
        else:
            assert not sparse.issparse(X_batch)


def test_balanced_batch_generator_function_no_return_indices(data):
    with pytest.raises(ValueError, match="needs to have an attribute"):
        balanced_batch_generator(
            *data,
            sampler=ClusterCentroids(estimator=KMeans(n_init=10)),
            batch_size=10,
            random_state=42,
        )


@pytest.mark.filterwarnings("ignore:`wait_time` is not used")  # keras 2.2.4
@pytest.mark.parametrize(
    "sampler, sample_weight",
    [
        (None, None),
        (RandomOverSampler(), None),
        (NearMiss(), None),
        (None, np.random.uniform(size=120)),
    ],
)
def test_balanced_batch_generator_function(data, sampler, sample_weight):
    X, y = data
    model = _build_keras_model(y.shape[1], X.shape[1])
    training_generator, steps_per_epoch = balanced_batch_generator(
        X,
        y,
        sample_weight=sample_weight,
        sampler=sampler,
        batch_size=10,
        random_state=42,
    )
    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
    )


@pytest.mark.parametrize("keep_sparse", [True, False])
def test_balanced_batch_generator_function_sparse(data, keep_sparse):
    X, y = data
    training_generator, steps_per_epoch = balanced_batch_generator(
        sparse.csr_matrix(X),
        y,
        keep_sparse=keep_sparse,
        batch_size=10,
        random_state=42,
    )
    for _ in range(steps_per_epoch):
        X_batch, _ = next(training_generator)
        if keep_sparse:
            assert sparse.issparse(X_batch)
        else:
            assert not sparse.issparse(X_batch)
