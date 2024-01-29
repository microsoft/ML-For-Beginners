import importlib
from collections import namedtuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
    ADAPTERS_MANAGER,
    ContainerAdapterProtocol,
    _get_output_config,
    _safe_set_output,
    _SetOutputMixin,
    _wrap_data_with_container,
    check_library_installed,
)
from sklearn.utils.fixes import CSR_CONTAINERS


def test_pandas_adapter():
    """Check pandas adapter has expected behavior."""
    pd = pytest.importorskip("pandas")
    X_np = np.asarray([[1, 0, 3], [0, 0, 1]])
    columns = np.asarray(["f0", "f1", "f2"], dtype=object)
    index = np.asarray([0, 1])
    X_df_orig = pd.DataFrame([[1, 2], [1, 3]], index=index)

    adapter = ADAPTERS_MANAGER.adapters["pandas"]
    X_container = adapter.create_container(X_np, X_df_orig, columns=lambda: columns)
    assert isinstance(X_container, pd.DataFrame)
    assert_array_equal(X_container.columns, columns)
    assert_array_equal(X_container.index, index)

    # Input dataframe's index does not change
    new_columns = np.asarray(["f0", "f1"], dtype=object)
    X_df = pd.DataFrame([[1, 2], [1, 3]], index=[10, 12])
    new_df = adapter.create_container(X_df, X_df_orig, columns=new_columns)
    assert_array_equal(new_df.columns, new_columns)
    assert_array_equal(new_df.index, X_df.index)

    assert adapter.is_supported_container(X_df)
    assert not adapter.is_supported_container(X_np)

    # adapter.update_columns updates the columns
    new_columns = np.array(["a", "c"], dtype=object)
    new_df = adapter.rename_columns(X_df, new_columns)
    assert_array_equal(new_df.columns, new_columns)

    # adapter.hstack stacks the dataframes horizontally.
    X_df_1 = pd.DataFrame([[1, 2, 5], [3, 4, 6]], columns=["a", "b", "e"])
    X_df_2 = pd.DataFrame([[4], [5]], columns=["c"])
    X_stacked = adapter.hstack([X_df_1, X_df_2])

    expected_df = pd.DataFrame(
        [[1, 2, 5, 4], [3, 4, 6, 5]], columns=["a", "b", "e", "c"]
    )
    pd.testing.assert_frame_equal(X_stacked, expected_df)


def test_polars_adapter():
    """Check Polars adapter has expected behavior."""
    pl = pytest.importorskip("polars")
    X_np = np.array([[1, 0, 3], [0, 0, 1]])
    columns = ["f1", "f2", "f3"]
    X_df_orig = pl.DataFrame(X_np, schema=columns, orient="row")

    adapter = ADAPTERS_MANAGER.adapters["polars"]
    X_container = adapter.create_container(X_np, X_df_orig, columns=lambda: columns)

    assert isinstance(X_container, pl.DataFrame)
    assert_array_equal(X_container.columns, columns)

    # Update columns with create_container
    new_columns = np.asarray(["a", "b", "c"], dtype=object)
    new_df = adapter.create_container(X_df_orig, X_df_orig, columns=new_columns)
    assert_array_equal(new_df.columns, new_columns)

    assert adapter.is_supported_container(X_df_orig)
    assert not adapter.is_supported_container(X_np)

    # adapter.update_columns updates the columns
    new_columns = np.array(["a", "c", "g"], dtype=object)
    new_df = adapter.rename_columns(X_df_orig, new_columns)
    assert_array_equal(new_df.columns, new_columns)

    # adapter.hstack stacks the dataframes horizontally.
    X_df_1 = pl.DataFrame([[1, 2, 5], [3, 4, 6]], schema=["a", "b", "e"], orient="row")
    X_df_2 = pl.DataFrame([[4], [5]], schema=["c"], orient="row")
    X_stacked = adapter.hstack([X_df_1, X_df_2])

    expected_df = pl.DataFrame(
        [[1, 2, 5, 4], [3, 4, 6, 5]], schema=["a", "b", "e", "c"], orient="row"
    )
    from polars.testing import assert_frame_equal

    assert_frame_equal(X_stacked, expected_df)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test__container_error_validation(csr_container):
    """Check errors in _wrap_data_with_container."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    X_csr = csr_container(X)
    match = "The transformer outputs a scipy sparse matrix."
    with config_context(transform_output="pandas"):
        with pytest.raises(ValueError, match=match):
            _wrap_data_with_container("transform", X_csr, X, StandardScaler())


class EstimatorWithoutSetOutputAndWithoutTransform:
    pass


class EstimatorNoSetOutputWithTransform:
    def transform(self, X, y=None):
        return X  # pragma: no cover


class EstimatorWithSetOutput(_SetOutputMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


def test__safe_set_output():
    """Check _safe_set_output works as expected."""

    # Estimator without transform will not raise when setting set_output for transform.
    est = EstimatorWithoutSetOutputAndWithoutTransform()
    _safe_set_output(est, transform="pandas")

    # Estimator with transform but without set_output will raise
    est = EstimatorNoSetOutputWithTransform()
    with pytest.raises(ValueError, match="Unable to configure output"):
        _safe_set_output(est, transform="pandas")

    est = EstimatorWithSetOutput().fit(np.asarray([[1, 2, 3]]))
    _safe_set_output(est, transform="pandas")
    config = _get_output_config("transform", est)
    assert config["dense"] == "pandas"

    _safe_set_output(est, transform="default")
    config = _get_output_config("transform", est)
    assert config["dense"] == "default"

    # transform is None is a no-op, so the config remains "default"
    _safe_set_output(est, transform=None)
    config = _get_output_config("transform", est)
    assert config["dense"] == "default"


class EstimatorNoSetOutputWithTransformNoFeatureNamesOut(_SetOutputMixin):
    def transform(self, X, y=None):
        return X  # pragma: no cover


def test_set_output_mixin():
    """Estimator without get_feature_names_out does not define `set_output`."""
    est = EstimatorNoSetOutputWithTransformNoFeatureNamesOut()
    assert not hasattr(est, "set_output")


def test__safe_set_output_error():
    """Check transform with invalid config."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])

    est = EstimatorWithSetOutput()
    _safe_set_output(est, transform="bad")

    msg = "output config must be in"
    with pytest.raises(ValueError, match=msg):
        est.transform(X)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def test_set_output_method(dataframe_lib):
    """Check that the output is a dataframe."""
    lib = pytest.importorskip(dataframe_lib)

    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)

    # transform=None is a no-op
    est2 = est.set_output(transform=None)
    assert est2 is est
    X_trans_np = est2.transform(X)
    assert isinstance(X_trans_np, np.ndarray)

    est.set_output(transform=dataframe_lib)

    X_trans_pd = est.transform(X)

    assert isinstance(X_trans_pd, lib.DataFrame)


def test_set_output_method_error():
    """Check transform fails with invalid transform."""

    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)
    est.set_output(transform="bad")

    msg = "output config must be in"
    with pytest.raises(ValueError, match=msg):
        est.transform(X)


@pytest.mark.parametrize("transform_output", ["pandas", "polars"])
def test__get_output_config(transform_output):
    """Check _get_output_config works as expected."""

    # Without a configuration set, the global config is used
    global_config = get_config()["transform_output"]
    config = _get_output_config("transform")
    assert config["dense"] == global_config

    with config_context(transform_output=transform_output):
        # with estimator=None, the global config is used
        config = _get_output_config("transform")
        assert config["dense"] == transform_output

        est = EstimatorNoSetOutputWithTransform()
        config = _get_output_config("transform", est)
        assert config["dense"] == transform_output

        est = EstimatorWithSetOutput()
        # If estimator has not config, use global config
        config = _get_output_config("transform", est)
        assert config["dense"] == transform_output

        # If estimator has a config, use local config
        est.set_output(transform="default")
        config = _get_output_config("transform", est)
        assert config["dense"] == "default"

    est.set_output(transform=transform_output)
    config = _get_output_config("transform", est)
    assert config["dense"] == transform_output


class EstimatorWithSetOutputNoAutoWrap(_SetOutputMixin, auto_wrap_output_keys=None):
    def transform(self, X, y=None):
        return X


def test_get_output_auto_wrap_false():
    """Check that auto_wrap_output_keys=None does not wrap."""
    est = EstimatorWithSetOutputNoAutoWrap()
    assert not hasattr(est, "set_output")

    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    assert X is est.transform(X)


def test_auto_wrap_output_keys_errors_with_incorrect_input():
    msg = "auto_wrap_output_keys must be None or a tuple of keys."
    with pytest.raises(ValueError, match=msg):

        class BadEstimator(_SetOutputMixin, auto_wrap_output_keys="bad_parameter"):
            pass


class AnotherMixin:
    def __init_subclass__(cls, custom_parameter, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.custom_parameter = custom_parameter


def test_set_output_mixin_custom_mixin():
    """Check that multiple init_subclasses passes parameters up."""

    class BothMixinEstimator(_SetOutputMixin, AnotherMixin, custom_parameter=123):
        def transform(self, X, y=None):
            return X

        def get_feature_names_out(self, input_features=None):
            return input_features

    est = BothMixinEstimator()
    assert est.custom_parameter == 123
    assert hasattr(est, "set_output")


def test_set_output_mro():
    """Check that multi-inheritance resolves to the correct class method.

    Non-regression test gh-25293.
    """

    class Base(_SetOutputMixin):
        def transform(self, X):
            return "Base"  # noqa

    class A(Base):
        pass

    class B(Base):
        def transform(self, X):
            return "B"

    class C(A, B):
        pass

    assert C().transform(None) == "B"


class EstimatorWithSetOutputIndex(_SetOutputMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        import pandas as pd

        # transform by giving output a new index.
        return pd.DataFrame(X.to_numpy(), index=[f"s{i}" for i in range(X.shape[0])])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


def test_set_output_pandas_keep_index():
    """Check that set_output does not override index.

    Non-regression test for gh-25730.
    """
    pd = pytest.importorskip("pandas")

    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[0, 1])
    est = EstimatorWithSetOutputIndex().set_output(transform="pandas")
    est.fit(X)

    X_trans = est.transform(X)
    assert_array_equal(X_trans.index, ["s0", "s1"])


class EstimatorReturnTuple(_SetOutputMixin):
    def __init__(self, OutputTuple):
        self.OutputTuple = OutputTuple

    def transform(self, X, y=None):
        return self.OutputTuple(X, 2 * X)


def test_set_output_named_tuple_out():
    """Check that namedtuples are kept by default."""
    Output = namedtuple("Output", "X, Y")
    X = np.asarray([[1, 2, 3]])
    est = EstimatorReturnTuple(OutputTuple=Output)
    X_trans = est.transform(X)

    assert isinstance(X_trans, Output)
    assert_array_equal(X_trans.X, X)
    assert_array_equal(X_trans.Y, 2 * X)


class EstimatorWithListInput(_SetOutputMixin):
    def fit(self, X, y=None):
        assert isinstance(X, list)
        self.n_features_in_ = len(X[0])
        return self

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def test_set_output_list_input(dataframe_lib):
    """Check set_output for list input.

    Non-regression test for #27037.
    """
    lib = pytest.importorskip(dataframe_lib)

    X = [[0, 1, 2, 3], [4, 5, 6, 7]]
    est = EstimatorWithListInput()
    est.set_output(transform=dataframe_lib)

    X_out = est.fit(X).transform(X)
    assert isinstance(X_out, lib.DataFrame)
    assert_array_equal(X_out.columns, ["X0", "X1", "X2", "X3"])


@pytest.mark.parametrize("name", sorted(ADAPTERS_MANAGER.adapters))
def test_adapter_class_has_interface(name):
    """Check adapters have the correct interface."""
    assert isinstance(ADAPTERS_MANAGER.adapters[name], ContainerAdapterProtocol)


def test_check_library_installed(monkeypatch):
    """Check import error changed."""
    orig_import_module = importlib.import_module

    def patched_import_module(name):
        if name == "pandas":
            raise ImportError()
        orig_import_module(name, package=None)

    monkeypatch.setattr(importlib, "import_module", patched_import_module)

    msg = "Setting output container to 'pandas' requires"
    with pytest.raises(ImportError, match=msg):
        check_library_installed("pandas")
