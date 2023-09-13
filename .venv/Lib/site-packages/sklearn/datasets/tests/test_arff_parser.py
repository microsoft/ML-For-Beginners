import textwrap
from io import BytesIO

import pytest

from sklearn.datasets._arff_parser import (
    _liac_arff_parser,
    _pandas_arff_parser,
    _post_process_frame,
    load_arff_from_gzip_file,
)


@pytest.mark.parametrize(
    "feature_names, target_names",
    [
        (
            [
                "col_int_as_integer",
                "col_int_as_numeric",
                "col_float_as_real",
                "col_float_as_numeric",
            ],
            ["col_categorical", "col_string"],
        ),
        (
            [
                "col_int_as_integer",
                "col_int_as_numeric",
                "col_float_as_real",
                "col_float_as_numeric",
            ],
            ["col_categorical"],
        ),
        (
            [
                "col_int_as_integer",
                "col_int_as_numeric",
                "col_float_as_real",
                "col_float_as_numeric",
            ],
            [],
        ),
    ],
)
def test_post_process_frame(feature_names, target_names):
    """Check the behaviour of the post-processing function for splitting a dataframe."""
    pd = pytest.importorskip("pandas")

    X_original = pd.DataFrame(
        {
            "col_int_as_integer": [1, 2, 3],
            "col_int_as_numeric": [1, 2, 3],
            "col_float_as_real": [1.0, 2.0, 3.0],
            "col_float_as_numeric": [1.0, 2.0, 3.0],
            "col_categorical": ["a", "b", "c"],
            "col_string": ["a", "b", "c"],
        }
    )

    X, y = _post_process_frame(X_original, feature_names, target_names)
    assert isinstance(X, pd.DataFrame)
    if len(target_names) >= 2:
        assert isinstance(y, pd.DataFrame)
    elif len(target_names) == 1:
        assert isinstance(y, pd.Series)
    else:
        assert y is None


def test_load_arff_from_gzip_file_error_parser():
    """An error will be raised if the parser is not known."""
    # None of the input parameters are required to be accurate since the check
    # of the parser will be carried out first.

    err_msg = "Unknown parser: 'xxx'. Should be 'liac-arff' or 'pandas'"
    with pytest.raises(ValueError, match=err_msg):
        load_arff_from_gzip_file("xxx", "xxx", "xxx", "xxx", "xxx", "xxx")


@pytest.mark.parametrize("parser_func", [_liac_arff_parser, _pandas_arff_parser])
def test_pandas_arff_parser_strip_single_quotes(parser_func):
    """Check that we properly strip single quotes from the data."""
    pd = pytest.importorskip("pandas")

    arff_file = BytesIO(textwrap.dedent("""
            @relation 'toy'
            @attribute 'cat_single_quote' {'A', 'B', 'C'}
            @attribute 'str_single_quote' string
            @attribute 'str_nested_quote' string
            @attribute 'class' numeric
            @data
            'A','some text','\"expect double quotes\"',0
            """).encode("utf-8"))

    columns_info = {
        "cat_single_quote": {
            "data_type": "nominal",
            "name": "cat_single_quote",
        },
        "str_single_quote": {
            "data_type": "string",
            "name": "str_single_quote",
        },
        "str_nested_quote": {
            "data_type": "string",
            "name": "str_nested_quote",
        },
        "class": {
            "data_type": "numeric",
            "name": "class",
        },
    }

    feature_names = [
        "cat_single_quote",
        "str_single_quote",
        "str_nested_quote",
    ]
    target_names = ["class"]

    # We don't strip single quotes for string columns with the pandas parser.
    expected_values = {
        "cat_single_quote": "A",
        "str_single_quote": (
            "some text" if parser_func is _liac_arff_parser else "'some text'"
        ),
        "str_nested_quote": (
            '"expect double quotes"'
            if parser_func is _liac_arff_parser
            else "'\"expect double quotes\"'"
        ),
        "class": 0,
    }

    _, _, frame, _ = parser_func(
        arff_file,
        output_arrays_type="pandas",
        openml_columns_info=columns_info,
        feature_names_to_select=feature_names,
        target_names_to_select=target_names,
    )

    assert frame.columns.tolist() == feature_names + target_names
    pd.testing.assert_series_equal(frame.iloc[0], pd.Series(expected_values, name=0))


@pytest.mark.parametrize("parser_func", [_liac_arff_parser, _pandas_arff_parser])
def test_pandas_arff_parser_strip_double_quotes(parser_func):
    """Check that we properly strip double quotes from the data."""
    pd = pytest.importorskip("pandas")

    arff_file = BytesIO(textwrap.dedent("""
            @relation 'toy'
            @attribute 'cat_double_quote' {"A", "B", "C"}
            @attribute 'str_double_quote' string
            @attribute 'str_nested_quote' string
            @attribute 'class' numeric
            @data
            "A","some text","\'expect double quotes\'",0
            """).encode("utf-8"))

    columns_info = {
        "cat_double_quote": {
            "data_type": "nominal",
            "name": "cat_double_quote",
        },
        "str_double_quote": {
            "data_type": "string",
            "name": "str_double_quote",
        },
        "str_nested_quote": {
            "data_type": "string",
            "name": "str_nested_quote",
        },
        "class": {
            "data_type": "numeric",
            "name": "class",
        },
    }

    feature_names = [
        "cat_double_quote",
        "str_double_quote",
        "str_nested_quote",
    ]
    target_names = ["class"]

    expected_values = {
        "cat_double_quote": "A",
        "str_double_quote": "some text",
        "str_nested_quote": "'expect double quotes'",
        "class": 0,
    }

    _, _, frame, _ = parser_func(
        arff_file,
        output_arrays_type="pandas",
        openml_columns_info=columns_info,
        feature_names_to_select=feature_names,
        target_names_to_select=target_names,
    )

    assert frame.columns.tolist() == feature_names + target_names
    pd.testing.assert_series_equal(frame.iloc[0], pd.Series(expected_values, name=0))


@pytest.mark.parametrize(
    "parser_func",
    [
        # internal quotes are not considered to follow the ARFF spec in LIAC ARFF
        pytest.param(_liac_arff_parser, marks=pytest.mark.xfail),
        _pandas_arff_parser,
    ],
)
def test_pandas_arff_parser_strip_no_quotes(parser_func):
    """Check that we properly parse with no quotes characters."""
    pd = pytest.importorskip("pandas")

    arff_file = BytesIO(textwrap.dedent("""
            @relation 'toy'
            @attribute 'cat_without_quote' {A, B, C}
            @attribute 'str_without_quote' string
            @attribute 'str_internal_quote' string
            @attribute 'class' numeric
            @data
            A,some text,'internal' quote,0
            """).encode("utf-8"))

    columns_info = {
        "cat_without_quote": {
            "data_type": "nominal",
            "name": "cat_without_quote",
        },
        "str_without_quote": {
            "data_type": "string",
            "name": "str_without_quote",
        },
        "str_internal_quote": {
            "data_type": "string",
            "name": "str_internal_quote",
        },
        "class": {
            "data_type": "numeric",
            "name": "class",
        },
    }

    feature_names = [
        "cat_without_quote",
        "str_without_quote",
        "str_internal_quote",
    ]
    target_names = ["class"]

    expected_values = {
        "cat_without_quote": "A",
        "str_without_quote": "some text",
        "str_internal_quote": "'internal' quote",
        "class": 0,
    }

    _, _, frame, _ = parser_func(
        arff_file,
        output_arrays_type="pandas",
        openml_columns_info=columns_info,
        feature_names_to_select=feature_names,
        target_names_to_select=target_names,
    )

    assert frame.columns.tolist() == feature_names + target_names
    pd.testing.assert_series_equal(frame.iloc[0], pd.Series(expected_values, name=0))
