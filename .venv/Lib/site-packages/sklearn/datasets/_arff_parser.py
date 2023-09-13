"""Implementation of ARFF parsers: via LIAC-ARFF and pandas."""
import itertools
import re
from collections import OrderedDict
from collections.abc import Generator
from typing import List

import numpy as np
import scipy as sp

from ..externals import _arff
from ..externals._arff import ArffSparseDataType
from ..utils import (
    _chunk_generator,
    check_pandas_support,
    get_chunk_n_rows,
)


def _split_sparse_columns(
    arff_data: ArffSparseDataType, include_columns: List
) -> ArffSparseDataType:
    """Obtains several columns from sparse ARFF representation. Additionally,
    the column indices are re-labelled, given the columns that are not
    included. (e.g., when including [1, 2, 3], the columns will be relabelled
    to [0, 1, 2]).

    Parameters
    ----------
    arff_data : tuple
        A tuple of three lists of equal size; first list indicating the value,
        second the x coordinate and the third the y coordinate.

    include_columns : list
        A list of columns to include.

    Returns
    -------
    arff_data_new : tuple
        Subset of arff data with only the include columns indicated by the
        include_columns argument.
    """
    arff_data_new: ArffSparseDataType = (list(), list(), list())
    reindexed_columns = {
        column_idx: array_idx for array_idx, column_idx in enumerate(include_columns)
    }
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            arff_data_new[0].append(val)
            arff_data_new[1].append(row_idx)
            arff_data_new[2].append(reindexed_columns[col_idx])
    return arff_data_new


def _sparse_data_to_array(
    arff_data: ArffSparseDataType, include_columns: List
) -> np.ndarray:
    # turns the sparse data back into an array (can't use toarray() function,
    # as this does only work on numeric data)
    num_obs = max(arff_data[1]) + 1
    y_shape = (num_obs, len(include_columns))
    reindexed_columns = {
        column_idx: array_idx for array_idx, column_idx in enumerate(include_columns)
    }
    # TODO: improve for efficiency
    y = np.empty(y_shape, dtype=np.float64)
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            y[row_idx, reindexed_columns[col_idx]] = val
    return y


def _post_process_frame(frame, feature_names, target_names):
    """Post process a dataframe to select the desired columns in `X` and `y`.

    Parameters
    ----------
    frame : dataframe
        The dataframe to split into `X` and `y`.

    feature_names : list of str
        The list of feature names to populate `X`.

    target_names : list of str
        The list of target names to populate `y`.

    Returns
    -------
    X : dataframe
        The dataframe containing the features.

    y : {series, dataframe} or None
        The series or dataframe containing the target.
    """
    X = frame[feature_names]
    if len(target_names) >= 2:
        y = frame[target_names]
    elif len(target_names) == 1:
        y = frame[target_names[0]]
    else:
        y = None
    return X, y


def _liac_arff_parser(
    gzip_file,
    output_arrays_type,
    openml_columns_info,
    feature_names_to_select,
    target_names_to_select,
    shape=None,
):
    """ARFF parser using the LIAC-ARFF library coded purely in Python.

    This parser is quite slow but consumes a generator. Currently it is needed
    to parse sparse datasets. For dense datasets, it is recommended to instead
    use the pandas-based parser, although it does not always handles the
    dtypes exactly the same.

    Parameters
    ----------
    gzip_file : GzipFile instance
        The file compressed to be read.

    output_arrays_type : {"numpy", "sparse", "pandas"}
        The type of the arrays that will be returned. The possibilities ara:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        A list of the feature names to be selected.

    target_names_to_select : list of str
        A list of the target names to be selected.

    Returns
    -------
    X : {ndarray, sparse matrix, dataframe}
        The data matrix.

    y : {ndarray, dataframe, series}
        The target.

    frame : dataframe or None
        A dataframe containing both `X` and `y`. `None` if
        `output_array_type != "pandas"`.

    categories : list of str or None
        The names of the features that are categorical. `None` if
        `output_array_type == "pandas"`.
    """

    def _io_to_generator(gzip_file):
        for line in gzip_file:
            yield line.decode("utf-8")

    stream = _io_to_generator(gzip_file)

    # find which type (dense or sparse) ARFF type we will have to deal with
    return_type = _arff.COO if output_arrays_type == "sparse" else _arff.DENSE_GEN
    # we should not let LIAC-ARFF to encode the nominal attributes with NumPy
    # arrays to have only numerical values.
    encode_nominal = not (output_arrays_type == "pandas")
    arff_container = _arff.load(
        stream, return_type=return_type, encode_nominal=encode_nominal
    )
    columns_to_select = feature_names_to_select + target_names_to_select

    categories = {
        name: cat
        for name, cat in arff_container["attributes"]
        if isinstance(cat, list) and name in columns_to_select
    }
    if output_arrays_type == "pandas":
        pd = check_pandas_support("fetch_openml with as_frame=True")

        columns_info = OrderedDict(arff_container["attributes"])
        columns_names = list(columns_info.keys())

        # calculate chunksize
        first_row = next(arff_container["data"])
        first_df = pd.DataFrame([first_row], columns=columns_names, copy=False)

        row_bytes = first_df.memory_usage(deep=True).sum()
        chunksize = get_chunk_n_rows(row_bytes)

        # read arff data with chunks
        columns_to_keep = [col for col in columns_names if col in columns_to_select]
        dfs = [first_df[columns_to_keep]]
        for data in _chunk_generator(arff_container["data"], chunksize):
            dfs.append(
                pd.DataFrame(data, columns=columns_names, copy=False)[columns_to_keep]
            )
        # dfs[0] contains only one row, which may not have enough data to infer to
        # column's dtype. Here we use `dfs[1]` to configure the dtype in dfs[0]
        if len(dfs) >= 2:
            dfs[0] = dfs[0].astype(dfs[1].dtypes)

        # liac-arff parser does not depend on NumPy and uses None to represent
        # missing values. To be consistent with the pandas parser, we replace
        # None with np.nan.
        frame = pd.concat(dfs, ignore_index=True).fillna(value=np.nan)
        del dfs, first_df

        # cast the columns frame
        dtypes = {}
        for name in frame.columns:
            column_dtype = openml_columns_info[name]["data_type"]
            if column_dtype.lower() == "integer":
                # Use a pandas extension array instead of np.int64 to be able
                # to support missing values.
                dtypes[name] = "Int64"
            elif column_dtype.lower() == "nominal":
                dtypes[name] = "category"
            else:
                dtypes[name] = frame.dtypes[name]
        frame = frame.astype(dtypes)

        X, y = _post_process_frame(
            frame, feature_names_to_select, target_names_to_select
        )
    else:
        arff_data = arff_container["data"]

        feature_indices_to_select = [
            int(openml_columns_info[col_name]["index"])
            for col_name in feature_names_to_select
        ]
        target_indices_to_select = [
            int(openml_columns_info[col_name]["index"])
            for col_name in target_names_to_select
        ]

        if isinstance(arff_data, Generator):
            if shape is None:
                raise ValueError(
                    "shape must be provided when arr['data'] is a Generator"
                )
            if shape[0] == -1:
                count = -1
            else:
                count = shape[0] * shape[1]
            data = np.fromiter(
                itertools.chain.from_iterable(arff_data),
                dtype="float64",
                count=count,
            )
            data = data.reshape(*shape)
            X = data[:, feature_indices_to_select]
            y = data[:, target_indices_to_select]
        elif isinstance(arff_data, tuple):
            arff_data_X = _split_sparse_columns(arff_data, feature_indices_to_select)
            num_obs = max(arff_data[1]) + 1
            X_shape = (num_obs, len(feature_indices_to_select))
            X = sp.sparse.coo_matrix(
                (arff_data_X[0], (arff_data_X[1], arff_data_X[2])),
                shape=X_shape,
                dtype=np.float64,
            )
            X = X.tocsr()
            y = _sparse_data_to_array(arff_data, target_indices_to_select)
        else:
            # This should never happen
            raise ValueError(
                f"Unexpected type for data obtained from arff: {type(arff_data)}"
            )

        is_classification = {
            col_name in categories for col_name in target_names_to_select
        }
        if not is_classification:
            # No target
            pass
        elif all(is_classification):
            y = np.hstack(
                [
                    np.take(
                        np.asarray(categories.pop(col_name), dtype="O"),
                        y[:, i : i + 1].astype(int, copy=False),
                    )
                    for i, col_name in enumerate(target_names_to_select)
                ]
            )
        elif any(is_classification):
            raise ValueError(
                "Mix of nominal and non-nominal targets is not currently supported"
            )

        # reshape y back to 1-D array, if there is only 1 target column;
        # back to None if there are not target columns
        if y.shape[1] == 1:
            y = y.reshape((-1,))
        elif y.shape[1] == 0:
            y = None

    if output_arrays_type == "pandas":
        return X, y, frame, None
    return X, y, None, categories


def _pandas_arff_parser(
    gzip_file,
    output_arrays_type,
    openml_columns_info,
    feature_names_to_select,
    target_names_to_select,
    read_csv_kwargs=None,
):
    """ARFF parser using `pandas.read_csv`.

    This parser uses the metadata fetched directly from OpenML and skips the metadata
    headers of ARFF file itself. The data is loaded as a CSV file.

    Parameters
    ----------
    gzip_file : GzipFile instance
        The GZip compressed file with the ARFF formatted payload.

    output_arrays_type : {"numpy", "sparse", "pandas"}
        The type of the arrays that will be returned. The possibilities are:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    openml_columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        A list of the feature names to be selected to build `X`.

    target_names_to_select : list of str
        A list of the target names to be selected to build `y`.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv`. It allows to overwrite
        the default options.

    Returns
    -------
    X : {ndarray, sparse matrix, dataframe}
        The data matrix.

    y : {ndarray, dataframe, series}
        The target.

    frame : dataframe or None
        A dataframe containing both `X` and `y`. `None` if
        `output_array_type != "pandas"`.

    categories : list of str or None
        The names of the features that are categorical. `None` if
        `output_array_type == "pandas"`.
    """
    import pandas as pd

    # read the file until the data section to skip the ARFF metadata headers
    for line in gzip_file:
        if line.decode("utf-8").lower().startswith("@data"):
            break

    dtypes = {}
    for name in openml_columns_info:
        column_dtype = openml_columns_info[name]["data_type"]
        if column_dtype.lower() == "integer":
            # Use Int64 to infer missing values from data
            # XXX: this line is not covered by our tests. Is this really needed?
            dtypes[name] = "Int64"
        elif column_dtype.lower() == "nominal":
            dtypes[name] = "category"
    # since we will not pass `names` when reading the ARFF file, we need to translate
    # `dtypes` from column names to column indices to pass to `pandas.read_csv`
    dtypes_positional = {
        col_idx: dtypes[name]
        for col_idx, name in enumerate(openml_columns_info)
        if name in dtypes
    }

    default_read_csv_kwargs = {
        "header": None,
        "index_col": False,  # always force pandas to not use the first column as index
        "na_values": ["?"],  # missing values are represented by `?`
        "keep_default_na": False,  # only `?` is a missing value given the ARFF specs
        "comment": "%",  # skip line starting by `%` since they are comments
        "quotechar": '"',  # delimiter to use for quoted strings
        "skipinitialspace": True,  # skip spaces after delimiter to follow ARFF specs
        "escapechar": "\\",
        "dtype": dtypes_positional,
    }
    read_csv_kwargs = {**default_read_csv_kwargs, **(read_csv_kwargs or {})}
    frame = pd.read_csv(gzip_file, **read_csv_kwargs)
    try:
        # Setting the columns while reading the file will select the N first columns
        # and not raise a ParserError. Instead, we set the columns after reading the
        # file and raise a ParserError if the number of columns does not match the
        # number of columns in the metadata given by OpenML.
        frame.columns = [name for name in openml_columns_info]
    except ValueError as exc:
        raise pd.errors.ParserError(
            "The number of columns provided by OpenML does not match the number of "
            "columns inferred by pandas when reading the file."
        ) from exc

    columns_to_select = feature_names_to_select + target_names_to_select
    columns_to_keep = [col for col in frame.columns if col in columns_to_select]
    frame = frame[columns_to_keep]

    # `pd.read_csv` automatically handles double quotes for quoting non-numeric
    # CSV cell values. Contrary to LIAC-ARFF, `pd.read_csv` cannot be configured to
    # consider either single quotes and double quotes as valid quoting chars at
    # the same time since this case does not occur in regular (non-ARFF) CSV files.
    # To mimic the behavior of LIAC-ARFF parser, we manually strip single quotes
    # on categories as a post-processing steps if needed.
    #
    # Note however that we intentionally do not attempt to do this kind of manual
    # post-processing of (non-categorical) string-typed columns because we cannot
    # resolve the ambiguity of the case of CSV cell with nesting quoting such as
    # `"'some string value'"` with pandas.
    single_quote_pattern = re.compile(r"^'(?P<contents>.*)'$")

    def strip_single_quotes(input_string):
        match = re.search(single_quote_pattern, input_string)
        if match is None:
            return input_string

        return match.group("contents")

    categorical_columns = [
        name
        for name, dtype in frame.dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype)
    ]
    for col in categorical_columns:
        frame[col] = frame[col].cat.rename_categories(strip_single_quotes)

    X, y = _post_process_frame(frame, feature_names_to_select, target_names_to_select)

    if output_arrays_type == "pandas":
        return X, y, frame, None
    else:
        X, y = X.to_numpy(), y.to_numpy()

    categories = {
        name: dtype.categories.tolist()
        for name, dtype in frame.dtypes.items()
        if isinstance(dtype, pd.CategoricalDtype)
    }
    return X, y, None, categories


def load_arff_from_gzip_file(
    gzip_file,
    parser,
    output_type,
    openml_columns_info,
    feature_names_to_select,
    target_names_to_select,
    shape=None,
    read_csv_kwargs=None,
):
    """Load a compressed ARFF file using a given parser.

    Parameters
    ----------
    gzip_file : GzipFile instance
        The file compressed to be read.

    parser : {"pandas", "liac-arff"}
        The parser used to parse the ARFF file. "pandas" is recommended
        but only supports loading dense datasets.

    output_type : {"numpy", "sparse", "pandas"}
        The type of the arrays that will be returned. The possibilities ara:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    openml_columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        A list of the feature names to be selected.

    target_names_to_select : list of str
        A list of the target names to be selected.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv`. It allows to overwrite
        the default options.

    Returns
    -------
    X : {ndarray, sparse matrix, dataframe}
        The data matrix.

    y : {ndarray, dataframe, series}
        The target.

    frame : dataframe or None
        A dataframe containing both `X` and `y`. `None` if
        `output_array_type != "pandas"`.

    categories : list of str or None
        The names of the features that are categorical. `None` if
        `output_array_type == "pandas"`.
    """
    if parser == "liac-arff":
        return _liac_arff_parser(
            gzip_file,
            output_type,
            openml_columns_info,
            feature_names_to_select,
            target_names_to_select,
            shape,
        )
    elif parser == "pandas":
        return _pandas_arff_parser(
            gzip_file,
            output_type,
            openml_columns_info,
            feature_names_to_select,
            target_names_to_select,
            read_csv_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown parser: '{parser}'. Should be 'liac-arff' or 'pandas'."
        )
