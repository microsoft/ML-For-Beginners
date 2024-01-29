import gzip
import hashlib
import json
import os
import shutil
import time
from contextlib import closing
from functools import wraps
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from warnings import warn

import numpy as np

from ..utils import (
    Bunch,
    check_pandas_support,  # noqa  # noqa
)
from ..utils._param_validation import (
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_params,
)
from . import get_data_home
from ._arff_parser import load_arff_from_gzip_file

__all__ = ["fetch_openml"]

_OPENML_PREFIX = "https://api.openml.org/"
_SEARCH_NAME = "api/v1/json/data/list/data_name/{}/limit/2"
_DATA_INFO = "api/v1/json/data/{}"
_DATA_FEATURES = "api/v1/json/data/features/{}"
_DATA_QUALITIES = "api/v1/json/data/qualities/{}"
_DATA_FILE = "data/v1/download/{}"

OpenmlQualitiesType = List[Dict[str, str]]
OpenmlFeaturesType = List[Dict[str, str]]


def _get_local_path(openml_path: str, data_home: str) -> str:
    return os.path.join(data_home, "openml.org", openml_path + ".gz")


def _retry_with_clean_cache(
    openml_path: str,
    data_home: Optional[str],
    no_retry_exception: Optional[Exception] = None,
) -> Callable:
    """If the first call to the decorated function fails, the local cached
    file is removed, and the function is called again. If ``data_home`` is
    ``None``, then the function is called once. We can provide a specific
    exception to not retry on using `no_retry_exception` parameter.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw):
            if data_home is None:
                return f(*args, **kw)
            try:
                return f(*args, **kw)
            except URLError:
                raise
            except Exception as exc:
                if no_retry_exception is not None and isinstance(
                    exc, no_retry_exception
                ):
                    raise
                warn("Invalid cache, redownloading file", RuntimeWarning)
                local_path = _get_local_path(openml_path, data_home)
                if os.path.exists(local_path):
                    os.unlink(local_path)
                return f(*args, **kw)

        return wrapper

    return decorator


def _retry_on_network_error(
    n_retries: int = 3, delay: float = 1.0, url: str = ""
) -> Callable:
    """If the function call results in a network error, call the function again
    up to ``n_retries`` times with a ``delay`` between each call. If the error
    has a 412 status code, don't call the function again as this is a specific
    OpenML error.
    The url parameter is used to give more information to the user about the
    error.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            retry_counter = n_retries
            while True:
                try:
                    return f(*args, **kwargs)
                except (URLError, TimeoutError) as e:
                    # 412 is a specific OpenML error code.
                    if isinstance(e, HTTPError) and e.code == 412:
                        raise
                    if retry_counter == 0:
                        raise
                    warn(
                        f"A network error occurred while downloading {url}. Retrying..."
                    )
                    retry_counter -= 1
                    time.sleep(delay)

        return wrapper

    return decorator


def _open_openml_url(
    openml_path: str, data_home: Optional[str], n_retries: int = 3, delay: float = 1.0
):
    """
    Returns a resource from OpenML.org. Caches it to data_home if required.

    Parameters
    ----------
    openml_path : str
        OpenML URL that will be accessed. This will be prefixes with
        _OPENML_PREFIX.

    data_home : str
        Directory to which the files will be cached. If None, no caching will
        be applied.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    result : stream
        A stream to the OpenML resource.
    """

    def is_gzip_encoded(_fsrc):
        return _fsrc.info().get("Content-Encoding", "") == "gzip"

    req = Request(_OPENML_PREFIX + openml_path)
    req.add_header("Accept-encoding", "gzip")

    if data_home is None:
        fsrc = _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(req)
        if is_gzip_encoded(fsrc):
            return gzip.GzipFile(fileobj=fsrc, mode="rb")
        return fsrc

    local_path = _get_local_path(openml_path, data_home)
    dir_name, file_name = os.path.split(local_path)
    if not os.path.exists(local_path):
        os.makedirs(dir_name, exist_ok=True)
        try:
            # Create a tmpdir as a subfolder of dir_name where the final file will
            # be moved to if the download is successful. This guarantees that the
            # renaming operation to the final location is atomic to ensure the
            # concurrence safety of the dataset caching mechanism.
            with TemporaryDirectory(dir=dir_name) as tmpdir:
                with closing(
                    _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(
                        req
                    )
                ) as fsrc:
                    opener: Callable
                    if is_gzip_encoded(fsrc):
                        opener = open
                    else:
                        opener = gzip.GzipFile
                    with opener(os.path.join(tmpdir, file_name), "wb") as fdst:
                        shutil.copyfileobj(fsrc, fdst)
                shutil.move(fdst.name, local_path)
        except Exception:
            if os.path.exists(local_path):
                os.unlink(local_path)
            raise

    # XXX: First time, decompression will not be necessary (by using fsrc), but
    # it will happen nonetheless
    return gzip.GzipFile(local_path, "rb")


class OpenMLError(ValueError):
    """HTTP 412 is a specific OpenML error code, indicating a generic error"""

    pass


def _get_json_content_from_openml_api(
    url: str,
    error_message: Optional[str],
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
) -> Dict:
    """
    Loads json data from the openml api.

    Parameters
    ----------
    url : str
        The URL to load from. Should be an official OpenML endpoint.

    error_message : str or None
        The error message to raise if an acceptable OpenML error is thrown
        (acceptable error is, e.g., data id not found. Other errors, like 404's
        will throw the native error message).

    data_home : str or None
        Location to cache the response. None if no cache is required.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    json_data : json
        the json result from the OpenML server if the call was successful.
        An exception otherwise.
    """

    @_retry_with_clean_cache(url, data_home=data_home)
    def _load_json():
        with closing(
            _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return _load_json()
    except HTTPError as error:
        # 412 is an OpenML specific error code, indicating a generic error
        # (e.g., data not found)
        if error.code != 412:
            raise error

    # 412 error, not in except for nicer traceback
    raise OpenMLError(error_message)


def _get_data_info_by_name(
    name: str,
    version: Union[int, str],
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
):
    """
    Utilizes the openml dataset listing api to find a dataset by
    name/version
    OpenML api function:
    https://www.openml.org/api_docs#!/data/get_data_list_data_name_data_name

    Parameters
    ----------
    name : str
        name of the dataset

    version : int or str
        If version is an integer, the exact name/version will be obtained from
        OpenML. If version is a string (value: "active") it will take the first
        version from OpenML that is annotated as active. Any other string
        values except "active" are treated as integer.

    data_home : str or None
        Location to cache the response. None if no cache is required.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    Returns
    -------
    first_dataset : json
        json representation of the first dataset object that adhired to the
        search criteria

    """
    if version == "active":
        # situation in which we return the oldest active version
        url = _SEARCH_NAME.format(name) + "/status/active/"
        error_msg = "No active dataset {} found.".format(name)
        json_data = _get_json_content_from_openml_api(
            url,
            error_msg,
            data_home=data_home,
            n_retries=n_retries,
            delay=delay,
        )
        res = json_data["data"]["dataset"]
        if len(res) > 1:
            first_version = version = res[0]["version"]
            warning_msg = (
                "Multiple active versions of the dataset matching the name"
                f" {name} exist. Versions may be fundamentally different, "
                f"returning version {first_version}. "
                "Available versions:\n"
            )
            for r in res:
                warning_msg += f"- version {r['version']}, status: {r['status']}\n"
                warning_msg += (
                    f"  url: https://www.openml.org/search?type=data&id={r['did']}\n"
                )
            warn(warning_msg)
        return res[0]

    # an integer version has been provided
    url = (_SEARCH_NAME + "/data_version/{}").format(name, version)
    try:
        json_data = _get_json_content_from_openml_api(
            url,
            error_message=None,
            data_home=data_home,
            n_retries=n_retries,
            delay=delay,
        )
    except OpenMLError:
        # we can do this in 1 function call if OpenML does not require the
        # specification of the dataset status (i.e., return datasets with a
        # given name / version regardless of active, deactivated, etc. )
        # TODO: feature request OpenML.
        url += "/status/deactivated"
        error_msg = "Dataset {} with version {} not found.".format(name, version)
        json_data = _get_json_content_from_openml_api(
            url,
            error_msg,
            data_home=data_home,
            n_retries=n_retries,
            delay=delay,
        )

    return json_data["data"]["dataset"][0]


def _get_data_description_by_id(
    data_id: int,
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
) -> Dict[str, Any]:
    # OpenML API function: https://www.openml.org/api_docs#!/data/get_data_id
    url = _DATA_INFO.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(
        url,
        error_message,
        data_home=data_home,
        n_retries=n_retries,
        delay=delay,
    )
    return json_data["data_set_description"]


def _get_data_features(
    data_id: int,
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
) -> OpenmlFeaturesType:
    # OpenML function:
    # https://www.openml.org/api_docs#!/data/get_data_features_id
    url = _DATA_FEATURES.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(
        url,
        error_message,
        data_home=data_home,
        n_retries=n_retries,
        delay=delay,
    )
    return json_data["data_features"]["feature"]


def _get_data_qualities(
    data_id: int,
    data_home: Optional[str],
    n_retries: int = 3,
    delay: float = 1.0,
) -> OpenmlQualitiesType:
    # OpenML API function:
    # https://www.openml.org/api_docs#!/data/get_data_qualities_id
    url = _DATA_QUALITIES.format(data_id)
    error_message = "Dataset with data_id {} not found.".format(data_id)
    json_data = _get_json_content_from_openml_api(
        url,
        error_message,
        data_home=data_home,
        n_retries=n_retries,
        delay=delay,
    )
    # the qualities might not be available, but we still try to process
    # the data
    return json_data.get("data_qualities", {}).get("quality", [])


def _get_num_samples(data_qualities: OpenmlQualitiesType) -> int:
    """Get the number of samples from data qualities.

    Parameters
    ----------
    data_qualities : list of dict
        Used to retrieve the number of instances (samples) in the dataset.

    Returns
    -------
    n_samples : int
        The number of samples in the dataset or -1 if data qualities are
        unavailable.
    """
    # If the data qualities are unavailable, we return -1
    default_n_samples = -1

    qualities = {d["name"]: d["value"] for d in data_qualities}
    return int(float(qualities.get("NumberOfInstances", default_n_samples)))


def _load_arff_response(
    url: str,
    data_home: Optional[str],
    parser: str,
    output_type: str,
    openml_columns_info: dict,
    feature_names_to_select: List[str],
    target_names_to_select: List[str],
    shape: Optional[Tuple[int, int]],
    md5_checksum: str,
    n_retries: int = 3,
    delay: float = 1.0,
    read_csv_kwargs: Optional[Dict] = None,
):
    """Load the ARFF data associated with the OpenML URL.

    In addition of loading the data, this function will also check the
    integrity of the downloaded file from OpenML using MD5 checksum.

    Parameters
    ----------
    url : str
        The URL of the ARFF file on OpenML.

    data_home : str
        The location where to cache the data.

    parser : {"liac-arff", "pandas"}
        The parser used to parse the ARFF file.

    output_type : {"numpy", "pandas", "sparse"}
        The type of the arrays that will be returned. The possibilities are:

        - `"numpy"`: both `X` and `y` will be NumPy arrays;
        - `"sparse"`: `X` will be sparse matrix and `y` will be a NumPy array;
        - `"pandas"`: `X` will be a pandas DataFrame and `y` will be either a
          pandas Series or DataFrame.

    openml_columns_info : dict
        The information provided by OpenML regarding the columns of the ARFF
        file.

    feature_names_to_select : list of str
        The list of the features to be selected.

    target_names_to_select : list of str
        The list of the target variables to be selected.

    shape : tuple or None
        With `parser="liac-arff"`, when using a generator to load the data,
        one needs to provide the shape of the data beforehand.

    md5_checksum : str
        The MD5 checksum provided by OpenML to check the data integrity.

    n_retries : int, default=3
        The number of times to retry downloading the data if it fails.

    delay : float, default=1.0
        The delay between two consecutive downloads in seconds.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv` when using the pandas parser.
        It allows to overwrite the default options.

        .. versionadded:: 1.3

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
    gzip_file = _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
    with closing(gzip_file):
        md5 = hashlib.md5()
        for chunk in iter(lambda: gzip_file.read(4096), b""):
            md5.update(chunk)
        actual_md5_checksum = md5.hexdigest()

    if actual_md5_checksum != md5_checksum:
        raise ValueError(
            f"md5 checksum of local file for {url} does not match description: "
            f"expected: {md5_checksum} but got {actual_md5_checksum}. "
            "Downloaded file could have been modified / corrupted, clean cache "
            "and retry..."
        )

    def _open_url_and_load_gzip_file(url, data_home, n_retries, delay, arff_params):
        gzip_file = _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
        with closing(gzip_file):
            return load_arff_from_gzip_file(gzip_file, **arff_params)

    arff_params: Dict = dict(
        parser=parser,
        output_type=output_type,
        openml_columns_info=openml_columns_info,
        feature_names_to_select=feature_names_to_select,
        target_names_to_select=target_names_to_select,
        shape=shape,
        read_csv_kwargs=read_csv_kwargs or {},
    )
    try:
        X, y, frame, categories = _open_url_and_load_gzip_file(
            url, data_home, n_retries, delay, arff_params
        )
    except Exception as exc:
        if parser != "pandas":
            raise

        from pandas.errors import ParserError

        if not isinstance(exc, ParserError):
            raise

        # A parsing error could come from providing the wrong quotechar
        # to pandas. By default, we use a double quote. Thus, we retry
        # with a single quote before to raise the error.
        arff_params["read_csv_kwargs"].update(quotechar="'")
        X, y, frame, categories = _open_url_and_load_gzip_file(
            url, data_home, n_retries, delay, arff_params
        )

    return X, y, frame, categories


def _download_data_to_bunch(
    url: str,
    sparse: bool,
    data_home: Optional[str],
    *,
    as_frame: bool,
    openml_columns_info: List[dict],
    data_columns: List[str],
    target_columns: List[str],
    shape: Optional[Tuple[int, int]],
    md5_checksum: str,
    n_retries: int = 3,
    delay: float = 1.0,
    parser: str,
    read_csv_kwargs: Optional[Dict] = None,
):
    """Download ARFF data, load it to a specific container and create to Bunch.

    This function has a mechanism to retry/cache/clean the data.

    Parameters
    ----------
    url : str
        The URL of the ARFF file on OpenML.

    sparse : bool
        Whether the dataset is expected to use the sparse ARFF format.

    data_home : str
        The location where to cache the data.

    as_frame : bool
        Whether or not to return the data into a pandas DataFrame.

    openml_columns_info : list of dict
        The information regarding the columns provided by OpenML for the
        ARFF dataset. The information is stored as a list of dictionaries.

    data_columns : list of str
        The list of the features to be selected.

    target_columns : list of str
        The list of the target variables to be selected.

    shape : tuple or None
        With `parser="liac-arff"`, when using a generator to load the data,
        one needs to provide the shape of the data beforehand.

    md5_checksum : str
        The MD5 checksum provided by OpenML to check the data integrity.

    n_retries : int, default=3
        Number of retries when HTTP errors are encountered. Error with status
        code 412 won't be retried as they represent OpenML generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    parser : {"liac-arff", "pandas"}
        The parser used to parse the ARFF file.

    read_csv_kwargs : dict, default=None
        Keyword arguments to pass to `pandas.read_csv` when using the pandas parser.
        It allows to overwrite the default options.

        .. versionadded:: 1.3

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

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
    # Prepare which columns and data types should be returned for the X and y
    features_dict = {feature["name"]: feature for feature in openml_columns_info}

    if sparse:
        output_type = "sparse"
    elif as_frame:
        output_type = "pandas"
    else:
        output_type = "numpy"

    # XXX: target columns should all be categorical or all numeric
    _verify_target_data_type(features_dict, target_columns)
    for name in target_columns:
        column_info = features_dict[name]
        n_missing_values = int(column_info["number_of_missing_values"])
        if n_missing_values > 0:
            raise ValueError(
                f"Target column '{column_info['name']}' has {n_missing_values} missing "
                "values. Missing values are not supported for target columns."
            )

    no_retry_exception = None
    if parser == "pandas":
        # If we get a ParserError with pandas, then we don't want to retry and we raise
        # early.
        from pandas.errors import ParserError

        no_retry_exception = ParserError

    X, y, frame, categories = _retry_with_clean_cache(
        url, data_home, no_retry_exception
    )(_load_arff_response)(
        url,
        data_home,
        parser=parser,
        output_type=output_type,
        openml_columns_info=features_dict,
        feature_names_to_select=data_columns,
        target_names_to_select=target_columns,
        shape=shape,
        md5_checksum=md5_checksum,
        n_retries=n_retries,
        delay=delay,
        read_csv_kwargs=read_csv_kwargs,
    )

    return Bunch(
        data=X,
        target=y,
        frame=frame,
        categories=categories,
        feature_names=data_columns,
        target_names=target_columns,
    )


def _verify_target_data_type(features_dict, target_columns):
    # verifies the data type of the y array in case there are multiple targets
    # (throws an error if these targets do not comply with sklearn support)
    if not isinstance(target_columns, list):
        raise ValueError("target_column should be list, got: %s" % type(target_columns))
    found_types = set()
    for target_column in target_columns:
        if target_column not in features_dict:
            raise KeyError(f"Could not find target_column='{target_column}'")
        if features_dict[target_column]["data_type"] == "numeric":
            found_types.add(np.float64)
        else:
            found_types.add(object)

        # note: we compare to a string, not boolean
        if features_dict[target_column]["is_ignore"] == "true":
            warn(f"target_column='{target_column}' has flag is_ignore.")
        if features_dict[target_column]["is_row_identifier"] == "true":
            warn(f"target_column='{target_column}' has flag is_row_identifier.")
    if len(found_types) > 1:
        raise ValueError(
            "Can only handle homogeneous multi-target datasets, "
            "i.e., all targets are either numeric or "
            "categorical."
        )


def _valid_data_column_names(features_list, target_columns):
    # logic for determining on which columns can be learned. Note that from the
    # OpenML guide follows that columns that have the `is_row_identifier` or
    # `is_ignore` flag, these can not be learned on. Also target columns are
    # excluded.
    valid_data_column_names = []
    for feature in features_list:
        if (
            feature["name"] not in target_columns
            and feature["is_ignore"] != "true"
            and feature["is_row_identifier"] != "true"
        ):
            valid_data_column_names.append(feature["name"])
    return valid_data_column_names


@validate_params(
    {
        "name": [str, None],
        "version": [Interval(Integral, 1, None, closed="left"), StrOptions({"active"})],
        "data_id": [Interval(Integral, 1, None, closed="left"), None],
        "data_home": [str, os.PathLike, None],
        "target_column": [str, list, None],
        "cache": [bool],
        "return_X_y": [bool],
        "as_frame": [bool, StrOptions({"auto"})],
        "n_retries": [Interval(Integral, 1, None, closed="left")],
        "delay": [Interval(Real, 0, None, closed="right")],
        "parser": [
            StrOptions({"auto", "pandas", "liac-arff"}),
        ],
        "read_csv_kwargs": [dict, None],
    },
    prefer_skip_nested_validation=True,
)
def fetch_openml(
    name: Optional[str] = None,
    *,
    version: Union[str, int] = "active",
    data_id: Optional[int] = None,
    data_home: Optional[Union[str, os.PathLike]] = None,
    target_column: Optional[Union[str, List]] = "default-target",
    cache: bool = True,
    return_X_y: bool = False,
    as_frame: Union[str, bool] = "auto",
    n_retries: int = 3,
    delay: float = 1.0,
    parser: str = "auto",
    read_csv_kwargs: Optional[Dict] = None,
):
    """Fetch dataset from openml by name or dataset id.

    Datasets are uniquely identified by either an integer ID or by a
    combination of name and version (i.e. there might be multiple
    versions of the 'iris' dataset). Please give either name or data_id
    (not both). In case a name is given, a version can also be
    provided.

    Read more in the :ref:`User Guide <openml>`.

    .. versionadded:: 0.20

    .. note:: EXPERIMENTAL

        The API is experimental (particularly the return value structure),
        and might have small backward-incompatible changes without notice
        or warning in future releases.

    Parameters
    ----------
    name : str, default=None
        String identifier of the dataset. Note that OpenML can have multiple
        datasets with the same name.

    version : int or 'active', default='active'
        Version of the dataset. Can only be provided if also ``name`` is given.
        If 'active' the oldest version that's still active is used. Since
        there may be more than one active version of a dataset, and those
        versions may fundamentally be different from one another, setting an
        exact version is highly recommended.

    data_id : int, default=None
        OpenML ID of the dataset. The most specific way of retrieving a
        dataset. If data_id is not given, name (and potential version) are
        used to obtain a dataset.

    data_home : str or path-like, default=None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    target_column : str, list or None, default='default-target'
        Specify the column name in the data to use as target. If
        'default-target', the standard target column a stored on the server
        is used. If ``None``, all columns are returned as data and the
        target is ``None``. If list (of strings), all columns with these names
        are returned as multi-target (Note: not all scikit-learn classifiers
        can handle all types of multi-output combinations).

    cache : bool, default=True
        Whether to cache the downloaded datasets into `data_home`.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` objects.

    as_frame : bool or 'auto', default='auto'
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.

        If `as_frame` is 'auto', the data and target will be converted to
        DataFrame or Series as if `as_frame` is set to True, unless the dataset
        is stored in sparse format.

        If `as_frame` is False, the data and target will be NumPy arrays and
        the `data` will only contain numerical values when `parser="liac-arff"`
        where the categories are provided in the attribute `categories` of the
        `Bunch` instance. When `parser="pandas"`, no ordinal encoding is made.

        .. versionchanged:: 0.24
           The default value of `as_frame` changed from `False` to `'auto'`
           in 0.24.

    n_retries : int, default=3
        Number of retries when HTTP errors or network timeouts are encountered.
        Error with status code 412 won't be retried as they represent OpenML
        generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    parser : {"auto", "pandas", "liac-arff"}, default="auto"
        Parser used to load the ARFF file. Two parsers are implemented:

        - `"pandas"`: this is the most efficient parser. However, it requires
          pandas to be installed and can only open dense datasets.
        - `"liac-arff"`: this is a pure Python ARFF parser that is much less
          memory- and CPU-efficient. It deals with sparse ARFF datasets.

        If `"auto"`, the parser is chosen automatically such that `"liac-arff"`
        is selected for sparse ARFF datasets, otherwise `"pandas"` is selected.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `parser` changes from `"liac-arff"` to
           `"auto"`.

    read_csv_kwargs : dict, default=None
        Keyword arguments passed to :func:`pandas.read_csv` when loading the data
        from a ARFF file and using the pandas parser. It can allow to
        overwrite some default parameters.

        .. versionadded:: 1.3

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : np.array, scipy.sparse.csr_matrix of floats, or pandas DataFrame
            The feature matrix. Categorical features are encoded as ordinals.
        target : np.array, pandas Series or DataFrame
            The regression target or classification labels, if applicable.
            Dtype is float if numeric, and object if categorical. If
            ``as_frame`` is True, ``target`` is a pandas object.
        DESCR : str
            The full description of the dataset.
        feature_names : list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.

        .. versionadded:: 0.22

        categories : dict or None
            Maps each categorical feature name to a list of values, such
            that the value encoded as i is ith in the list. If ``as_frame``
            is True, this is None.
        details : dict
            More metadata from OpenML.
        frame : pandas DataFrame
            Only present when `as_frame=True`. DataFrame with ``data`` and
            ``target``.

    (data, target) : tuple if ``return_X_y`` is True

        .. note:: EXPERIMENTAL

            This interface is **experimental** and subsequent releases may
            change attributes without notice (although there should only be
            minor changes to ``data`` and ``target``).

        Missing values in the 'data' are represented as NaN's. Missing values
        in 'target' are represented as NaN's (numerical target) or None
        (categorical target).

    Notes
    -----
    The `"pandas"` and `"liac-arff"` parsers can lead to different data types
    in the output. The notable differences are the following:

    - The `"liac-arff"` parser always encodes categorical features as `str` objects.
      To the contrary, the `"pandas"` parser instead infers the type while
      reading and numerical categories will be casted into integers whenever
      possible.
    - The `"liac-arff"` parser uses float64 to encode numerical features
      tagged as 'REAL' and 'NUMERICAL' in the metadata. The `"pandas"`
      parser instead infers if these numerical features corresponds
      to integers and uses panda's Integer extension dtype.
    - In particular, classification datasets with integer categories are
      typically loaded as such `(0, 1, ...)` with the `"pandas"` parser while
      `"liac-arff"` will force the use of string encoded class labels such as
      `"0"`, `"1"` and so on.
    - The `"pandas"` parser will not strip single quotes - i.e. `'` - from
      string columns. For instance, a string `'my string'` will be kept as is
      while the `"liac-arff"` parser will strip the single quotes. For
      categorical columns, the single quotes are stripped from the values.

    In addition, when `as_frame=False` is used, the `"liac-arff"` parser
    returns ordinally encoded data where the categories are provided in the
    attribute `categories` of the `Bunch` instance. Instead, `"pandas"` returns
    a NumPy array were the categories are not encoded.
    """
    if cache is False:
        # no caching will be applied
        data_home = None
    else:
        data_home = get_data_home(data_home=data_home)
        data_home = join(str(data_home), "openml")

    # check valid function arguments. data_id XOR (name, version) should be
    # provided
    if name is not None:
        # OpenML is case-insensitive, but the caching mechanism is not
        # convert all data names (str) to lower case
        name = name.lower()
        if data_id is not None:
            raise ValueError(
                "Dataset data_id={} and name={} passed, but you can only "
                "specify a numeric data_id or a name, not "
                "both.".format(data_id, name)
            )
        data_info = _get_data_info_by_name(
            name, version, data_home, n_retries=n_retries, delay=delay
        )
        data_id = data_info["did"]
    elif data_id is not None:
        # from the previous if statement, it is given that name is None
        if version != "active":
            raise ValueError(
                "Dataset data_id={} and version={} passed, but you can only "
                "specify a numeric data_id or a version, not "
                "both.".format(data_id, version)
            )
    else:
        raise ValueError(
            "Neither name nor data_id are provided. Please provide name or data_id."
        )

    data_description = _get_data_description_by_id(data_id, data_home)
    if data_description["status"] != "active":
        warn(
            "Version {} of dataset {} is inactive, meaning that issues have "
            "been found in the dataset. Try using a newer version from "
            "this URL: {}".format(
                data_description["version"],
                data_description["name"],
                data_description["url"],
            )
        )
    if "error" in data_description:
        warn(
            "OpenML registered a problem with the dataset. It might be "
            "unusable. Error: {}".format(data_description["error"])
        )
    if "warning" in data_description:
        warn(
            "OpenML raised a warning on the dataset. It might be "
            "unusable. Warning: {}".format(data_description["warning"])
        )

    return_sparse = data_description["format"].lower() == "sparse_arff"
    as_frame = not return_sparse if as_frame == "auto" else as_frame
    if parser == "auto":
        parser_ = "liac-arff" if return_sparse else "pandas"
    else:
        parser_ = parser

    if parser_ == "pandas":
        try:
            check_pandas_support("`fetch_openml`")
        except ImportError as exc:
            if as_frame:
                err_msg = (
                    "Returning pandas objects requires pandas to be installed. "
                    "Alternatively, explicitly set `as_frame=False` and "
                    "`parser='liac-arff'`."
                )
            else:
                err_msg = (
                    f"Using `parser={parser!r}` wit dense data requires pandas to be "
                    "installed. Alternatively, explicitly set `parser='liac-arff'`."
                )
            raise ImportError(err_msg) from exc

    if return_sparse:
        if as_frame:
            raise ValueError(
                "Sparse ARFF datasets cannot be loaded with as_frame=True. "
                "Use as_frame=False or as_frame='auto' instead."
            )
        if parser_ == "pandas":
            raise ValueError(
                f"Sparse ARFF datasets cannot be loaded with parser={parser!r}. "
                "Use parser='liac-arff' or parser='auto' instead."
            )

    # download data features, meta-info about column types
    features_list = _get_data_features(data_id, data_home)

    if not as_frame:
        for feature in features_list:
            if "true" in (feature["is_ignore"], feature["is_row_identifier"]):
                continue
            if feature["data_type"] == "string":
                raise ValueError(
                    "STRING attributes are not supported for "
                    "array representation. Try as_frame=True"
                )

    if target_column == "default-target":
        # determines the default target based on the data feature results
        # (which is currently more reliable than the data description;
        # see issue: https://github.com/openml/OpenML/issues/768)
        target_columns = [
            feature["name"]
            for feature in features_list
            if feature["is_target"] == "true"
        ]
    elif isinstance(target_column, str):
        # for code-simplicity, make target_column by default a list
        target_columns = [target_column]
    elif target_column is None:
        target_columns = []
    else:
        # target_column already is of type list
        target_columns = target_column
    data_columns = _valid_data_column_names(features_list, target_columns)

    shape: Optional[Tuple[int, int]]
    # determine arff encoding to return
    if not return_sparse:
        # The shape must include the ignored features to keep the right indexes
        # during the arff data conversion.
        data_qualities = _get_data_qualities(data_id, data_home)
        shape = _get_num_samples(data_qualities), len(features_list)
    else:
        shape = None

    # obtain the data
    url = _DATA_FILE.format(data_description["file_id"])
    bunch = _download_data_to_bunch(
        url,
        return_sparse,
        data_home,
        as_frame=bool(as_frame),
        openml_columns_info=features_list,
        shape=shape,
        target_columns=target_columns,
        data_columns=data_columns,
        md5_checksum=data_description["md5_checksum"],
        n_retries=n_retries,
        delay=delay,
        parser=parser_,
        read_csv_kwargs=read_csv_kwargs,
    )

    if return_X_y:
        return bunch.data, bunch.target

    description = "{}\n\nDownloaded from openml.org.".format(
        data_description.pop("description")
    )

    bunch.update(
        DESCR=description,
        details=data_description,
        url="https://www.openml.org/d/{}".format(data_id),
    )

    return bunch
