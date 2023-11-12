from statsmodels.compat.python import lrange

from io import StringIO
from os import environ, makedirs
from os.path import abspath, dirname, exists, expanduser, join
import shutil
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
from pandas import Index, read_csv, read_stata


def webuse(data, baseurl='https://www.stata-press.com/data/r11/', as_df=True):
    """
    Download and return an example dataset from Stata.

    Parameters
    ----------
    data : str
        Name of dataset to fetch.
    baseurl : str
        The base URL to the stata datasets.
    as_df : bool
        Deprecated. Always returns a DataFrame

    Returns
    -------
    dta : DataFrame
        A DataFrame containing the Stata dataset.

    Examples
    --------
    >>> dta = webuse('auto')

    Notes
    -----
    Make sure baseurl has trailing forward slash. Does not do any
    error checking in response URLs.
    """
    url = urljoin(baseurl, data+'.dta')
    return read_stata(url)


class Dataset(dict):
    def __init__(self, **kw):
        # define some default attributes, so pylint can find them
        self.endog = None
        self.exog = None
        self.data = None
        self.names = None

        dict.__init__(self, kw)
        self.__dict__ = self
        # Some datasets have string variables. If you want a raw_data
        # attribute you must create this in the dataset's load function.
        try:  # some datasets have string variables
            self.raw_data = self.data.astype(float)
        except:
            pass

    def __repr__(self):
        return str(self.__class__)


def process_pandas(data, endog_idx=0, exog_idx=None, index_idx=None):
    names = data.columns

    if isinstance(endog_idx, int):
        endog_name = names[endog_idx]
        endog = data[endog_name].copy()
        if exog_idx is None:
            exog = data.drop([endog_name], axis=1)
        else:
            exog = data[names[exog_idx]].copy()
    else:
        endog = data.loc[:, endog_idx].copy()
        endog_name = list(endog.columns)
        if exog_idx is None:
            exog = data.drop(endog_name, axis=1)
        elif isinstance(exog_idx, int):
            exog = data[names[exog_idx]].copy()
        else:
            exog = data[names[exog_idx]].copy()

    if index_idx is not None:  # NOTE: will have to be improved for dates
        index = Index(data.iloc[:, index_idx])
        endog.index = index
        exog.index = index.copy()
        data = data.set_index(names[index_idx])

    exog_name = list(exog.columns)
    dataset = Dataset(data=data, names=list(names), endog=endog,
                      exog=exog, endog_name=endog_name, exog_name=exog_name)
    return dataset


def _maybe_reset_index(data):
    """
    All the Rdatasets have the integer row.labels from R if there is no
    real index. Strip this for a zero-based index
    """
    if data.index.equals(Index(lrange(1, len(data) + 1))):
        data = data.reset_index(drop=True)
    return data


def _get_cache(cache):
    if cache is False:
        # do not do any caching or load from cache
        cache = None
    elif cache is True:  # use default dir for cache
        cache = get_data_home(None)
    else:
        cache = get_data_home(cache)
    return cache


def _cache_it(data, cache_path):
    import zlib
    with open(cache_path, "wb") as zf:
        zf.write(zlib.compress(data))


def _open_cache(cache_path):
    import zlib
    # return as bytes object encoded in utf-8 for cross-compat of cached
    with open(cache_path, 'rb') as zf:
        return zlib.decompress(zf.read())


def _urlopen_cached(url, cache):
    """
    Tries to load data from cache location otherwise downloads it. If it
    downloads the data and cache is not None then it will put the downloaded
    data in the cache path.
    """
    from_cache = False
    if cache is not None:
        file_name = url.split("://")[-1].replace('/', ',')
        file_name = file_name.split('.')
        if len(file_name) > 1:
            file_name[-2] += '-v2'
        else:
            file_name[0] += '-v2'
        file_name = '.'.join(file_name) + ".zip"
        cache_path = join(cache, file_name)
        try:
            data = _open_cache(cache_path)
            from_cache = True
        except:
            pass

    # not using the cache or did not find it in cache
    if not from_cache:
        data = urlopen(url, timeout=3).read()
        if cache is not None:  # then put it in the cache
            _cache_it(data, cache_path)
    return data, from_cache


def _get_data(base_url, dataname, cache, extension="csv"):
    url = base_url + (dataname + ".%s") % extension
    try:
        data, from_cache = _urlopen_cached(url, cache)
    except HTTPError as err:
        if '404' in str(err):
            raise ValueError("Dataset %s was not found." % dataname)
        else:
            raise err

    data = data.decode('utf-8', 'strict')
    return StringIO(data), from_cache


def _get_dataset_meta(dataname, package, cache):
    # get the index, you'll probably want this cached because you have
    # to download info about all the data to get info about any of the data...
    index_url = ("https://raw.githubusercontent.com/vincentarelbundock/"
                 "Rdatasets/master/datasets.csv")
    data, _ = _urlopen_cached(index_url, cache)
    data = data.decode('utf-8', 'strict')
    index = read_csv(StringIO(data))
    idx = np.logical_and(index.Item == dataname, index.Package == package)
    if not idx.any():
        raise ValueError(
            f"Item {dataname} from Package {package} was not found. Check "
            f"the CSV file at {index_url} to verify the Item and Package."
        )
    dataset_meta = index.loc[idx]
    return dataset_meta["Title"].iloc[0]


def get_rdataset(dataname, package="datasets", cache=False):
    """download and return R dataset

    Parameters
    ----------
    dataname : str
        The name of the dataset you want to download
    package : str
        The package in which the dataset is found. The default is the core
        'datasets' package.
    cache : bool or str
        If True, will download this data into the STATSMODELS_DATA folder.
        The default location is a folder called statsmodels_data in the
        user home folder. Otherwise, you can specify a path to a folder to
        use for caching the data. If False, the data will not be cached.

    Returns
    -------
    dataset : Dataset
        A `statsmodels.data.utils.Dataset` instance. This objects has
        attributes:

        * data - A pandas DataFrame containing the data
        * title - The dataset title
        * package - The package from which the data came
        * from_cache - Whether not cached data was retrieved
        * __doc__ - The verbatim R documentation.

    Notes
    -----
    If the R dataset has an integer index. This is reset to be zero-based.
    Otherwise the index is preserved. The caching facilities are dumb. That
    is, no download dates, e-tags, or otherwise identifying information
    is checked to see if the data should be downloaded again or not. If the
    dataset is in the cache, it's used.
    """
    # NOTE: use raw github bc html site might not be most up to date
    data_base_url = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
                     "master/csv/"+package+"/")
    docs_base_url = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
                     "master/doc/"+package+"/rst/")
    cache = _get_cache(cache)
    data, from_cache = _get_data(data_base_url, dataname, cache)
    data = read_csv(data, index_col=0)
    data = _maybe_reset_index(data)

    title = _get_dataset_meta(dataname, package, cache)
    doc, _ = _get_data(docs_base_url, dataname, cache, "rst")

    return Dataset(data=data, __doc__=doc.read(), package=package, title=title,
                   from_cache=from_cache)

# The below function were taken from sklearn


def get_data_home(data_home=None):
    """Return the path of the statsmodels data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'statsmodels_data'
    in the user home folder.

    Alternatively, it can be set by the 'STATSMODELS_DATA' environment
    variable or programatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = environ.get('STATSMODELS_DATA',
                                join('~', 'statsmodels_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def clear_data_home(data_home=None):
    """Delete all the content of the data home cache."""
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def check_internet(url=None):
    """Check if internet is available"""
    url = "https://github.com" if url is None else url
    try:
        urlopen(url)
    except URLError as err:
        return False
    return True


def strip_column_names(df):
    """
    Remove leading and trailing single quotes

    Parameters
    ----------
    df : DataFrame
        DataFrame to process

    Returns
    -------
    df : DataFrame
        DataFrame with stripped column names

    Notes
    -----
    In-place modification
    """
    columns = []
    for c in df:
        if c.startswith('\'') and c.endswith('\''):
            c = c[1:-1]
        elif c.startswith('\''):
            c = c[1:]
        elif c.endswith('\''):
            c = c[:-1]
        columns.append(c)
    df.columns = columns
    return df


def load_csv(base_file, csv_name, sep=',', convert_float=False):
    """Standard simple csv loader"""
    filepath = dirname(abspath(base_file))
    filename = join(filepath,csv_name)
    engine = 'python' if sep != ',' else 'c'
    float_precision = {}
    if engine == 'c':
        float_precision = {'float_precision': 'high'}
    data = read_csv(filename, sep=sep, engine=engine, **float_precision)
    if convert_float:
        data = data.astype(float)
    return data
