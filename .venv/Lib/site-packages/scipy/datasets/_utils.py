import os
import shutil
from ._registry import method_files_map

try:
    import platformdirs
except ImportError:
    platformdirs = None  # type: ignore[assignment]


def _clear_cache(datasets, cache_dir=None, method_map=None):
    if method_map is None:
        # Use SciPy Datasets method map
        method_map = method_files_map
    if cache_dir is None:
        # Use default cache_dir path
        if platformdirs is None:
            # platformdirs is pooch dependency
            raise ImportError("Missing optional dependency 'pooch' required "
                              "for scipy.datasets module. Please use pip or "
                              "conda to install 'pooch'.")
        cache_dir = platformdirs.user_cache_dir("scipy-data")

    if not os.path.exists(cache_dir):
        print(f"Cache Directory {cache_dir} doesn't exist. Nothing to clear.")
        return

    if datasets is None:
        print(f"Cleaning the cache directory {cache_dir}!")
        shutil.rmtree(cache_dir)
    else:
        if not isinstance(datasets, (list, tuple)):
            # single dataset method passed should be converted to list
            datasets = [datasets, ]
        for dataset in datasets:
            assert callable(dataset)
            dataset_name = dataset.__name__  # Name of the dataset method
            if dataset_name not in method_map:
                raise ValueError(f"Dataset method {dataset_name} doesn't "
                                 "exist. Please check if the passed dataset "
                                 "is a subset of the following dataset "
                                 f"methods: {list(method_map.keys())}")

            data_files = method_map[dataset_name]
            data_filepaths = [os.path.join(cache_dir, file)
                              for file in data_files]
            for data_filepath in data_filepaths:
                if os.path.exists(data_filepath):
                    print("Cleaning the file "
                          f"{os.path.split(data_filepath)[1]} "
                          f"for dataset {dataset_name}")
                    os.remove(data_filepath)
                else:
                    print(f"Path {data_filepath} doesn't exist. "
                          "Nothing to clear.")


def clear_cache(datasets=None):
    """
    Cleans the scipy datasets cache directory.

    If a scipy.datasets method or a list/tuple of the same is
    provided, then clear_cache removes all the data files
    associated to the passed dataset method callable(s).

    By default, it removes all the cached data files.

    Parameters
    ----------
    datasets : callable or list/tuple of callable or None

    Examples
    --------
    >>> from scipy import datasets
    >>> ascent_array = datasets.ascent()
    >>> ascent_array.shape
    (512, 512)
    >>> datasets.clear_cache([datasets.ascent])
    Cleaning the file ascent.dat for dataset ascent
    """
    _clear_cache(datasets)
