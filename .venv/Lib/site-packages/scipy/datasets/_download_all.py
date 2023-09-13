"""
Platform independent script to download all the
`scipy.datasets` module data files.
This doesn't require a full scipy build.

Run: python _download_all.py <download_dir>
"""

import argparse
try:
    import pooch
except ImportError:
    pooch = None


if __package__ is None or __package__ == '':
    # Running as python script, use absolute import
    import _registry  # type: ignore
else:
    # Running as python module, use relative import
    from . import _registry


def download_all(path=None):
    """
    Utility method to download all the dataset files
    for `scipy.datasets` module.

    Parameters
    ----------
    path : str, optional
        Directory path to download all the dataset files.
        If None, default to the system cache_dir detected by pooch.
    """
    if pooch is None:
        raise ImportError("Missing optional dependency 'pooch' required "
                          "for scipy.datasets module. Please use pip or "
                          "conda to install 'pooch'.")
    if path is None:
        path = pooch.os_cache('scipy-data')
    for dataset_name, dataset_hash in _registry.registry.items():
        pooch.retrieve(url=_registry.registry_urls[dataset_name],
                       known_hash=dataset_hash,
                       fname=dataset_name, path=path)


def main():
    parser = argparse.ArgumentParser(description='Download SciPy data files.')
    parser.add_argument("path", nargs='?', type=str,
                        default=pooch.os_cache('scipy-data'),
                        help="Directory path to download all the data files.")
    args = parser.parse_args()
    download_all(args.path)


if __name__ == "__main__":
    main()
