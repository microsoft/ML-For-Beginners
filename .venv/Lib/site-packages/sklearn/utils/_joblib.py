import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore")
    # joblib imports may raise DeprecationWarning on certain Python
    # versions
    import joblib
    from joblib import (
        Memory,
        Parallel,
        __version__,
        cpu_count,
        delayed,
        dump,
        effective_n_jobs,
        hash,
        load,
        logger,
        parallel_backend,
        register_parallel_backend,
    )


__all__ = [
    "parallel_backend",
    "register_parallel_backend",
    "cpu_count",
    "Parallel",
    "Memory",
    "delayed",
    "effective_n_jobs",
    "hash",
    "logger",
    "dump",
    "load",
    "joblib",
    "__version__",
]
