"""
Pyodide and other single-threaded Python builds will be missing the
_multiprocessing module. Test that joblib still works in this environment.
"""

import os
import subprocess
import sys


def test_missing_multiprocessing(tmp_path):
    """
    Test that import joblib works even if _multiprocessing is missing.

    pytest has already imported everything from joblib. The most reasonable way
    to test importing joblib with modified environment is to invoke a separate
    Python process. This also ensures that we don't break other tests by
    importing a bad `_multiprocessing` module.
    """
    (tmp_path / "_multiprocessing.py").write_text(
        'raise ImportError("No _multiprocessing module!")'
    )
    env = dict(os.environ)
    # For subprocess, use current sys.path with our custom version of
    # multiprocessing inserted.
    env["PYTHONPATH"] = ":".join([str(tmp_path)] + sys.path)
    subprocess.check_call(
        [sys.executable, "-c",
         "import joblib, math; "
         "joblib.Parallel(n_jobs=1)("
         "joblib.delayed(math.sqrt)(i**2) for i in range(10))"
         ], env=env)
