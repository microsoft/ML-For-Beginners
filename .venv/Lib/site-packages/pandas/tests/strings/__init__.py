import numpy as np

import pandas as pd

object_pyarrow_numpy = ("object", "string[pyarrow_numpy]")


def _convert_na_value(ser, expected):
    if ser.dtype != object:
        if ser.dtype.storage == "pyarrow_numpy":
            expected = expected.fillna(np.nan)
        else:
            # GH#18463
            expected = expected.fillna(pd.NA)
    return expected
