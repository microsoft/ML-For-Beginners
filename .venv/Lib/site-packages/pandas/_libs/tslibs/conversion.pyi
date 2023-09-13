from datetime import (
    datetime,
    tzinfo,
)

import numpy as np

DT64NS_DTYPE: np.dtype
TD64NS_DTYPE: np.dtype

def precision_from_unit(
    unit: str,
) -> tuple[int, int]: ...  # (int64_t, _)
def localize_pydatetime(dt: datetime, tz: tzinfo | None) -> datetime: ...
