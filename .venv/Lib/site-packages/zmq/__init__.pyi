from typing import List

from . import backend, sugar

COPY_THRESHOLD: int
DRAFT_API: bool
__version__: str

# mypy doesn't like overwriting symbols with * so be explicit
# about what comes from backend, not from sugar
# see tools/backend_imports.py to generate this list
# note: `x as x` is required for re-export
# see https://github.com/python/mypy/issues/2190
from .backend import IPC_PATH_MAX_LEN as IPC_PATH_MAX_LEN
from .backend import curve_keypair as curve_keypair
from .backend import curve_public as curve_public
from .backend import device as device
from .backend import has as has
from .backend import proxy as proxy
from .backend import proxy_steerable as proxy_steerable
from .backend import strerror as strerror
from .backend import zmq_errno as zmq_errno
from .backend import zmq_poll as zmq_poll
from .constants import *
from .error import *
from .sugar import *

def get_includes() -> List[str]: ...
def get_library_dirs() -> List[str]: ...
