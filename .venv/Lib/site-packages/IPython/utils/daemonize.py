from warnings import warn

warn("IPython.utils.daemonize has moved to ipyparallel.apps.daemonize since IPython 4.0", DeprecationWarning, stacklevel=2)
from ipyparallel.apps.daemonize import daemonize
