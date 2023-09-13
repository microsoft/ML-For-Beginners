from warnings import warn

warn("IPython.utils.localinterfaces has moved to jupyter_client.localinterfaces", stacklevel=2)

from jupyter_client.localinterfaces import *
