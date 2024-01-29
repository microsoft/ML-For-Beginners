"""
The :mod:`sklearn.utils.metadata_routing` module includes utilities to route
metadata within scikit-learn estimators.
"""

# This module is not a separate sub-folder since that would result in a circular
# import issue.
#
# Author: Adrin Jalali <adrin.jalali@gmail.com>
# License: BSD 3 clause

from ._metadata_requests import WARN, UNUSED, UNCHANGED  # noqa
from ._metadata_requests import get_routing_for_object  # noqa
from ._metadata_requests import MetadataRouter  # noqa
from ._metadata_requests import MetadataRequest  # noqa
from ._metadata_requests import MethodMapping  # noqa
from ._metadata_requests import process_routing  # noqa
from ._metadata_requests import _MetadataRequester  # noqa
from ._metadata_requests import _routing_enabled  # noqa
from ._metadata_requests import _raise_for_params  # noqa
from ._metadata_requests import _RoutingNotSupportedMixin  # noqa
from ._metadata_requests import _raise_for_unsupported_routing  # noqa
