"""The version information for jupyter client."""
import re
from typing import List, Union

__version__ = "8.3.1"

# Build up version_info tuple for backwards compatibility
pattern = r'(?P<major>\d+).(?P<minor>\d+).(?P<patch>\d+)(?P<rest>.*)'
match = re.match(pattern, __version__)
if match:
    parts: List[Union[int, str]] = [int(match[part]) for part in ['major', 'minor', 'patch']]
    if match['rest']:
        parts.append(match['rest'])
else:
    parts = []
version_info = tuple(parts)


protocol_version_info = (5, 3)
protocol_version = "%i.%i" % protocol_version_info
