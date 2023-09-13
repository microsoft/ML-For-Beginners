"""Shared constants.
"""

# Because inprocess communication is not networked, we can use a common Session
# key everywhere. This is not just the empty bytestring to avoid tripping
# certain security checks in the rest of Jupyter that assumes that empty keys
# are insecure.
INPROCESS_KEY = b"inprocess"
