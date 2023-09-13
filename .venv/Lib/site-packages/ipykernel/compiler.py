"""Compiler helpers for the debugger."""
import os
import sys
import tempfile

from IPython.core.compilerop import CachingCompiler


def murmur2_x86(data, seed):
    """Get the murmur2 hash."""
    m = 0x5BD1E995
    data = [chr(d) for d in str.encode(data, "utf8")]
    length = len(data)
    h = seed ^ length
    rounded_end = length & 0xFFFFFFFC
    for i in range(0, rounded_end, 4):
        k = (
            (ord(data[i]) & 0xFF)
            | ((ord(data[i + 1]) & 0xFF) << 8)
            | ((ord(data[i + 2]) & 0xFF) << 16)
            | (ord(data[i + 3]) << 24)
        )
        k = (k * m) & 0xFFFFFFFF
        k ^= k >> 24
        k = (k * m) & 0xFFFFFFFF

        h = (h * m) & 0xFFFFFFFF
        h ^= k

    val = length & 0x03
    k = 0
    if val == 3:
        k = (ord(data[rounded_end + 2]) & 0xFF) << 16
    if val in [2, 3]:
        k |= (ord(data[rounded_end + 1]) & 0xFF) << 8
    if val in [1, 2, 3]:
        k |= ord(data[rounded_end]) & 0xFF
        h ^= k
        h = (h * m) & 0xFFFFFFFF

    h ^= h >> 13
    h = (h * m) & 0xFFFFFFFF
    h ^= h >> 15

    return h


convert_to_long_pathname = lambda filename: filename  # noqa

if sys.platform == "win32":
    try:
        import ctypes
        from ctypes.wintypes import DWORD, LPCWSTR, LPWSTR, MAX_PATH

        _GetLongPathName = ctypes.windll.kernel32.GetLongPathNameW
        _GetLongPathName.argtypes = [LPCWSTR, LPWSTR, DWORD]
        _GetLongPathName.restype = DWORD

        def _convert_to_long_pathname(filename):
            buf = ctypes.create_unicode_buffer(MAX_PATH)
            rv = _GetLongPathName(filename, buf, MAX_PATH)
            if rv != 0 and rv <= MAX_PATH:
                filename = buf.value
            return filename

        # test that it works so if there are any issues we fail just once here
        _convert_to_long_pathname(__file__)
    except Exception:
        pass
    else:
        convert_to_long_pathname = _convert_to_long_pathname


def get_tmp_directory():
    """Get a temp directory."""
    tmp_dir = convert_to_long_pathname(tempfile.gettempdir())
    pid = os.getpid()
    return tmp_dir + os.sep + "ipykernel_" + str(pid)


def get_tmp_hash_seed():
    """Get a temp hash seed."""
    hash_seed = 0xC70F6907
    return hash_seed


def get_file_name(code):
    """Get a file name."""
    cell_name = os.environ.get("IPYKERNEL_CELL_NAME")
    if cell_name is None:
        name = murmur2_x86(code, get_tmp_hash_seed())
        cell_name = get_tmp_directory() + os.sep + str(name) + ".py"
    return cell_name


class XCachingCompiler(CachingCompiler):
    """A custom caching compiler."""

    def __init__(self, *args, **kwargs):
        """Initialize the compiler."""
        super().__init__(*args, **kwargs)
        self.log = None

    def get_code_name(self, raw_code, code, number):
        """Get the code name."""
        return get_file_name(raw_code)
