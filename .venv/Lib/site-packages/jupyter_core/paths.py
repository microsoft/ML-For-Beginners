"""Path utility functions."""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# Derived from IPython.utils.path, which is
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import platformdirs

from .utils import deprecation

pjoin = os.path.join

# Capitalize Jupyter in paths only on Windows and MacOS
APPNAME = "Jupyter" if sys.platform in ("win32", "darwin") else "jupyter"

# UF_HIDDEN is a stat flag not defined in the stat module.
# It is used by BSD to indicate hidden files.
UF_HIDDEN = getattr(stat, "UF_HIDDEN", 32768)


def envset(name: str, default: Optional[bool] = False) -> Optional[bool]:
    """Return the boolean value of a given environment variable.

    An environment variable is considered set if it is assigned to a value
    other than 'no', 'n', 'false', 'off', '0', or '0.0' (case insensitive)

    If the environment variable is not defined, the default value is returned.
    """
    if name not in os.environ:
        return default

    return os.environ[name].lower() not in ["no", "n", "false", "off", "0", "0.0"]


def use_platform_dirs() -> bool:
    """Determine if platformdirs should be used for system-specific paths.

    We plan for this to default to False in jupyter_core version 5 and to True
    in jupyter_core version 6.
    """
    return envset("JUPYTER_PLATFORM_DIRS", False)  # type:ignore[return-value]


def get_home_dir() -> str:
    """Get the real path of the home directory"""
    homedir = os.path.expanduser("~")
    # Next line will make things work even when /home/ is a symlink to
    # /usr/home as it is on FreeBSD, for example
    homedir = str(Path(homedir).resolve())
    return homedir


_dtemps: Dict[str, str] = {}


def _do_i_own(path: str) -> bool:
    """Return whether the current user owns the given path"""
    p = Path(path).resolve()

    # walk up to first existing parent
    while not p.exists() and p != p.parent:
        p = p.parent

    # simplest check: owner by name
    # not always implemented or available
    try:
        return p.owner() == os.getlogin()
    except Exception:  # noqa
        pass

    if hasattr(os, 'geteuid'):
        try:
            st = p.stat()
            return st.st_uid == os.geteuid()
        except (NotImplementedError, OSError):
            # geteuid not always implemented
            pass

    # no ownership checks worked, check write access
    return os.access(p, os.W_OK)


def prefer_environment_over_user() -> bool:
    """Determine if environment-level paths should take precedence over user-level paths."""
    # If JUPYTER_PREFER_ENV_PATH is defined, that signals user intent, so return its value
    if "JUPYTER_PREFER_ENV_PATH" in os.environ:
        return envset("JUPYTER_PREFER_ENV_PATH")  # type:ignore[return-value]

    # If we are in a Python virtualenv, default to True (see https://docs.python.org/3/library/venv.html#venv-def)
    if sys.prefix != sys.base_prefix and _do_i_own(sys.prefix):
        return True

    # If sys.prefix indicates Python comes from a conda/mamba environment that is not the root environment, default to True
    if (
        "CONDA_PREFIX" in os.environ
        and sys.prefix.startswith(os.environ["CONDA_PREFIX"])
        and os.environ.get("CONDA_DEFAULT_ENV", "base") != "base"
        and _do_i_own(sys.prefix)
    ):
        return True

    return False


def _mkdtemp_once(name: str) -> str:
    """Make or reuse a temporary directory.

    If this is called with the same name in the same process, it will return
    the same directory.
    """
    try:
        return _dtemps[name]
    except KeyError:
        d = _dtemps[name] = tempfile.mkdtemp(prefix=name + "-")
        return d


def jupyter_config_dir() -> str:
    """Get the Jupyter config directory for this platform and user.

    Returns JUPYTER_CONFIG_DIR if defined, otherwise the appropriate
    directory for the platform.
    """

    env = os.environ
    if env.get("JUPYTER_NO_CONFIG"):
        return _mkdtemp_once("jupyter-clean-cfg")

    if env.get("JUPYTER_CONFIG_DIR"):
        return env["JUPYTER_CONFIG_DIR"]

    if use_platform_dirs():
        return platformdirs.user_config_dir(APPNAME, appauthor=False)

    home_dir = get_home_dir()
    return pjoin(home_dir, ".jupyter")


def jupyter_data_dir() -> str:
    """Get the config directory for Jupyter data files for this platform and user.

    These are non-transient, non-configuration files.

    Returns JUPYTER_DATA_DIR if defined, else a platform-appropriate path.
    """
    env = os.environ

    if env.get("JUPYTER_DATA_DIR"):
        return env["JUPYTER_DATA_DIR"]

    if use_platform_dirs():
        return platformdirs.user_data_dir(APPNAME, appauthor=False)

    home = get_home_dir()

    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Jupyter")
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA", None)
        if appdata:
            return str(Path(appdata, "jupyter").resolve())
        else:
            return pjoin(jupyter_config_dir(), "data")
    else:
        # Linux, non-OS X Unix, AIX, etc.
        xdg = env.get("XDG_DATA_HOME", None)
        if not xdg:
            xdg = pjoin(home, ".local", "share")
        return pjoin(xdg, "jupyter")


def jupyter_runtime_dir() -> str:
    """Return the runtime dir for transient jupyter files.

    Returns JUPYTER_RUNTIME_DIR if defined.

    The default is now (data_dir)/runtime on all platforms;
    we no longer use XDG_RUNTIME_DIR after various problems.
    """
    env = os.environ

    if env.get("JUPYTER_RUNTIME_DIR"):
        return env["JUPYTER_RUNTIME_DIR"]

    return pjoin(jupyter_data_dir(), "runtime")


if use_platform_dirs():
    SYSTEM_JUPYTER_PATH = platformdirs.site_data_dir(
        APPNAME, appauthor=False, multipath=True
    ).split(os.pathsep)
else:
    deprecation(
        "Jupyter is migrating its paths to use standard platformdirs\n"
        "given by the platformdirs library.  To remove this warning and\n"
        "see the appropriate new directories, set the environment variable\n"
        "`JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.\n"
        "The use of platformdirs will be the default in `jupyter_core` v6"
    )
    if os.name == "nt":
        programdata = os.environ.get("PROGRAMDATA", None)
        if programdata:
            SYSTEM_JUPYTER_PATH = [pjoin(programdata, "jupyter")]
        else:  # PROGRAMDATA is not defined by default on XP.
            SYSTEM_JUPYTER_PATH = [os.path.join(sys.prefix, "share", "jupyter")]
    else:
        SYSTEM_JUPYTER_PATH = [
            "/usr/local/share/jupyter",
            "/usr/share/jupyter",
        ]

ENV_JUPYTER_PATH: List[str] = [os.path.join(sys.prefix, "share", "jupyter")]


def jupyter_path(*subdirs: str) -> List[str]:
    """Return a list of directories to search for data files

    JUPYTER_PATH environment variable has highest priority.

    If the JUPYTER_PREFER_ENV_PATH environment variable is set, the environment-level
    directories will have priority over user-level directories.

    If the Python site.ENABLE_USER_SITE variable is True, we also add the
    appropriate Python user site subdirectory to the user-level directories.


    If ``*subdirs`` are given, that subdirectory will be added to each element.

    Examples:

    >>> jupyter_path()
    ['~/.local/jupyter', '/usr/local/share/jupyter']
    >>> jupyter_path('kernels')
    ['~/.local/jupyter/kernels', '/usr/local/share/jupyter/kernels']
    """

    paths: List[str] = []

    # highest priority is explicit environment variable
    if os.environ.get("JUPYTER_PATH"):
        paths.extend(p.rstrip(os.sep) for p in os.environ["JUPYTER_PATH"].split(os.pathsep))

    # Next is environment or user, depending on the JUPYTER_PREFER_ENV_PATH flag
    user = [jupyter_data_dir()]
    if site.ENABLE_USER_SITE:
        # Check if site.getuserbase() exists to be compatible with virtualenv,
        # which often does not have this method.
        userbase: Optional[str]
        userbase = site.getuserbase() if hasattr(site, "getuserbase") else site.USER_BASE

        if userbase:
            userdir = os.path.join(userbase, "share", "jupyter")
            if userdir not in user:
                user.append(userdir)

    env = [p for p in ENV_JUPYTER_PATH if p not in SYSTEM_JUPYTER_PATH]

    if prefer_environment_over_user():
        paths.extend(env)
        paths.extend(user)
    else:
        paths.extend(user)
        paths.extend(env)

    # finally, system
    paths.extend(SYSTEM_JUPYTER_PATH)

    # add subdir, if requested
    if subdirs:
        paths = [pjoin(p, *subdirs) for p in paths]
    return paths


if use_platform_dirs():
    SYSTEM_CONFIG_PATH = platformdirs.site_config_dir(
        APPNAME, appauthor=False, multipath=True
    ).split(os.pathsep)
else:
    if os.name == "nt":  # noqa
        programdata = os.environ.get("PROGRAMDATA", None)
        if programdata:  # noqa
            SYSTEM_CONFIG_PATH = [os.path.join(programdata, "jupyter")]
        else:  # PROGRAMDATA is not defined by default on XP.
            SYSTEM_CONFIG_PATH = []
    else:
        SYSTEM_CONFIG_PATH = [
            "/usr/local/etc/jupyter",
            "/etc/jupyter",
        ]
ENV_CONFIG_PATH: List[str] = [os.path.join(sys.prefix, "etc", "jupyter")]


def jupyter_config_path() -> List[str]:
    """Return the search path for Jupyter config files as a list.

    If the JUPYTER_PREFER_ENV_PATH environment variable is set, the
    environment-level directories will have priority over user-level
    directories.

    If the Python site.ENABLE_USER_SITE variable is True, we also add the
    appropriate Python user site subdirectory to the user-level directories.
    """
    if os.environ.get("JUPYTER_NO_CONFIG"):
        # jupyter_config_dir makes a blank config when JUPYTER_NO_CONFIG is set.
        return [jupyter_config_dir()]

    paths: List[str] = []

    # highest priority is explicit environment variable
    if os.environ.get("JUPYTER_CONFIG_PATH"):
        paths.extend(p.rstrip(os.sep) for p in os.environ["JUPYTER_CONFIG_PATH"].split(os.pathsep))

    # Next is environment or user, depending on the JUPYTER_PREFER_ENV_PATH flag
    user = [jupyter_config_dir()]
    if site.ENABLE_USER_SITE:
        userbase: Optional[str]
        # Check if site.getuserbase() exists to be compatible with virtualenv,
        # which often does not have this method.
        userbase = site.getuserbase() if hasattr(site, "getuserbase") else site.USER_BASE

        if userbase:
            userdir = os.path.join(userbase, "etc", "jupyter")
            if userdir not in user:
                user.append(userdir)

    env = [p for p in ENV_CONFIG_PATH if p not in SYSTEM_CONFIG_PATH]

    if prefer_environment_over_user():
        paths.extend(env)
        paths.extend(user)
    else:
        paths.extend(user)
        paths.extend(env)

    # Finally, system path
    paths.extend(SYSTEM_CONFIG_PATH)
    return paths


def exists(path: str) -> bool:
    """Replacement for `os.path.exists` which works for host mapped volumes
    on Windows containers
    """
    try:
        os.lstat(path)
    except OSError:
        return False
    return True


def is_file_hidden_win(abs_path: str, stat_res: Optional[Any] = None) -> bool:
    """Is a file hidden?

    This only checks the file itself; it should be called in combination with
    checking the directory containing the file.

    Use is_hidden() instead to check the file and its parent directories.

    Parameters
    ----------
    abs_path : unicode
        The absolute path to check.
    stat_res : os.stat_result, optional
        The result of calling stat() on abs_path. If not passed, this function
        will call stat() internally.
    """
    if os.path.basename(abs_path).startswith("."):
        return True

    if stat_res is None:
        try:
            stat_res = os.stat(abs_path)
        except OSError as e:
            if e.errno == errno.ENOENT:
                return False
            raise

    try:
        if stat_res.st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN:  # type:ignore
            return True
    except AttributeError:
        # allow AttributeError on PyPy for Windows
        # 'stat_result' object has no attribute 'st_file_attributes'
        # https://foss.heptapod.net/pypy/pypy/-/issues/3469
        warnings.warn(
            "hidden files are not detectable on this system, so no file will be marked as hidden.",
            stacklevel=2,
        )
        pass

    return False


def is_file_hidden_posix(abs_path: str, stat_res: Optional[Any] = None) -> bool:
    """Is a file hidden?

    This only checks the file itself; it should be called in combination with
    checking the directory containing the file.

    Use is_hidden() instead to check the file and its parent directories.

    Parameters
    ----------
    abs_path : unicode
        The absolute path to check.
    stat_res : os.stat_result, optional
        The result of calling stat() on abs_path. If not passed, this function
        will call stat() internally.
    """
    if os.path.basename(abs_path).startswith("."):
        return True

    if stat_res is None or stat.S_ISLNK(stat_res.st_mode):
        try:
            stat_res = os.stat(abs_path)
        except OSError as e:
            if e.errno == errno.ENOENT:
                return False
            raise

    # check that dirs can be listed
    if stat.S_ISDIR(stat_res.st_mode):  # type:ignore[misc]  # noqa
        # use x-access, not actual listing, in case of slow/large listings
        if not os.access(abs_path, os.X_OK | os.R_OK):
            return True

    # check UF_HIDDEN
    if getattr(stat_res, "st_flags", 0) & UF_HIDDEN:
        return True

    return False


if sys.platform == "win32":
    is_file_hidden = is_file_hidden_win
else:
    is_file_hidden = is_file_hidden_posix


def is_hidden(abs_path: str, abs_root: str = "") -> bool:
    """Is a file hidden or contained in a hidden directory?

    This will start with the rightmost path element and work backwards to the
    given root to see if a path is hidden or in a hidden directory. Hidden is
    determined by either name starting with '.' or the UF_HIDDEN flag as
    reported by stat.

    If abs_path is the same directory as abs_root, it will be visible even if
    that is a hidden folder. This only checks the visibility of files
    and directories *within* abs_root.

    Parameters
    ----------
    abs_path : unicode
        The absolute path to check for hidden directories.
    abs_root : unicode
        The absolute path of the root directory in which hidden directories
        should be checked for.
    """
    abs_path = os.path.normpath(abs_path)
    abs_root = os.path.normpath(abs_root)

    if abs_path == abs_root:
        return False

    if is_file_hidden(abs_path):
        return True

    if not abs_root:
        abs_root = abs_path.split(os.sep, 1)[0] + os.sep
    inside_root = abs_path[len(abs_root) :]
    if any(part.startswith(".") for part in inside_root.split(os.sep)):
        return True

    # check UF_HIDDEN on any location up to root.
    # is_file_hidden() already checked the file, so start from its parent dir
    path = os.path.dirname(abs_path)
    while path and path.startswith(abs_root) and path != abs_root:
        if not exists(path):
            path = os.path.dirname(path)
            continue
        try:
            # may fail on Windows junctions
            st = os.lstat(path)
        except OSError:
            return True
        if getattr(st, "st_flags", 0) & UF_HIDDEN:
            return True
        path = os.path.dirname(path)

    return False


def win32_restrict_file_to_user(fname: str) -> None:
    """Secure a windows file to read-only access for the user.
    Follows guidance from win32 library creator:
    http://timgolden.me.uk/python/win32_how_do_i/add-security-to-a-file.html

    This method should be executed against an already generated file which
    has no secrets written to it yet.

    Parameters
    ----------

    fname : unicode
        The path to the file to secure
    """
    try:
        import win32api
    except ImportError:
        return _win32_restrict_file_to_user_ctypes(fname)

    import ntsecuritycon as con
    import win32security

    # everyone, _domain, _type = win32security.LookupAccountName("", "Everyone")
    admins = win32security.CreateWellKnownSid(win32security.WinBuiltinAdministratorsSid)
    user, _domain, _type = win32security.LookupAccountName(
        "", win32api.GetUserNameEx(win32api.NameSamCompatible)
    )

    sd = win32security.GetFileSecurity(fname, win32security.DACL_SECURITY_INFORMATION)

    dacl = win32security.ACL()
    # dacl.AddAccessAllowedAce(win32security.ACL_REVISION, con.FILE_ALL_ACCESS, everyone)
    dacl.AddAccessAllowedAce(
        win32security.ACL_REVISION,
        con.FILE_GENERIC_READ | con.FILE_GENERIC_WRITE | con.DELETE,
        user,
    )
    dacl.AddAccessAllowedAce(win32security.ACL_REVISION, con.FILE_ALL_ACCESS, admins)

    sd.SetSecurityDescriptorDacl(1, dacl, 0)
    win32security.SetFileSecurity(fname, win32security.DACL_SECURITY_INFORMATION, sd)


def _win32_restrict_file_to_user_ctypes(fname: str) -> None:  # noqa
    """Secure a windows file to read-only access for the user.

    Follows guidance from win32 library creator:
    http://timgolden.me.uk/python/win32_how_do_i/add-security-to-a-file.html

    This method should be executed against an already generated file which
    has no secrets written to it yet.

    Parameters
    ----------

    fname : unicode
        The path to the file to secure
    """
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)  # type:ignore[attr-defined]
    secur32 = ctypes.WinDLL("secur32", use_last_error=True)  # type:ignore[attr-defined]

    NameSamCompatible = 2
    WinBuiltinAdministratorsSid = 26
    DACL_SECURITY_INFORMATION = 4
    ACL_REVISION = 2
    ERROR_INSUFFICIENT_BUFFER = 122
    ERROR_MORE_DATA = 234

    SYNCHRONIZE = 0x100000
    DELETE = 0x00010000
    STANDARD_RIGHTS_REQUIRED = 0xF0000
    STANDARD_RIGHTS_READ = 0x20000
    STANDARD_RIGHTS_WRITE = 0x20000
    FILE_READ_DATA = 1
    FILE_READ_EA = 8
    FILE_READ_ATTRIBUTES = 128
    FILE_WRITE_DATA = 2
    FILE_APPEND_DATA = 4
    FILE_WRITE_EA = 16
    FILE_WRITE_ATTRIBUTES = 256
    FILE_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0x1FF
    FILE_GENERIC_READ = (
        STANDARD_RIGHTS_READ | FILE_READ_DATA | FILE_READ_ATTRIBUTES | FILE_READ_EA | SYNCHRONIZE
    )
    FILE_GENERIC_WRITE = (
        STANDARD_RIGHTS_WRITE
        | FILE_WRITE_DATA
        | FILE_WRITE_ATTRIBUTES
        | FILE_WRITE_EA
        | FILE_APPEND_DATA
        | SYNCHRONIZE
    )

    class ACL(ctypes.Structure):
        _fields_ = [
            ("AclRevision", wintypes.BYTE),
            ("Sbz1", wintypes.BYTE),
            ("AclSize", wintypes.WORD),
            ("AceCount", wintypes.WORD),
            ("Sbz2", wintypes.WORD),
        ]

    PSID = ctypes.c_void_p
    PACL = ctypes.POINTER(ACL)
    PSECURITY_DESCRIPTOR = ctypes.POINTER(wintypes.BYTE)

    def _nonzero_success(result, func, args):
        if not result:
            raise ctypes.WinError(ctypes.get_last_error())  # type:ignore[attr-defined]
        return args

    secur32.GetUserNameExW.errcheck = _nonzero_success
    secur32.GetUserNameExW.restype = wintypes.BOOL
    secur32.GetUserNameExW.argtypes = (
        ctypes.c_int,  # EXTENDED_NAME_FORMAT NameFormat
        wintypes.LPWSTR,  # LPWSTR lpNameBuffer,
        wintypes.PULONG,  # PULONG nSize
    )

    advapi32.CreateWellKnownSid.errcheck = _nonzero_success
    advapi32.CreateWellKnownSid.restype = wintypes.BOOL
    advapi32.CreateWellKnownSid.argtypes = (
        wintypes.DWORD,  # WELL_KNOWN_SID_TYPE WellKnownSidType
        PSID,  # PSID DomainSid
        PSID,  # PSID pSid
        wintypes.PDWORD,  # DWORD *cbSid
    )

    advapi32.LookupAccountNameW.errcheck = _nonzero_success
    advapi32.LookupAccountNameW.restype = wintypes.BOOL
    advapi32.LookupAccountNameW.argtypes = (
        wintypes.LPWSTR,  # LPCWSTR lpSystemName
        wintypes.LPWSTR,  # LPCWSTR lpAccountName
        PSID,  # PSID Sid
        wintypes.LPDWORD,  # LPDWORD cbSid
        wintypes.LPWSTR,  # LPCWSTR ReferencedDomainName
        wintypes.LPDWORD,  # LPDWORD cchReferencedDomainName
        wintypes.LPDWORD,  # PSID_NAME_USE peUse
    )

    advapi32.AddAccessAllowedAce.errcheck = _nonzero_success
    advapi32.AddAccessAllowedAce.restype = wintypes.BOOL
    advapi32.AddAccessAllowedAce.argtypes = (
        PACL,  # PACL pAcl
        wintypes.DWORD,  # DWORD dwAceRevision
        wintypes.DWORD,  # DWORD AccessMask
        PSID,  # PSID pSid
    )

    advapi32.SetSecurityDescriptorDacl.errcheck = _nonzero_success
    advapi32.SetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.SetSecurityDescriptorDacl.argtypes = (
        PSECURITY_DESCRIPTOR,  # PSECURITY_DESCRIPTOR pSecurityDescriptor
        wintypes.BOOL,  # BOOL bDaclPresent
        PACL,  # PACL pDacl
        wintypes.BOOL,  # BOOL bDaclDefaulted
    )

    advapi32.GetFileSecurityW.errcheck = _nonzero_success
    advapi32.GetFileSecurityW.restype = wintypes.BOOL
    advapi32.GetFileSecurityW.argtypes = (
        wintypes.LPCWSTR,  # LPCWSTR lpFileName
        wintypes.DWORD,  # SECURITY_INFORMATION RequestedInformation
        PSECURITY_DESCRIPTOR,  # PSECURITY_DESCRIPTOR pSecurityDescriptor
        wintypes.DWORD,  # DWORD nLength
        wintypes.LPDWORD,  # LPDWORD lpnLengthNeeded
    )

    advapi32.SetFileSecurityW.errcheck = _nonzero_success
    advapi32.SetFileSecurityW.restype = wintypes.BOOL
    advapi32.SetFileSecurityW.argtypes = (
        wintypes.LPCWSTR,  # LPCWSTR lpFileName
        wintypes.DWORD,  # SECURITY_INFORMATION SecurityInformation
        PSECURITY_DESCRIPTOR,  # PSECURITY_DESCRIPTOR pSecurityDescriptor
    )

    advapi32.MakeAbsoluteSD.errcheck = _nonzero_success
    advapi32.MakeAbsoluteSD.restype = wintypes.BOOL
    advapi32.MakeAbsoluteSD.argtypes = (
        PSECURITY_DESCRIPTOR,  # pSelfRelativeSecurityDescriptor
        PSECURITY_DESCRIPTOR,  # pAbsoluteSecurityDescriptor
        wintypes.LPDWORD,  # LPDWORD lpdwAbsoluteSecurityDescriptorSize
        PACL,  # PACL pDacl
        wintypes.LPDWORD,  # LPDWORD lpdwDaclSize
        PACL,  # PACL pSacl
        wintypes.LPDWORD,  # LPDWORD lpdwSaclSize
        PSID,  # PSID pOwner
        wintypes.LPDWORD,  # LPDWORD lpdwOwnerSize
        PSID,  # PSID pPrimaryGroup
        wintypes.LPDWORD,  # LPDWORD lpdwPrimaryGroupSize
    )

    advapi32.MakeSelfRelativeSD.errcheck = _nonzero_success
    advapi32.MakeSelfRelativeSD.restype = wintypes.BOOL
    advapi32.MakeSelfRelativeSD.argtypes = (
        PSECURITY_DESCRIPTOR,  # pAbsoluteSecurityDescriptor
        PSECURITY_DESCRIPTOR,  # pSelfRelativeSecurityDescriptor
        wintypes.LPDWORD,  # LPDWORD lpdwBufferLength
    )

    advapi32.InitializeAcl.errcheck = _nonzero_success
    advapi32.InitializeAcl.restype = wintypes.BOOL
    advapi32.InitializeAcl.argtypes = (
        PACL,  # PACL pAcl,
        wintypes.DWORD,  # DWORD nAclLength,
        wintypes.DWORD,  # DWORD dwAclRevision
    )

    def CreateWellKnownSid(WellKnownSidType):
        # return a SID for predefined aliases
        pSid = (ctypes.c_char * 1)()
        cbSid = wintypes.DWORD()
        try:
            advapi32.CreateWellKnownSid(WellKnownSidType, None, pSid, ctypes.byref(cbSid))
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:  # type:ignore[attr-defined]
                raise
            pSid = (ctypes.c_char * cbSid.value)()
            advapi32.CreateWellKnownSid(WellKnownSidType, None, pSid, ctypes.byref(cbSid))
        return pSid[:]

    def GetUserNameEx(NameFormat):
        # return the user or other security principal associated with
        # the calling thread
        nSize = ctypes.pointer(ctypes.c_ulong(0))
        try:
            secur32.GetUserNameExW(NameFormat, None, nSize)
        except OSError as e:
            if e.winerror != ERROR_MORE_DATA:  # type:ignore[attr-defined]
                raise
        if not nSize.contents.value:
            return None
        lpNameBuffer = ctypes.create_unicode_buffer(nSize.contents.value)
        secur32.GetUserNameExW(NameFormat, lpNameBuffer, nSize)
        return lpNameBuffer.value

    def LookupAccountName(lpSystemName, lpAccountName):
        # return a security identifier (SID) for an account on a system
        # and the name of the domain on which the account was found
        cbSid = wintypes.DWORD(0)
        cchReferencedDomainName = wintypes.DWORD(0)
        peUse = wintypes.DWORD(0)
        try:
            advapi32.LookupAccountNameW(
                lpSystemName,
                lpAccountName,
                None,
                ctypes.byref(cbSid),
                None,
                ctypes.byref(cchReferencedDomainName),
                ctypes.byref(peUse),
            )
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:  # type:ignore[attr-defined]
                raise
        Sid = ctypes.create_unicode_buffer("", cbSid.value)
        pSid = ctypes.cast(ctypes.pointer(Sid), wintypes.LPVOID)
        lpReferencedDomainName = ctypes.create_unicode_buffer("", cchReferencedDomainName.value + 1)
        success = advapi32.LookupAccountNameW(
            lpSystemName,
            lpAccountName,
            pSid,
            ctypes.byref(cbSid),
            lpReferencedDomainName,
            ctypes.byref(cchReferencedDomainName),
            ctypes.byref(peUse),
        )
        if not success:
            raise ctypes.WinError()  # type:ignore[attr-defined]
        return pSid, lpReferencedDomainName.value, peUse.value

    def AddAccessAllowedAce(pAcl, dwAceRevision, AccessMask, pSid):
        # add an access-allowed access control entry (ACE)
        # to an access control list (ACL)
        advapi32.AddAccessAllowedAce(pAcl, dwAceRevision, AccessMask, pSid)

    def GetFileSecurity(lpFileName, RequestedInformation):
        # return information about the security of a file or directory
        nLength = wintypes.DWORD(0)
        try:
            advapi32.GetFileSecurityW(
                lpFileName,
                RequestedInformation,
                None,
                0,
                ctypes.byref(nLength),
            )
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:  # type:ignore[attr-defined]
                raise
        if not nLength.value:
            return None
        pSecurityDescriptor = (wintypes.BYTE * nLength.value)()
        advapi32.GetFileSecurityW(
            lpFileName,
            RequestedInformation,
            pSecurityDescriptor,
            nLength,
            ctypes.byref(nLength),
        )
        return pSecurityDescriptor

    def SetFileSecurity(lpFileName, RequestedInformation, pSecurityDescriptor):
        # set the security of a file or directory object
        advapi32.SetFileSecurityW(lpFileName, RequestedInformation, pSecurityDescriptor)

    def SetSecurityDescriptorDacl(pSecurityDescriptor, bDaclPresent, pDacl, bDaclDefaulted):
        # set information in a discretionary access control list (DACL)
        advapi32.SetSecurityDescriptorDacl(pSecurityDescriptor, bDaclPresent, pDacl, bDaclDefaulted)

    def MakeAbsoluteSD(pSelfRelativeSecurityDescriptor):
        # return a security descriptor in absolute format
        # by using a security descriptor in self-relative format as a template
        pAbsoluteSecurityDescriptor = None
        lpdwAbsoluteSecurityDescriptorSize = wintypes.DWORD(0)
        pDacl = None
        lpdwDaclSize = wintypes.DWORD(0)
        pSacl = None
        lpdwSaclSize = wintypes.DWORD(0)
        pOwner = None
        lpdwOwnerSize = wintypes.DWORD(0)
        pPrimaryGroup = None
        lpdwPrimaryGroupSize = wintypes.DWORD(0)
        try:
            advapi32.MakeAbsoluteSD(
                pSelfRelativeSecurityDescriptor,
                pAbsoluteSecurityDescriptor,
                ctypes.byref(lpdwAbsoluteSecurityDescriptorSize),
                pDacl,
                ctypes.byref(lpdwDaclSize),
                pSacl,
                ctypes.byref(lpdwSaclSize),
                pOwner,
                ctypes.byref(lpdwOwnerSize),
                pPrimaryGroup,
                ctypes.byref(lpdwPrimaryGroupSize),
            )
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:  # type:ignore[attr-defined]
                raise
        pAbsoluteSecurityDescriptor = (wintypes.BYTE * lpdwAbsoluteSecurityDescriptorSize.value)()
        pDaclData = (wintypes.BYTE * lpdwDaclSize.value)()
        pDacl = ctypes.cast(pDaclData, PACL).contents
        pSaclData = (wintypes.BYTE * lpdwSaclSize.value)()
        pSacl = ctypes.cast(pSaclData, PACL).contents
        pOwnerData = (wintypes.BYTE * lpdwOwnerSize.value)()
        pOwner = ctypes.cast(pOwnerData, PSID)
        pPrimaryGroupData = (wintypes.BYTE * lpdwPrimaryGroupSize.value)()
        pPrimaryGroup = ctypes.cast(pPrimaryGroupData, PSID)
        advapi32.MakeAbsoluteSD(
            pSelfRelativeSecurityDescriptor,
            pAbsoluteSecurityDescriptor,
            ctypes.byref(lpdwAbsoluteSecurityDescriptorSize),
            pDacl,
            ctypes.byref(lpdwDaclSize),
            pSacl,
            ctypes.byref(lpdwSaclSize),
            pOwner,
            lpdwOwnerSize,
            pPrimaryGroup,
            ctypes.byref(lpdwPrimaryGroupSize),
        )
        return pAbsoluteSecurityDescriptor

    def MakeSelfRelativeSD(pAbsoluteSecurityDescriptor):
        # return a security descriptor in self-relative format
        # by using a security descriptor in absolute format as a template
        pSelfRelativeSecurityDescriptor = None
        lpdwBufferLength = wintypes.DWORD(0)
        try:
            advapi32.MakeSelfRelativeSD(
                pAbsoluteSecurityDescriptor,
                pSelfRelativeSecurityDescriptor,
                ctypes.byref(lpdwBufferLength),
            )
        except OSError as e:
            if e.winerror != ERROR_INSUFFICIENT_BUFFER:  # type:ignore[attr-defined]
                raise
        pSelfRelativeSecurityDescriptor = (wintypes.BYTE * lpdwBufferLength.value)()
        advapi32.MakeSelfRelativeSD(
            pAbsoluteSecurityDescriptor,
            pSelfRelativeSecurityDescriptor,
            ctypes.byref(lpdwBufferLength),
        )
        return pSelfRelativeSecurityDescriptor

    def NewAcl():
        # return a new, initialized ACL (access control list) structure
        nAclLength = 32767  # TODO: calculate this: ctypes.sizeof(ACL) + ?
        acl_data = ctypes.create_string_buffer(nAclLength)
        pAcl = ctypes.cast(acl_data, PACL).contents
        advapi32.InitializeAcl(pAcl, nAclLength, ACL_REVISION)
        return pAcl

    SidAdmins = CreateWellKnownSid(WinBuiltinAdministratorsSid)
    SidUser = LookupAccountName("", GetUserNameEx(NameSamCompatible))[0]

    Acl = NewAcl()
    AddAccessAllowedAce(Acl, ACL_REVISION, FILE_ALL_ACCESS, SidAdmins)
    AddAccessAllowedAce(
        Acl,
        ACL_REVISION,
        FILE_GENERIC_READ | FILE_GENERIC_WRITE | DELETE,
        SidUser,
    )

    SelfRelativeSD = GetFileSecurity(fname, DACL_SECURITY_INFORMATION)
    AbsoluteSD = MakeAbsoluteSD(SelfRelativeSD)
    SetSecurityDescriptorDacl(AbsoluteSD, 1, Acl, 0)
    SelfRelativeSD = MakeSelfRelativeSD(AbsoluteSD)

    SetFileSecurity(fname, DACL_SECURITY_INFORMATION, SelfRelativeSD)


def get_file_mode(fname: str) -> int:
    """Retrieves the file mode corresponding to fname in a filesystem-tolerant manner.

    Parameters
    ----------

    fname : unicode
        The path to the file to get mode from

    """
    # Some filesystems (e.g., CIFS) auto-enable the execute bit on files.  As a result, we
    # should tolerate the execute bit on the file's owner when validating permissions - thus
    # the missing least significant bit on the third octal digit. In addition, we also tolerate
    # the sticky bit being set, so the lsb from the fourth octal digit is also removed.
    return (
        stat.S_IMODE(os.stat(fname).st_mode) & 0o6677
    )  # Use 4 octal digits since S_IMODE does the same


allow_insecure_writes = os.getenv("JUPYTER_ALLOW_INSECURE_WRITES", "false").lower() in ("true", "1")


@contextmanager
def secure_write(fname: str, binary: bool = False) -> Iterator[Any]:
    """Opens a file in the most restricted pattern available for
    writing content. This limits the file mode to `0o0600` and yields
    the resulting opened filed handle.

    Parameters
    ----------

    fname : unicode
        The path to the file to write

    binary: boolean
        Indicates that the file is binary
    """
    mode = "wb" if binary else "w"
    encoding = None if binary else "utf-8"
    open_flag = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
    try:
        os.remove(fname)
    except OSError:
        # Skip any issues with the file not existing
        pass

    if os.name == "nt":
        if allow_insecure_writes:
            # Mounted file systems can have a number of failure modes inside this block.
            # For windows machines in insecure mode we simply skip this to avoid failures :/
            issue_insecure_write_warning()
        else:
            # Python on windows does not respect the group and public bits for chmod, so we need
            # to take additional steps to secure the contents.
            # Touch file pre-emptively to avoid editing permissions in open files in Windows
            fd = os.open(fname, open_flag, 0o0600)
            os.close(fd)
            open_flag = os.O_WRONLY | os.O_TRUNC
            win32_restrict_file_to_user(fname)

    with os.fdopen(os.open(fname, open_flag, 0o0600), mode, encoding=encoding) as f:
        if os.name != "nt":
            # Enforce that the file got the requested permissions before writing
            file_mode = get_file_mode(fname)
            if file_mode != 0o0600:  # noqa
                if allow_insecure_writes:
                    issue_insecure_write_warning()
                else:
                    msg = (
                        "Permissions assignment failed for secure file: '{file}'."
                        " Got '{permissions}' instead of '0o0600'.".format(
                            file=fname, permissions=oct(file_mode)
                        )
                    )
                    raise RuntimeError(msg)
        yield f


def issue_insecure_write_warning() -> None:
    """Issue an insecure write warning."""

    def format_warning(msg, *args, **kwargs):
        return str(msg) + "\n"

    warnings.formatwarning = format_warning  # type:ignore[assignment]
    warnings.warn(
        "WARNING: Insecure writes have been enabled via environment variable "
        "'JUPYTER_ALLOW_INSECURE_WRITES'! If this is not intended, remove the "
        "variable or set its value to 'False'.",
        stacklevel=2,
    )
