'''
This module was created to get information available in the interpreter, such as libraries,
paths, etc.

what is what:
sys.builtin_module_names: contains the builtin modules embeeded in python (rigth now, we specify all manually).
sys.prefix: A string giving the site-specific directory prefix where the platform independent Python files are installed

format is something as
EXECUTABLE:python.exe|libs@compiled_dlls$builtin_mods

all internal are separated by |
'''
import sys

try:
    import os.path

    def fully_normalize_path(path):
        '''fixes the path so that the format of the path really reflects the directories in the system
        '''
        return os.path.normpath(path)

    join = os.path.join
except:  # ImportError or AttributeError.

    # See: http://stackoverflow.com/questions/10254353/error-while-installing-jython-for-pydev
    def fully_normalize_path(path):
        '''fixes the path so that the format of the path really reflects the directories in the system
        '''
        return path

    def join(a, b):
        if a.endswith('/') or a.endswith('\\'):
            return a + b
        return a + '/' + b

IS_PYTHON_3_ONWARDS = 0

try:
    IS_PYTHON_3_ONWARDS = sys.version_info[0] >= 3
except:
    # That's OK, not all versions of python have sys.version_info
    pass

try:
    # Just check if False and True are defined (depends on version, not whether it's jython/python)
    False
    True
except:
    exec ('True, False = 1,0')  # An exec is used so that python 3k does not give a syntax error

if sys.platform == "cygwin":

    import ctypes

    def native_path(path):
        MAX_PATH = 512  # On cygwin NT, its 260 lately, but just need BIG ENOUGH buffer
        '''Get the native form of the path, like c:\\Foo for /cygdrive/c/Foo'''

        retval = ctypes.create_string_buffer(MAX_PATH)
        path = fully_normalize_path(path)
        path = tobytes(path)
        CCP_POSIX_TO_WIN_A = 0
        cygwin1dll = ctypes.cdll.LoadLibrary('cygwin1.dll')
        cygwin1dll.cygwin_conv_path(CCP_POSIX_TO_WIN_A, path, retval, MAX_PATH)

        return retval.value

else:

    def native_path(path):
        return fully_normalize_path(path)


def __getfilesystemencoding():
    '''
    Note: there's a copy of this method in _pydev_filesystem_encoding.py
    '''
    try:
        ret = sys.getfilesystemencoding()
        if not ret:
            raise RuntimeError('Unable to get encoding.')
        return ret
    except:
        try:
            # Handle Jython
            from java.lang import System  # @UnresolvedImport
            env = System.getProperty("os.name").lower()
            if env.find('win') != -1:
                return 'ISO-8859-1'  # mbcs does not work on Jython, so, use a (hopefully) suitable replacement
            return 'utf-8'
        except:
            pass

        # Only available from 2.3 onwards.
        if sys.platform == 'win32':
            return 'mbcs'
        return 'utf-8'


def getfilesystemencoding():
    try:
        ret = __getfilesystemencoding()

        # Check if the encoding is actually there to be used!
        if hasattr('', 'encode'):
            ''.encode(ret)
        if hasattr('', 'decode'):
            ''.decode(ret)

        return ret
    except:
        return 'utf-8'


file_system_encoding = getfilesystemencoding()

if IS_PYTHON_3_ONWARDS:
    unicode_type = str
    bytes_type = bytes

else:
    unicode_type = unicode
    bytes_type = str


def tounicode(s):
    if hasattr(s, 'decode'):
        if not isinstance(s, unicode_type):
            # Depending on the platform variant we may have decode on string or not.
            return s.decode(file_system_encoding)
    return s


def tobytes(s):
    if hasattr(s, 'encode'):
        if not isinstance(s, bytes_type):
            return s.encode(file_system_encoding)
    return s


def toasciimxl(s):
    # output for xml without a declared encoding

    # As the output is xml, we have to encode chars (< and > are ok as they're not accepted in the filesystem name --
    # if it was allowed, we'd have to do things more selectively so that < and > don't get wrongly replaced).
    s = s.replace("&", "&amp;")

    try:
        ret = s.encode('ascii', 'xmlcharrefreplace')
    except:
        # use workaround
        ret = ''
        for c in s:
            try:
                ret += c.encode('ascii')
            except:
                try:
                    # Python 2: unicode is a valid identifier
                    ret += unicode("&#%d;") % ord(c)
                except:
                    # Python 3: a string is already unicode, so, just doing it directly should work.
                    ret += "&#%d;" % ord(c)
    return ret


if __name__ == '__main__':
    try:
        # just give some time to get the reading threads attached (just in case)
        import time
        time.sleep(0.1)
    except:
        pass

    try:
        executable = tounicode(native_path(sys.executable))
    except:
        executable = tounicode(sys.executable)

    if sys.platform == "cygwin" and not executable.endswith(tounicode('.exe')):
        executable += tounicode('.exe')

    try:
        major = str(sys.version_info[0])
        minor = str(sys.version_info[1])
    except AttributeError:
        # older versions of python don't have version_info
        import string
        s = string.split(sys.version, ' ')[0]
        s = string.split(s, '.')
        major = s[0]
        minor = s[1]

    s = tounicode('%s.%s') % (tounicode(major), tounicode(minor))

    contents = [tounicode('<xml>')]
    contents.append(tounicode('<version>%s</version>') % (tounicode(s),))

    contents.append(tounicode('<executable>%s</executable>') % tounicode(executable))

    # this is the new implementation to get the system folders
    # (still need to check if it works in linux)
    # (previously, we were getting the executable dir, but that is not always correct...)
    prefix = tounicode(native_path(sys.prefix))
    # print_ 'prefix is', prefix

    result = []

    path_used = sys.path
    try:
        path_used = path_used[1:]  # Use a copy (and don't include the directory of this script as a path.)
    except:
        pass  # just ignore it...

    for p in path_used:
        p = tounicode(native_path(p))

        try:
            import string  # to be compatible with older versions
            if string.find(p, prefix) == 0:  # was startswith
                result.append((p, True))
            else:
                result.append((p, False))
        except (ImportError, AttributeError):
            # python 3k also does not have it
            # jython may not have it (depending on how are things configured)
            if p.startswith(prefix):  # was startswith
                result.append((p, True))
            else:
                result.append((p, False))

    for p, b in result:
        if b:
            contents.append(tounicode('<lib path="ins">%s</lib>') % (p,))
        else:
            contents.append(tounicode('<lib path="out">%s</lib>') % (p,))

    # no compiled libs
    # nor forced libs

    for builtinMod in sys.builtin_module_names:
        contents.append(tounicode('<forced_lib>%s</forced_lib>') % tounicode(builtinMod))

    contents.append(tounicode('</xml>'))
    unic = tounicode('\n').join(contents)
    inasciixml = toasciimxl(unic)
    if IS_PYTHON_3_ONWARDS:
        # This is the 'official' way of writing binary output in Py3K (see: http://bugs.python.org/issue4571)
        sys.stdout.buffer.write(inasciixml)
    else:
        sys.stdout.write(inasciixml)

    sys.stdout.flush()
    sys.stderr.flush()
