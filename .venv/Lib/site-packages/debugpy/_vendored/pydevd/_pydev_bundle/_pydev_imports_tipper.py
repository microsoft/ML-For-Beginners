import inspect
import os.path
import sys

from _pydev_bundle._pydev_tipper_common import do_find
from _pydevd_bundle.pydevd_utils import hasattr_checked, dir_checked

from inspect import getfullargspec


def getargspec(*args, **kwargs):
    arg_spec = getfullargspec(*args, **kwargs)
    return arg_spec.args, arg_spec.varargs, arg_spec.varkw, arg_spec.defaults, arg_spec.kwonlyargs or [], arg_spec.kwonlydefaults or {}


# completion types.
TYPE_IMPORT = '0'
TYPE_CLASS = '1'
TYPE_FUNCTION = '2'
TYPE_ATTR = '3'
TYPE_BUILTIN = '4'
TYPE_PARAM = '5'


def _imp(name, log=None):
    try:
        return __import__(name)
    except:
        if '.' in name:
            sub = name[0:name.rfind('.')]

            if log is not None:
                log.add_content('Unable to import', name, 'trying with', sub)
                log.add_exception()

            return _imp(sub, log)
        else:
            s = 'Unable to import module: %s - sys.path: %s' % (str(name), sys.path)
            if log is not None:
                log.add_content(s)
                log.add_exception()

            raise ImportError(s)


IS_IPY = False
if sys.platform == 'cli':
    IS_IPY = True
    _old_imp = _imp

    def _imp(name, log=None):
        # We must add a reference in clr for .Net
        import clr  # @UnresolvedImport
        initial_name = name
        while '.' in name:
            try:
                clr.AddReference(name)
                break  # If it worked, that's OK.
            except:
                name = name[0:name.rfind('.')]
        else:
            try:
                clr.AddReference(name)
            except:
                pass  # That's OK (not dot net module).

        return _old_imp(initial_name, log)


def get_file(mod):
    f = None
    try:
        f = inspect.getsourcefile(mod) or inspect.getfile(mod)
    except:
        try:
            f = getattr(mod, '__file__', None)
        except:
            f = None
        if f and f.lower(f[-4:]) in ['.pyc', '.pyo']:
            filename = f[:-4] + '.py'
            if os.path.exists(filename):
                f = filename

    return f


def Find(name, log=None):
    f = None

    mod = _imp(name, log)
    parent = mod
    foundAs = ''

    if inspect.ismodule(mod):
        f = get_file(mod)

    components = name.split('.')

    old_comp = None
    for comp in components[1:]:
        try:
            # this happens in the following case:
            # we have mx.DateTime.mxDateTime.mxDateTime.pyd
            # but after importing it, mx.DateTime.mxDateTime shadows access to mxDateTime.pyd
            mod = getattr(mod, comp)
        except AttributeError:
            if old_comp != comp:
                raise

        if inspect.ismodule(mod):
            f = get_file(mod)
        else:
            if len(foundAs) > 0:
                foundAs = foundAs + '.'
            foundAs = foundAs + comp

        old_comp = comp

    return f, mod, parent, foundAs


def search_definition(data):
    '''@return file, line, col
    '''

    data = data.replace('\n', '')
    if data.endswith('.'):
        data = data.rstrip('.')
    f, mod, parent, foundAs = Find(data)
    try:
        return do_find(f, mod), foundAs
    except:
        return do_find(f, parent), foundAs


def generate_tip(data, log=None):
    data = data.replace('\n', '')
    if data.endswith('.'):
        data = data.rstrip('.')

    f, mod, parent, foundAs = Find(data, log)
    # print_ >> open('temp.txt', 'w'), f
    tips = generate_imports_tip_for_module(mod)
    return f, tips


def check_char(c):
    if c == '-' or c == '.':
        return '_'
    return c


_SENTINEL = object()


def generate_imports_tip_for_module(obj_to_complete, dir_comps=None, getattr=getattr, filter=lambda name:True):
    '''
        @param obj_to_complete: the object from where we should get the completions
        @param dir_comps: if passed, we should not 'dir' the object and should just iterate those passed as kwonly_arg parameter
        @param getattr: the way to get kwonly_arg given object from the obj_to_complete (used for the completer)
        @param filter: kwonly_arg callable that receives the name and decides if it should be appended or not to the results
        @return: list of tuples, so that each tuple represents kwonly_arg completion with:
            name, doc, args, type (from the TYPE_* constants)
    '''
    ret = []

    if dir_comps is None:
        dir_comps = dir_checked(obj_to_complete)
        if hasattr_checked(obj_to_complete, '__dict__'):
            dir_comps.append('__dict__')
        if hasattr_checked(obj_to_complete, '__class__'):
            dir_comps.append('__class__')

    get_complete_info = True

    if len(dir_comps) > 1000:
        # ok, we don't want to let our users wait forever...
        # no complete info for you...

        get_complete_info = False

    dontGetDocsOn = (float, int, str, tuple, list, dict)
    dontGetattrOn = (dict, list, set, tuple)
    for d in dir_comps:

        if d is None:
            continue

        if not filter(d):
            continue

        args = ''

        try:
            try:
                if isinstance(obj_to_complete, dontGetattrOn):
                    raise Exception('Since python 3.9, e.g. "dict[str]" will return'
                                    " a dict that's only supposed to take strings. "
                                    'Interestingly, e.g. dict["val"] is also valid '
                                    'and presumably represents a dict that only takes '
                                    'keys that are "val". This breaks our check for '
                                    'class attributes.')
                obj = getattr(obj_to_complete.__class__, d)
            except:
                obj = getattr(obj_to_complete, d)
        except:  # just ignore and get it without additional info
            ret.append((d, '', args, TYPE_BUILTIN))
        else:

            if get_complete_info:
                try:
                    retType = TYPE_BUILTIN

                    # check if we have to get docs
                    getDoc = True
                    for class_ in dontGetDocsOn:

                        if isinstance(obj, class_):
                            getDoc = False
                            break

                    doc = ''
                    if getDoc:
                        # no need to get this info... too many constants are defined and
                        # makes things much slower (passing all that through sockets takes quite some time)
                        try:
                            doc = inspect.getdoc(obj)
                            if doc is None:
                                doc = ''
                        except:  # may happen on jython when checking java classes (so, just ignore it)
                            doc = ''

                    if inspect.ismethod(obj) or inspect.isbuiltin(obj) or inspect.isfunction(obj) or inspect.isroutine(obj):
                        try:
                            args, vargs, kwargs, defaults, kwonly_args, kwonly_defaults = getargspec(obj)

                            args = args[:]

                            for kwonly_arg in kwonly_args:
                                default = kwonly_defaults.get(kwonly_arg, _SENTINEL)
                                if default is not _SENTINEL:
                                    args.append('%s=%s' % (kwonly_arg, default))
                                else:
                                    args.append(str(kwonly_arg))

                            args = '(%s)' % (', '.join(args))
                        except TypeError:
                            # ok, let's see if we can get the arguments from the doc
                            args, doc = signature_from_docstring(doc, getattr(obj, '__name__', None))

                        retType = TYPE_FUNCTION

                    elif inspect.isclass(obj):
                        retType = TYPE_CLASS

                    elif inspect.ismodule(obj):
                        retType = TYPE_IMPORT

                    else:
                        retType = TYPE_ATTR

                    # add token and doc to return - assure only strings.
                    ret.append((d, doc, args, retType))

                except:  # just ignore and get it without aditional info
                    ret.append((d, '', args, TYPE_BUILTIN))

            else:  # get_complete_info == False
                if inspect.ismethod(obj) or inspect.isbuiltin(obj) or inspect.isfunction(obj) or inspect.isroutine(obj):
                    retType = TYPE_FUNCTION

                elif inspect.isclass(obj):
                    retType = TYPE_CLASS

                elif inspect.ismodule(obj):
                    retType = TYPE_IMPORT

                else:
                    retType = TYPE_ATTR
                # ok, no complete info, let's try to do this as fast and clean as possible
                # so, no docs for this kind of information, only the signatures
                ret.append((d, '', str(args), retType))

    return ret


def signature_from_docstring(doc, obj_name):
    args = '()'
    try:
        found = False
        if len(doc) > 0:
            if IS_IPY:
                # Handle case where we have the situation below
                # sort(self, object cmp, object key)
                # sort(self, object cmp, object key, bool reverse)
                # sort(self)
                # sort(self, object cmp)

                # Or: sort(self: list, cmp: object, key: object)
                # sort(self: list, cmp: object, key: object, reverse: bool)
                # sort(self: list)
                # sort(self: list, cmp: object)
                if obj_name:
                    name = obj_name + '('

                    # Fix issue where it was appearing sort(aa)sort(bb)sort(cc) in the same line.
                    lines = doc.splitlines()
                    if len(lines) == 1:
                        c = doc.count(name)
                        if c > 1:
                            doc = ('\n' + name).join(doc.split(name))

                    major = ''
                    for line in doc.splitlines():
                        if line.startswith(name) and line.endswith(')'):
                            if len(line) > len(major):
                                major = line
                    if major:
                        args = major[major.index('('):]
                        found = True

            if not found:
                i = doc.find('->')
                if i < 0:
                    i = doc.find('--')
                    if i < 0:
                        i = doc.find('\n')
                        if i < 0:
                            i = doc.find('\r')

                if i > 0:
                    s = doc[0:i]
                    s = s.strip()

                    # let's see if we have a docstring in the first line
                    if s[-1] == ')':
                        start = s.find('(')
                        if start >= 0:
                            end = s.find('[')
                            if end <= 0:
                                end = s.find(')')
                                if end <= 0:
                                    end = len(s)

                            args = s[start:end]
                            if not args[-1] == ')':
                                args = args + ')'

                            # now, get rid of unwanted chars
                            l = len(args) - 1
                            r = []
                            for i in range(len(args)):
                                if i == 0 or i == l:
                                    r.append(args[i])
                                else:
                                    r.append(check_char(args[i]))

                            args = ''.join(r)

            if IS_IPY:
                if args.startswith('(self:'):
                    i = args.find(',')
                    if i >= 0:
                        args = '(self' + args[i:]
                    else:
                        args = '(self)'
                i = args.find(')')
                if i > 0:
                    args = args[:i + 1]

    except:
        pass
    return args, doc
