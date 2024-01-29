# encoding: utf-8
"""Implementations for various useful completers.

These are all loaded by default by IPython.
"""
#-----------------------------------------------------------------------------
#  Copyright (C) 2010-2011 The IPython Development Team.
#
#  Distributed under the terms of the BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib imports
import glob
import inspect
import os
import re
import sys
from importlib import import_module
from importlib.machinery import all_suffixes


# Third-party imports
from time import time
from zipimport import zipimporter

# Our own imports
from .completer import expand_user, compress_user
from .error import TryNext
from ..utils._process_common import arg_split

# FIXME: this should be pulled in with the right call via the component system
from IPython import get_ipython

from typing import List

#-----------------------------------------------------------------------------
# Globals and constants
#-----------------------------------------------------------------------------
_suffixes = all_suffixes()

# Time in seconds after which the rootmodules will be stored permanently in the
# ipython ip.db database (kept in the user's .ipython dir).
TIMEOUT_STORAGE = 2

# Time in seconds after which we give up
TIMEOUT_GIVEUP = 20

# Regular expression for the python import statement
import_re = re.compile(r'(?P<name>[^\W\d]\w*?)'
                       r'(?P<package>[/\\]__init__)?'
                       r'(?P<suffix>%s)$' %
                       r'|'.join(re.escape(s) for s in _suffixes))

# RE for the ipython %run command (python + ipython scripts)
magic_run_re = re.compile(r'.*(\.ipy|\.ipynb|\.py[w]?)$')

#-----------------------------------------------------------------------------
# Local utilities
#-----------------------------------------------------------------------------

def module_list(path):
    """
    Return the list containing the names of the modules available in the given
    folder.
    """
    # sys.path has the cwd as an empty string, but isdir/listdir need it as '.'
    if path == '':
        path = '.'

    # A few local constants to be used in loops below
    pjoin = os.path.join

    if os.path.isdir(path):
        # Build a list of all files in the directory and all files
        # in its subdirectories. For performance reasons, do not
        # recurse more than one level into subdirectories.
        files = []
        for root, dirs, nondirs in os.walk(path, followlinks=True):
            subdir = root[len(path)+1:]
            if subdir:
                files.extend(pjoin(subdir, f) for f in nondirs)
                dirs[:] = [] # Do not recurse into additional subdirectories.
            else:
                files.extend(nondirs)

    else:
        try:
            files = list(zipimporter(path)._files.keys())
        except:
            files = []

    # Build a list of modules which match the import_re regex.
    modules = []
    for f in files:
        m = import_re.match(f)
        if m:
            modules.append(m.group('name'))
    return list(set(modules))


def get_root_modules():
    """
    Returns a list containing the names of all the modules available in the
    folders of the pythonpath.

    ip.db['rootmodules_cache'] maps sys.path entries to list of modules.
    """
    ip = get_ipython()
    if ip is None:
        # No global shell instance to store cached list of modules.
        # Don't try to scan for modules every time.
        return list(sys.builtin_module_names)

    if getattr(ip.db, "_mock", False):
        rootmodules_cache = {}
    else:
        rootmodules_cache = ip.db.get("rootmodules_cache", {})
    rootmodules = list(sys.builtin_module_names)
    start_time = time()
    store = False
    for path in sys.path:
        try:
            modules = rootmodules_cache[path]
        except KeyError:
            modules = module_list(path)
            try:
                modules.remove('__init__')
            except ValueError:
                pass
            if path not in ('', '.'): # cwd modules should not be cached
                rootmodules_cache[path] = modules
            if time() - start_time > TIMEOUT_STORAGE and not store:
                store = True
                print("\nCaching the list of root modules, please wait!")
                print("(This will only be done once - type '%rehashx' to "
                      "reset cache!)\n")
                sys.stdout.flush()
            if time() - start_time > TIMEOUT_GIVEUP:
                print("This is taking too long, we give up.\n")
                return []
        rootmodules.extend(modules)
    if store:
        ip.db['rootmodules_cache'] = rootmodules_cache
    rootmodules = list(set(rootmodules))
    return rootmodules


def is_importable(module, attr, only_modules):
    if only_modules:
        return inspect.ismodule(getattr(module, attr))
    else:
        return not(attr[:2] == '__' and attr[-2:] == '__')

def is_possible_submodule(module, attr):
    try:
        obj = getattr(module, attr)
    except AttributeError:
        # Is possilby an unimported submodule
        return True
    except TypeError:
        # https://github.com/ipython/ipython/issues/9678
        return False
    return inspect.ismodule(obj)


def try_import(mod: str, only_modules=False) -> List[str]:
    """
    Try to import given module and return list of potential completions.
    """
    mod = mod.rstrip('.')
    try:
        m = import_module(mod)
    except:
        return []

    m_is_init = '__init__' in (getattr(m, '__file__', '') or '')

    completions = []
    if (not hasattr(m, '__file__')) or (not only_modules) or m_is_init:
        completions.extend( [attr for attr in dir(m) if
                             is_importable(m, attr, only_modules)])

    m_all = getattr(m, "__all__", [])
    if only_modules:
        completions.extend(attr for attr in m_all if is_possible_submodule(m, attr))
    else:
        completions.extend(m_all)

    if m_is_init:
        file_ = m.__file__
        completions.extend(module_list(os.path.dirname(file_)))
    completions_set = {c for c in completions if isinstance(c, str)}
    completions_set.discard('__init__')
    return list(completions_set)


#-----------------------------------------------------------------------------
# Completion-related functions.
#-----------------------------------------------------------------------------

def quick_completer(cmd, completions):
    r""" Easily create a trivial completer for a command.

    Takes either a list of completions, or all completions in string (that will
    be split on whitespace).

    Example::

        [d:\ipython]|1> import ipy_completers
        [d:\ipython]|2> ipy_completers.quick_completer('foo', ['bar','baz'])
        [d:\ipython]|3> foo b<TAB>
        bar baz
        [d:\ipython]|3> foo ba
    """

    if isinstance(completions, str):
        completions = completions.split()

    def do_complete(self, event):
        return completions

    get_ipython().set_hook('complete_command',do_complete, str_key = cmd)

def module_completion(line):
    """
    Returns a list containing the completion possibilities for an import line.

    The line looks like this :
    'import xml.d'
    'from xml.dom import'
    """

    words = line.split(' ')
    nwords = len(words)

    # from whatever <tab> -> 'import '
    if nwords == 3 and words[0] == 'from':
        return ['import ']

    # 'from xy<tab>' or 'import xy<tab>'
    if nwords < 3 and (words[0] in {'%aimport', 'import', 'from'}) :
        if nwords == 1:
            return get_root_modules()
        mod = words[1].split('.')
        if len(mod) < 2:
            return get_root_modules()
        completion_list = try_import('.'.join(mod[:-1]), True)
        return ['.'.join(mod[:-1] + [el]) for el in completion_list]

    # 'from xyz import abc<tab>'
    if nwords >= 3 and words[0] == 'from':
        mod = words[1]
        return try_import(mod)

#-----------------------------------------------------------------------------
# Completers
#-----------------------------------------------------------------------------
# These all have the func(self, event) signature to be used as custom
# completers

def module_completer(self,event):
    """Give completions after user has typed 'import ...' or 'from ...'"""

    # This works in all versions of python.  While 2.5 has
    # pkgutil.walk_packages(), that particular routine is fairly dangerous,
    # since it imports *EVERYTHING* on sys.path.  That is: a) very slow b) full
    # of possibly problematic side effects.
    # This search the folders in the sys.path for available modules.

    return module_completion(event.line)

# FIXME: there's a lot of logic common to the run, cd and builtin file
# completers, that is currently reimplemented in each.

def magic_run_completer(self, event):
    """Complete files that end in .py or .ipy or .ipynb for the %run command.
    """
    comps = arg_split(event.line, strict=False)
    # relpath should be the current token that we need to complete.
    if (len(comps) > 1) and (not event.line.endswith(' ')):
        relpath = comps[-1].strip("'\"")
    else:
        relpath = ''

    #print("\nev=", event)  # dbg
    #print("rp=", relpath)  # dbg
    #print('comps=', comps)  # dbg

    lglob = glob.glob
    isdir = os.path.isdir
    relpath, tilde_expand, tilde_val = expand_user(relpath)

    # Find if the user has already typed the first filename, after which we
    # should complete on all files, since after the first one other files may
    # be arguments to the input script.

    if any(magic_run_re.match(c) for c in comps):
        matches =  [f.replace('\\','/') + ('/' if isdir(f) else '')
                            for f in lglob(relpath+'*')]
    else:
        dirs = [f.replace('\\','/') + "/" for f in lglob(relpath+'*') if isdir(f)]
        pys =  [f.replace('\\','/')
                for f in lglob(relpath+'*.py') + lglob(relpath+'*.ipy') +
                lglob(relpath+'*.ipynb') + lglob(relpath + '*.pyw')]

        matches = dirs + pys

    #print('run comp:', dirs+pys) # dbg
    return [compress_user(p, tilde_expand, tilde_val) for p in matches]


def cd_completer(self, event):
    """Completer function for cd, which only returns directories."""
    ip = get_ipython()
    relpath = event.symbol

    #print(event) # dbg
    if event.line.endswith('-b') or ' -b ' in event.line:
        # return only bookmark completions
        bkms = self.db.get('bookmarks', None)
        if bkms:
            return bkms.keys()
        else:
            return []

    if event.symbol == '-':
        width_dh = str(len(str(len(ip.user_ns['_dh']) + 1)))
        # jump in directory history by number
        fmt = '-%0' + width_dh +'d [%s]'
        ents = [ fmt % (i,s) for i,s in enumerate(ip.user_ns['_dh'])]
        if len(ents) > 1:
            return ents
        return []

    if event.symbol.startswith('--'):
        return ["--" + os.path.basename(d) for d in ip.user_ns['_dh']]

    # Expand ~ in path and normalize directory separators.
    relpath, tilde_expand, tilde_val = expand_user(relpath)
    relpath = relpath.replace('\\','/')

    found = []
    for d in [f.replace('\\','/') + '/' for f in glob.glob(relpath+'*')
              if os.path.isdir(f)]:
        if ' ' in d:
            # we don't want to deal with any of that, complex code
            # for this is elsewhere
            raise TryNext

        found.append(d)

    if not found:
        if os.path.isdir(relpath):
            return [compress_user(relpath, tilde_expand, tilde_val)]

        # if no completions so far, try bookmarks
        bks = self.db.get('bookmarks',{})
        bkmatches = [s for s in bks if s.startswith(event.symbol)]
        if bkmatches:
            return bkmatches

        raise TryNext

    return [compress_user(p, tilde_expand, tilde_val) for p in found]

def reset_completer(self, event):
    "A completer for %reset magic"
    return '-f -s in out array dhist'.split()
