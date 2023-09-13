'''
Helper module to do refactoring to convert names to pep8.
'''
import re
import os
import names_to_rename

_CAMEL_RE = re.compile(r'(?<=[a-z])([A-Z])')
_CAMEL_DEF_RE = re.compile(r'(def )((([A-Z0-9]+|[a-z0-9])[a-z][a-z0-9]*[A-Z]|[a-z0-9]*[A-Z][A-Z0-9]*[a-z])[A-Za-z0-9]*)')


def _normalize(name):
    return _CAMEL_RE.sub(lambda x: '_' + x.group(1).lower(), name).lower()


def find_matches_in_contents(contents):
    return [x[1] for x in re.findall(_CAMEL_DEF_RE, contents)]


def iter_files_in_dir(dirname):
    for root, dirs, files in os.walk(dirname):
        for name in ('pydevd_attach_to_process', '.git', 'stubs', 'pydev_ipython', 'third_party', 'pydev_ipython'):
            try:
                dirs.remove(name)
            except:
                pass
        for filename in files:
            if filename.endswith('.py') and filename not in ('rename_pep8.py', 'names_to_rename.py'):
                path = os.path.join(root, filename)
                with open(path, 'rb') as stream:
                    initial_contents = stream.read()

                yield path, initial_contents


def find_matches():
    found = set()
    for path, initial_contents in iter_files_in_dir(os.path.dirname(os.path.dirname(__file__))):
        found.update(find_matches_in_contents(initial_contents))
    print('\n'.join(sorted(found)))
    print('Total', len(found))


def substitute_contents(re_name_to_new_val, initial_contents):
    contents = initial_contents
    for key, val in re_name_to_new_val.iteritems():
        contents = re.sub(key, val, contents)
    return contents


def make_replace():
    re_name_to_new_val = load_re_to_new_val(names_to_rename.NAMES)
    # traverse root directory, and list directories as dirs and files as files
    for path, initial_contents in iter_files_in_dir(os.path.dirname(os.path.dirname(__file__))):
        contents = substitute_contents(re_name_to_new_val, initial_contents)
        if contents != initial_contents:
            print('Changed something at: %s' % (path,))

            for val in re_name_to_new_val.itervalues():
                # Check in initial contents to see if it already existed!
                if re.findall(r'\b%s\b' % (val,), initial_contents):
                    raise AssertionError('Error in:\n%s\n%s is already being used (and changes may conflict).' % (path, val,))

            with open(path, 'wb') as stream:
                stream.write(contents)


def load_re_to_new_val(names):
    name_to_new_val = {}
    for n in names.splitlines():
        n = n.strip()
        if not n.startswith('#') and n:
            name_to_new_val[r'\b' + n + r'\b'] = _normalize(n)
    return name_to_new_val


def test():
    assert _normalize('RestoreSysSetTraceFunc') == 'restore_sys_set_trace_func'
    assert _normalize('restoreSysSetTraceFunc') == 'restore_sys_set_trace_func'
    assert _normalize('Restore') == 'restore'
    matches = find_matches_in_contents('''
    def CamelCase()
    def camelCase()
    def ignore()
    def ignore_this()
    def Camel()
    def CamelCaseAnother()
    ''')
    assert matches == ['CamelCase', 'camelCase', 'Camel', 'CamelCaseAnother']
    re_name_to_new_val = load_re_to_new_val('''
# Call -- skip
# Call1 -- skip
# Call2 -- skip
# Call3 -- skip
# Call4 -- skip
CustomFramesContainerInit
DictContains
DictItems
DictIterItems
DictIterValues
DictKeys
DictPop
DictValues
''')
    assert re_name_to_new_val == {'\\bDictPop\\b': 'dict_pop', '\\bDictItems\\b': 'dict_items', '\\bDictIterValues\\b': 'dict_iter_values', '\\bDictKeys\\b': 'dict_keys', '\\bDictContains\\b': 'dict_contains', '\\bDictIterItems\\b': 'dict_iter_items', '\\bCustomFramesContainerInit\\b': 'custom_frames_container_init', '\\bDictValues\\b': 'dict_values'}
    assert substitute_contents(re_name_to_new_val, '''
CustomFramesContainerInit
DictContains
DictItems
DictIterItems
DictIterValues
DictKeys
DictPop
DictValues
''') == '''
custom_frames_container_init
dict_contains
dict_items
dict_iter_items
dict_iter_values
dict_keys
dict_pop
dict_values
'''


if __name__ == '__main__':
#     find_matches()
    make_replace()
#     test()

