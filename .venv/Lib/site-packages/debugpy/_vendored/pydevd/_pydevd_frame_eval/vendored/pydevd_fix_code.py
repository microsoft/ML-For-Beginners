def _fix_contents(filename, contents):
    import re

    contents = re.sub(
        r"from bytecode", r'from _pydevd_frame_eval.vendored.bytecode', contents, flags=re.MULTILINE
    )

    contents = re.sub(
        r"import bytecode", r'from _pydevd_frame_eval.vendored import bytecode', contents, flags=re.MULTILINE
    )

    # This test will import the wrong setup (we're not interested in it).
    contents = re.sub(
        r"def test_version\(self\):", r'def skip_test_version(self):', contents, flags=re.MULTILINE
    )

    if filename.startswith('test_'):
        if 'pytestmark' not in contents:
            pytest_mark = '''
import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
pytestmark = pytest.mark.skipif(not IS_PY36_OR_GREATER or not IS_CPYTHON or not TEST_CYTHON, reason='Requires CPython >= 3.6')
'''
            contents = pytest_mark + contents
    return contents


def main():
    import os

    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(os.path.dirname(__file__)):
        path = root.split(os.sep)
        for filename in files:
            if filename.endswith('.py') and filename != 'pydevd_fix_code.py':
                with open(os.path.join(root, filename), 'r') as stream:
                    contents = stream.read()

                new_contents = _fix_contents(filename, contents)
                if contents != new_contents:
                    print('fixed ', os.path.join(root, filename))
                    with open(os.path.join(root, filename), 'w') as stream:
                        stream.write(new_contents)

#             print(len(path) * '---', filename)


if __name__ == '__main__':
    main()