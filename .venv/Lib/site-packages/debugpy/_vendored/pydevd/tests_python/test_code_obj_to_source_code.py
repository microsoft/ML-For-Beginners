from _pydevd_bundle.pydevd_code_to_source import code_obj_to_source
import pytest

# i.e.: Skip these tests (this is a work in progress / proof of concept / not ready to be used).
pytestmark = pytest.mark.skip


def check(obtained, expected, strip_return_none=True):
    keepends = False
    obtained_lines = list(obtained.rstrip().splitlines(keepends))
    if strip_return_none:
        obtained_lines = [x.replace('return None', '') for x in obtained_lines]

    expected_lines = list(expected.rstrip().splitlines(keepends))

    assert obtained_lines == expected_lines


def test_code_obj_to_source_make_class_and_func():
    code = '''
class MyClass(object, other_class):
    def my_func(self):
        print(self)
'''
    expected = '''
__module__=__name____qualname__=MyClassMyClass=(def MyClass():MyClass,object,other_class,)
    def MyClass.my_func(self):
        print(self)
'''

    co = compile(code, '<unused>', 'exec')

    contents = code_obj_to_source(co)
    check(contents, expected)


def test_code_obj_to_source_lambda():
    code = 'my_func = lambda arg: (2,)'

    co = compile(code, '<unused>', 'exec')

    contents = code_obj_to_source(co)
    check(contents, 'my_func=<lambda>(arg):return return (2,)None')


def test_code_obj_to_source_make_func():
    code = '''

def my_func(arg1, arg2=2, arg3=3):
    some_call(arg1)
'''

    co = compile(code, '<unused>', 'exec')

    contents = code_obj_to_source(co)
    check(contents, code)


def test_code_obj_to_source_call_func():
    code = 'a=call1(call2(arg1))'

    co = compile(code, '<unused>', 'exec')

    contents = code_obj_to_source(co)
    check(contents, code)


def test_for_list_comp():
    code = '[x for x in range(10)]'

    co = compile(code, '<unused>', 'exec')

    contents = code_obj_to_source(co)
    check(contents, code)


def test_code_obj_to_source_for():
    code = 'for i in range(10):\n    print(i)'

    co = compile(code, '<unused>', 'exec')

    contents = code_obj_to_source(co)
    check(contents, code)


def test_code_obj_to_source_call_func2():
    code = '''a=call1(
call2(
arg1))
'''

    co = compile(code, '<unused>', 'exec')

    contents = code_obj_to_source(co)
    check(contents, code)
