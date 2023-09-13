# coding: utf-8
from io import StringIO
import os.path
import sys
import traceback

from _pydevd_bundle.pydevd_collect_bytecode_info import collect_try_except_info, \
    collect_return_info, code_to_bytecode_representation
from tests_python.debugger_unittest import IS_CPYTHON, IS_PYPY
from _pydevd_bundle.pydevd_constants import IS_PY38_OR_GREATER, IS_JYTHON
from tests_python.debug_constants import IS_PY311_OR_GREATER


def _method_call_with_error():
    try:
        _method_reraise()
    except:
        raise


def _method_except_local():
    Foo = AssertionError
    try:  # SETUP_EXCEPT (to except line)
        raise AssertionError()
    except Foo as exc:
        # DUP_TOP, LOAD_FAST (x), COMPARE_OP (exception match), POP_JUMP_IF_FALSE
        pass


def _method_reraise():
    try:  # SETUP_EXCEPT (to except line)
        raise AssertionError()
    except AssertionError as e:  # POP_TOP
        raise e


def _method_return_with_error():
    _method_call_with_error()


def _method_return_with_error2():
    try:
        _method_call_with_error()
    except:
        return


def _method_simple_raise_any_except():
    try:  # SETUP_EXCEPT (to except line)
        raise AssertionError()
    except:  # POP_TOP
        pass


def _method_simple_raise_any_except_return_on_raise():
    # Note how the tracing the debugger has is equal to the tracing from _method_simple_raise_any_except
    # but this one resulted in an unhandled exception while the other one didn't.
    try:  # SETUP_EXCEPT (to except line)
        raise AssertionError()
    except:  # POP_TOP
        raise  # RAISE_VARARGS


def _method_simple_raise_local_load():
    x = AssertionError
    try:  # SETUP_EXCEPT (to except line)
        raise AssertionError()
    except x as exc:
        # DUP_TOP, LOAD_GLOBAL (NameError), LOAD_GLOBAL(AssertionError), BUILD_TUPLE,
        # COMPARE_OP (exception match), POP_JUMP_IF_FALSE
        pass


def _method_simple_raise_multi_except():
    try:  # SETUP_EXCEPT (to except line)
        raise AssertionError()
    except (NameError, AssertionError) as exc:
        # DUP_TOP, LOAD_FAST (x), COMPARE_OP (exception match), POP_JUMP_IF_FALSE
        pass


def _method_simple_raise_unmatched_except():
    try:  # SETUP_EXCEPT (to except line)
        raise AssertionError()
    except NameError:  # DUP_TOP, LOAD_GLOBAL (NameError), COMPARE_OP (exception match), POP_JUMP_IF_FALSE
        pass


class _Tracer(object):

    def __init__(self, partial_info=False):
        self.partial_info = partial_info
        self.stream = StringIO()
        self._in_print = False

    def tracer_printer(self, frame, event, arg):
        if self._in_print:
            return None
        self._in_print = True
        try:
            if arg is not None:
                if event == 'exception':
                    arg = arg[0].__name__
                elif arg is not None:
                    arg = str(arg)
            if arg is None:
                arg = ''

            if self.partial_info:
                s = ' '.join((
                    os.path.basename(frame.f_code.co_filename),
                    event.upper() if event != 'line' else event,
                    arg,
                ))
            else:
                s = ' '.join((
                    str(frame.f_lineno),
                    frame.f_code.co_name,
                    os.path.basename(frame.f_code.co_filename),
                    event.upper() if event != 'line' else event,
                    arg,
                ))
            self.writeln(s)
        except:
            traceback.print_exc()
        self._in_print = False
        return self.tracer_printer

    def writeln(self, s):
        self.write(s)
        self.write('\n')

    def write(self, s):
        if isinstance(s, bytes):
            s = s.decode('utf-8')
        self.stream.write(s)

    def call(self, c):
        sys.settrace(self.tracer_printer)
        try:
            c()
        except:
            pass
        sys.settrace(None)
        return self.stream.getvalue()


import pytest


class _ExcVerifier(object):

    def __init__(self, pyfile):
        self.pyfile = pyfile

    def check(self, method, expected_as_str, expected_as_str_source_version=None, update_try_except_infos=None):
        code = method.__code__

        try_except_infos = sorted(collect_try_except_info(code, use_func_first_line=True), key=lambda t:t.try_line)
        if IS_CPYTHON or IS_PYPY:
            if update_try_except_infos is not None:
                update_try_except_infos(try_except_infos)

            if sys.version_info[:2] not in ((3, 10), (3, 11)):
                assert str(try_except_infos) == expected_as_str
            from _pydevd_bundle.pydevd_collect_bytecode_info import collect_try_except_info_from_source

            expected_as_str_source_version = expected_as_str_source_version or expected_as_str
            try_except_infos = collect_try_except_info_from_source(self.pyfile(method))
            if update_try_except_infos is not None:
                update_try_except_infos(try_except_infos)
            assert str(try_except_infos) == expected_as_str_source_version
        else:
            assert try_except_infos == []


@pytest.fixture
def exc_verifier(pyfile):
    return _ExcVerifier(pyfile)


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_collect_try_except_info(data_regression, pyfile):
    from _pydevd_bundle.pydevd_collect_bytecode_info import collect_try_except_info_from_source
    method_to_info = {}
    method_to_info_from_source = {}
    for key, method in sorted(dict(globals()).items()):
        if key.startswith('_method'):

            info = collect_try_except_info_from_source(pyfile(method))
            method_to_info_from_source[key] = sorted(str(x) for x in info)

            info = collect_try_except_info(method.__code__, use_func_first_line=True)
            method_to_info[key] = sorted(str(x) for x in info)

    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        data_regression.check(method_to_info)

    data_regression.check(method_to_info_from_source)


def test_collect_try_except_info2(exc_verifier):

    def method():
        try:
            raise AssertionError
        except:
            _a = 10
            raise
        finally:
            _b = 20
        _c = 20

    exc_verifier.check(method, '[{try:1 except 3 end block 5 raises: 5}]')


def test_collect_try_except_info3(exc_verifier):

    def method():
        get_exc_class = lambda:AssertionError
        try:  # SETUP_EXCEPT (to except line)
            raise AssertionError()
        except get_exc_class() \
                as e:  # POP_TOP
            raise e

    exc_verifier.check(method, '[{try:2 except 4 end block 6}]')


def test_collect_try_except_info4(exc_verifier):

    def method():
        for i in range(2):
            try:
                raise AssertionError()
            except AssertionError:
                if i == 1:
                    try:
                        raise
                    except:
                        pass

        _foo = 10

    exc_verifier.check(
        method,
        '[{try:2 except 4 end block 9 raises: 7}, {try:6 except 8 end block 9 raises: 7}]',
        '[{try:2 except 4 end block 9 raises: 7}, {try:6 except 8 end block 9}]',
    )


def test_collect_try_except_info4a(exc_verifier):

    def method():
        for i in range(2):
            try:
                raise AssertionError()
            except:
                if i == 1:
                    try:
                        raise
                    except:
                        pass

        _foo = 10

    exc_verifier.check(method,
        '[{try:2 except 4 end block 9 raises: 7}, {try:6 except 8 end block 9 raises: 7}]',
        '[{try:2 except 4 end block 9 raises: 7}, {try:6 except 8 end block 9}]',
    )


def test_collect_try_except_info_raise_unhandled7(exc_verifier):

    def raise_unhandled7():
        try:
            raise AssertionError()
        except AssertionError:
            try:
                raise AssertionError()
            except RuntimeError:
                pass

    exc_verifier.check(raise_unhandled7, '[{try:1 except 3 end block 7}, {try:4 except 6 end block 7}]')


def test_collect_try_except_info_raise_unhandled10(exc_verifier):

    def raise_unhandled10():
        for i in range(2):
            try:
                raise AssertionError()
            except AssertionError:
                if i == 1:
                    try:
                        raise
                    except RuntimeError:
                        pass

    exc_verifier.check(
        raise_unhandled10,
        '[{try:2 except 4 end block 9 raises: 7}, {try:6 except 8 end block 9 raises: 7}]',
        '[{try:2 except 4 end block 9 raises: 7}, {try:6 except 8 end block 9}]',
    )


def test_collect_try_except_info_return_on_except(exc_verifier):

    def method():
        try:  # SETUP_EXCEPT (to except line)
            try:  # SETUP_EXCEPT (to except line)
                raise AssertionError()
            except:  # POP_TOP
                raise
        except:  # POP_TOP
            return (
                1,
                2
            )

    def update_try_except_infos(try_except_infos):
        for try_except_info in try_except_infos:
            # On 3.7/3.8 the last bytecode actually has a different start line.
            if try_except_info.except_end_line in (7, 8):
                try_except_info.except_end_line = 9

    try_except_info_for_source = '[{try:1 except 6 end block 10}, {try:2 except 4 end block 5 raises: 5}]'
    if sys.version_info[:2] <= (3, 7):
        # The ast doesn't have end_lineno, so, the end block must be calculated based on children lineno (and thus is a bit different).
        try_except_info_for_source = '[{try:1 except 6 end block 9}, {try:2 except 4 end block 5 raises: 5}]'

    exc_verifier.check(
        method,
        '[{try:1 except 6 end block 9 raises: 5}, {try:2 except 4 end block 5 raises: 5}]',
        try_except_info_for_source,
        update_try_except_infos=update_try_except_infos
    )


def test_collect_try_except_info_with(exc_verifier):

    def try_except_with():
        try:
            with object():
                pass
        except AssertionError:
            pass

    exc_verifier.check(try_except_with, '[{try:1 except 4 end block 5}]')


def test_collect_try_except_info_in_single_line_1(exc_verifier):

    def try_except_single_line():
        try:range()
        except:
            return False
        return True

    exc_verifier.check(try_except_single_line, '[{try:1 except 2 end block 3}]')


def test_collect_try_except_info_in_single_line_2(exc_verifier):

    def try_except_single_line():
        try:range()
        except: return False
        return True

    exc_verifier.check(try_except_single_line, '[{try:1 except 2 end block 2}]')


def test_collect_try_except_info_multiple_except(exc_verifier):

    def try_except_with():
        try:
            pass
        except AssertionError:
            a = 1
        except RuntimeError:
            a = 2
        except:
            a = 3

    exc_verifier.check(try_except_with, '[{try:1 except 3 end block 8}]')


def test_collect_try_except_info_async_for():
    if IS_PY311_OR_GREATER:
        pytest.skip('On Python 3.11 we just support collecting info from the AST.')

    # Not valid on Python 2.
    code_str = '''
async def try_except_with():
    try:
        async for a in object():
            b = 10
        else:
            b = 20
    except AssertionError:
        pass
'''

    namespace = {}
    exec(code_str, namespace, namespace)
    code = namespace['try_except_with'].__code__

    lst = sorted(collect_try_except_info(code, use_func_first_line=True), key=lambda t:t.try_line)
    if IS_CPYTHON or IS_PYPY:
        if IS_PY38_OR_GREATER:
            assert str(lst) == '[{try:1 except 6 end block 7}]'
        else:
            # Before Python 3.8 the async for does a try..except StopAsyncIteration internally.
            assert str(lst) in (
                '[{try:1 except 6 end block 7}, {try:2 except 2 end block 7}]',
                '[{try:1 except 6 end block 7}, {try:2 except 2 end block 2}]'
            )

        # The version from the contents should always be correct.
        from _pydevd_bundle.pydevd_collect_bytecode_info import collect_try_except_info_from_contents
        assert str(collect_try_except_info_from_contents(code_str)) == '[{try:3 except 8 end block 9}]'

    else:
        assert lst == []


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_collect_return_info():

    def method():
        return 1

    assert str(collect_return_info(method.__code__, use_func_first_line=True)) == '[{return: 1}]'

    def method2():
        pass

    assert str(collect_return_info(method2.__code__, use_func_first_line=True)) == '[{return: 1}]'

    def method3():
        yield 1
        yield 2

    assert str(collect_return_info(method3.__code__, use_func_first_line=True)) == '[{return: 2}]'

    def method4():
        return (1,
                2,
                3,
                4)

    assert str(collect_return_info(method4.__code__, use_func_first_line=True)) == \
        '[{return: 1}]' if IS_PY38_OR_GREATER else '[{return: 4}]'

    def method5():
        return \
            \
            1

    assert str(collect_return_info(method5.__code__, use_func_first_line=True)) == \
        '[{return: 1}]' if IS_PY38_OR_GREATER else '[{return: 3}]'

    code = '''
def method():
    if a:
        yield 1
        yield 2
        return 1
    else:
        pass
'''

    scope = {}
    exec(code, scope)
    assert str(collect_return_info(scope['method'].__code__, use_func_first_line=True)) == \
        '[{return: 4}, {return: 6}]'


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr():

    def method4():
        return (1,
                2,
                3,
                call('tnh %s' % 1))

    contents = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert contents.count('\n') == 4, 'Found:%s' % (contents,)


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr_many():

    def method4():
        a = call()
        if a == 20:
            [x for x in call()]

        def method2():
            for x in y:
                yield x
            raise AssertionError

        return (1,
                2,
                3,
                call('tnh 1' % 1))

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    # print(new_repr)


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr2():

    def method():
        print(10)

        def method4(a, b):
            return (1,
                    2,
                    3,
                    call('somestr %s' % 1))

        print(20)

    s = code_to_bytecode_representation(method.__code__, use_func_first_line=True)
    assert s.count('\n') == 9, 'Expected 9 lines. Found: %s in:>>\n%s\n<<' % (s.count('\n'), s)
    assert 'somestr' in s  # i.e.: the contents of the inner code have been added too


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr_simple_method_calls():

    def method4():
        call()
        a = 10
        call(1, 2, 3, a, b, "x")

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert 'call()' in new_repr
    assert 'call(1, 2, 3, a, b, \'x\')' in new_repr
    assert 'NULL' not in new_repr


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr_assign():

    def method4():
        a = call()
        return call()

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert 'a = call()' in new_repr
    assert 'return call()' in new_repr


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr_tuple():

    def method4():
        return (1, 2, call(3, 4))

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert 'return (1, 2, call(3, 4))' in new_repr


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_build_tuple():

    def method4():
        return call(1, (call2(), 2))

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert 'return call(1, (call2(), 2))' in new_repr


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr_return_tuple():

    def method4():
        return (1, 2, 3, a)

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert 'return (1, 2, 3, a)' in new_repr


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr_return_tuple_with_call():

    def method4():
        return (1, 2, 3, a())

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert 'return (1, 2, 3, a())' in new_repr


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr_attr():

    def method4():
        call(a.b.c)

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert 'call(a.b.c)' in new_repr


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_cls_method():

    def method4():

        class B:

            def method(self):
                self.a.b.c

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert 'self.a.b.c' in new_repr


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not have bytecode support.')
def test_simple_code_to_bytecode_repr_unicode():

    def method4():
        return 'áéíóú'

    new_repr = code_to_bytecode_representation(method4.__code__, use_func_first_line=True)
    assert repr('áéíóú') in new_repr


def _create_entry(instruction):
    argval = instruction.argval
    return dict(
        opname=instruction.opname,
        argval=argval,
        starts_line=instruction.starts_line,
        is_jump_target=instruction.is_jump_target,
    )


def debug_test_iter_bytecode(data_regression):
    # Note: not run by default, only to help visualizing bytecode and comparing differences among versions.
    import dis

    basename = 'test_iter_bytecode.py%s%s' % (sys.version_info[:2])
    method_to_info = {}
    for key, method in sorted(dict(globals()).items()):
        if key.startswith('_method'):
            info = []

            if sys.version_info[0] < 3:
                from _pydevd_bundle.pydevd_collect_bytecode_info import _iter_as_bytecode_as_instructions_py2
                iter_in = _iter_as_bytecode_as_instructions_py2(method.__code__)
            else:
                iter_in = dis.Bytecode(method)

            for instruction in iter_in:
                info.append(_create_entry(instruction))

            msg = []
            for d in info:
                line = []
                for k, v in sorted(d.items()):
                    line.append(u'%s=%s' % (k, v))
                msg.append(u'   '.join(line))
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            method_to_info[key] = msg

    data_regression.check(method_to_info, basename=basename)


def debug_test_tracing_output():  # Note: not run by default, only to debug tracing.
    from collections import defaultdict
    output_to_method_names = defaultdict(list)
    for key, val in sorted(dict(globals()).items()):
        if key.startswith('_method'):
            tracer = _Tracer(partial_info=False)
            output_to_method_names[tracer.call(val)].append(val.__name__)

    # Notes:
    #
    # Seen as the same by the tracing (so, we inspect the bytecode to disambiguate).
    #     _method_simple_raise_any_except
    #     _method_simple_raise_any_except_return_on_raise
    #     _method_simple_raise_multi_except
    #
    # The return with an exception is always None
    #
    # It's not possible to disambiguate from a return None, pass or raise just with the tracing
    # (only a raise with an exception is gotten by the debugger).
    for output, method_names in sorted(output_to_method_names.items(), key=lambda x:(-len(x[1]), ''.join(x[1]))):
        print('=' * 80)
        print('    %s ' % (', '.join(method_names),))
        print('=' * 80)
        print(output)
