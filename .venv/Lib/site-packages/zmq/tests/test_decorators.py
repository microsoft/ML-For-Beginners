import threading

from pytest import fixture, raises

import zmq
from zmq.decorators import context, socket
from zmq.tests import BaseZMQTestCase, term_context

##############################################
#  Test cases for @context
##############################################


@fixture(autouse=True)
def term_context_instance(request):
    request.addfinalizer(lambda: term_context(zmq.Context.instance(), timeout=10))


def test_ctx():
    @context()
    def test(ctx):
        assert isinstance(ctx, zmq.Context), ctx

    test()


def test_ctx_orig_args():
    @context()
    def f(foo, bar, ctx, baz=None):
        assert isinstance(ctx, zmq.Context), ctx
        assert foo == 42
        assert bar is True
        assert baz == 'mock'

    f(42, True, baz='mock')


def test_ctx_arg_naming():
    @context('myctx')
    def test(myctx):
        assert isinstance(myctx, zmq.Context), myctx

    test()


def test_ctx_args():
    @context('ctx', 5)
    def test(ctx):
        assert isinstance(ctx, zmq.Context), ctx
        assert ctx.IO_THREADS == 5, ctx.IO_THREADS

    test()


def test_ctx_arg_kwarg():
    @context('ctx', io_threads=5)
    def test(ctx):
        assert isinstance(ctx, zmq.Context), ctx
        assert ctx.IO_THREADS == 5, ctx.IO_THREADS

    test()


def test_ctx_kw_naming():
    @context(name='myctx')
    def test(myctx):
        assert isinstance(myctx, zmq.Context), myctx

    test()


def test_ctx_kwargs():
    @context(name='ctx', io_threads=5)
    def test(ctx):
        assert isinstance(ctx, zmq.Context), ctx
        assert ctx.IO_THREADS == 5, ctx.IO_THREADS

    test()


def test_ctx_kwargs_default():
    @context(name='ctx', io_threads=5)
    def test(ctx=None):
        assert isinstance(ctx, zmq.Context), ctx
        assert ctx.IO_THREADS == 5, ctx.IO_THREADS

    test()


def test_ctx_keyword_miss():
    @context(name='ctx')
    def test(other_name):
        pass  # the keyword ``ctx`` not found

    with raises(TypeError):
        test()


def test_ctx_multi_assign():
    @context(name='ctx')
    def test(ctx):
        pass  # explosion

    with raises(TypeError):
        test('mock')


def test_ctx_reinit():
    result = {'foo': None, 'bar': None}

    @context()
    def f(key, ctx):
        assert isinstance(ctx, zmq.Context), ctx
        result[key] = ctx

    foo_t = threading.Thread(target=f, args=('foo',))
    bar_t = threading.Thread(target=f, args=('bar',))

    foo_t.start()
    bar_t.start()

    foo_t.join()
    bar_t.join()

    assert result['foo'] is not None, result
    assert result['bar'] is not None, result
    assert result['foo'] is not result['bar'], result


def test_ctx_multi_thread():
    @context()
    @context()
    def f(foo, bar):
        assert isinstance(foo, zmq.Context), foo
        assert isinstance(bar, zmq.Context), bar

        assert len(set(map(id, [foo, bar]))) == 2, set(map(id, [foo, bar]))

    threads = [threading.Thread(target=f) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]


##############################################
#  Test cases for @socket
##############################################


def test_ctx_skt():
    @context()
    @socket(zmq.PUB)
    def test(ctx, skt):
        assert isinstance(ctx, zmq.Context), ctx
        assert isinstance(skt, zmq.Socket), skt
        assert skt.type == zmq.PUB

    test()


def test_skt_name():
    @context()
    @socket('myskt', zmq.PUB)
    def test(ctx, myskt):
        assert isinstance(myskt, zmq.Socket), myskt
        assert isinstance(ctx, zmq.Context), ctx
        assert myskt.type == zmq.PUB

    test()


def test_skt_kwarg():
    @context()
    @socket(zmq.PUB, name='myskt')
    def test(ctx, myskt):
        assert isinstance(myskt, zmq.Socket), myskt
        assert isinstance(ctx, zmq.Context), ctx
        assert myskt.type == zmq.PUB

    test()


def test_ctx_skt_name():
    @context('ctx')
    @socket('skt', zmq.PUB, context_name='ctx')
    def test(ctx, skt):
        assert isinstance(skt, zmq.Socket), skt
        assert isinstance(ctx, zmq.Context), ctx
        assert skt.type == zmq.PUB

    test()


def test_skt_default_ctx():
    @socket(zmq.PUB)
    def test(skt):
        assert isinstance(skt, zmq.Socket), skt
        assert skt.context is zmq.Context.instance()
        assert skt.type == zmq.PUB

    test()


def test_skt_reinit():
    result = {'foo': None, 'bar': None}

    @socket(zmq.PUB)
    def f(key, skt):
        assert isinstance(skt, zmq.Socket), skt

        result[key] = skt

    foo_t = threading.Thread(target=f, args=('foo',))
    bar_t = threading.Thread(target=f, args=('bar',))

    foo_t.start()
    bar_t.start()

    foo_t.join()
    bar_t.join()

    assert result['foo'] is not None, result
    assert result['bar'] is not None, result
    assert result['foo'] is not result['bar'], result


def test_ctx_skt_reinit():
    result = {'foo': {'ctx': None, 'skt': None}, 'bar': {'ctx': None, 'skt': None}}

    @context()
    @socket(zmq.PUB)
    def f(key, ctx, skt):
        assert isinstance(ctx, zmq.Context), ctx
        assert isinstance(skt, zmq.Socket), skt

        result[key]['ctx'] = ctx
        result[key]['skt'] = skt

    foo_t = threading.Thread(target=f, args=('foo',))
    bar_t = threading.Thread(target=f, args=('bar',))

    foo_t.start()
    bar_t.start()

    foo_t.join()
    bar_t.join()

    assert result['foo']['ctx'] is not None, result
    assert result['foo']['skt'] is not None, result
    assert result['bar']['ctx'] is not None, result
    assert result['bar']['skt'] is not None, result
    assert result['foo']['ctx'] is not result['bar']['ctx'], result
    assert result['foo']['skt'] is not result['bar']['skt'], result


def test_skt_type_miss():
    @context()
    @socket('myskt')
    def f(ctx, myskt):
        pass  # the socket type is missing

    with raises(TypeError):
        f()


def test_multi_skts():
    @socket(zmq.PUB)
    @socket(zmq.SUB)
    @socket(zmq.PUSH)
    def test(pub, sub, push):
        assert isinstance(pub, zmq.Socket), pub
        assert isinstance(sub, zmq.Socket), sub
        assert isinstance(push, zmq.Socket), push

        assert pub.context is zmq.Context.instance()
        assert sub.context is zmq.Context.instance()
        assert push.context is zmq.Context.instance()

        assert pub.type == zmq.PUB
        assert sub.type == zmq.SUB
        assert push.type == zmq.PUSH

    test()


def test_multi_skts_single_ctx():
    @context()
    @socket(zmq.PUB)
    @socket(zmq.SUB)
    @socket(zmq.PUSH)
    def test(ctx, pub, sub, push):
        assert isinstance(ctx, zmq.Context), ctx
        assert isinstance(pub, zmq.Socket), pub
        assert isinstance(sub, zmq.Socket), sub
        assert isinstance(push, zmq.Socket), push

        assert pub.context is ctx
        assert sub.context is ctx
        assert push.context is ctx

        assert pub.type == zmq.PUB
        assert sub.type == zmq.SUB
        assert push.type == zmq.PUSH

    test()


def test_multi_skts_with_name():
    @socket('foo', zmq.PUSH)
    @socket('bar', zmq.SUB)
    @socket('baz', zmq.PUB)
    def test(foo, bar, baz):
        assert isinstance(foo, zmq.Socket), foo
        assert isinstance(bar, zmq.Socket), bar
        assert isinstance(baz, zmq.Socket), baz

        assert foo.context is zmq.Context.instance()
        assert bar.context is zmq.Context.instance()
        assert baz.context is zmq.Context.instance()

        assert foo.type == zmq.PUSH
        assert bar.type == zmq.SUB
        assert baz.type == zmq.PUB

    test()


def test_func_return():
    @context()
    def f(ctx):
        assert isinstance(ctx, zmq.Context), ctx
        return 'something'

    assert f() == 'something'


def test_skt_multi_thread():
    @socket(zmq.PUB)
    @socket(zmq.SUB)
    @socket(zmq.PUSH)
    def f(pub, sub, push):
        assert isinstance(pub, zmq.Socket), pub
        assert isinstance(sub, zmq.Socket), sub
        assert isinstance(push, zmq.Socket), push

        assert pub.context is zmq.Context.instance()
        assert sub.context is zmq.Context.instance()
        assert push.context is zmq.Context.instance()

        assert pub.type == zmq.PUB
        assert sub.type == zmq.SUB
        assert push.type == zmq.PUSH

        assert len(set(map(id, [pub, sub, push]))) == 3

    threads = [threading.Thread(target=f) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]


class TestMethodDecorators(BaseZMQTestCase):
    @context()
    @socket(zmq.PUB)
    @socket(zmq.SUB)
    def multi_skts_method(self, ctx, pub, sub, foo='bar'):
        assert isinstance(self, TestMethodDecorators), self
        assert isinstance(ctx, zmq.Context), ctx
        assert isinstance(pub, zmq.Socket), pub
        assert isinstance(sub, zmq.Socket), sub
        assert foo == 'bar'

        assert pub.context is ctx
        assert sub.context is ctx

        assert pub.type == zmq.PUB
        assert sub.type == zmq.SUB

    def test_multi_skts_method(self):
        self.multi_skts_method()

    def test_multi_skts_method_other_args(self):
        @socket(zmq.PUB)
        @socket(zmq.SUB)
        def f(foo, pub, sub, bar=None):
            assert isinstance(pub, zmq.Socket), pub
            assert isinstance(sub, zmq.Socket), sub

            assert foo == 'mock'
            assert bar == 'fake'

            assert pub.context is zmq.Context.instance()
            assert sub.context is zmq.Context.instance()

            assert pub.type == zmq.PUB
            assert sub.type == zmq.SUB

        f('mock', bar='fake')
