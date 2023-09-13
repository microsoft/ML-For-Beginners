"""tests for extending pyzmq"""

import zmq


class CustomSocket(zmq.Socket):
    custom_attr: int

    def __init__(self, context, socket_type, custom_attr: int = 0):
        super().__init__(context, socket_type)
        self.custom_attr = custom_attr


class CustomContext(zmq.Context):
    extra_arg: str
    _socket_class = CustomSocket

    def __init__(self, extra_arg: str = 'x'):
        super().__init__()
        self.extra_arg = extra_arg


def test_custom_context():
    ctx = CustomContext('s')
    assert isinstance(ctx, CustomContext)

    assert ctx.extra_arg == 's'
    s = ctx.socket(zmq.PUSH, custom_attr=10)
    assert isinstance(s, CustomSocket)
    assert s.custom_attr == 10
    assert s.context is ctx
    assert s.type == zmq.PUSH
    s.close()
    ctx.term()
