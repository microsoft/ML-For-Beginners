from zmq.eventloop import zmqstream
from zmq.green.eventloop.ioloop import IOLoop


class ZMQStream(zmqstream.ZMQStream):
    def __init__(self, socket, io_loop=None):
        io_loop = io_loop or IOLoop.instance()
        super().__init__(socket, io_loop=io_loop)


__all__ = ["ZMQStream"]
