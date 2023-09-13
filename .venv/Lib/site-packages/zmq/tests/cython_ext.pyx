from zmq cimport Context, Frame, Socket, libzmq


cdef inline Frame c_send_recv(Socket a, Socket b, bytes to_send):
    cdef Frame msg = Frame(to_send)
    a.send(msg)
    cdef Frame recvd = b.recv(flags=0, copy=False)
    return recvd


cpdef bytes send_recv_test(bytes to_send):
    cdef Context ctx = Context()
    cdef Socket a = Socket(ctx, libzmq.ZMQ_PUSH)
    cdef Socket b = Socket(ctx, libzmq.ZMQ_PULL)
    url = 'inproc://test'
    a.bind(url)
    b.connect(url)
    cdef Frame recvd_frame = c_send_recv(a, b, to_send)
    a.close()
    b.close()
    ctx.term()
    cdef bytes recvd_bytes = recvd_frame.bytes
    return recvd_bytes
