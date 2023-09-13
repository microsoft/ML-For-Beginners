from _pydev_bundle._pydev_saved_modules import socket
import sys

IS_JYTHON = sys.platform.find('java') != -1

_cache = None


def get_localhost():
    '''
    Should return 127.0.0.1 in ipv4 and ::1 in ipv6

    localhost is not used because on windows vista/windows 7, there can be issues where the resolving doesn't work
    properly and takes a lot of time (had this issue on the pyunit server).

    Using the IP directly solves the problem.
    '''
    # TODO: Needs better investigation!

    global _cache
    if _cache is None:
        try:
            for addr_info in socket.getaddrinfo("localhost", 80, 0, 0, socket.SOL_TCP):
                config = addr_info[4]
                if config[0] == '127.0.0.1':
                    _cache = '127.0.0.1'
                    return _cache
        except:
            # Ok, some versions of Python don't have getaddrinfo or SOL_TCP... Just consider it 127.0.0.1 in this case.
            _cache = '127.0.0.1'
        else:
            _cache = 'localhost'

    return _cache


def get_socket_names(n_sockets, close=False):
    socket_names = []
    sockets = []
    for _ in range(n_sockets):
        if IS_JYTHON:
            # Although the option which would be pure java *should* work for Jython, the socket being returned is still 0
            # (i.e.: it doesn't give the local port bound, only the original port, which was 0).
            from java.net import ServerSocket
            sock = ServerSocket(0)
            socket_name = get_localhost(), sock.getLocalPort()
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((get_localhost(), 0))
            socket_name = sock.getsockname()

        sockets.append(sock)
        socket_names.append(socket_name)

    if close:
        for s in sockets:
            s.close()
    return socket_names


def get_socket_name(close=False):
    return get_socket_names(1, close)[0]


if __name__ == '__main__':
    print(get_socket_name())
