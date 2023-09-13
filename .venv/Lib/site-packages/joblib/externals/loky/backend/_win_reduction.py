###############################################################################
# Extra reducers for Windows system and connections objects
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from multiprocessing/reduction.py (17/02/2017)
#  * Add adapted reduction for LokyProcesses and socket/PipeConnection
#
import socket
from multiprocessing import connection
from multiprocessing.reduction import _reduce_socket

from .reduction import register

# register reduction for win32 communication objects
register(socket.socket, _reduce_socket)
register(connection.Connection, connection.reduce_connection)
register(connection.PipeConnection, connection.reduce_pipe_connection)
