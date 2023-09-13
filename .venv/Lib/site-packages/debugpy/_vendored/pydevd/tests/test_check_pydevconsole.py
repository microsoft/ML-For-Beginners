import threading
import unittest
import os
import pytest
import pydevconsole

from _pydev_bundle.pydev_imports import xmlrpclib, SimpleXMLRPCServer
from _pydev_bundle.pydev_localhost import get_localhost

try:
    raw_input
    raw_input_name = 'raw_input'
except NameError:
    raw_input_name = 'input'
    
try:
    from IPython import core  # @UnusedImport
    has_ipython = True
except:
    has_ipython = False


#=======================================================================================================================
# Test
#=======================================================================================================================
@pytest.mark.skipif(os.environ.get('TRAVIS') == 'true' or not has_ipython, reason='Too flaky on Travis (and requires IPython).')
class Test(unittest.TestCase):

    def start_client_thread(self, client_port):
        class ClientThread(threading.Thread):
            def __init__(self, client_port):
                threading.Thread.__init__(self)
                self.client_port = client_port

            def run(self):
                class HandleRequestInput:
                    def RequestInput(self):
                        client_thread.requested_input = True
                        return 'RequestInput: OK'

                    def NotifyFinished(self, *args, **kwargs):
                        client_thread.notified_finished += 1
                        return 1

                handle_request_input = HandleRequestInput()

                from _pydev_bundle import pydev_localhost
                self.client_server = client_server = SimpleXMLRPCServer((pydev_localhost.get_localhost(), self.client_port), logRequests=False)
                client_server.register_function(handle_request_input.RequestInput)
                client_server.register_function(handle_request_input.NotifyFinished)
                client_server.serve_forever()

            def shutdown(self):
                return
                self.client_server.shutdown()

        client_thread = ClientThread(client_port)
        client_thread.requested_input = False
        client_thread.notified_finished = 0
        client_thread.daemon = True
        client_thread.start()
        return client_thread


    def get_free_addresses(self):
        from _pydev_bundle.pydev_localhost import get_socket_names
        socket_names = get_socket_names(2, close=True)
        return [socket_name[1] for socket_name in socket_names]

    def test_server(self):
        # Just making sure that the singleton is created in this thread.
        from _pydev_bundle.pydev_ipython_console_011 import get_pydev_frontend
        get_pydev_frontend(get_localhost(), 0)

        client_port, server_port = self.get_free_addresses()
        class ServerThread(threading.Thread):
            def __init__(self, client_port, server_port):
                threading.Thread.__init__(self)
                self.client_port = client_port
                self.server_port = server_port

            def run(self):
                from _pydev_bundle import pydev_localhost
                print('Starting server with:', pydev_localhost.get_localhost(), self.server_port, self.client_port)
                pydevconsole.start_server(pydev_localhost.get_localhost(), self.server_port, self.client_port)
        server_thread = ServerThread(client_port, server_port)
        server_thread.daemon = True
        server_thread.start()

        client_thread = self.start_client_thread(client_port) #@UnusedVariable

        try:
            import time
            time.sleep(.3) #let's give it some time to start the threads

            from _pydev_bundle import pydev_localhost
            server = xmlrpclib.Server('http://%s:%s' % (pydev_localhost.get_localhost(), server_port))
            server.execLine("import sys; print('Running with: %s %s' % (sys.executable or sys.platform, sys.version))")
            server.execLine('class Foo:')
            server.execLine('    pass')
            server.execLine('')
            server.execLine('foo = Foo()')
            server.execLine('a = %s()' % raw_input_name)
            initial = time.time()
            while not client_thread.requested_input:
                if time.time() - initial > 2:
                    raise AssertionError('Did not get the return asked before the timeout.')
                time.sleep(.1)
            frame_xml = server.getFrame()
            self.assertTrue('RequestInput' in frame_xml, 'Did not fid RequestInput in:\n%s' % (frame_xml,))
        finally:
            client_thread.shutdown()

