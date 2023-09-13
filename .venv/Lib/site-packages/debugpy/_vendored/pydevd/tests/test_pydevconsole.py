import threading
import unittest
import sys
import pydevconsole
from _pydev_bundle.pydev_imports import xmlrpclib, SimpleXMLRPCServer
from _pydevd_bundle import pydevd_io
from contextlib import contextmanager
import pytest

try:
    from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT  # @UnusedImport
    CAN_EVALUATE_TOP_LEVEL_ASYNC = True
except:
    CAN_EVALUATE_TOP_LEVEL_ASYNC = False


#=======================================================================================================================
# Test
#=======================================================================================================================
class Test(unittest.TestCase):

    @contextmanager
    def interpreter(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = pydevd_io.IOBuf()
        sys.stderr = pydevd_io.IOBuf()
        try:
            sys.stdout.encoding = sys.stdin.encoding
            sys.stderr.encoding = sys.stdin.encoding
        except AttributeError:
            # In Python 3 encoding is not writable (whereas in Python 2 it doesn't exist).
            pass

        try:
            client_port, _server_port = self.get_free_addresses()
            client_thread = self.start_client_thread(client_port)  # @UnusedVariable
            import time
            time.sleep(.3)  # let's give it some time to start the threads

            from _pydev_bundle import pydev_localhost
            interpreter = pydevconsole.InterpreterInterface(pydev_localhost.get_localhost(), client_port, threading.current_thread())
            yield interpreter
        except:
            # if there's some error, print the output to the actual output.
            self.original_stdout.write(sys.stdout.getvalue())
            self.original_stderr.write(sys.stderr.getvalue())
            raise
        finally:
            sys.stderr = self.original_stderr
            sys.stdout = self.original_stdout

    def test_console_hello(self):
        with self.interpreter() as interpreter:
            (result,) = interpreter.hello("Hello pydevconsole")
            self.assertEqual(result, "Hello eclipse")

    @pytest.mark.skipif(not CAN_EVALUATE_TOP_LEVEL_ASYNC, reason='Requires top-level async.')
    def test_console_async(self):
        with self.interpreter() as interpreter:
            from _pydev_bundle.pydev_console_utils import CodeFragment
            more = interpreter.add_exec(CodeFragment('''
async def async_func(a):
    return a
'''))
            assert not more
            assert not sys.stderr.getvalue()
            assert not sys.stdout.getvalue()

            more = interpreter.add_exec(CodeFragment('''x = await async_func(1111)'''))
            assert not more
            assert not sys.stderr.getvalue()
            assert not sys.stdout.getvalue()

            more = interpreter.add_exec(CodeFragment('''print(x)'''))
            assert not more
            assert not sys.stderr.getvalue()
            assert '1111' in sys.stdout.getvalue()

    def test_console_requests(self):
        with self.interpreter() as interpreter:
            from _pydev_bundle.pydev_console_utils import CodeFragment
            interpreter.add_exec(CodeFragment('class Foo:\n    CONSTANT=1\n'))
            interpreter.add_exec(CodeFragment('foo=Foo()'))
            interpreter.add_exec(CodeFragment('foo.__doc__=None'))
            interpreter.add_exec(CodeFragment('val = input()'))
            interpreter.add_exec(CodeFragment('50'))
            interpreter.add_exec(CodeFragment('print (val)'))
            found = sys.stdout.getvalue().split()
            try:
                self.assertEqual(['50', 'input_request'], found)
            except:
                try:
                    self.assertEqual(['input_request'], found)  # IPython
                except:
                    self.assertEqual([u'50', u'input_request'], found[1:])  # IPython 5.1
                    self.assertTrue(found[0].startswith(u'Out'))

            comps = interpreter.getCompletions('foo.', 'foo.')
            self.assertTrue(
                ('CONSTANT', '', '', '3') in comps or ('CONSTANT', '', '', '4') in comps, \
                'Found: %s' % comps
            )

            comps = interpreter.getCompletions('"".', '"".')
            self.assertTrue(
                ('__add__', 'x.__add__(y) <==> x+y', '', '3') in comps or
                ('__add__', '', '', '4') in comps or
                ('__add__', 'x.__add__(y) <==> x+y\r\nx.__add__(y) <==> x+y', '()', '2') in comps or
                ('__add__', 'x.\n__add__(y) <==> x+yx.\n__add__(y) <==> x+y', '()', '2'),
                'Did not find __add__ in : %s' % (comps,)
            )

            completions = interpreter.getCompletions('', '')
            for c in completions:
                if c[0] == 'AssertionError':
                    break
            else:
                self.fail('Could not find AssertionError')

            completions = interpreter.getCompletions('Assert', 'Assert')
            for c in completions:
                if c[0] == 'RuntimeError':
                    self.fail('Did not expect to find RuntimeError there')

            assert ('__doc__', None, '', '3') not in interpreter.getCompletions('foo.CO', 'foo.')

            comps = interpreter.getCompletions('va', 'va')
            assert ('val', '', '', '3') in comps or ('val', '', '', '4') in comps

            interpreter.add_exec(CodeFragment('s = "mystring"'))

            desc = interpreter.getDescription('val')
            self.assertTrue(desc.find('str(object) -> string') >= 0 or
                         desc == "'input_request'" or
                         desc.find('str(string[, encoding[, errors]]) -> str') >= 0 or
                         desc.find('str(Char* value)') >= 0 or
                         desc.find('str(object=\'\') -> string') >= 0 or
                         desc.find('str(value: Char*)') >= 0 or
                         desc.find('str(object=\'\') -> str') >= 0 or
                         desc.find('The most base type') >= 0  # Jython 2.7 is providing this :P
                         ,
                         'Could not find what was needed in %s' % desc)

            desc = interpreter.getDescription('val.join')
            self.assertTrue(desc.find('S.join(sequence) -> string') >= 0 or
                         desc.find('S.join(sequence) -> str') >= 0 or
                         desc.find('S.join(iterable) -> string') >= 0 or
                         desc == "<builtin method 'join'>"  or
                         desc == "<built-in method join of str object>" or
                         desc.find('str join(str self, list sequence)') >= 0 or
                         desc.find('S.join(iterable) -> str') >= 0 or
                         desc.find('join(self: str, sequence: list) -> str') >= 0 or
                         desc.find('Concatenate any number of strings.') >= 0 or
                         desc.find('bound method str.join') >= 0,  # PyPy
                         "Could not recognize: %s" % (desc,))

    def start_client_thread(self, client_port):

        class ClientThread(threading.Thread):

            def __init__(self, client_port):
                threading.Thread.__init__(self)
                self.client_port = client_port

            def run(self):

                class HandleRequestInput:

                    def RequestInput(self):
                        client_thread.requested_input = True
                        return 'input_request'

                    def NotifyFinished(self, *args, **kwargs):
                        client_thread.notified_finished += 1
                        return 1

                handle_request_input = HandleRequestInput()

                from _pydev_bundle import pydev_localhost
                client_server = SimpleXMLRPCServer((pydev_localhost.get_localhost(), self.client_port), logRequests=False)
                client_server.register_function(handle_request_input.RequestInput)
                client_server.register_function(handle_request_input.NotifyFinished)
                client_server.serve_forever()

        client_thread = ClientThread(client_port)
        client_thread.requested_input = False
        client_thread.notified_finished = 0
        client_thread.daemon = True
        client_thread.start()
        return client_thread

    def start_debugger_server_thread(self, debugger_port, socket_code):

        class DebuggerServerThread(threading.Thread):

            def __init__(self, debugger_port, socket_code):
                threading.Thread.__init__(self)
                self.debugger_port = debugger_port
                self.socket_code = socket_code

            def run(self):
                import socket
                s = socket.socket()
                s.bind(('', debugger_port))
                s.listen(1)
                socket, unused_addr = s.accept()
                socket_code(socket)

        debugger_thread = DebuggerServerThread(debugger_port, socket_code)
        debugger_thread.daemon = True
        debugger_thread.start()
        return debugger_thread

    def get_free_addresses(self):
        from _pydev_bundle.pydev_localhost import get_socket_names
        socket_names = get_socket_names(2, True)
        port0 = socket_names[0][1]
        port1 = socket_names[1][1]

        assert port0 != port1
        assert port0 > 0
        assert port1 > 0

        return port0, port1

    def test_server(self):
        self.original_stdout = sys.stdout
        sys.stdout = pydevd_io.IOBuf()
        try:
            client_port, server_port = self.get_free_addresses()

            class ServerThread(threading.Thread):

                def __init__(self, client_port, server_port):
                    threading.Thread.__init__(self)
                    self.client_port = client_port
                    self.server_port = server_port

                def run(self):
                    from _pydev_bundle import pydev_localhost
                    pydevconsole.start_server(pydev_localhost.get_localhost(), self.server_port, self.client_port)

            server_thread = ServerThread(client_port, server_port)
            server_thread.daemon = True
            server_thread.start()

            client_thread = self.start_client_thread(client_port)  # @UnusedVariable

            import time
            time.sleep(.3)  # let's give it some time to start the threads
            sys.stdout = pydevd_io.IOBuf()

            from _pydev_bundle import pydev_localhost
            server = xmlrpclib.Server('http://%s:%s' % (pydev_localhost.get_localhost(), server_port))
            server.execLine('class Foo:')
            server.execLine('    pass')
            server.execLine('')
            server.execLine('foo = Foo()')
            server.execLine('a = input()')
            server.execLine('print (a)')
            initial = time.time()
            while not client_thread.requested_input:
                if time.time() - initial > 2:
                    raise AssertionError('Did not get the return asked before the timeout.')
                time.sleep(.1)

            found = sys.stdout.getvalue()
            while ['input_request'] != found.split():
                found += sys.stdout.getvalue()
                if time.time() - initial > 2:
                    break
                time.sleep(.1)
            self.assertEqual(['input_request'], found.split())
        finally:
            sys.stdout = self.original_stdout

