'''
@author Fabio Zadrozny
'''
import sys
import unittest
import socket
import urllib
import pytest
import pycompletionserver


IS_JYTHON = sys.platform.find('java') != -1
DEBUG = 0

def dbg(s):
    if DEBUG:
        sys.stdout.write('TEST %s\n' % s)

@pytest.mark.skipif(not IS_JYTHON, reason='Jython related test')
class TestJython(unittest.TestCase):

    def test_it(self):
        dbg('ok')

    
    def test_message(self):
        t = pycompletionserver.CompletionServer(0)
        t.exit_process_on_kill = False

        l = []
        l.append(('Def', 'description'  , 'args'))
        l.append(('Def1', 'description1', 'args1'))
        l.append(('Def2', 'description2', 'args2'))

        msg = t.processor.format_completion_message('test_jyserver.py', l)

        self.assertEqual('@@COMPLETIONS(test_jyserver.py,(Def,description,args),(Def1,description1,args1),(Def2,description2,args2))END@@', msg)

        l = []
        l.append(('Def', 'desc,,r,,i()ption', ''))
        l.append(('Def(1', 'descriptio(n1', ''))
        l.append(('De,f)2', 'de,s,c,ription2', ''))
        msg = t.processor.format_completion_message(None, l)
        expected = '@@COMPLETIONS(None,(Def,desc%2C%2Cr%2C%2Ci%28%29ption, ),(Def%281,descriptio%28n1, ),(De%2Cf%292,de%2Cs%2Cc%2Cription2, ))END@@'

        self.assertEqual(expected, msg)

    
    def test_completion_sockets_and_messages(self):
        dbg('test_completion_sockets_and_messages')
        t, socket = self.create_connections()
        self.socket = socket
        dbg('connections created')

        try:
            #now that we have the connections all set up, check the code completion messages.
            msg = urllib.quote_plus('math')

            toWrite = '@@IMPORTS:%sEND@@' % msg
            dbg('writing' + str(toWrite))
            socket.send(toWrite)  #math completions
            completions = self.read_msg()
            dbg(urllib.unquote_plus(completions))

            start = '@@COMPLETIONS('
            self.assertTrue(completions.startswith(start), '%s DOESNT START WITH %s' % (completions, start))
            self.assertTrue(completions.find('@@COMPLETIONS') != -1)
            self.assertTrue(completions.find('END@@') != -1)


            msg = urllib.quote_plus('__builtin__.str')
            toWrite = '@@IMPORTS:%sEND@@' % msg
            dbg('writing' + str(toWrite))
            socket.send(toWrite)  #math completions
            completions = self.read_msg()
            dbg(urllib.unquote_plus(completions))

            start = '@@COMPLETIONS('
            self.assertTrue(completions.startswith(start), '%s DOESNT START WITH %s' % (completions, start))
            self.assertTrue(completions.find('@@COMPLETIONS') != -1)
            self.assertTrue(completions.find('END@@') != -1)



        finally:
            try:
                self.send_kill_msg(socket)


                while not t.ended:
                    pass  #wait until it receives the message and quits.


                socket.close()
            except:
                pass

    def get_free_port(self):
        from _pydev_bundle.pydev_localhost import get_socket_name
        return get_socket_name(close=True)[1]

    def create_connections(self):
        '''
        Creates the connections needed for testing.
        '''
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((pycompletionserver.HOST, 0))
        server.listen(1)  #socket to receive messages.

        from thread import start_new_thread
        t = pycompletionserver.CompletionServer(server.getsockname()[1])
        t.exit_process_on_kill = False

        start_new_thread(t.run, ())

        sock, _addr = server.accept()

        return t, sock

    def read_msg(self):
        msg = '@@PROCESSING_END@@'
        while msg.startswith('@@PROCESSING'):
            msg = self.socket.recv(1024)
            if msg.startswith('@@PROCESSING:'):
                dbg('Status msg:' + str(msg))

        while msg.find('END@@') == -1:
            msg += self.socket.recv(1024)

        return msg

    def send_kill_msg(self, socket):
        socket.send(pycompletionserver.MSG_KILL_SERVER)




# Run for jython in command line:
# c:\bin\jython2.7.0\bin\jython.exe -m py.test tests\test_jyserver.py
