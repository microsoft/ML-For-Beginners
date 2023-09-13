import sys
from _pydev_bundle._pydev_saved_modules import thread
import pycompletionserver
import socket
from urllib.parse import quote_plus

start_new_thread = thread.start_new_thread

BUILTIN_MOD = 'builtins'


def send(s, msg):
    s.send(bytearray(msg, 'utf-8'))


import unittest


class TestCPython(unittest.TestCase):

    def test_message(self):
        t = pycompletionserver.CompletionServer(0)

        l = []
        l.append(('Def', 'description'  , 'args'))
        l.append(('Def1', 'description1', 'args1'))
        l.append(('Def2', 'description2', 'args2'))

        msg = t.processor.format_completion_message(None, l)

        self.assertEqual('@@COMPLETIONS(None,(Def,description,args),(Def1,description1,args1),(Def2,description2,args2))END@@', msg)
        l = []
        l.append(('Def', 'desc,,r,,i()ption', ''))
        l.append(('Def(1', 'descriptio(n1', ''))
        l.append(('De,f)2', 'de,s,c,ription2', ''))
        msg = t.processor.format_completion_message(None, l)
        self.assertEqual('@@COMPLETIONS(None,(Def,desc%2C%2Cr%2C%2Ci%28%29ption, ),(Def%281,descriptio%28n1, ),(De%2Cf%292,de%2Cs%2Cc%2Cription2, ))END@@', msg)

    def create_connections(self):
        '''
        Creates the connections needed for testing.
        '''
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((pycompletionserver.HOST, 0))
        server.listen(1)  # socket to receive messages.

        t = pycompletionserver.CompletionServer(server.getsockname()[1])
        t.exit_process_on_kill = False
        start_new_thread(t.run, ())

        s, _addr = server.accept()

        return t, s

    def read_msg(self):
        finish = False
        msg = ''
        while finish == False:
            m = self.socket.recv(1024 * 4)
            m = m.decode('utf-8')
            if m.startswith('@@PROCESSING'):
                sys.stdout.write('Status msg: %s\n' % (msg,))
            else:
                msg += m

            if msg.find('END@@') != -1:
                finish = True

        return msg

    def test_completion_sockets_and_messages(self):
        t, socket = self.create_connections()
        self.socket = socket

        try:
            # now that we have the connections all set up, check the code completion messages.
            msg = quote_plus('math')
            send(socket, '@@IMPORTS:%sEND@@' % msg)  # math completions
            completions = self.read_msg()
            # print_ unquote_plus(completions)

            # math is a builtin and because of that, it starts with None as a file
            start = '@@COMPLETIONS(None,(__doc__,'
            start_2 = '@@COMPLETIONS(None,(__name__,'
            if ('/math.so,' in completions or
                '/math.cpython-33m.so,' in completions or
                '/math.cpython-34m.so,' in completions or
                'math.cpython-35m' in completions or
                'math.cpython-36m' in completions or
                'math.cpython-37m' in completions or
                'math.cpython-38' in completions or
                'math.cpython-39' in completions or
                'math.cpython-310' in completions or
                'math.cpython-311' in completions
                ):
                return
            self.assertTrue(completions.startswith(start) or completions.startswith(start_2), '%s DOESNT START WITH %s' % (completions, (start, start_2)))

            self.assertTrue('@@COMPLETIONS' in completions)
            self.assertTrue('END@@' in completions)

            # now, test i
            msg = quote_plus('%s.list' % BUILTIN_MOD)
            send(socket, "@@IMPORTS:%s\nEND@@" % msg)
            found = self.read_msg()
            self.assertTrue('sort' in found, 'Could not find sort in: %s' % (found,))

            # now, test search
            msg = quote_plus('inspect.ismodule')
            send(socket, '@@SEARCH%sEND@@' % msg)  # math completions
            found = self.read_msg()
            self.assertTrue('inspect.py' in found)
            for i in range(33, 100):
                if str(i) in found:
                    break
            else:
                self.fail('Could not find the ismodule line in %s' % (found,))

            # now, test search
            msg = quote_plus('inspect.CO_NEWLOCALS')
            send(socket, '@@SEARCH%sEND@@' % msg)  # math completions
            found = self.read_msg()
            self.assertTrue('inspect.py' in found)
            self.assertTrue('CO_NEWLOCALS' in found)

            # now, test search
            msg = quote_plus('inspect.BlockFinder.tokeneater')
            send(socket, '@@SEARCH%sEND@@' % msg)
            found = self.read_msg()
            self.assertTrue('inspect.py' in found)
#            self.assertTrue('CO_NEWLOCALS' in found)

        # reload modules test
#        send(socket, '@@RELOAD_MODULES_END@@')
#        ok = self.read_msg()
#        self.assertEqual('@@MSG_OK_END@@' , ok)
#        this test is not executed because it breaks our current enviroment.

        finally:
            try:
                sys.stdout.write('succedded...sending kill msg\n')
                self.send_kill_msg(socket)

#                while not hasattr(t, 'ended'):
#                    pass #wait until it receives the message and quits.

                socket.close()
                self.socket.close()
            except:
                pass

    def send_kill_msg(self, socket):
        socket.send(pycompletionserver.MSG_KILL_SERVER)

