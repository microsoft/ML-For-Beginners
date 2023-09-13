'''
Entry-point module to start the code-completion server for PyDev.

@author Fabio Zadrozny
'''
from _pydevd_bundle.pydevd_constants import IS_JYTHON

if IS_JYTHON:
    import java.lang  # @UnresolvedImport
    SERVER_NAME = 'jycompletionserver'
    from _pydev_bundle import _pydev_jy_imports_tipper
    _pydev_imports_tipper = _pydev_jy_imports_tipper

else:
    # it is python
    SERVER_NAME = 'pycompletionserver'
    from _pydev_bundle import _pydev_imports_tipper

from _pydev_bundle._pydev_saved_modules import socket

import sys
if sys.platform == "darwin":
    # See: https://sourceforge.net/projects/pydev/forums/forum/293649/topic/3454227
    try:
        import _CF  # Don't fail if it doesn't work -- do it because it must be loaded on the main thread! @UnresolvedImport @UnusedImport
    except:
        pass

# initial sys.path
_sys_path = []
for p in sys.path:
    # changed to be compatible with 1.5
    _sys_path.append(p)

# initial sys.modules
_sys_modules = {}
for name, mod in sys.modules.items():
    _sys_modules[name] = mod

import traceback

from io import StringIO

from urllib.parse import quote_plus, unquote_plus

INFO1 = 1
INFO2 = 2
WARN = 4
ERROR = 8

DEBUG = INFO1 | ERROR


def dbg(s, prior):
    if prior & DEBUG != 0:
        sys.stdout.write('%s\n' % (s,))
#        f = open('c:/temp/test.txt', 'a')
#        print_ >> f, s
#        f.close()


from _pydev_bundle import pydev_localhost
HOST = pydev_localhost.get_localhost()  # Symbolic name meaning the local host

MSG_KILL_SERVER = '@@KILL_SERVER_END@@'
MSG_COMPLETIONS = '@@COMPLETIONS'
MSG_END = 'END@@'
MSG_INVALID_REQUEST = '@@INVALID_REQUEST'
MSG_JYTHON_INVALID_REQUEST = '@@JYTHON_INVALID_REQUEST'
MSG_CHANGE_DIR = '@@CHANGE_DIR:'
MSG_OK = '@@MSG_OK_END@@'
MSG_IMPORTS = '@@IMPORTS:'
MSG_PYTHONPATH = '@@PYTHONPATH_END@@'
MSG_CHANGE_PYTHONPATH = '@@CHANGE_PYTHONPATH:'
MSG_JEDI = '@@MSG_JEDI:'
MSG_SEARCH = '@@SEARCH'

BUFFER_SIZE = 1024

currDirModule = None


def complete_from_dir(directory):
    '''
    This is necessary so that we get the imports from the same directory where the file
    we are completing is located.
    '''
    global currDirModule
    if currDirModule is not None:
        if len(sys.path) > 0 and sys.path[0] == currDirModule:
            del sys.path[0]

    currDirModule = directory
    sys.path.insert(0, directory)


def change_python_path(pythonpath):
    '''Changes the pythonpath (clears all the previous pythonpath)

    @param pythonpath: string with paths separated by |
    '''

    split = pythonpath.split('|')
    sys.path = []
    for path in split:
        path = path.strip()
        if len(path) > 0:
            sys.path.append(path)


class Processor:

    def __init__(self):
        # nothing to do
        return

    def remove_invalid_chars(self, msg):
        try:
            msg = str(msg)
        except UnicodeDecodeError:
            pass

        if msg:
            try:
                return quote_plus(msg)
            except:
                sys.stdout.write('error making quote plus in %s\n' % (msg,))
                raise
        return ' '

    def format_completion_message(self, defFile, completionsList):
        '''
        Format the completions suggestions in the following format:
        @@COMPLETIONS(modFile(token,description),(token,description),(token,description))END@@
        '''
        compMsg = []
        compMsg.append('%s' % defFile)
        for tup in completionsList:
            compMsg.append(',')

            compMsg.append('(')
            compMsg.append(str(self.remove_invalid_chars(tup[0])))  # token
            compMsg.append(',')
            compMsg.append(self.remove_invalid_chars(tup[1]))  # description

            if(len(tup) > 2):
                compMsg.append(',')
                compMsg.append(self.remove_invalid_chars(tup[2]))  # args - only if function.

            if(len(tup) > 3):
                compMsg.append(',')
                compMsg.append(self.remove_invalid_chars(tup[3]))  # TYPE

            compMsg.append(')')

        return '%s(%s)%s' % (MSG_COMPLETIONS, ''.join(compMsg), MSG_END)


class Exit(Exception):
    pass


class CompletionServer:

    def __init__(self, port):
        self.ended = False
        self.port = port
        self.socket = None  # socket to send messages.
        self.exit_process_on_kill = True
        self.processor = Processor()

    def connect_to_server(self):
        from _pydev_bundle._pydev_saved_modules import socket

        self.socket = s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((HOST, self.port))
        except:
            sys.stderr.write('Error on connect_to_server with parameters: host: %s port: %s\n' % (HOST, self.port))
            raise

    def get_completions_message(self, defFile, completionsList):
        '''
        get message with completions.
        '''
        return self.processor.format_completion_message(defFile, completionsList)

    def get_token_and_data(self, data):
        '''
        When we receive this, we have 'token):data'
        '''
        token = ''
        for c in data:
            if c != ')':
                token = token + c
            else:
                break;

        return token, data.lstrip(token + '):')

    def emulated_sendall(self, msg):
        MSGLEN = 1024 * 20

        totalsent = 0
        while totalsent < MSGLEN:
            sent = self.socket.send(msg[totalsent:])
            if sent == 0:
                return
            totalsent = totalsent + sent

    def send(self, msg):
        self.socket.sendall(bytearray(msg, 'utf-8'))

    def run(self):
        # Echo server program
        try:
            from _pydev_bundle import _pydev_log
            log = _pydev_log.Log()

            dbg(SERVER_NAME + ' connecting to java server on %s (%s)' % (HOST, self.port) , INFO1)
            # after being connected, create a socket as a client.
            self.connect_to_server()

            dbg(SERVER_NAME + ' Connected to java server', INFO1)

            while not self.ended:
                data = ''

                while data.find(MSG_END) == -1:
                    received = self.socket.recv(BUFFER_SIZE)
                    if len(received) == 0:
                        raise Exit()  # ok, connection ended
                    data = data + received.decode('utf-8')

                try:
                    try:
                        if data.find(MSG_KILL_SERVER) != -1:
                            dbg(SERVER_NAME + ' kill message received', INFO1)
                            # break if we received kill message.
                            self.ended = True
                            raise Exit()

                        dbg(SERVER_NAME + ' starting keep alive thread', INFO2)

                        if data.find(MSG_PYTHONPATH) != -1:
                            comps = []
                            for p in _sys_path:
                                comps.append((p, ' '))
                            self.send(self.get_completions_message(None, comps))

                        else:
                            data = data[:data.rfind(MSG_END)]

                            if data.startswith(MSG_IMPORTS):
                                data = data[len(MSG_IMPORTS):]
                                data = unquote_plus(data)
                                defFile, comps = _pydev_imports_tipper.generate_tip(data, log)
                                self.send(self.get_completions_message(defFile, comps))

                            elif data.startswith(MSG_CHANGE_PYTHONPATH):
                                data = data[len(MSG_CHANGE_PYTHONPATH):]
                                data = unquote_plus(data)
                                change_python_path(data)
                                self.send(MSG_OK)

                            elif data.startswith(MSG_JEDI):
                                data = data[len(MSG_JEDI):]
                                data = unquote_plus(data)
                                line, column, encoding, path, source = data.split('|', 4)
                                try:
                                    import jedi  # @UnresolvedImport
                                except:
                                    self.send(self.get_completions_message(None, [('Error on import jedi', 'Error importing jedi', '')]))
                                else:
                                    script = jedi.Script(
                                        # Line +1 because it expects lines 1-based (and col 0-based)
                                        source=source,
                                        line=int(line) + 1,
                                        column=int(column),
                                        source_encoding=encoding,
                                        path=path,
                                    )
                                    lst = []
                                    for completion in script.completions():
                                        t = completion.type
                                        if t == 'class':
                                            t = '1'

                                        elif t == 'function':
                                            t = '2'

                                        elif t == 'import':
                                            t = '0'

                                        elif t == 'keyword':
                                            continue  # Keywords are already handled in PyDev

                                        elif t == 'statement':
                                            t = '3'

                                        else:
                                            t = '-1'

                                        # gen list(tuple(name, doc, args, type))
                                        lst.append((completion.name, '', '', t))
                                    self.send(self.get_completions_message('empty', lst))

                            elif data.startswith(MSG_SEARCH):
                                data = data[len(MSG_SEARCH):]
                                data = unquote_plus(data)
                                (f, line, col), foundAs = _pydev_imports_tipper.search_definition(data)
                                self.send(self.get_completions_message(f, [(line, col, foundAs)]))

                            elif data.startswith(MSG_CHANGE_DIR):
                                data = data[len(MSG_CHANGE_DIR):]
                                data = unquote_plus(data)
                                complete_from_dir(data)
                                self.send(MSG_OK)

                            else:
                                self.send(MSG_INVALID_REQUEST)
                    except Exit:
                        e = sys.exc_info()[1]
                        msg = self.get_completions_message(None, [('Exit:', 'SystemExit', '')])
                        try:
                            self.send(msg)
                        except socket.error:
                            pass  # Ok, may be closed already

                        raise e  # raise original error.

                    except:
                        dbg(SERVER_NAME + ' exception occurred', ERROR)
                        s = StringIO()
                        traceback.print_exc(file=s)

                        err = s.getvalue()
                        dbg(SERVER_NAME + ' received error: ' + str(err), ERROR)
                        msg = self.get_completions_message(None, [('ERROR:', '%s\nLog:%s' % (err, log.get_contents()), '')])
                        try:
                            self.send(msg)
                        except socket.error:
                            pass  # Ok, may be closed already

                finally:
                    log.clear_log()

            self.socket.close()
            self.ended = True
            raise Exit()  # connection broken

        except Exit:
            if self.exit_process_on_kill:
                sys.exit(0)
            # No need to log SystemExit error
        except:
            s = StringIO()
            exc_info = sys.exc_info()

            traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], limit=None, file=s)
            err = s.getvalue()
            dbg(SERVER_NAME + ' received error: ' + str(err), ERROR)
            raise


if __name__ == '__main__':

    port = int(sys.argv[1])  # this is from where we want to receive messages.

    t = CompletionServer(port)
    dbg(SERVER_NAME + ' will start', INFO1)
    t.run()
