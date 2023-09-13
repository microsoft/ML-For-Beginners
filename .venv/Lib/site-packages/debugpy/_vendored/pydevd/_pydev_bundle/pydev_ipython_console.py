import sys
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface

import traceback

# Uncomment to force PyDev standard shell.
# raise ImportError()

from _pydev_bundle.pydev_ipython_console_011 import get_pydev_frontend


#=======================================================================================================================
# InterpreterInterface
#=======================================================================================================================
class InterpreterInterface(BaseInterpreterInterface):
    '''
        The methods in this class should be registered in the xml-rpc server.
    '''

    def __init__(self, host, client_port, main_thread, show_banner=True, connect_status_queue=None):
        BaseInterpreterInterface.__init__(self, main_thread, connect_status_queue)
        self.client_port = client_port
        self.host = host
        self.interpreter = get_pydev_frontend(host, client_port)
        self._input_error_printed = False
        self.notification_succeeded = False
        self.notification_tries = 0
        self.notification_max_tries = 3
        self.show_banner = show_banner

        self.notify_about_magic()

    def get_greeting_msg(self):
        if self.show_banner:
            self.interpreter.show_banner()
        return self.interpreter.get_greeting_msg()

    def do_add_exec(self, code_fragment):
        self.notify_about_magic()
        if code_fragment.text.rstrip().endswith('??'):
            print('IPython-->')
        try:
            res = bool(self.interpreter.add_exec(code_fragment.text))
        finally:
            if code_fragment.text.rstrip().endswith('??'):
                print('<--IPython')

        return res

    def get_namespace(self):
        return self.interpreter.get_namespace()

    def getCompletions(self, text, act_tok):
        return self.interpreter.getCompletions(text, act_tok)

    def close(self):
        sys.exit(0)

    def notify_about_magic(self):
        if not self.notification_succeeded:
            self.notification_tries += 1
            if self.notification_tries > self.notification_max_tries:
                return
            completions = self.getCompletions("%", "%")
            magic_commands = [x[0] for x in completions]

            server = self.get_server()

            if server is not None:
                try:
                    server.NotifyAboutMagic(magic_commands, self.interpreter.is_automagic())
                    self.notification_succeeded = True
                except:
                    self.notification_succeeded = False

    def get_ipython_hidden_vars_dict(self):
        try:
            if hasattr(self.interpreter, 'ipython') and hasattr(self.interpreter.ipython, 'user_ns_hidden'):
                user_ns_hidden = self.interpreter.ipython.user_ns_hidden
                if isinstance(user_ns_hidden, dict):
                    # Since IPython 2 dict `user_ns_hidden` contains hidden variables and values
                    user_hidden_dict = user_ns_hidden.copy()
                else:
                    # In IPython 1.x `user_ns_hidden` used to be a set with names of hidden variables
                    user_hidden_dict = dict([(key, val) for key, val in self.interpreter.ipython.user_ns.items()
                                             if key in user_ns_hidden])

                # while `_`, `__` and `___` were not initialized, they are not presented in `user_ns_hidden`
                user_hidden_dict.setdefault('_', '')
                user_hidden_dict.setdefault('__', '')
                user_hidden_dict.setdefault('___', '')

                return user_hidden_dict
        except:
            # Getting IPython variables shouldn't break loading frame variables
            traceback.print_exc()

