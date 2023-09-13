import traceback
import sys
from io import StringIO


class Log:

    def __init__(self):
        self._contents = []

    def add_content(self, *content):
        self._contents.append(' '.join(content))

    def add_exception(self):
        s = StringIO()
        exc_info = sys.exc_info()
        traceback.print_exception(exc_info[0], exc_info[1], exc_info[2], limit=None, file=s)
        self._contents.append(s.getvalue())

    def get_contents(self):
        return '\n'.join(self._contents)

    def clear_log(self):
        del self._contents[:]
