import sys
import os

try:
    from _debugger_case_dont_trace import call_me_back
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from _debugger_case_dont_trace import call_me_back


def my_callback():
    print('trace me')  # Break here


if __name__ == '__main__':
    call_me_back(my_callback)
    call_me_back(my_callback)
    print('TEST SUCEEDED!')
