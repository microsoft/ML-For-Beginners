import linecache
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import _debugger_case_stepping

linecache.updatecache(_debugger_case_stepping.__file__)
assert linecache.getline(_debugger_case_stepping.__file__, 1)
_debugger_case_stepping.Call()

print('TEST SUCEEDED')
