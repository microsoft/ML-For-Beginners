import os
import sys
import platform

TEST_CYTHON = os.getenv('PYDEVD_USE_CYTHON', None) == 'YES'
PYDEVD_TEST_VM = os.getenv('PYDEVD_TEST_VM', None)

IS_PY36_OR_GREATER = sys.version_info[0:2] >= (3, 6)
IS_PY311_OR_GREATER = sys.version_info[0:2] >= (3, 11)
IS_CPYTHON = platform.python_implementation() == 'CPython'

TODO_PY311 = IS_PY311_OR_GREATER  # Code which needs to be fixed in 3.11 should use this constant.

IS_PY36 = False
if sys.version_info[0] == 3 and sys.version_info[1] == 6:
    IS_PY36 = True

TEST_DJANGO = False
TEST_FLASK = False
TEST_CHERRYPY = False
TEST_GEVENT = False

try:
    import django
    TEST_DJANGO = True
except:
    pass

try:
    import flask
    TEST_FLASK = True
except:
    pass

try:
    import cherrypy
    TEST_CHERRYPY = True
except:
    pass

try:
    import gevent
    TEST_GEVENT = True
except:
    pass
