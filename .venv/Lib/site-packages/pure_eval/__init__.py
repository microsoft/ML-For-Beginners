from .core import Evaluator, CannotEval, group_expressions, is_expression_interesting
from .my_getattr_static import getattr_static

try:
    from .version import __version__
except ImportError:
    # version.py is auto-generated with the git tag when building
    __version__ = "???"
