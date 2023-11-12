from .linear_model import yule_walker

from statsmodels.tools._testing import PytestTester

__all__ = ['yule_walker', 'test']

test = PytestTester()
