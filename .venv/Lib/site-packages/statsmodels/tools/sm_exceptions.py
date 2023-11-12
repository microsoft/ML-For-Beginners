"""
Contains custom errors and warnings.

Errors should derive from Exception or another custom error. Custom errors are
only needed it standard errors, for example ValueError or TypeError, are not
accurate descriptions of the reason for the error.

Warnings should derive from either an existing warning or another custom
warning, and should usually be accompanied by a sting using the format
warning_name_doc that services as a generic message to use when the warning is
raised.
"""

import warnings


# Errors
class PerfectSeparationError(Exception):
    """
    Error due to perfect prediction in discrete models
    """

    pass


class MissingDataError(Exception):
    """
    Error raised if variables contain missing values when forbidden
    """

    pass


class X13NotFoundError(Exception):
    """
    Error locating the X13 binary
    """

    pass


class X13Error(Exception):
    """
    Error when running modes using X13
    """

    pass


class ParseError(Exception):
    """
    Error when parsing a docstring.
    """

    def __str__(self):
        message = self.args[0]
        if hasattr(self, "docstring"):
            message = f"{message} in {self.docstring}"
        return message


# Warning
class X13Warning(Warning):
    """
    Unexpected conditions when using X13
    """
    pass


class IOWarning(RuntimeWarning):
    """
    Resource not deleted
    """

    pass


class ModuleUnavailableWarning(Warning):
    """
    Non-fatal import error
    """
    pass


module_unavailable_doc = """
The module {0} is not available. Cannot run in parallel.
"""


class ModelWarning(UserWarning):
    """
    Base internal Warning class to simplify end-user filtering
    """
    pass


class ConvergenceWarning(ModelWarning):
    """
    Nonlinear optimizer failed to converge to a unique solution
    """
    pass


convergence_doc = """
Failed to converge on a solution.
"""


class CacheWriteWarning(ModelWarning):
    """
    Attempting to write to a read-only cached value
    """

    pass


class IterationLimitWarning(ModelWarning):
    """
    Iteration limit reached without convergence
    """

    pass


iteration_limit_doc = """
Maximum iteration reached.
"""


class InvalidTestWarning(ModelWarning):
    """
    Test not applicable to model
    """

    pass


class NotImplementedWarning(ModelWarning):
    """
    Non-fatal function non-implementation
    """

    pass


class OutputWarning(ModelWarning):
    """
    Function output contains atypical values
    """

    pass


class DomainWarning(ModelWarning):
    """
    Variables are not compliant with required domain constraints
    """

    pass


class ValueWarning(ModelWarning):
    """
    Non-fatal out-of-range value given
    """

    pass


class EstimationWarning(ModelWarning):
    """
    Unexpected condition encountered during estimation
    """

    pass


class SingularMatrixWarning(ModelWarning):
    """
    Non-fatal matrix inversion affects output results
    """

    pass


class HypothesisTestWarning(ModelWarning):
    """
    Issue occurred when performing hypothesis test
    """

    pass


class InterpolationWarning(ModelWarning):
    """
    Table granularity and limits restrict interpolation
    """

    pass


class PrecisionWarning(ModelWarning):
    """
    Numerical implementation affects precision
    """

    pass


class SpecificationWarning(ModelWarning):
    """
    Non-fatal model specification issue
    """

    pass


class HessianInversionWarning(ModelWarning):
    """
    Hessian noninvertible and standard errors unavailable
    """

    pass


class CollinearityWarning(ModelWarning):
    """
    Variables are highly collinear
    """

    pass


class PerfectSeparationWarning(ModelWarning):
    """
    Perfect separation or prediction
    """

    pass


class InfeasibleTestError(RuntimeError):
    """
    Test statistic cannot be computed
    """

    pass


recarray_exception = """
recarray support has been removed from statsmodels. Use pandas DataFrames
for structured data.
"""


warnings.simplefilter("always", ModelWarning)
warnings.simplefilter("always", ConvergenceWarning)
warnings.simplefilter("always", CacheWriteWarning)
warnings.simplefilter("always", IterationLimitWarning)
warnings.simplefilter("always", InvalidTestWarning)
