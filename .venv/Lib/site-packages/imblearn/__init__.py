"""Toolbox for imbalanced dataset in machine learning.

``imbalanced-learn`` is a set of python methods to deal with imbalanced
datset in machine learning and pattern recognition.

Subpackages
-----------
combine
    Module which provides methods based on over-sampling and under-sampling.
ensemble
    Module which provides methods generating an ensemble of
    under-sampled subsets.
exceptions
    Module including custom warnings and error clases used across
    imbalanced-learn.
keras
    Module which provides custom generator, layers for deep learning using
    keras.
metrics
    Module which provides metrics to quantified the classification performance
    with imbalanced dataset.
over_sampling
    Module which provides methods to over-sample a dataset.
tensorflow
    Module which provides custom generator, layers for deep learning using
    tensorflow.
under-sampling
    Module which provides methods to under-sample a dataset.
utils
    Module including various utilities.
pipeline
    Module which allowing to create pipeline with scikit-learn estimators.
"""
import importlib
import sys
import types

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    # mypy error: Cannot determine type of '__SKLEARN_SETUP__'
    __IMBLEARN_SETUP__  # type: ignore
except NameError:
    __IMBLEARN_SETUP__ = False

if __IMBLEARN_SETUP__:
    sys.stderr.write("Partial import of imblearn during the build process.\n")
    # We are not importing the rest of scikit-learn during the build
    # process, as it may not be compiled yet
else:
    from . import (
        combine,
        ensemble,
        exceptions,
        metrics,
        over_sampling,
        pipeline,
        tensorflow,
        under_sampling,
        utils,
    )
    from ._version import __version__
    from .base import FunctionSampler
    from .utils._show_versions import show_versions  # noqa: F401

    # FIXME: When we get Python 3.7 as minimal version, we will need to switch to
    # the following solution:
    # https://snarky.ca/lazy-importing-in-python-3-7/
    class LazyLoader(types.ModuleType):
        """Lazily import a module, mainly to avoid pulling in large dependencies.

        Adapted from TensorFlow:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
        python/util/lazy_loader.py
        """

        def __init__(self, local_name, parent_module_globals, name, warning=None):
            self._local_name = local_name
            self._parent_module_globals = parent_module_globals
            self._warning = warning

            super(LazyLoader, self).__init__(name)

        def _load(self):
            """Load the module and insert it into the parent's globals."""
            # Import the target module and insert it into the parent's namespace
            module = importlib.import_module(self.__name__)
            self._parent_module_globals[self._local_name] = module

            # Update this object's dict so that if someone keeps a reference to the
            #   LazyLoader, lookups are efficient (__getattr__ is only called on
            #   lookups that fail).
            self.__dict__.update(module.__dict__)

            return module

        def __getattr__(self, item):
            module = self._load()
            return getattr(module, item)

        def __dir__(self):
            module = self._load()
            return dir(module)

    # delay the import of keras since we are going to import either tensorflow
    # or keras
    keras = LazyLoader("keras", globals(), "imblearn.keras")

    __all__ = [
        "combine",
        "ensemble",
        "exceptions",
        "keras",
        "metrics",
        "over_sampling",
        "tensorflow",
        "under_sampling",
        "utils",
        "pipeline",
        "FunctionSampler",
        "__version__",
    ]
