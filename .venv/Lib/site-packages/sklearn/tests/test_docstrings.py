import re
from inspect import signature
from typing import Optional

import pytest

# make it possible to discover experimental estimators when calling `all_estimators`
from sklearn.experimental import (
    enable_halving_search_cv,  # noqa
    enable_iterative_imputer,  # noqa
)
from sklearn.utils.discovery import all_displays, all_estimators, all_functions

numpydoc_validation = pytest.importorskip("numpydoc.validate")


def get_all_methods():
    estimators = all_estimators()
    displays = all_displays()
    for name, Klass in estimators + displays:
        if name.startswith("_"):
            # skip private classes
            continue
        methods = []
        for name in dir(Klass):
            if name.startswith("_"):
                continue
            method_obj = getattr(Klass, name)
            if hasattr(method_obj, "__call__") or isinstance(method_obj, property):
                methods.append(name)
        methods.append(None)

        for method in sorted(methods, key=str):
            yield Klass, method


def get_all_functions_names():
    functions = all_functions()
    for _, func in functions:
        # exclude functions from utils.fixex since they come from external packages
        if "utils.fixes" not in func.__module__:
            yield f"{func.__module__}.{func.__name__}"


def filter_errors(errors, method, Klass=None):
    """
    Ignore some errors based on the method type.

    These rules are specific for scikit-learn."""
    for code, message in errors:
        # We ignore following error code,
        #  - RT02: The first line of the Returns section
        #    should contain only the type, ..
        #   (as we may need refer to the name of the returned
        #    object)
        #  - GL01: Docstring text (summary) should start in the line
        #    immediately after the opening quotes (not in the same line,
        #    or leaving a blank line in between)
        #  - GL02: If there's a blank line, it should be before the
        #    first line of the Returns section, not after (it allows to have
        #    short docstrings for properties).

        if code in ["RT02", "GL01", "GL02"]:
            continue

        # Ignore PR02: Unknown parameters for properties. We sometimes use
        # properties for ducktyping, i.e. SGDClassifier.predict_proba
        # Ignore GL08: Parsing of the method signature failed, possibly because this is
        # a property. Properties are sometimes used for deprecated attributes and the
        # attribute is already documented in the class docstring.
        #
        # All error codes:
        # https://numpydoc.readthedocs.io/en/latest/validation.html#built-in-validation-checks
        if code in ("PR02", "GL08") and Klass is not None and method is not None:
            method_obj = getattr(Klass, method)
            if isinstance(method_obj, property):
                continue

        # Following codes are only taken into account for the
        # top level class docstrings:
        #  - ES01: No extended summary found
        #  - SA01: See Also section not found
        #  - EX01: No examples section found

        if method is not None and code in ["EX01", "SA01", "ES01"]:
            continue
        yield code, message


def repr_errors(res, Klass=None, method: Optional[str] = None) -> str:
    """Pretty print original docstring and the obtained errors

    Parameters
    ----------
    res : dict
        result of numpydoc.validate.validate
    Klass : {Estimator, Display, None}
        estimator object or None
    method : str
        if estimator is not None, either the method name or None.

    Returns
    -------
    str
       String representation of the error.
    """
    if method is None:
        if hasattr(Klass, "__init__"):
            method = "__init__"
        elif Klass is None:
            raise ValueError("At least one of Klass, method should be provided")
        else:
            raise NotImplementedError

    if Klass is not None:
        obj = getattr(Klass, method)
        try:
            obj_signature = str(signature(obj))
        except TypeError:
            # In particular we can't parse the signature of properties
            obj_signature = (
                "\nParsing of the method signature failed, "
                "possibly because this is a property."
            )

        obj_name = Klass.__name__ + "." + method
    else:
        obj_signature = ""
        obj_name = method

    msg = "\n\n" + "\n\n".join(
        [
            str(res["file"]),
            obj_name + obj_signature,
            res["docstring"],
            "# Errors",
            "\n".join(
                " - {}: {}".format(code, message) for code, message in res["errors"]
            ),
        ]
    )
    return msg


@pytest.mark.parametrize("function_name", get_all_functions_names())
def test_function_docstring(function_name, request):
    """Check function docstrings using numpydoc."""
    res = numpydoc_validation.validate(function_name)

    res["errors"] = list(filter_errors(res["errors"], method="function"))

    if res["errors"]:
        msg = repr_errors(res, method=f"Tested function: {function_name}")

        raise ValueError(msg)


@pytest.mark.parametrize("Klass, method", get_all_methods())
def test_docstring(Klass, method, request):
    base_import_path = Klass.__module__
    import_path = [base_import_path, Klass.__name__]
    if method is not None:
        import_path.append(method)

    import_path = ".".join(import_path)

    res = numpydoc_validation.validate(import_path)

    res["errors"] = list(filter_errors(res["errors"], method, Klass=Klass))

    if res["errors"]:
        msg = repr_errors(res, Klass, method)

        raise ValueError(msg)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate docstring with numpydoc.")
    parser.add_argument("import_path", help="Import path to validate")

    args = parser.parse_args()

    res = numpydoc_validation.validate(args.import_path)

    import_path_sections = args.import_path.split(".")
    # When applied to classes, detect class method. For functions
    # method = None.
    # TODO: this detection can be improved. Currently we assume that we have
    # class # methods if the second path element before last is in camel case.
    if len(import_path_sections) >= 2 and re.match(
        r"(?:[A-Z][a-z]*)+", import_path_sections[-2]
    ):
        method = import_path_sections[-1]
    else:
        method = None

    res["errors"] = list(filter_errors(res["errors"], method))

    if res["errors"]:
        msg = repr_errors(res, method=args.import_path)

        print(msg)
        sys.exit(1)
    else:
        print("All docstring checks passed for {}!".format(args.import_path))
