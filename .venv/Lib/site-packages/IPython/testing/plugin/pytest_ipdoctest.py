# Based on Pytest doctest.py
# Original license:
# The MIT License (MIT)
#
# Copyright (c) 2004-2021 Holger Krekel and others
"""Discover and run ipdoctests in modules and test files."""
import builtins
import bdb
import inspect
import os
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning

if TYPE_CHECKING:
    import doctest

DOCTEST_REPORT_CHOICE_NONE = "none"
DOCTEST_REPORT_CHOICE_CDIFF = "cdiff"
DOCTEST_REPORT_CHOICE_NDIFF = "ndiff"
DOCTEST_REPORT_CHOICE_UDIFF = "udiff"
DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE = "only_first_failure"

DOCTEST_REPORT_CHOICES = (
    DOCTEST_REPORT_CHOICE_NONE,
    DOCTEST_REPORT_CHOICE_CDIFF,
    DOCTEST_REPORT_CHOICE_NDIFF,
    DOCTEST_REPORT_CHOICE_UDIFF,
    DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE,
)

# Lazy definition of runner class
RUNNER_CLASS = None
# Lazy definition of output checker class
CHECKER_CLASS: Optional[Type["IPDoctestOutputChecker"]] = None


def pytest_addoption(parser: Parser) -> None:
    parser.addini(
        "ipdoctest_optionflags",
        "option flags for ipdoctests",
        type="args",
        default=["ELLIPSIS"],
    )
    parser.addini(
        "ipdoctest_encoding", "encoding used for ipdoctest files", default="utf-8"
    )
    group = parser.getgroup("collect")
    group.addoption(
        "--ipdoctest-modules",
        action="store_true",
        default=False,
        help="run ipdoctests in all .py modules",
        dest="ipdoctestmodules",
    )
    group.addoption(
        "--ipdoctest-report",
        type=str.lower,
        default="udiff",
        help="choose another output format for diffs on ipdoctest failure",
        choices=DOCTEST_REPORT_CHOICES,
        dest="ipdoctestreport",
    )
    group.addoption(
        "--ipdoctest-glob",
        action="append",
        default=[],
        metavar="pat",
        help="ipdoctests file matching pattern, default: test*.txt",
        dest="ipdoctestglob",
    )
    group.addoption(
        "--ipdoctest-ignore-import-errors",
        action="store_true",
        default=False,
        help="ignore ipdoctest ImportErrors",
        dest="ipdoctest_ignore_import_errors",
    )
    group.addoption(
        "--ipdoctest-continue-on-failure",
        action="store_true",
        default=False,
        help="for a given ipdoctest, continue to run after the first failure",
        dest="ipdoctest_continue_on_failure",
    )


def pytest_unconfigure() -> None:
    global RUNNER_CLASS

    RUNNER_CLASS = None


def pytest_collect_file(
    file_path: Path,
    parent: Collector,
) -> Optional[Union["IPDoctestModule", "IPDoctestTextfile"]]:
    config = parent.config
    if file_path.suffix == ".py":
        if config.option.ipdoctestmodules and not any(
            (_is_setup_py(file_path), _is_main_py(file_path))
        ):
            mod: IPDoctestModule = IPDoctestModule.from_parent(parent, path=file_path)
            return mod
    elif _is_ipdoctest(config, file_path, parent):
        txt: IPDoctestTextfile = IPDoctestTextfile.from_parent(parent, path=file_path)
        return txt
    return None


if int(pytest.__version__.split(".")[0]) < 7:
    _collect_file = pytest_collect_file

    def pytest_collect_file(
        path,
        parent: Collector,
    ) -> Optional[Union["IPDoctestModule", "IPDoctestTextfile"]]:
        return _collect_file(Path(path), parent)

    _import_path = import_path

    def import_path(path, root):
        import py.path

        return _import_path(py.path.local(path))


def _is_setup_py(path: Path) -> bool:
    if path.name != "setup.py":
        return False
    contents = path.read_bytes()
    return b"setuptools" in contents or b"distutils" in contents


def _is_ipdoctest(config: Config, path: Path, parent: Collector) -> bool:
    if path.suffix in (".txt", ".rst") and parent.session.isinitpath(path):
        return True
    globs = config.getoption("ipdoctestglob") or ["test*.txt"]
    return any(fnmatch_ex(glob, path) for glob in globs)


def _is_main_py(path: Path) -> bool:
    return path.name == "__main__.py"


class ReprFailDoctest(TerminalRepr):
    def __init__(
        self, reprlocation_lines: Sequence[Tuple[ReprFileLocation, Sequence[str]]]
    ) -> None:
        self.reprlocation_lines = reprlocation_lines

    def toterminal(self, tw: TerminalWriter) -> None:
        for reprlocation, lines in self.reprlocation_lines:
            for line in lines:
                tw.line(line)
            reprlocation.toterminal(tw)


class MultipleDoctestFailures(Exception):
    def __init__(self, failures: Sequence["doctest.DocTestFailure"]) -> None:
        super().__init__()
        self.failures = failures


def _init_runner_class() -> Type["IPDocTestRunner"]:
    import doctest
    from .ipdoctest import IPDocTestRunner

    class PytestDoctestRunner(IPDocTestRunner):
        """Runner to collect failures.

        Note that the out variable in this case is a list instead of a
        stdout-like object.
        """

        def __init__(
            self,
            checker: Optional["IPDoctestOutputChecker"] = None,
            verbose: Optional[bool] = None,
            optionflags: int = 0,
            continue_on_failure: bool = True,
        ) -> None:
            super().__init__(checker=checker, verbose=verbose, optionflags=optionflags)
            self.continue_on_failure = continue_on_failure

        def report_failure(
            self,
            out,
            test: "doctest.DocTest",
            example: "doctest.Example",
            got: str,
        ) -> None:
            failure = doctest.DocTestFailure(test, example, got)
            if self.continue_on_failure:
                out.append(failure)
            else:
                raise failure

        def report_unexpected_exception(
            self,
            out,
            test: "doctest.DocTest",
            example: "doctest.Example",
            exc_info: Tuple[Type[BaseException], BaseException, types.TracebackType],
        ) -> None:
            if isinstance(exc_info[1], OutcomeException):
                raise exc_info[1]
            if isinstance(exc_info[1], bdb.BdbQuit):
                outcomes.exit("Quitting debugger")
            failure = doctest.UnexpectedException(test, example, exc_info)
            if self.continue_on_failure:
                out.append(failure)
            else:
                raise failure

    return PytestDoctestRunner


def _get_runner(
    checker: Optional["IPDoctestOutputChecker"] = None,
    verbose: Optional[bool] = None,
    optionflags: int = 0,
    continue_on_failure: bool = True,
) -> "IPDocTestRunner":
    # We need this in order to do a lazy import on doctest
    global RUNNER_CLASS
    if RUNNER_CLASS is None:
        RUNNER_CLASS = _init_runner_class()
    # Type ignored because the continue_on_failure argument is only defined on
    # PytestDoctestRunner, which is lazily defined so can't be used as a type.
    return RUNNER_CLASS(  # type: ignore
        checker=checker,
        verbose=verbose,
        optionflags=optionflags,
        continue_on_failure=continue_on_failure,
    )


class IPDoctestItem(pytest.Item):
    def __init__(
        self,
        name: str,
        parent: "Union[IPDoctestTextfile, IPDoctestModule]",
        runner: Optional["IPDocTestRunner"] = None,
        dtest: Optional["doctest.DocTest"] = None,
    ) -> None:
        super().__init__(name, parent)
        self.runner = runner
        self.dtest = dtest
        self.obj = None
        self.fixture_request: Optional[FixtureRequest] = None

    @classmethod
    def from_parent(  # type: ignore
        cls,
        parent: "Union[IPDoctestTextfile, IPDoctestModule]",
        *,
        name: str,
        runner: "IPDocTestRunner",
        dtest: "doctest.DocTest",
    ):
        # incompatible signature due to imposed limits on subclass
        """The public named constructor."""
        return super().from_parent(name=name, parent=parent, runner=runner, dtest=dtest)

    def setup(self) -> None:
        if self.dtest is not None:
            self.fixture_request = _setup_fixtures(self)
            globs = dict(getfixture=self.fixture_request.getfixturevalue)
            for name, value in self.fixture_request.getfixturevalue(
                "ipdoctest_namespace"
            ).items():
                globs[name] = value
            self.dtest.globs.update(globs)

            from .ipdoctest import IPExample

            if isinstance(self.dtest.examples[0], IPExample):
                # for IPython examples *only*, we swap the globals with the ipython
                # namespace, after updating it with the globals (which doctest
                # fills with the necessary info from the module being tested).
                self._user_ns_orig = {}
                self._user_ns_orig.update(_ip.user_ns)
                _ip.user_ns.update(self.dtest.globs)
                # We must remove the _ key in the namespace, so that Python's
                # doctest code sets it naturally
                _ip.user_ns.pop("_", None)
                _ip.user_ns["__builtins__"] = builtins
                self.dtest.globs = _ip.user_ns

    def teardown(self) -> None:
        from .ipdoctest import IPExample

        # Undo the test.globs reassignment we made
        if isinstance(self.dtest.examples[0], IPExample):
            self.dtest.globs = {}
            _ip.user_ns.clear()
            _ip.user_ns.update(self._user_ns_orig)
            del self._user_ns_orig

        self.dtest.globs.clear()

    def runtest(self) -> None:
        assert self.dtest is not None
        assert self.runner is not None
        _check_all_skipped(self.dtest)
        self._disable_output_capturing_for_darwin()
        failures: List["doctest.DocTestFailure"] = []

        # exec(compile(..., "single", ...), ...) puts result in builtins._
        had_underscore_value = hasattr(builtins, "_")
        underscore_original_value = getattr(builtins, "_", None)

        # Save our current directory and switch out to the one where the
        # test was originally created, in case another doctest did a
        # directory change.  We'll restore this in the finally clause.
        curdir = os.getcwd()
        os.chdir(self.fspath.dirname)
        try:
            # Type ignored because we change the type of `out` from what
            # ipdoctest expects.
            self.runner.run(self.dtest, out=failures, clear_globs=False)  # type: ignore[arg-type]
        finally:
            os.chdir(curdir)
            if had_underscore_value:
                setattr(builtins, "_", underscore_original_value)
            elif hasattr(builtins, "_"):
                delattr(builtins, "_")

        if failures:
            raise MultipleDoctestFailures(failures)

    def _disable_output_capturing_for_darwin(self) -> None:
        """Disable output capturing. Otherwise, stdout is lost to ipdoctest (pytest#985)."""
        if platform.system() != "Darwin":
            return
        capman = self.config.pluginmanager.getplugin("capturemanager")
        if capman:
            capman.suspend_global_capture(in_=True)
            out, err = capman.read_global_capture()
            sys.stdout.write(out)
            sys.stderr.write(err)

    # TODO: Type ignored -- breaks Liskov Substitution.
    def repr_failure(  # type: ignore[override]
        self,
        excinfo: ExceptionInfo[BaseException],
    ) -> Union[str, TerminalRepr]:
        import doctest

        failures: Optional[
            Sequence[Union[doctest.DocTestFailure, doctest.UnexpectedException]]
        ] = None
        if isinstance(
            excinfo.value, (doctest.DocTestFailure, doctest.UnexpectedException)
        ):
            failures = [excinfo.value]
        elif isinstance(excinfo.value, MultipleDoctestFailures):
            failures = excinfo.value.failures

        if failures is None:
            return super().repr_failure(excinfo)

        reprlocation_lines = []
        for failure in failures:
            example = failure.example
            test = failure.test
            filename = test.filename
            if test.lineno is None:
                lineno = None
            else:
                lineno = test.lineno + example.lineno + 1
            message = type(failure).__name__
            # TODO: ReprFileLocation doesn't expect a None lineno.
            reprlocation = ReprFileLocation(filename, lineno, message)  # type: ignore[arg-type]
            checker = _get_checker()
            report_choice = _get_report_choice(self.config.getoption("ipdoctestreport"))
            if lineno is not None:
                assert failure.test.docstring is not None
                lines = failure.test.docstring.splitlines(False)
                # add line numbers to the left of the error message
                assert test.lineno is not None
                lines = [
                    "%03d %s" % (i + test.lineno + 1, x) for (i, x) in enumerate(lines)
                ]
                # trim docstring error lines to 10
                lines = lines[max(example.lineno - 9, 0) : example.lineno + 1]
            else:
                lines = [
                    "EXAMPLE LOCATION UNKNOWN, not showing all tests of that example"
                ]
                indent = ">>>"
                for line in example.source.splitlines():
                    lines.append(f"??? {indent} {line}")
                    indent = "..."
            if isinstance(failure, doctest.DocTestFailure):
                lines += checker.output_difference(
                    example, failure.got, report_choice
                ).split("\n")
            else:
                inner_excinfo = ExceptionInfo.from_exc_info(failure.exc_info)
                lines += ["UNEXPECTED EXCEPTION: %s" % repr(inner_excinfo.value)]
                lines += [
                    x.strip("\n") for x in traceback.format_exception(*failure.exc_info)
                ]
            reprlocation_lines.append((reprlocation, lines))
        return ReprFailDoctest(reprlocation_lines)

    def reportinfo(self) -> Tuple[Union["os.PathLike[str]", str], Optional[int], str]:
        assert self.dtest is not None
        return self.path, self.dtest.lineno, "[ipdoctest] %s" % self.name

    if int(pytest.__version__.split(".")[0]) < 7:

        @property
        def path(self) -> Path:
            return Path(self.fspath)


def _get_flag_lookup() -> Dict[str, int]:
    import doctest

    return dict(
        DONT_ACCEPT_TRUE_FOR_1=doctest.DONT_ACCEPT_TRUE_FOR_1,
        DONT_ACCEPT_BLANKLINE=doctest.DONT_ACCEPT_BLANKLINE,
        NORMALIZE_WHITESPACE=doctest.NORMALIZE_WHITESPACE,
        ELLIPSIS=doctest.ELLIPSIS,
        IGNORE_EXCEPTION_DETAIL=doctest.IGNORE_EXCEPTION_DETAIL,
        COMPARISON_FLAGS=doctest.COMPARISON_FLAGS,
        ALLOW_UNICODE=_get_allow_unicode_flag(),
        ALLOW_BYTES=_get_allow_bytes_flag(),
        NUMBER=_get_number_flag(),
    )


def get_optionflags(parent):
    optionflags_str = parent.config.getini("ipdoctest_optionflags")
    flag_lookup_table = _get_flag_lookup()
    flag_acc = 0
    for flag in optionflags_str:
        flag_acc |= flag_lookup_table[flag]
    return flag_acc


def _get_continue_on_failure(config):
    continue_on_failure = config.getvalue("ipdoctest_continue_on_failure")
    if continue_on_failure:
        # We need to turn off this if we use pdb since we should stop at
        # the first failure.
        if config.getvalue("usepdb"):
            continue_on_failure = False
    return continue_on_failure


class IPDoctestTextfile(pytest.Module):
    obj = None

    def collect(self) -> Iterable[IPDoctestItem]:
        import doctest
        from .ipdoctest import IPDocTestParser

        # Inspired by doctest.testfile; ideally we would use it directly,
        # but it doesn't support passing a custom checker.
        encoding = self.config.getini("ipdoctest_encoding")
        text = self.path.read_text(encoding)
        filename = str(self.path)
        name = self.path.name
        globs = {"__name__": "__main__"}

        optionflags = get_optionflags(self)

        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )

        parser = IPDocTestParser()
        test = parser.get_doctest(text, globs, name, filename, 0)
        if test.examples:
            yield IPDoctestItem.from_parent(
                self, name=test.name, runner=runner, dtest=test
            )

    if int(pytest.__version__.split(".")[0]) < 7:

        @property
        def path(self) -> Path:
            return Path(self.fspath)

        @classmethod
        def from_parent(
            cls,
            parent,
            *,
            fspath=None,
            path: Optional[Path] = None,
            **kw,
        ):
            if path is not None:
                import py.path

                fspath = py.path.local(path)
            return super().from_parent(parent=parent, fspath=fspath, **kw)


def _check_all_skipped(test: "doctest.DocTest") -> None:
    """Raise pytest.skip() if all examples in the given DocTest have the SKIP
    option set."""
    import doctest

    all_skipped = all(x.options.get(doctest.SKIP, False) for x in test.examples)
    if all_skipped:
        pytest.skip("all docstests skipped by +SKIP option")


def _is_mocked(obj: object) -> bool:
    """Return if an object is possibly a mock object by checking the
    existence of a highly improbable attribute."""
    return (
        safe_getattr(obj, "pytest_mock_example_attribute_that_shouldnt_exist", None)
        is not None
    )


@contextmanager
def _patch_unwrap_mock_aware() -> Generator[None, None, None]:
    """Context manager which replaces ``inspect.unwrap`` with a version
    that's aware of mock objects and doesn't recurse into them."""
    real_unwrap = inspect.unwrap

    def _mock_aware_unwrap(
        func: Callable[..., Any], *, stop: Optional[Callable[[Any], Any]] = None
    ) -> Any:
        try:
            if stop is None or stop is _is_mocked:
                return real_unwrap(func, stop=_is_mocked)
            _stop = stop
            return real_unwrap(func, stop=lambda obj: _is_mocked(obj) or _stop(func))
        except Exception as e:
            warnings.warn(
                "Got %r when unwrapping %r.  This is usually caused "
                "by a violation of Python's object protocol; see e.g. "
                "https://github.com/pytest-dev/pytest/issues/5080" % (e, func),
                PytestWarning,
            )
            raise

    inspect.unwrap = _mock_aware_unwrap
    try:
        yield
    finally:
        inspect.unwrap = real_unwrap


class IPDoctestModule(pytest.Module):
    def collect(self) -> Iterable[IPDoctestItem]:
        import doctest
        from .ipdoctest import DocTestFinder, IPDocTestParser

        class MockAwareDocTestFinder(DocTestFinder):
            """A hackish ipdoctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456
            https://bugs.python.org/issue25532
            """

            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct
                line number is returned. This will be reported upstream. #8796
                """
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)

                if hasattr(obj, "__wrapped__"):
                    # Get the main obj in case of it being wrapped
                    obj = inspect.unwrap(obj)

                # Type ignored because this is a private function.
                return super()._find_lineno(  # type:ignore[misc]
                    obj,
                    source_lines,
                )

            def _find(
                self, tests, obj, name, module, source_lines, globs, seen
            ) -> None:
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    # Type ignored because this is a private function.
                    super()._find(  # type:ignore[misc]
                        tests, obj, name, module, source_lines, globs, seen
                    )

        if self.path.name == "conftest.py":
            if int(pytest.__version__.split(".")[0]) < 7:
                module = self.config.pluginmanager._importconftest(
                    self.path,
                    self.config.getoption("importmode"),
                )
            else:
                module = self.config.pluginmanager._importconftest(
                    self.path,
                    self.config.getoption("importmode"),
                    rootpath=self.config.rootpath,
                )
        else:
            try:
                module = import_path(self.path, root=self.config.rootpath)
            except ImportError:
                if self.config.getvalue("ipdoctest_ignore_import_errors"):
                    pytest.skip("unable to import module %r" % self.path)
                else:
                    raise
        # Uses internal doctest module parsing mechanism.
        finder = MockAwareDocTestFinder(parser=IPDocTestParser())
        optionflags = get_optionflags(self)
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )

        for test in finder.find(module, module.__name__):
            if test.examples:  # skip empty ipdoctests
                yield IPDoctestItem.from_parent(
                    self, name=test.name, runner=runner, dtest=test
                )

    if int(pytest.__version__.split(".")[0]) < 7:

        @property
        def path(self) -> Path:
            return Path(self.fspath)

        @classmethod
        def from_parent(
            cls,
            parent,
            *,
            fspath=None,
            path: Optional[Path] = None,
            **kw,
        ):
            if path is not None:
                import py.path

                fspath = py.path.local(path)
            return super().from_parent(parent=parent, fspath=fspath, **kw)


def _setup_fixtures(doctest_item: IPDoctestItem) -> FixtureRequest:
    """Used by IPDoctestTextfile and IPDoctestItem to setup fixture information."""

    def func() -> None:
        pass

    doctest_item.funcargs = {}  # type: ignore[attr-defined]
    fm = doctest_item.session._fixturemanager
    doctest_item._fixtureinfo = fm.getfixtureinfo(  # type: ignore[attr-defined]
        node=doctest_item, func=func, cls=None, funcargs=False
    )
    fixture_request = FixtureRequest(doctest_item, _ispytest=True)
    fixture_request._fillfixtures()
    return fixture_request


def _init_checker_class() -> Type["IPDoctestOutputChecker"]:
    import doctest
    import re
    from .ipdoctest import IPDoctestOutputChecker

    class LiteralsOutputChecker(IPDoctestOutputChecker):
        # Based on doctest_nose_plugin.py from the nltk project
        # (https://github.com/nltk/nltk) and on the "numtest" doctest extension
        # by Sebastien Boisgerault (https://github.com/boisgera/numtest).

        _unicode_literal_re = re.compile(r"(\W|^)[uU]([rR]?[\'\"])", re.UNICODE)
        _bytes_literal_re = re.compile(r"(\W|^)[bB]([rR]?[\'\"])", re.UNICODE)
        _number_re = re.compile(
            r"""
            (?P<number>
              (?P<mantissa>
                (?P<integer1> [+-]?\d*)\.(?P<fraction>\d+)
                |
                (?P<integer2> [+-]?\d+)\.
              )
              (?:
                [Ee]
                (?P<exponent1> [+-]?\d+)
              )?
              |
              (?P<integer3> [+-]?\d+)
              (?:
                [Ee]
                (?P<exponent2> [+-]?\d+)
              )
            )
            """,
            re.VERBOSE,
        )

        def check_output(self, want: str, got: str, optionflags: int) -> bool:
            if super().check_output(want, got, optionflags):
                return True

            allow_unicode = optionflags & _get_allow_unicode_flag()
            allow_bytes = optionflags & _get_allow_bytes_flag()
            allow_number = optionflags & _get_number_flag()

            if not allow_unicode and not allow_bytes and not allow_number:
                return False

            def remove_prefixes(regex: Pattern[str], txt: str) -> str:
                return re.sub(regex, r"\1\2", txt)

            if allow_unicode:
                want = remove_prefixes(self._unicode_literal_re, want)
                got = remove_prefixes(self._unicode_literal_re, got)

            if allow_bytes:
                want = remove_prefixes(self._bytes_literal_re, want)
                got = remove_prefixes(self._bytes_literal_re, got)

            if allow_number:
                got = self._remove_unwanted_precision(want, got)

            return super().check_output(want, got, optionflags)

        def _remove_unwanted_precision(self, want: str, got: str) -> str:
            wants = list(self._number_re.finditer(want))
            gots = list(self._number_re.finditer(got))
            if len(wants) != len(gots):
                return got
            offset = 0
            for w, g in zip(wants, gots):
                fraction: Optional[str] = w.group("fraction")
                exponent: Optional[str] = w.group("exponent1")
                if exponent is None:
                    exponent = w.group("exponent2")
                precision = 0 if fraction is None else len(fraction)
                if exponent is not None:
                    precision -= int(exponent)
                if float(w.group()) == approx(float(g.group()), abs=10**-precision):
                    # They're close enough. Replace the text we actually
                    # got with the text we want, so that it will match when we
                    # check the string literally.
                    got = (
                        got[: g.start() + offset] + w.group() + got[g.end() + offset :]
                    )
                    offset += w.end() - w.start() - (g.end() - g.start())
            return got

    return LiteralsOutputChecker


def _get_checker() -> "IPDoctestOutputChecker":
    """Return a IPDoctestOutputChecker subclass that supports some
    additional options:

    * ALLOW_UNICODE and ALLOW_BYTES options to ignore u'' and b''
      prefixes (respectively) in string literals. Useful when the same
      ipdoctest should run in Python 2 and Python 3.

    * NUMBER to ignore floating-point differences smaller than the
      precision of the literal number in the ipdoctest.

    An inner class is used to avoid importing "ipdoctest" at the module
    level.
    """
    global CHECKER_CLASS
    if CHECKER_CLASS is None:
        CHECKER_CLASS = _init_checker_class()
    return CHECKER_CLASS()


def _get_allow_unicode_flag() -> int:
    """Register and return the ALLOW_UNICODE flag."""
    import doctest

    return doctest.register_optionflag("ALLOW_UNICODE")


def _get_allow_bytes_flag() -> int:
    """Register and return the ALLOW_BYTES flag."""
    import doctest

    return doctest.register_optionflag("ALLOW_BYTES")


def _get_number_flag() -> int:
    """Register and return the NUMBER flag."""
    import doctest

    return doctest.register_optionflag("NUMBER")


def _get_report_choice(key: str) -> int:
    """Return the actual `ipdoctest` module flag value.

    We want to do it as late as possible to avoid importing `ipdoctest` and all
    its dependencies when parsing options, as it adds overhead and breaks tests.
    """
    import doctest

    return {
        DOCTEST_REPORT_CHOICE_UDIFF: doctest.REPORT_UDIFF,
        DOCTEST_REPORT_CHOICE_CDIFF: doctest.REPORT_CDIFF,
        DOCTEST_REPORT_CHOICE_NDIFF: doctest.REPORT_NDIFF,
        DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE: doctest.REPORT_ONLY_FIRST_FAILURE,
        DOCTEST_REPORT_CHOICE_NONE: 0,
    }[key]


@pytest.fixture(scope="session")
def ipdoctest_namespace() -> Dict[str, Any]:
    """Fixture that returns a :py:class:`dict` that will be injected into the
    namespace of ipdoctests."""
    return dict()
