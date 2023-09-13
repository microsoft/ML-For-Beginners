from functools import reduce
import gc
import io
import locale  # system locale module, not tornado.locale
import logging
import operator
import textwrap
import sys
import unittest
import warnings

from tornado.httpclient import AsyncHTTPClient
from tornado.httpserver import HTTPServer
from tornado.netutil import Resolver
from tornado.options import define, add_parse_callback, options


TEST_MODULES = [
    "tornado.httputil.doctests",
    "tornado.iostream.doctests",
    "tornado.util.doctests",
    "tornado.test.asyncio_test",
    "tornado.test.auth_test",
    "tornado.test.autoreload_test",
    "tornado.test.concurrent_test",
    "tornado.test.curl_httpclient_test",
    "tornado.test.escape_test",
    "tornado.test.gen_test",
    "tornado.test.http1connection_test",
    "tornado.test.httpclient_test",
    "tornado.test.httpserver_test",
    "tornado.test.httputil_test",
    "tornado.test.import_test",
    "tornado.test.ioloop_test",
    "tornado.test.iostream_test",
    "tornado.test.locale_test",
    "tornado.test.locks_test",
    "tornado.test.netutil_test",
    "tornado.test.log_test",
    "tornado.test.options_test",
    "tornado.test.process_test",
    "tornado.test.queues_test",
    "tornado.test.routing_test",
    "tornado.test.simple_httpclient_test",
    "tornado.test.tcpclient_test",
    "tornado.test.tcpserver_test",
    "tornado.test.template_test",
    "tornado.test.testing_test",
    "tornado.test.twisted_test",
    "tornado.test.util_test",
    "tornado.test.web_test",
    "tornado.test.websocket_test",
    "tornado.test.wsgi_test",
]


def all():
    return unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)


def test_runner_factory(stderr):
    class TornadoTextTestRunner(unittest.TextTestRunner):
        def __init__(self, *args, **kwargs):
            kwargs["stream"] = stderr
            super().__init__(*args, **kwargs)

        def run(self, test):
            result = super().run(test)
            if result.skipped:
                skip_reasons = set(reason for (test, reason) in result.skipped)
                self.stream.write(  # type: ignore
                    textwrap.fill(
                        "Some tests were skipped because: %s"
                        % ", ".join(sorted(skip_reasons))
                    )
                )
                self.stream.write("\n")  # type: ignore
            return result

    return TornadoTextTestRunner


class LogCounter(logging.Filter):
    """Counts the number of WARNING or higher log records."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info_count = self.warning_count = self.error_count = 0

    def filter(self, record):
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1
        elif record.levelno >= logging.INFO:
            self.info_count += 1
        return True


class CountingStderr(io.IOBase):
    def __init__(self, real):
        self.real = real
        self.byte_count = 0

    def write(self, data):
        self.byte_count += len(data)
        return self.real.write(data)

    def flush(self):
        return self.real.flush()


def main():
    # Be strict about most warnings (This is set in our test running
    # scripts to catch import-time warnings, but set it again here to
    # be sure). This also turns on warnings that are ignored by
    # default, including DeprecationWarnings and python 3.2's
    # ResourceWarnings.
    warnings.filterwarnings("error")
    # setuptools sometimes gives ImportWarnings about things that are on
    # sys.path even if they're not being used.
    warnings.filterwarnings("ignore", category=ImportWarning)
    # Tornado generally shouldn't use anything deprecated, but some of
    # our dependencies do (last match wins).
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("error", category=DeprecationWarning, module=r"tornado\..*")
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings(
        "error", category=PendingDeprecationWarning, module=r"tornado\..*"
    )
    # The unittest module is aggressive about deprecating redundant methods,
    # leaving some without non-deprecated spellings that work on both
    # 2.7 and 3.2
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message="Please use assert.* instead"
    )
    warnings.filterwarnings(
        "ignore",
        category=PendingDeprecationWarning,
        message="Please use assert.* instead",
    )
    # Twisted 15.0.0 triggers some warnings on py3 with -bb.
    warnings.filterwarnings("ignore", category=BytesWarning, module=r"twisted\..*")
    if (3,) < sys.version_info < (3, 6):
        # Prior to 3.6, async ResourceWarnings were rather noisy
        # and even
        # `python3.4 -W error -c 'import asyncio; asyncio.get_event_loop()'`
        # would generate a warning.
        warnings.filterwarnings(
            "ignore", category=ResourceWarning, module=r"asyncio\..*"
        )
    # This deprecation warning is introduced in Python 3.8 and is
    # triggered by pycurl. Unforunately, because it is raised in the C
    # layer it can't be filtered by module and we must match the
    # message text instead (Tornado's C module uses PY_SSIZE_T_CLEAN
    # so it's not at risk of running into this issue).
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message="PY_SSIZE_T_CLEAN will be required",
    )

    logging.getLogger("tornado.access").setLevel(logging.CRITICAL)

    define(
        "httpclient",
        type=str,
        default=None,
        callback=lambda s: AsyncHTTPClient.configure(
            s, defaults=dict(allow_ipv6=False)
        ),
    )
    define("httpserver", type=str, default=None, callback=HTTPServer.configure)
    define("resolver", type=str, default=None, callback=Resolver.configure)
    define(
        "debug_gc",
        type=str,
        multiple=True,
        help="A comma-separated list of gc module debug constants, "
        "e.g. DEBUG_STATS or DEBUG_COLLECTABLE,DEBUG_OBJECTS",
        callback=lambda values: gc.set_debug(
            reduce(operator.or_, (getattr(gc, v) for v in values))
        ),
    )
    define(
        "fail-if-logs",
        default=True,
        help="If true, fail the tests if any log output is produced (unless captured by ExpectLog)",
    )

    def set_locale(x):
        locale.setlocale(locale.LC_ALL, x)

    define("locale", type=str, default=None, callback=set_locale)

    log_counter = LogCounter()
    add_parse_callback(lambda: logging.getLogger().handlers[0].addFilter(log_counter))

    # Certain errors (especially "unclosed resource" errors raised in
    # destructors) go directly to stderr instead of logging. Count
    # anything written by anything but the test runner as an error.
    orig_stderr = sys.stderr
    counting_stderr = CountingStderr(orig_stderr)
    sys.stderr = counting_stderr  # type: ignore

    import tornado.testing

    kwargs = {}

    # HACK:  unittest.main will make its own changes to the warning
    # configuration, which may conflict with the settings above
    # or command-line flags like -bb.  Passing warnings=False
    # suppresses this behavior, although this looks like an implementation
    # detail.  http://bugs.python.org/issue15626
    kwargs["warnings"] = False

    kwargs["testRunner"] = test_runner_factory(orig_stderr)
    try:
        tornado.testing.main(**kwargs)
    finally:
        # The tests should run clean; consider it a failure if they
        # logged anything at info level or above.
        if (
            log_counter.info_count > 0
            or log_counter.warning_count > 0
            or log_counter.error_count > 0
            or counting_stderr.byte_count > 0
        ):
            logging.error(
                "logged %d infos, %d warnings, %d errors, and %d bytes to stderr",
                log_counter.info_count,
                log_counter.warning_count,
                log_counter.error_count,
                counting_stderr.byte_count,
            )
            if options.fail_if_logs:
                sys.exit(1)


if __name__ == "__main__":
    main()
