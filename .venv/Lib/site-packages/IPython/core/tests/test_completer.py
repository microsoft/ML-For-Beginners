# encoding: utf-8
"""Tests for the IPython tab-completion machinery."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import pytest
import sys
import textwrap
import unittest

from contextlib import contextmanager

from traitlets.config.loader import Config
from IPython import get_ipython
from IPython.core import completer
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from IPython.utils.generics import complete_object
from IPython.testing import decorators as dec

from IPython.core.completer import (
    Completion,
    provisionalcompleter,
    match_dict_keys,
    _deduplicate_completions,
    _match_number_in_dict_key_prefix,
    completion_matcher,
    SimpleCompletion,
    CompletionContext,
)

# -----------------------------------------------------------------------------
# Test functions
# -----------------------------------------------------------------------------


def recompute_unicode_ranges():
    """
    utility to recompute the largest unicode range without any characters

    use to recompute the gap in the global _UNICODE_RANGES of completer.py
    """
    import itertools
    import unicodedata

    valid = []
    for c in range(0, 0x10FFFF + 1):
        try:
            unicodedata.name(chr(c))
        except ValueError:
            continue
        valid.append(c)

    def ranges(i):
        for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield b[0][1], b[-1][1]

    rg = list(ranges(valid))
    lens = []
    gap_lens = []
    pstart, pstop = 0, 0
    for start, stop in rg:
        lens.append(stop - start)
        gap_lens.append(
            (
                start - pstop,
                hex(pstop + 1),
                hex(start),
                f"{round((start - pstop)/0xe01f0*100)}%",
            )
        )
        pstart, pstop = start, stop

    return sorted(gap_lens)[-1]


def test_unicode_range():
    """
    Test that the ranges we test for unicode names give the same number of
    results than testing the full length.
    """
    from IPython.core.completer import _unicode_name_compute, _UNICODE_RANGES

    expected_list = _unicode_name_compute([(0, 0x110000)])
    test = _unicode_name_compute(_UNICODE_RANGES)
    len_exp = len(expected_list)
    len_test = len(test)

    # do not inline the len() or on error pytest will try to print the 130 000 +
    # elements.
    message = None
    if len_exp != len_test or len_exp > 131808:
        size, start, stop, prct = recompute_unicode_ranges()
        message = f"""_UNICODE_RANGES likely wrong and need updating. This is
        likely due to a new release of Python. We've find that the biggest gap
        in unicode characters has reduces in size to be {size} characters
        ({prct}), from {start}, to {stop}. In completer.py likely update to

            _UNICODE_RANGES = [(32, {start}), ({stop}, 0xe01f0)]

        And update the assertion below to use

            len_exp <= {len_exp}
        """
    assert len_exp == len_test, message

    # fail if new unicode symbols have been added.
    assert len_exp <= 143668, message


@contextmanager
def greedy_completion():
    ip = get_ipython()
    greedy_original = ip.Completer.greedy
    try:
        ip.Completer.greedy = True
        yield
    finally:
        ip.Completer.greedy = greedy_original


@contextmanager
def evaluation_policy(evaluation: str):
    ip = get_ipython()
    evaluation_original = ip.Completer.evaluation
    try:
        ip.Completer.evaluation = evaluation
        yield
    finally:
        ip.Completer.evaluation = evaluation_original


@contextmanager
def custom_matchers(matchers):
    ip = get_ipython()
    try:
        ip.Completer.custom_matchers.extend(matchers)
        yield
    finally:
        ip.Completer.custom_matchers.clear()


def test_protect_filename():
    if sys.platform == "win32":
        pairs = [
            ("abc", "abc"),
            (" abc", '" abc"'),
            ("a bc", '"a bc"'),
            ("a  bc", '"a  bc"'),
            ("  bc", '"  bc"'),
        ]
    else:
        pairs = [
            ("abc", "abc"),
            (" abc", r"\ abc"),
            ("a bc", r"a\ bc"),
            ("a  bc", r"a\ \ bc"),
            ("  bc", r"\ \ bc"),
            # On posix, we also protect parens and other special characters.
            ("a(bc", r"a\(bc"),
            ("a)bc", r"a\)bc"),
            ("a( )bc", r"a\(\ \)bc"),
            ("a[1]bc", r"a\[1\]bc"),
            ("a{1}bc", r"a\{1\}bc"),
            ("a#bc", r"a\#bc"),
            ("a?bc", r"a\?bc"),
            ("a=bc", r"a\=bc"),
            ("a\\bc", r"a\\bc"),
            ("a|bc", r"a\|bc"),
            ("a;bc", r"a\;bc"),
            ("a:bc", r"a\:bc"),
            ("a'bc", r"a\'bc"),
            ("a*bc", r"a\*bc"),
            ('a"bc', r"a\"bc"),
            ("a^bc", r"a\^bc"),
            ("a&bc", r"a\&bc"),
        ]
    # run the actual tests
    for s1, s2 in pairs:
        s1p = completer.protect_filename(s1)
        assert s1p == s2


def check_line_split(splitter, test_specs):
    for part1, part2, split in test_specs:
        cursor_pos = len(part1)
        line = part1 + part2
        out = splitter.split_line(line, cursor_pos)
        assert out == split

def test_line_split():
    """Basic line splitter test with default specs."""
    sp = completer.CompletionSplitter()
    # The format of the test specs is: part1, part2, expected answer.  Parts 1
    # and 2 are joined into the 'line' sent to the splitter, as if the cursor
    # was at the end of part1.  So an empty part2 represents someone hitting
    # tab at the end of the line, the most common case.
    t = [
        ("run some/scrip", "", "some/scrip"),
        ("run scripts/er", "ror.py foo", "scripts/er"),
        ("echo $HOM", "", "HOM"),
        ("print sys.pa", "", "sys.pa"),
        ("print(sys.pa", "", "sys.pa"),
        ("execfile('scripts/er", "", "scripts/er"),
        ("a[x.", "", "x."),
        ("a[x.", "y", "x."),
        ('cd "some_file/', "", "some_file/"),
    ]
    check_line_split(sp, t)
    # Ensure splitting works OK with unicode by re-running the tests with
    # all inputs turned into unicode
    check_line_split(sp, [map(str, p) for p in t])


class NamedInstanceClass:
    instances = {}

    def __init__(self, name):
        self.instances[name] = self

    @classmethod
    def _ipython_key_completions_(cls):
        return cls.instances.keys()


class KeyCompletable:
    def __init__(self, things=()):
        self.things = things

    def _ipython_key_completions_(self):
        return list(self.things)


class TestCompleter(unittest.TestCase):
    def setUp(self):
        """
        We want to silence all PendingDeprecationWarning when testing the completer
        """
        self._assertwarns = self.assertWarns(PendingDeprecationWarning)
        self._assertwarns.__enter__()

    def tearDown(self):
        try:
            self._assertwarns.__exit__(None, None, None)
        except AssertionError:
            pass

    def test_custom_completion_error(self):
        """Test that errors from custom attribute completers are silenced."""
        ip = get_ipython()

        class A:
            pass

        ip.user_ns["x"] = A()

        @complete_object.register(A)
        def complete_A(a, existing_completions):
            raise TypeError("this should be silenced")

        ip.complete("x.")

    def test_custom_completion_ordering(self):
        """Test that errors from custom attribute completers are silenced."""
        ip = get_ipython()

        _, matches = ip.complete('in')
        assert matches.index('input') < matches.index('int')

        def complete_example(a):
            return ['example2', 'example1']

        ip.Completer.custom_completers.add_re('ex*', complete_example)
        _, matches = ip.complete('ex')
        assert matches.index('example2') < matches.index('example1')

    def test_unicode_completions(self):
        ip = get_ipython()
        # Some strings that trigger different types of completion.  Check them both
        # in str and unicode forms
        s = ["ru", "%ru", "cd /", "floa", "float(x)/"]
        for t in s + list(map(str, s)):
            # We don't need to check exact completion values (they may change
            # depending on the state of the namespace, but at least no exceptions
            # should be thrown and the return value should be a pair of text, list
            # values.
            text, matches = ip.complete(t)
            self.assertIsInstance(text, str)
            self.assertIsInstance(matches, list)

    def test_latex_completions(self):
        from IPython.core.latex_symbols import latex_symbols
        import random

        ip = get_ipython()
        # Test some random unicode symbols
        keys = random.sample(sorted(latex_symbols), 10)
        for k in keys:
            text, matches = ip.complete(k)
            self.assertEqual(text, k)
            self.assertEqual(matches, [latex_symbols[k]])
        # Test a more complex line
        text, matches = ip.complete("print(\\alpha")
        self.assertEqual(text, "\\alpha")
        self.assertEqual(matches[0], latex_symbols["\\alpha"])
        # Test multiple matching latex symbols
        text, matches = ip.complete("\\al")
        self.assertIn("\\alpha", matches)
        self.assertIn("\\aleph", matches)

    def test_latex_no_results(self):
        """
        forward latex should really return nothing in either field if nothing is found.
        """
        ip = get_ipython()
        text, matches = ip.Completer.latex_matches("\\really_i_should_match_nothing")
        self.assertEqual(text, "")
        self.assertEqual(matches, ())

    def test_back_latex_completion(self):
        ip = get_ipython()

        # do not return more than 1 matches for \beta, only the latex one.
        name, matches = ip.complete("\\β")
        self.assertEqual(matches, ["\\beta"])

    def test_back_unicode_completion(self):
        ip = get_ipython()

        name, matches = ip.complete("\\Ⅴ")
        self.assertEqual(matches, ["\\ROMAN NUMERAL FIVE"])

    def test_forward_unicode_completion(self):
        ip = get_ipython()

        name, matches = ip.complete("\\ROMAN NUMERAL FIVE")
        self.assertEqual(matches, ["Ⅴ"])  # This is not a V
        self.assertEqual(matches, ["\u2164"])  # same as above but explicit.

    def test_delim_setting(self):
        sp = completer.CompletionSplitter()
        sp.delims = " "
        self.assertEqual(sp.delims, " ")
        self.assertEqual(sp._delim_expr, r"[\ ]")

    def test_spaces(self):
        """Test with only spaces as split chars."""
        sp = completer.CompletionSplitter()
        sp.delims = " "
        t = [("foo", "", "foo"), ("run foo", "", "foo"), ("run foo", "bar", "foo")]
        check_line_split(sp, t)

    def test_has_open_quotes1(self):
        for s in ["'", "'''", "'hi' '"]:
            self.assertEqual(completer.has_open_quotes(s), "'")

    def test_has_open_quotes2(self):
        for s in ['"', '"""', '"hi" "']:
            self.assertEqual(completer.has_open_quotes(s), '"')

    def test_has_open_quotes3(self):
        for s in ["''", "''' '''", "'hi' 'ipython'"]:
            self.assertFalse(completer.has_open_quotes(s))

    def test_has_open_quotes4(self):
        for s in ['""', '""" """', '"hi" "ipython"']:
            self.assertFalse(completer.has_open_quotes(s))

    @pytest.mark.xfail(
        sys.platform == "win32", reason="abspath completions fail on Windows"
    )
    def test_abspath_file_completions(self):
        ip = get_ipython()
        with TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, "foo")
            suffixes = ["1", "2"]
            names = [prefix + s for s in suffixes]
            for n in names:
                open(n, "w", encoding="utf-8").close()

            # Check simple completion
            c = ip.complete(prefix)[1]
            self.assertEqual(c, names)

            # Now check with a function call
            cmd = 'a = f("%s' % prefix
            c = ip.complete(prefix, cmd)[1]
            comp = [prefix + s for s in suffixes]
            self.assertEqual(c, comp)

    def test_local_file_completions(self):
        ip = get_ipython()
        with TemporaryWorkingDirectory():
            prefix = "./foo"
            suffixes = ["1", "2"]
            names = [prefix + s for s in suffixes]
            for n in names:
                open(n, "w", encoding="utf-8").close()

            # Check simple completion
            c = ip.complete(prefix)[1]
            self.assertEqual(c, names)

            # Now check with a function call
            cmd = 'a = f("%s' % prefix
            c = ip.complete(prefix, cmd)[1]
            comp = {prefix + s for s in suffixes}
            self.assertTrue(comp.issubset(set(c)))

    def test_quoted_file_completions(self):
        ip = get_ipython()

        def _(text):
            return ip.Completer._complete(
                cursor_line=0, cursor_pos=len(text), full_text=text
            )["IPCompleter.file_matcher"]["completions"]

        with TemporaryWorkingDirectory():
            name = "foo'bar"
            open(name, "w", encoding="utf-8").close()

            # Don't escape Windows
            escaped = name if sys.platform == "win32" else "foo\\'bar"

            # Single quote matches embedded single quote
            c = _("open('foo")[0]
            self.assertEqual(c.text, escaped)

            # Double quote requires no escape
            c = _('open("foo')[0]
            self.assertEqual(c.text, name)

            # No quote requires an escape
            c = _("%ls foo")[0]
            self.assertEqual(c.text, escaped)

    @pytest.mark.xfail(
        sys.version_info.releaselevel in ("alpha",),
        reason="Parso does not yet parse 3.13",
    )
    def test_all_completions_dups(self):
        """
        Make sure the output of `IPCompleter.all_completions` does not have
        duplicated prefixes.
        """
        ip = get_ipython()
        c = ip.Completer
        ip.ex("class TestClass():\n\ta=1\n\ta1=2")
        for jedi_status in [True, False]:
            with provisionalcompleter():
                ip.Completer.use_jedi = jedi_status
                matches = c.all_completions("TestCl")
                assert matches == ["TestClass"], (jedi_status, matches)
                matches = c.all_completions("TestClass.")
                assert len(matches) > 2, (jedi_status, matches)
                matches = c.all_completions("TestClass.a")
                assert matches == ['TestClass.a', 'TestClass.a1'], jedi_status

    @pytest.mark.xfail(
        sys.version_info.releaselevel in ("alpha",),
        reason="Parso does not yet parse 3.13",
    )
    def test_jedi(self):
        """
        A couple of issue we had with Jedi
        """
        ip = get_ipython()

        def _test_complete(reason, s, comp, start=None, end=None):
            l = len(s)
            start = start if start is not None else l
            end = end if end is not None else l
            with provisionalcompleter():
                ip.Completer.use_jedi = True
                completions = set(ip.Completer.completions(s, l))
                ip.Completer.use_jedi = False
                assert Completion(start, end, comp) in completions, reason

        def _test_not_complete(reason, s, comp):
            l = len(s)
            with provisionalcompleter():
                ip.Completer.use_jedi = True
                completions = set(ip.Completer.completions(s, l))
                ip.Completer.use_jedi = False
                assert Completion(l, l, comp) not in completions, reason

        import jedi

        jedi_version = tuple(int(i) for i in jedi.__version__.split(".")[:3])
        if jedi_version > (0, 10):
            _test_complete("jedi >0.9 should complete and not crash", "a=1;a.", "real")
        _test_complete("can infer first argument", 'a=(1,"foo");a[0].', "real")
        _test_complete("can infer second argument", 'a=(1,"foo");a[1].', "capitalize")
        _test_complete("cover duplicate completions", "im", "import", 0, 2)

        _test_not_complete("does not mix types", 'a=(1,"foo");a[0].', "capitalize")

    @pytest.mark.xfail(
        sys.version_info.releaselevel in ("alpha",),
        reason="Parso does not yet parse 3.13",
    )
    def test_completion_have_signature(self):
        """
        Lets make sure jedi is capable of pulling out the signature of the function we are completing.
        """
        ip = get_ipython()
        with provisionalcompleter():
            ip.Completer.use_jedi = True
            completions = ip.Completer.completions("ope", 3)
            c = next(completions)  # should be `open`
            ip.Completer.use_jedi = False
        assert "file" in c.signature, "Signature of function was not found by completer"
        assert (
            "encoding" in c.signature
        ), "Signature of function was not found by completer"

    @pytest.mark.xfail(
        sys.version_info.releaselevel in ("alpha",),
        reason="Parso does not yet parse 3.13",
    )
    def test_completions_have_type(self):
        """
        Lets make sure matchers provide completion type.
        """
        ip = get_ipython()
        with provisionalcompleter():
            ip.Completer.use_jedi = False
            completions = ip.Completer.completions("%tim", 3)
            c = next(completions)  # should be `%time` or similar
        assert c.type == "magic", "Type of magic was not assigned by completer"

    @pytest.mark.xfail(reason="Known failure on jedi<=0.18.0")
    def test_deduplicate_completions(self):
        """
        Test that completions are correctly deduplicated (even if ranges are not the same)
        """
        ip = get_ipython()
        ip.ex(
            textwrap.dedent(
                """
        class Z:
            zoo = 1
        """
            )
        )
        with provisionalcompleter():
            ip.Completer.use_jedi = True
            l = list(
                _deduplicate_completions("Z.z", ip.Completer.completions("Z.z", 3))
            )
            ip.Completer.use_jedi = False

        assert len(l) == 1, "Completions (Z.z<tab>) correctly deduplicate: %s " % l
        assert l[0].text == "zoo"  # and not `it.accumulate`

    @pytest.mark.xfail(
        sys.version_info.releaselevel in ("alpha",),
        reason="Parso does not yet parse 3.13",
    )
    def test_greedy_completions(self):
        """
        Test the capability of the Greedy completer.

        Most of the test here does not really show off the greedy completer, for proof
        each of the text below now pass with Jedi. The greedy completer is capable of more.

        See the :any:`test_dict_key_completion_contexts`

        """
        ip = get_ipython()
        ip.ex("a=list(range(5))")
        ip.ex("d = {'a b': str}")
        _, c = ip.complete(".", line="a[0].")
        self.assertFalse(".real" in c, "Shouldn't have completed on a[0]: %s" % c)

        def _(line, cursor_pos, expect, message, completion):
            with greedy_completion(), provisionalcompleter():
                ip.Completer.use_jedi = False
                _, c = ip.complete(".", line=line, cursor_pos=cursor_pos)
                self.assertIn(expect, c, message % c)

                ip.Completer.use_jedi = True
                with provisionalcompleter():
                    completions = ip.Completer.completions(line, cursor_pos)
                self.assertIn(completion, completions)

        with provisionalcompleter():
            _(
                "a[0].",
                5,
                ".real",
                "Should have completed on a[0].: %s",
                Completion(5, 5, "real"),
            )
            _(
                "a[0].r",
                6,
                ".real",
                "Should have completed on a[0].r: %s",
                Completion(5, 6, "real"),
            )

            _(
                "a[0].from_",
                10,
                ".from_bytes",
                "Should have completed on a[0].from_: %s",
                Completion(5, 10, "from_bytes"),
            )
            _(
                "assert str.star",
                14,
                "str.startswith",
                "Should have completed on `assert str.star`: %s",
                Completion(11, 14, "startswith"),
            )
            _(
                "d['a b'].str",
                12,
                ".strip",
                "Should have completed on `d['a b'].str`: %s",
                Completion(9, 12, "strip"),
            )

    def test_omit__names(self):
        # also happens to test IPCompleter as a configurable
        ip = get_ipython()
        ip._hidden_attr = 1
        ip._x = {}
        c = ip.Completer
        ip.ex("ip=get_ipython()")
        cfg = Config()
        cfg.IPCompleter.omit__names = 0
        c.update_config(cfg)
        with provisionalcompleter():
            c.use_jedi = False
            s, matches = c.complete("ip.")
            self.assertIn("ip.__str__", matches)
            self.assertIn("ip._hidden_attr", matches)

            # c.use_jedi = True
            # completions = set(c.completions('ip.', 3))
            # self.assertIn(Completion(3, 3, '__str__'), completions)
            # self.assertIn(Completion(3,3, "_hidden_attr"), completions)

        cfg = Config()
        cfg.IPCompleter.omit__names = 1
        c.update_config(cfg)
        with provisionalcompleter():
            c.use_jedi = False
            s, matches = c.complete("ip.")
            self.assertNotIn("ip.__str__", matches)
            # self.assertIn('ip._hidden_attr', matches)

            # c.use_jedi = True
            # completions = set(c.completions('ip.', 3))
            # self.assertNotIn(Completion(3,3,'__str__'), completions)
            # self.assertIn(Completion(3,3, "_hidden_attr"), completions)

        cfg = Config()
        cfg.IPCompleter.omit__names = 2
        c.update_config(cfg)
        with provisionalcompleter():
            c.use_jedi = False
            s, matches = c.complete("ip.")
            self.assertNotIn("ip.__str__", matches)
            self.assertNotIn("ip._hidden_attr", matches)

            # c.use_jedi = True
            # completions = set(c.completions('ip.', 3))
            # self.assertNotIn(Completion(3,3,'__str__'), completions)
            # self.assertNotIn(Completion(3,3, "_hidden_attr"), completions)

        with provisionalcompleter():
            c.use_jedi = False
            s, matches = c.complete("ip._x.")
            self.assertIn("ip._x.keys", matches)

            # c.use_jedi = True
            # completions = set(c.completions('ip._x.', 6))
            # self.assertIn(Completion(6,6, "keys"), completions)

        del ip._hidden_attr
        del ip._x

    def test_limit_to__all__False_ok(self):
        """
        Limit to all is deprecated, once we remove it this test can go away. 
        """
        ip = get_ipython()
        c = ip.Completer
        c.use_jedi = False
        ip.ex("class D: x=24")
        ip.ex("d=D()")
        cfg = Config()
        cfg.IPCompleter.limit_to__all__ = False
        c.update_config(cfg)
        s, matches = c.complete("d.")
        self.assertIn("d.x", matches)

    def test_get__all__entries_ok(self):
        class A:
            __all__ = ["x", 1]

        words = completer.get__all__entries(A())
        self.assertEqual(words, ["x"])

    def test_get__all__entries_no__all__ok(self):
        class A:
            pass

        words = completer.get__all__entries(A())
        self.assertEqual(words, [])

    def test_func_kw_completions(self):
        ip = get_ipython()
        c = ip.Completer
        c.use_jedi = False
        ip.ex("def myfunc(a=1,b=2): return a+b")
        s, matches = c.complete(None, "myfunc(1,b")
        self.assertIn("b=", matches)
        # Simulate completing with cursor right after b (pos==10):
        s, matches = c.complete(None, "myfunc(1,b)", 10)
        self.assertIn("b=", matches)
        s, matches = c.complete(None, 'myfunc(a="escaped\\")string",b')
        self.assertIn("b=", matches)
        # builtin function
        s, matches = c.complete(None, "min(k, k")
        self.assertIn("key=", matches)

    def test_default_arguments_from_docstring(self):
        ip = get_ipython()
        c = ip.Completer
        kwd = c._default_arguments_from_docstring("min(iterable[, key=func]) -> value")
        self.assertEqual(kwd, ["key"])
        # with cython type etc
        kwd = c._default_arguments_from_docstring(
            "Minuit.migrad(self, int ncall=10000, resume=True, int nsplit=1)\n"
        )
        self.assertEqual(kwd, ["ncall", "resume", "nsplit"])
        # white spaces
        kwd = c._default_arguments_from_docstring(
            "\n Minuit.migrad(self, int ncall=10000, resume=True, int nsplit=1)\n"
        )
        self.assertEqual(kwd, ["ncall", "resume", "nsplit"])

    def test_line_magics(self):
        ip = get_ipython()
        c = ip.Completer
        s, matches = c.complete(None, "lsmag")
        self.assertIn("%lsmagic", matches)
        s, matches = c.complete(None, "%lsmag")
        self.assertIn("%lsmagic", matches)

    def test_cell_magics(self):
        from IPython.core.magic import register_cell_magic

        @register_cell_magic
        def _foo_cellm(line, cell):
            pass

        ip = get_ipython()
        c = ip.Completer

        s, matches = c.complete(None, "_foo_ce")
        self.assertIn("%%_foo_cellm", matches)
        s, matches = c.complete(None, "%%_foo_ce")
        self.assertIn("%%_foo_cellm", matches)

    def test_line_cell_magics(self):
        from IPython.core.magic import register_line_cell_magic

        @register_line_cell_magic
        def _bar_cellm(line, cell):
            pass

        ip = get_ipython()
        c = ip.Completer

        # The policy here is trickier, see comments in completion code.  The
        # returned values depend on whether the user passes %% or not explicitly,
        # and this will show a difference if the same name is both a line and cell
        # magic.
        s, matches = c.complete(None, "_bar_ce")
        self.assertIn("%_bar_cellm", matches)
        self.assertIn("%%_bar_cellm", matches)
        s, matches = c.complete(None, "%_bar_ce")
        self.assertIn("%_bar_cellm", matches)
        self.assertIn("%%_bar_cellm", matches)
        s, matches = c.complete(None, "%%_bar_ce")
        self.assertNotIn("%_bar_cellm", matches)
        self.assertIn("%%_bar_cellm", matches)

    def test_magic_completion_order(self):
        ip = get_ipython()
        c = ip.Completer

        # Test ordering of line and cell magics.
        text, matches = c.complete("timeit")
        self.assertEqual(matches, ["%timeit", "%%timeit"])

    def test_magic_completion_shadowing(self):
        ip = get_ipython()
        c = ip.Completer
        c.use_jedi = False

        # Before importing matplotlib, %matplotlib magic should be the only option.
        text, matches = c.complete("mat")
        self.assertEqual(matches, ["%matplotlib"])

        # The newly introduced name should shadow the magic.
        ip.run_cell("matplotlib = 1")
        text, matches = c.complete("mat")
        self.assertEqual(matches, ["matplotlib"])

        # After removing matplotlib from namespace, the magic should again be
        # the only option.
        del ip.user_ns["matplotlib"]
        text, matches = c.complete("mat")
        self.assertEqual(matches, ["%matplotlib"])

    def test_magic_completion_shadowing_explicit(self):
        """
        If the user try to complete a shadowed magic, and explicit % start should
        still return the completions.
        """
        ip = get_ipython()
        c = ip.Completer

        # Before importing matplotlib, %matplotlib magic should be the only option.
        text, matches = c.complete("%mat")
        self.assertEqual(matches, ["%matplotlib"])

        ip.run_cell("matplotlib = 1")

        # After removing matplotlib from namespace, the magic should still be
        # the only option.
        text, matches = c.complete("%mat")
        self.assertEqual(matches, ["%matplotlib"])

    def test_magic_config(self):
        ip = get_ipython()
        c = ip.Completer

        s, matches = c.complete(None, "conf")
        self.assertIn("%config", matches)
        s, matches = c.complete(None, "conf")
        self.assertNotIn("AliasManager", matches)
        s, matches = c.complete(None, "config ")
        self.assertIn("AliasManager", matches)
        s, matches = c.complete(None, "%config ")
        self.assertIn("AliasManager", matches)
        s, matches = c.complete(None, "config Ali")
        self.assertListEqual(["AliasManager"], matches)
        s, matches = c.complete(None, "%config Ali")
        self.assertListEqual(["AliasManager"], matches)
        s, matches = c.complete(None, "config AliasManager")
        self.assertListEqual(["AliasManager"], matches)
        s, matches = c.complete(None, "%config AliasManager")
        self.assertListEqual(["AliasManager"], matches)
        s, matches = c.complete(None, "config AliasManager.")
        self.assertIn("AliasManager.default_aliases", matches)
        s, matches = c.complete(None, "%config AliasManager.")
        self.assertIn("AliasManager.default_aliases", matches)
        s, matches = c.complete(None, "config AliasManager.de")
        self.assertListEqual(["AliasManager.default_aliases"], matches)
        s, matches = c.complete(None, "config AliasManager.de")
        self.assertListEqual(["AliasManager.default_aliases"], matches)

    def test_magic_color(self):
        ip = get_ipython()
        c = ip.Completer

        s, matches = c.complete(None, "colo")
        self.assertIn("%colors", matches)
        s, matches = c.complete(None, "colo")
        self.assertNotIn("NoColor", matches)
        s, matches = c.complete(None, "%colors")  # No trailing space
        self.assertNotIn("NoColor", matches)
        s, matches = c.complete(None, "colors ")
        self.assertIn("NoColor", matches)
        s, matches = c.complete(None, "%colors ")
        self.assertIn("NoColor", matches)
        s, matches = c.complete(None, "colors NoCo")
        self.assertListEqual(["NoColor"], matches)
        s, matches = c.complete(None, "%colors NoCo")
        self.assertListEqual(["NoColor"], matches)

    def test_match_dict_keys(self):
        """
        Test that match_dict_keys works on a couple of use case does return what
        expected, and does not crash
        """
        delims = " \t\n`!@#$^&*()=+[{]}\\|;:'\",<>?"

        def match(*args, **kwargs):
            quote, offset, matches = match_dict_keys(*args, delims=delims, **kwargs)
            return quote, offset, list(matches)

        keys = ["foo", b"far"]
        assert match(keys, "b'") == ("'", 2, ["far"])
        assert match(keys, "b'f") == ("'", 2, ["far"])
        assert match(keys, 'b"') == ('"', 2, ["far"])
        assert match(keys, 'b"f') == ('"', 2, ["far"])

        assert match(keys, "'") == ("'", 1, ["foo"])
        assert match(keys, "'f") == ("'", 1, ["foo"])
        assert match(keys, '"') == ('"', 1, ["foo"])
        assert match(keys, '"f') == ('"', 1, ["foo"])

        # Completion on first item of tuple
        keys = [("foo", 1111), ("foo", 2222), (3333, "bar"), (3333, "test")]
        assert match(keys, "'f") == ("'", 1, ["foo"])
        assert match(keys, "33") == ("", 0, ["3333"])

        # Completion on numbers
        keys = [
            0xDEADBEEF,
            1111,
            1234,
            "1999",
            0b10101,
            22,
        ]  # 0xDEADBEEF = 3735928559; 0b10101 = 21
        assert match(keys, "0xdead") == ("", 0, ["0xdeadbeef"])
        assert match(keys, "1") == ("", 0, ["1111", "1234"])
        assert match(keys, "2") == ("", 0, ["21", "22"])
        assert match(keys, "0b101") == ("", 0, ["0b10101", "0b10110"])

        # Should yield on variables
        assert match(keys, "a_variable") == ("", 0, [])

        # Should pass over invalid literals
        assert match(keys, "'' ''") == ("", 0, [])

    def test_match_dict_keys_tuple(self):
        """
        Test that match_dict_keys called with extra prefix works on a couple of use case,
        does return what expected, and does not crash.
        """
        delims = " \t\n`!@#$^&*()=+[{]}\\|;:'\",<>?"

        keys = [("foo", "bar"), ("foo", "oof"), ("foo", b"bar"), ('other', 'test')]

        def match(*args, extra=None, **kwargs):
            quote, offset, matches = match_dict_keys(
                *args, delims=delims, extra_prefix=extra, **kwargs
            )
            return quote, offset, list(matches)

        # Completion on first key == "foo"
        assert match(keys, "'", extra=("foo",)) == ("'", 1, ["bar", "oof"])
        assert match(keys, '"', extra=("foo",)) == ('"', 1, ["bar", "oof"])
        assert match(keys, "'o", extra=("foo",)) == ("'", 1, ["oof"])
        assert match(keys, '"o', extra=("foo",)) == ('"', 1, ["oof"])
        assert match(keys, "b'", extra=("foo",)) == ("'", 2, ["bar"])
        assert match(keys, 'b"', extra=("foo",)) == ('"', 2, ["bar"])
        assert match(keys, "b'b", extra=("foo",)) == ("'", 2, ["bar"])
        assert match(keys, 'b"b', extra=("foo",)) == ('"', 2, ["bar"])

        # No Completion
        assert match(keys, "'", extra=("no_foo",)) == ("'", 1, [])
        assert match(keys, "'", extra=("fo",)) == ("'", 1, [])

        keys = [("foo1", "foo2", "foo3", "foo4"), ("foo1", "foo2", "bar", "foo4")]
        assert match(keys, "'foo", extra=("foo1",)) == ("'", 1, ["foo2"])
        assert match(keys, "'foo", extra=("foo1", "foo2")) == ("'", 1, ["foo3"])
        assert match(keys, "'foo", extra=("foo1", "foo2", "foo3")) == ("'", 1, ["foo4"])
        assert match(keys, "'foo", extra=("foo1", "foo2", "foo3", "foo4")) == (
            "'",
            1,
            [],
        )

        keys = [("foo", 1111), ("foo", "2222"), (3333, "bar"), (3333, 4444)]
        assert match(keys, "'", extra=("foo",)) == ("'", 1, ["2222"])
        assert match(keys, "", extra=("foo",)) == ("", 0, ["1111", "'2222'"])
        assert match(keys, "'", extra=(3333,)) == ("'", 1, ["bar"])
        assert match(keys, "", extra=(3333,)) == ("", 0, ["'bar'", "4444"])
        assert match(keys, "'", extra=("3333",)) == ("'", 1, [])
        assert match(keys, "33") == ("", 0, ["3333"])

    def test_dict_key_completion_closures(self):
        ip = get_ipython()
        complete = ip.Completer.complete
        ip.Completer.auto_close_dict_keys = True

        ip.user_ns["d"] = {
            # tuple only
            ("aa", 11): None,
            # tuple and non-tuple
            ("bb", 22): None,
            "bb": None,
            # non-tuple only
            "cc": None,
            # numeric tuple only
            (77, "x"): None,
            # numeric tuple and non-tuple
            (88, "y"): None,
            88: None,
            # numeric non-tuple only
            99: None,
        }

        _, matches = complete(line_buffer="d[")
        # should append `, ` if matches a tuple only
        self.assertIn("'aa', ", matches)
        # should not append anything if matches a tuple and an item
        self.assertIn("'bb'", matches)
        # should append `]` if matches and item only
        self.assertIn("'cc']", matches)

        # should append `, ` if matches a tuple only
        self.assertIn("77, ", matches)
        # should not append anything if matches a tuple and an item
        self.assertIn("88", matches)
        # should append `]` if matches and item only
        self.assertIn("99]", matches)

        _, matches = complete(line_buffer="d['aa', ")
        # should restrict matches to those matching tuple prefix
        self.assertIn("11]", matches)
        self.assertNotIn("'bb'", matches)
        self.assertNotIn("'bb', ", matches)
        self.assertNotIn("'bb']", matches)
        self.assertNotIn("'cc'", matches)
        self.assertNotIn("'cc', ", matches)
        self.assertNotIn("'cc']", matches)
        ip.Completer.auto_close_dict_keys = False

    def test_dict_key_completion_string(self):
        """Test dictionary key completion for string keys"""
        ip = get_ipython()
        complete = ip.Completer.complete

        ip.user_ns["d"] = {"abc": None}

        # check completion at different stages
        _, matches = complete(line_buffer="d[")
        self.assertIn("'abc'", matches)
        self.assertNotIn("'abc']", matches)

        _, matches = complete(line_buffer="d['")
        self.assertIn("abc", matches)
        self.assertNotIn("abc']", matches)

        _, matches = complete(line_buffer="d['a")
        self.assertIn("abc", matches)
        self.assertNotIn("abc']", matches)

        # check use of different quoting
        _, matches = complete(line_buffer='d["')
        self.assertIn("abc", matches)
        self.assertNotIn('abc"]', matches)

        _, matches = complete(line_buffer='d["a')
        self.assertIn("abc", matches)
        self.assertNotIn('abc"]', matches)

        # check sensitivity to following context
        _, matches = complete(line_buffer="d[]", cursor_pos=2)
        self.assertIn("'abc'", matches)

        _, matches = complete(line_buffer="d['']", cursor_pos=3)
        self.assertIn("abc", matches)
        self.assertNotIn("abc'", matches)
        self.assertNotIn("abc']", matches)

        # check multiple solutions are correctly returned and that noise is not
        ip.user_ns["d"] = {
            "abc": None,
            "abd": None,
            "bad": None,
            object(): None,
            5: None,
            ("abe", None): None,
            (None, "abf"): None
        }

        _, matches = complete(line_buffer="d['a")
        self.assertIn("abc", matches)
        self.assertIn("abd", matches)
        self.assertNotIn("bad", matches)
        self.assertNotIn("abe", matches)
        self.assertNotIn("abf", matches)
        assert not any(m.endswith(("]", '"', "'")) for m in matches), matches

        # check escaping and whitespace
        ip.user_ns["d"] = {"a\nb": None, "a'b": None, 'a"b': None, "a word": None}
        _, matches = complete(line_buffer="d['a")
        self.assertIn("a\\nb", matches)
        self.assertIn("a\\'b", matches)
        self.assertIn('a"b', matches)
        self.assertIn("a word", matches)
        assert not any(m.endswith(("]", '"', "'")) for m in matches), matches

        # - can complete on non-initial word of the string
        _, matches = complete(line_buffer="d['a w")
        self.assertIn("word", matches)

        # - understands quote escaping
        _, matches = complete(line_buffer="d['a\\'")
        self.assertIn("b", matches)

        # - default quoting should work like repr
        _, matches = complete(line_buffer="d[")
        self.assertIn('"a\'b"', matches)

        # - when opening quote with ", possible to match with unescaped apostrophe
        _, matches = complete(line_buffer="d[\"a'")
        self.assertIn("b", matches)

        # need to not split at delims that readline won't split at
        if "-" not in ip.Completer.splitter.delims:
            ip.user_ns["d"] = {"before-after": None}
            _, matches = complete(line_buffer="d['before-af")
            self.assertIn("before-after", matches)

        # check completion on tuple-of-string keys at different stage - on first key
        ip.user_ns["d"] = {('foo', 'bar'): None}
        _, matches = complete(line_buffer="d[")
        self.assertIn("'foo'", matches)
        self.assertNotIn("'foo']", matches)
        self.assertNotIn("'bar'", matches)
        self.assertNotIn("foo", matches)
        self.assertNotIn("bar", matches)

        # - match the prefix
        _, matches = complete(line_buffer="d['f")
        self.assertIn("foo", matches)
        self.assertNotIn("foo']", matches)
        self.assertNotIn('foo"]', matches)
        _, matches = complete(line_buffer="d['foo")
        self.assertIn("foo", matches)

        # - can complete on second key
        _, matches = complete(line_buffer="d['foo', ")
        self.assertIn("'bar'", matches)
        _, matches = complete(line_buffer="d['foo', 'b")
        self.assertIn("bar", matches)
        self.assertNotIn("foo", matches)

        # - does not propose missing keys
        _, matches = complete(line_buffer="d['foo', 'f")
        self.assertNotIn("bar", matches)
        self.assertNotIn("foo", matches)

        # check sensitivity to following context
        _, matches = complete(line_buffer="d['foo',]", cursor_pos=8)
        self.assertIn("'bar'", matches)
        self.assertNotIn("bar", matches)
        self.assertNotIn("'foo'", matches)
        self.assertNotIn("foo", matches)

        _, matches = complete(line_buffer="d['']", cursor_pos=3)
        self.assertIn("foo", matches)
        assert not any(m.endswith(("]", '"', "'")) for m in matches), matches

        _, matches = complete(line_buffer='d[""]', cursor_pos=3)
        self.assertIn("foo", matches)
        assert not any(m.endswith(("]", '"', "'")) for m in matches), matches

        _, matches = complete(line_buffer='d["foo","]', cursor_pos=9)
        self.assertIn("bar", matches)
        assert not any(m.endswith(("]", '"', "'")) for m in matches), matches

        _, matches = complete(line_buffer='d["foo",]', cursor_pos=8)
        self.assertIn("'bar'", matches)
        self.assertNotIn("bar", matches)

        # Can complete with longer tuple keys
        ip.user_ns["d"] = {('foo', 'bar', 'foobar'): None}

        # - can complete second key
        _, matches = complete(line_buffer="d['foo', 'b")
        self.assertIn("bar", matches)
        self.assertNotIn("foo", matches)
        self.assertNotIn("foobar", matches)

        # - can complete third key
        _, matches = complete(line_buffer="d['foo', 'bar', 'fo")
        self.assertIn("foobar", matches)
        self.assertNotIn("foo", matches)
        self.assertNotIn("bar", matches)

    def test_dict_key_completion_numbers(self):
        ip = get_ipython()
        complete = ip.Completer.complete

        ip.user_ns["d"] = {
            0xDEADBEEF: None,  # 3735928559
            1111: None,
            1234: None,
            "1999": None,
            0b10101: None,  # 21
            22: None,
        }
        _, matches = complete(line_buffer="d[1")
        self.assertIn("1111", matches)
        self.assertIn("1234", matches)
        self.assertNotIn("1999", matches)
        self.assertNotIn("'1999'", matches)

        _, matches = complete(line_buffer="d[0xdead")
        self.assertIn("0xdeadbeef", matches)

        _, matches = complete(line_buffer="d[2")
        self.assertIn("21", matches)
        self.assertIn("22", matches)

        _, matches = complete(line_buffer="d[0b101")
        self.assertIn("0b10101", matches)
        self.assertIn("0b10110", matches)

    def test_dict_key_completion_contexts(self):
        """Test expression contexts in which dict key completion occurs"""
        ip = get_ipython()
        complete = ip.Completer.complete
        d = {"abc": None}
        ip.user_ns["d"] = d

        class C:
            data = d

        ip.user_ns["C"] = C
        ip.user_ns["get"] = lambda: d
        ip.user_ns["nested"] = {"x": d}

        def assert_no_completion(**kwargs):
            _, matches = complete(**kwargs)
            self.assertNotIn("abc", matches)
            self.assertNotIn("abc'", matches)
            self.assertNotIn("abc']", matches)
            self.assertNotIn("'abc'", matches)
            self.assertNotIn("'abc']", matches)

        def assert_completion(**kwargs):
            _, matches = complete(**kwargs)
            self.assertIn("'abc'", matches)
            self.assertNotIn("'abc']", matches)

        # no completion after string closed, even if reopened
        assert_no_completion(line_buffer="d['a'")
        assert_no_completion(line_buffer='d["a"')
        assert_no_completion(line_buffer="d['a' + ")
        assert_no_completion(line_buffer="d['a' + '")

        # completion in non-trivial expressions
        assert_completion(line_buffer="+ d[")
        assert_completion(line_buffer="(d[")
        assert_completion(line_buffer="C.data[")

        # nested dict completion
        assert_completion(line_buffer="nested['x'][")

        with evaluation_policy("minimal"):
            with pytest.raises(AssertionError):
                assert_completion(line_buffer="nested['x'][")

        # greedy flag
        def assert_completion(**kwargs):
            _, matches = complete(**kwargs)
            self.assertIn("get()['abc']", matches)

        assert_no_completion(line_buffer="get()[")
        with greedy_completion():
            assert_completion(line_buffer="get()[")
            assert_completion(line_buffer="get()['")
            assert_completion(line_buffer="get()['a")
            assert_completion(line_buffer="get()['ab")
            assert_completion(line_buffer="get()['abc")

    def test_dict_key_completion_bytes(self):
        """Test handling of bytes in dict key completion"""
        ip = get_ipython()
        complete = ip.Completer.complete

        ip.user_ns["d"] = {"abc": None, b"abd": None}

        _, matches = complete(line_buffer="d[")
        self.assertIn("'abc'", matches)
        self.assertIn("b'abd'", matches)

        if False:  # not currently implemented
            _, matches = complete(line_buffer="d[b")
            self.assertIn("b'abd'", matches)
            self.assertNotIn("b'abc'", matches)

            _, matches = complete(line_buffer="d[b'")
            self.assertIn("abd", matches)
            self.assertNotIn("abc", matches)

            _, matches = complete(line_buffer="d[B'")
            self.assertIn("abd", matches)
            self.assertNotIn("abc", matches)

            _, matches = complete(line_buffer="d['")
            self.assertIn("abc", matches)
            self.assertNotIn("abd", matches)

    def test_dict_key_completion_unicode_py3(self):
        """Test handling of unicode in dict key completion"""
        ip = get_ipython()
        complete = ip.Completer.complete

        ip.user_ns["d"] = {"a\u05d0": None}

        # query using escape
        if sys.platform != "win32":
            # Known failure on Windows
            _, matches = complete(line_buffer="d['a\\u05d0")
            self.assertIn("u05d0", matches)  # tokenized after \\

        # query using character
        _, matches = complete(line_buffer="d['a\u05d0")
        self.assertIn("a\u05d0", matches)

        with greedy_completion():
            # query using escape
            _, matches = complete(line_buffer="d['a\\u05d0")
            self.assertIn("d['a\\u05d0']", matches)  # tokenized after \\

            # query using character
            _, matches = complete(line_buffer="d['a\u05d0")
            self.assertIn("d['a\u05d0']", matches)

    @dec.skip_without("numpy")
    def test_struct_array_key_completion(self):
        """Test dict key completion applies to numpy struct arrays"""
        import numpy

        ip = get_ipython()
        complete = ip.Completer.complete
        ip.user_ns["d"] = numpy.array([], dtype=[("hello", "f"), ("world", "f")])
        _, matches = complete(line_buffer="d['")
        self.assertIn("hello", matches)
        self.assertIn("world", matches)
        # complete on the numpy struct itself
        dt = numpy.dtype(
            [("my_head", [("my_dt", ">u4"), ("my_df", ">u4")]), ("my_data", ">f4", 5)]
        )
        x = numpy.zeros(2, dtype=dt)
        ip.user_ns["d"] = x[1]
        _, matches = complete(line_buffer="d['")
        self.assertIn("my_head", matches)
        self.assertIn("my_data", matches)

        def completes_on_nested():
            ip.user_ns["d"] = numpy.zeros(2, dtype=dt)
            _, matches = complete(line_buffer="d[1]['my_head']['")
            self.assertTrue(any(["my_dt" in m for m in matches]))
            self.assertTrue(any(["my_df" in m for m in matches]))
        # complete on a nested level
        with greedy_completion():
            completes_on_nested()

        with evaluation_policy("limited"):
            completes_on_nested()

        with evaluation_policy("minimal"):
            with pytest.raises(AssertionError):
                completes_on_nested()

    @dec.skip_without("pandas")
    def test_dataframe_key_completion(self):
        """Test dict key completion applies to pandas DataFrames"""
        import pandas

        ip = get_ipython()
        complete = ip.Completer.complete
        ip.user_ns["d"] = pandas.DataFrame({"hello": [1], "world": [2]})
        _, matches = complete(line_buffer="d['")
        self.assertIn("hello", matches)
        self.assertIn("world", matches)
        _, matches = complete(line_buffer="d.loc[:, '")
        self.assertIn("hello", matches)
        self.assertIn("world", matches)
        _, matches = complete(line_buffer="d.loc[1:, '")
        self.assertIn("hello", matches)
        _, matches = complete(line_buffer="d.loc[1:1, '")
        self.assertIn("hello", matches)
        _, matches = complete(line_buffer="d.loc[1:1:-1, '")
        self.assertIn("hello", matches)
        _, matches = complete(line_buffer="d.loc[::, '")
        self.assertIn("hello", matches)

    def test_dict_key_completion_invalids(self):
        """Smoke test cases dict key completion can't handle"""
        ip = get_ipython()
        complete = ip.Completer.complete

        ip.user_ns["no_getitem"] = None
        ip.user_ns["no_keys"] = []
        ip.user_ns["cant_call_keys"] = dict
        ip.user_ns["empty"] = {}
        ip.user_ns["d"] = {"abc": 5}

        _, matches = complete(line_buffer="no_getitem['")
        _, matches = complete(line_buffer="no_keys['")
        _, matches = complete(line_buffer="cant_call_keys['")
        _, matches = complete(line_buffer="empty['")
        _, matches = complete(line_buffer="name_error['")
        _, matches = complete(line_buffer="d['\\")  # incomplete escape

    def test_object_key_completion(self):
        ip = get_ipython()
        ip.user_ns["key_completable"] = KeyCompletable(["qwerty", "qwick"])

        _, matches = ip.Completer.complete(line_buffer="key_completable['qw")
        self.assertIn("qwerty", matches)
        self.assertIn("qwick", matches)

    def test_class_key_completion(self):
        ip = get_ipython()
        NamedInstanceClass("qwerty")
        NamedInstanceClass("qwick")
        ip.user_ns["named_instance_class"] = NamedInstanceClass

        _, matches = ip.Completer.complete(line_buffer="named_instance_class['qw")
        self.assertIn("qwerty", matches)
        self.assertIn("qwick", matches)

    def test_tryimport(self):
        """
        Test that try-import don't crash on trailing dot, and import modules before
        """
        from IPython.core.completerlib import try_import

        assert try_import("IPython.")

    def test_aimport_module_completer(self):
        ip = get_ipython()
        _, matches = ip.complete("i", "%aimport i")
        self.assertIn("io", matches)
        self.assertNotIn("int", matches)

    def test_nested_import_module_completer(self):
        ip = get_ipython()
        _, matches = ip.complete(None, "import IPython.co", 17)
        self.assertIn("IPython.core", matches)
        self.assertNotIn("import IPython.core", matches)
        self.assertNotIn("IPython.display", matches)

    def test_import_module_completer(self):
        ip = get_ipython()
        _, matches = ip.complete("i", "import i")
        self.assertIn("io", matches)
        self.assertNotIn("int", matches)

    def test_from_module_completer(self):
        ip = get_ipython()
        _, matches = ip.complete("B", "from io import B", 16)
        self.assertIn("BytesIO", matches)
        self.assertNotIn("BaseException", matches)

    def test_snake_case_completion(self):
        ip = get_ipython()
        ip.Completer.use_jedi = False
        ip.user_ns["some_three"] = 3
        ip.user_ns["some_four"] = 4
        _, matches = ip.complete("s_", "print(s_f")
        self.assertIn("some_three", matches)
        self.assertIn("some_four", matches)

    def test_mix_terms(self):
        ip = get_ipython()
        from textwrap import dedent

        ip.Completer.use_jedi = False
        ip.ex(
            dedent(
                """
            class Test:
                def meth(self, meth_arg1):
                    print("meth")

                def meth_1(self, meth1_arg1, meth1_arg2):
                    print("meth1")

                def meth_2(self, meth2_arg1, meth2_arg2):
                    print("meth2")
            test = Test()
            """
            )
        )
        _, matches = ip.complete(None, "test.meth(")
        self.assertIn("meth_arg1=", matches)
        self.assertNotIn("meth2_arg1=", matches)

    def test_percent_symbol_restrict_to_magic_completions(self):
        ip = get_ipython()
        completer = ip.Completer
        text = "%a"

        with provisionalcompleter():
            completer.use_jedi = True
            completions = completer.completions(text, len(text))
            for c in completions:
                self.assertEqual(c.text[0], "%")

    def test_fwd_unicode_restricts(self):
        ip = get_ipython()
        completer = ip.Completer
        text = "\\ROMAN NUMERAL FIVE"

        with provisionalcompleter():
            completer.use_jedi = True
            completions = [
                completion.text for completion in completer.completions(text, len(text))
            ]
            self.assertEqual(completions, ["\u2164"])

    def test_dict_key_restrict_to_dicts(self):
        """Test that dict key suppresses non-dict completion items"""
        ip = get_ipython()
        c = ip.Completer
        d = {"abc": None}
        ip.user_ns["d"] = d

        text = 'd["a'

        def _():
            with provisionalcompleter():
                c.use_jedi = True
                return [
                    completion.text for completion in c.completions(text, len(text))
                ]

        completions = _()
        self.assertEqual(completions, ["abc"])

        # check that it can be disabled in granular manner:
        cfg = Config()
        cfg.IPCompleter.suppress_competing_matchers = {
            "IPCompleter.dict_key_matcher": False
        }
        c.update_config(cfg)

        completions = _()
        self.assertIn("abc", completions)
        self.assertGreater(len(completions), 1)

    def test_matcher_suppression(self):
        @completion_matcher(identifier="a_matcher")
        def a_matcher(text):
            return ["completion_a"]

        @completion_matcher(identifier="b_matcher", api_version=2)
        def b_matcher(context: CompletionContext):
            text = context.token
            result = {"completions": [SimpleCompletion("completion_b")]}

            if text == "suppress c":
                result["suppress"] = {"c_matcher"}

            if text.startswith("suppress all"):
                result["suppress"] = True
                if text == "suppress all but c":
                    result["do_not_suppress"] = {"c_matcher"}
                if text == "suppress all but a":
                    result["do_not_suppress"] = {"a_matcher"}

            return result

        @completion_matcher(identifier="c_matcher")
        def c_matcher(text):
            return ["completion_c"]

        with custom_matchers([a_matcher, b_matcher, c_matcher]):
            ip = get_ipython()
            c = ip.Completer

            def _(text, expected):
                c.use_jedi = False
                s, matches = c.complete(text)
                self.assertEqual(expected, matches)

            _("do not suppress", ["completion_a", "completion_b", "completion_c"])
            _("suppress all", ["completion_b"])
            _("suppress all but a", ["completion_a", "completion_b"])
            _("suppress all but c", ["completion_b", "completion_c"])

            def configure(suppression_config):
                cfg = Config()
                cfg.IPCompleter.suppress_competing_matchers = suppression_config
                c.update_config(cfg)

            # test that configuration takes priority over the run-time decisions

            configure(False)
            _("suppress all", ["completion_a", "completion_b", "completion_c"])

            configure({"b_matcher": False})
            _("suppress all", ["completion_a", "completion_b", "completion_c"])

            configure({"a_matcher": False})
            _("suppress all", ["completion_b"])

            configure({"b_matcher": True})
            _("do not suppress", ["completion_b"])

            configure(True)
            _("do not suppress", ["completion_a"])

    def test_matcher_suppression_with_iterator(self):
        @completion_matcher(identifier="matcher_returning_iterator")
        def matcher_returning_iterator(text):
            return iter(["completion_iter"])

        @completion_matcher(identifier="matcher_returning_list")
        def matcher_returning_list(text):
            return ["completion_list"]

        with custom_matchers([matcher_returning_iterator, matcher_returning_list]):
            ip = get_ipython()
            c = ip.Completer

            def _(text, expected):
                c.use_jedi = False
                s, matches = c.complete(text)
                self.assertEqual(expected, matches)

            def configure(suppression_config):
                cfg = Config()
                cfg.IPCompleter.suppress_competing_matchers = suppression_config
                c.update_config(cfg)

            configure(False)
            _("---", ["completion_iter", "completion_list"])

            configure(True)
            _("---", ["completion_iter"])

            configure(None)
            _("--", ["completion_iter", "completion_list"])

    @pytest.mark.xfail(
        sys.version_info.releaselevel in ("alpha",),
        reason="Parso does not yet parse 3.13",
    )
    def test_matcher_suppression_with_jedi(self):
        ip = get_ipython()
        c = ip.Completer
        c.use_jedi = True

        def configure(suppression_config):
            cfg = Config()
            cfg.IPCompleter.suppress_competing_matchers = suppression_config
            c.update_config(cfg)

        def _():
            with provisionalcompleter():
                matches = [completion.text for completion in c.completions("dict.", 5)]
                self.assertIn("keys", matches)

        configure(False)
        _()

        configure(True)
        _()

        configure(None)
        _()

    def test_matcher_disabling(self):
        @completion_matcher(identifier="a_matcher")
        def a_matcher(text):
            return ["completion_a"]

        @completion_matcher(identifier="b_matcher")
        def b_matcher(text):
            return ["completion_b"]

        def _(expected):
            s, matches = c.complete("completion_")
            self.assertEqual(expected, matches)

        with custom_matchers([a_matcher, b_matcher]):
            ip = get_ipython()
            c = ip.Completer

            _(["completion_a", "completion_b"])

            cfg = Config()
            cfg.IPCompleter.disable_matchers = ["b_matcher"]
            c.update_config(cfg)

            _(["completion_a"])

            cfg.IPCompleter.disable_matchers = []
            c.update_config(cfg)

    def test_matcher_priority(self):
        @completion_matcher(identifier="a_matcher", priority=0, api_version=2)
        def a_matcher(text):
            return {"completions": [SimpleCompletion("completion_a")], "suppress": True}

        @completion_matcher(identifier="b_matcher", priority=2, api_version=2)
        def b_matcher(text):
            return {"completions": [SimpleCompletion("completion_b")], "suppress": True}

        def _(expected):
            s, matches = c.complete("completion_")
            self.assertEqual(expected, matches)

        with custom_matchers([a_matcher, b_matcher]):
            ip = get_ipython()
            c = ip.Completer

            _(["completion_b"])
            a_matcher.matcher_priority = 3
            _(["completion_a"])


@pytest.mark.parametrize(
    "input, expected",
    [
        ["1.234", "1.234"],
        # should match signed numbers
        ["+1", "+1"],
        ["-1", "-1"],
        ["-1.0", "-1.0"],
        ["-1.", "-1."],
        ["+1.", "+1."],
        [".1", ".1"],
        # should not match non-numbers
        ["1..", None],
        ["..", None],
        [".1.", None],
        # should match after comma
        [",1", "1"],
        [", 1", "1"],
        [", .1", ".1"],
        [", +.1", "+.1"],
        # should not match after trailing spaces
        [".1 ", None],
        # some complex cases
        ["0b_0011_1111_0100_1110", "0b_0011_1111_0100_1110"],
        ["0xdeadbeef", "0xdeadbeef"],
        ["0b_1110_0101", "0b_1110_0101"],
        # should not match if in an operation
        ["1 + 1", None],
        [", 1 + 1", None],
    ],
)
def test_match_numeric_literal_for_dict_key(input, expected):
    assert _match_number_in_dict_key_prefix(input) == expected
