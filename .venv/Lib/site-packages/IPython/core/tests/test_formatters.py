"""Tests for the Formatters."""

from math import pi

try:
    import numpy
except:
    numpy = None
import pytest

from IPython import get_ipython
from traitlets.config import Config
from IPython.core.formatters import (
    PlainTextFormatter, HTMLFormatter, PDFFormatter, _mod_name_key,
    DisplayFormatter, JSONFormatter,
)
from IPython.utils.io import capture_output

class A(object):
    def __repr__(self):
        return 'A()'

class B(A):
    def __repr__(self):
        return 'B()'

class C:
    pass

class BadRepr(object):
    def __repr__(self):
        raise ValueError("bad repr")

class BadPretty(object):
    _repr_pretty_ = None

class GoodPretty(object):
    def _repr_pretty_(self, pp, cycle):
        pp.text('foo')

    def __repr__(self):
        return 'GoodPretty()'

def foo_printer(obj, pp, cycle):
    pp.text('foo')

def test_pretty():
    f = PlainTextFormatter()
    f.for_type(A, foo_printer)
    assert f(A()) == "foo"
    assert f(B()) == "B()"
    assert f(GoodPretty()) == "foo"
    # Just don't raise an exception for the following:
    f(BadPretty())

    f.pprint = False
    assert f(A()) == "A()"
    assert f(B()) == "B()"
    assert f(GoodPretty()) == "GoodPretty()"


def test_deferred():
    f = PlainTextFormatter()

def test_precision():
    """test various values for float_precision."""
    f = PlainTextFormatter()
    assert f(pi) == repr(pi)
    f.float_precision = 0
    if numpy:
        po = numpy.get_printoptions()
        assert po["precision"] == 0
    assert f(pi) == "3"
    f.float_precision = 2
    if numpy:
        po = numpy.get_printoptions()
        assert po["precision"] == 2
    assert f(pi) == "3.14"
    f.float_precision = "%g"
    if numpy:
        po = numpy.get_printoptions()
        assert po["precision"] == 2
    assert f(pi) == "3.14159"
    f.float_precision = "%e"
    assert f(pi) == "3.141593e+00"
    f.float_precision = ""
    if numpy:
        po = numpy.get_printoptions()
        assert po["precision"] == 8
    assert f(pi) == repr(pi)


def test_bad_precision():
    """test various invalid values for float_precision."""
    f = PlainTextFormatter()
    def set_fp(p):
        f.float_precision = p

    pytest.raises(ValueError, set_fp, "%")
    pytest.raises(ValueError, set_fp, "%.3f%i")
    pytest.raises(ValueError, set_fp, "foo")
    pytest.raises(ValueError, set_fp, -1)

def test_for_type():
    f = PlainTextFormatter()
    
    # initial return, None
    assert f.for_type(C, foo_printer) is None
    # no func queries
    assert f.for_type(C) is foo_printer
    # shouldn't change anything
    assert f.for_type(C) is foo_printer
    # None should do the same
    assert f.for_type(C, None) is foo_printer
    assert f.for_type(C, None) is foo_printer

def test_for_type_string():
    f = PlainTextFormatter()
    
    type_str = '%s.%s' % (C.__module__, 'C')
    
    # initial return, None
    assert f.for_type(type_str, foo_printer) is None
    # no func queries
    assert f.for_type(type_str) is foo_printer
    assert _mod_name_key(C) in f.deferred_printers
    assert f.for_type(C) is foo_printer
    assert _mod_name_key(C) not in f.deferred_printers
    assert C in f.type_printers

def test_for_type_by_name():
    f = PlainTextFormatter()
    
    mod = C.__module__
    
    # initial return, None
    assert f.for_type_by_name(mod, "C", foo_printer) is None
    # no func queries
    assert f.for_type_by_name(mod, "C") is foo_printer
    # shouldn't change anything
    assert f.for_type_by_name(mod, "C") is foo_printer
    # None should do the same
    assert f.for_type_by_name(mod, "C", None) is foo_printer
    assert f.for_type_by_name(mod, "C", None) is foo_printer


def test_lookup():
    f = PlainTextFormatter()
    
    f.for_type(C, foo_printer)
    assert f.lookup(C()) is foo_printer
    with pytest.raises(KeyError):
        f.lookup(A())

def test_lookup_string():
    f = PlainTextFormatter()
    type_str = '%s.%s' % (C.__module__, 'C')
    
    f.for_type(type_str, foo_printer)
    assert f.lookup(C()) is foo_printer
    # should move from deferred to imported dict
    assert _mod_name_key(C) not in f.deferred_printers
    assert C in f.type_printers

def test_lookup_by_type():
    f = PlainTextFormatter()
    f.for_type(C, foo_printer)
    assert f.lookup_by_type(C) is foo_printer
    with pytest.raises(KeyError):
        f.lookup_by_type(A)

def test_lookup_by_type_string():
    f = PlainTextFormatter()
    type_str = '%s.%s' % (C.__module__, 'C')
    f.for_type(type_str, foo_printer)
    
    # verify insertion
    assert _mod_name_key(C) in f.deferred_printers
    assert C not in f.type_printers
    
    assert f.lookup_by_type(type_str) is foo_printer
    # lookup by string doesn't cause import
    assert _mod_name_key(C) in f.deferred_printers
    assert C not in f.type_printers
    
    assert f.lookup_by_type(C) is foo_printer
    # should move from deferred to imported dict
    assert _mod_name_key(C) not in f.deferred_printers
    assert C in f.type_printers

def test_in_formatter():
    f = PlainTextFormatter()
    f.for_type(C, foo_printer)
    type_str = '%s.%s' % (C.__module__, 'C')
    assert C in f
    assert type_str in f

def test_string_in_formatter():
    f = PlainTextFormatter()
    type_str = '%s.%s' % (C.__module__, 'C')
    f.for_type(type_str, foo_printer)
    assert type_str in f
    assert C in f

def test_pop():
    f = PlainTextFormatter()
    f.for_type(C, foo_printer)
    assert f.lookup_by_type(C) is foo_printer
    assert f.pop(C, None) is foo_printer
    f.for_type(C, foo_printer)
    assert f.pop(C) is foo_printer
    with pytest.raises(KeyError):
        f.lookup_by_type(C)
    with pytest.raises(KeyError):
        f.pop(C)
    with pytest.raises(KeyError):
        f.pop(A)
    assert f.pop(A, None) is None

def test_pop_string():
    f = PlainTextFormatter()
    type_str = '%s.%s' % (C.__module__, 'C')
    
    with pytest.raises(KeyError):
        f.pop(type_str)
    
    f.for_type(type_str, foo_printer)
    f.pop(type_str)
    with pytest.raises(KeyError):
        f.lookup_by_type(C)
    with pytest.raises(KeyError):
        f.pop(type_str)

    f.for_type(C, foo_printer)
    assert f.pop(type_str, None) is foo_printer
    with pytest.raises(KeyError):
        f.lookup_by_type(C)
    with pytest.raises(KeyError):
        f.pop(type_str)
    assert f.pop(type_str, None) is None
    

def test_error_method():
    f = HTMLFormatter()
    class BadHTML(object):
        def _repr_html_(self):
            raise ValueError("Bad HTML")
    bad = BadHTML()
    with capture_output() as captured:
        result = f(bad)
    assert result is None
    assert "Traceback" in captured.stdout
    assert "Bad HTML" in captured.stdout
    assert "_repr_html_" in captured.stdout

def test_nowarn_notimplemented():
    f = HTMLFormatter()
    class HTMLNotImplemented(object):
        def _repr_html_(self):
            raise NotImplementedError
    h = HTMLNotImplemented()
    with capture_output() as captured:
        result = f(h)
    assert result is None
    assert "" == captured.stderr
    assert "" == captured.stdout


def test_warn_error_for_type():
    f = HTMLFormatter()
    f.for_type(int, lambda i: name_error)
    with capture_output() as captured:
        result = f(5)
    assert result is None
    assert "Traceback" in captured.stdout
    assert "NameError" in captured.stdout
    assert "name_error" in captured.stdout

def test_error_pretty_method():
    f = PlainTextFormatter()
    class BadPretty(object):
        def _repr_pretty_(self):
            return "hello"
    bad = BadPretty()
    with capture_output() as captured:
        result = f(bad)
    assert result is None
    assert "Traceback" in captured.stdout
    assert "_repr_pretty_" in captured.stdout
    assert "given" in captured.stdout
    assert "argument" in captured.stdout


def test_bad_repr_traceback():
    f = PlainTextFormatter()
    bad = BadRepr()
    with capture_output() as captured:
        result = f(bad)
    # catches error, returns None
    assert result is None
    assert "Traceback" in captured.stdout
    assert "__repr__" in captured.stdout
    assert "ValueError" in captured.stdout


class MakePDF(object):
    def _repr_pdf_(self):
        return 'PDF'

def test_pdf_formatter():
    pdf = MakePDF()
    f = PDFFormatter()
    assert f(pdf) == "PDF"


def test_print_method_bound():
    f = HTMLFormatter()
    class MyHTML(object):
        def _repr_html_(self):
            return "hello"
    with capture_output() as captured:
        result = f(MyHTML)
    assert result is None
    assert "FormatterWarning" not in captured.stderr

    with capture_output() as captured:
        result = f(MyHTML())
    assert result == "hello"
    assert captured.stderr == ""


def test_print_method_weird():

    class TextMagicHat(object):
        def __getattr__(self, key):
            return key

    f = HTMLFormatter()

    text_hat = TextMagicHat()
    assert text_hat._repr_html_ == "_repr_html_"
    with capture_output() as captured:
        result = f(text_hat)
    
    assert result is None
    assert "FormatterWarning" not in captured.stderr

    class CallableMagicHat(object):
        def __getattr__(self, key):
            return lambda : key
    
    call_hat = CallableMagicHat()
    with capture_output() as captured:
        result = f(call_hat)

    assert result is None

    class BadReprArgs(object):
        def _repr_html_(self, extra, args):
            return "html"
    
    bad = BadReprArgs()
    with capture_output() as captured:
        result = f(bad)
    
    assert result is None
    assert "FormatterWarning" not in captured.stderr


def test_format_config():
    """config objects don't pretend to support fancy reprs with lazy attrs"""
    f = HTMLFormatter()
    cfg = Config()
    with capture_output() as captured:
        result = f(cfg)
    assert result is None
    assert captured.stderr == ""

    with capture_output() as captured:
        result = f(Config)
    assert result is None
    assert captured.stderr == ""


def test_pretty_max_seq_length():
    f = PlainTextFormatter(max_seq_length=1)
    lis = list(range(3))
    text = f(lis)
    assert text == "[0, ...]"
    f.max_seq_length = 0
    text = f(lis)
    assert text == "[0, 1, 2]"
    text = f(list(range(1024)))
    lines = text.splitlines()
    assert len(lines) == 1024


def test_ipython_display_formatter():
    """Objects with _ipython_display_ defined bypass other formatters"""
    f = get_ipython().display_formatter
    catcher = []
    class SelfDisplaying(object):
        def _ipython_display_(self):
            catcher.append(self)

    class NotSelfDisplaying(object):
        def __repr__(self):
            return "NotSelfDisplaying"
        
        def _ipython_display_(self):
            raise NotImplementedError
    
    save_enabled = f.ipython_display_formatter.enabled
    f.ipython_display_formatter.enabled = True
    
    yes = SelfDisplaying()
    no = NotSelfDisplaying()

    d, md = f.format(no)
    assert d == {"text/plain": repr(no)}
    assert md == {}
    assert catcher == []

    d, md = f.format(yes)
    assert d == {}
    assert md == {}
    assert catcher == [yes]

    f.ipython_display_formatter.enabled = save_enabled


def test_repr_mime():
    class HasReprMime(object):
        def _repr_mimebundle_(self, include=None, exclude=None):
            return {
                'application/json+test.v2': {
                    'x': 'y'
                },
                'plain/text' : '<HasReprMime>',
                'image/png' : 'i-overwrite'
            }

        def _repr_png_(self):
            return 'should-be-overwritten'
        def _repr_html_(self):
            return '<b>hi!</b>'
    
    f = get_ipython().display_formatter
    html_f = f.formatters['text/html']
    save_enabled = html_f.enabled
    html_f.enabled = True
    obj = HasReprMime()
    d, md = f.format(obj)
    html_f.enabled = save_enabled

    assert sorted(d) == [
        "application/json+test.v2",
        "image/png",
        "plain/text",
        "text/html",
        "text/plain",
    ]
    assert md == {}

    d, md = f.format(obj, include={"image/png"})
    assert list(d.keys()) == [
        "image/png"
    ], "Include should filter out even things from repr_mimebundle"

    assert d["image/png"] == "i-overwrite", "_repr_mimebundle_ take precedence"


def test_pass_correct_include_exclude():
    class Tester(object):

        def __init__(self, include=None, exclude=None):
            self.include = include
            self.exclude = exclude

        def _repr_mimebundle_(self, include, exclude, **kwargs):
            if include and (include != self.include):
                raise ValueError('include got modified: display() may be broken.')
            if exclude and (exclude != self.exclude):
                raise ValueError('exclude got modified: display() may be broken.')

            return None

    include = {'a', 'b', 'c'}
    exclude = {'c', 'e' , 'f'}

    f = get_ipython().display_formatter
    f.format(Tester(include=include, exclude=exclude), include=include, exclude=exclude)
    f.format(Tester(exclude=exclude), exclude=exclude)
    f.format(Tester(include=include), include=include)


def test_repr_mime_meta():
    class HasReprMimeMeta(object):
        def _repr_mimebundle_(self, include=None, exclude=None):
            data = {
                'image/png': 'base64-image-data',
            }
            metadata = {
                'image/png': {
                    'width': 5,
                    'height': 10,
                }
            }
            return (data, metadata)
    
    f = get_ipython().display_formatter
    obj = HasReprMimeMeta()
    d, md = f.format(obj)
    assert sorted(d) == ["image/png", "text/plain"]
    assert md == {
        "image/png": {
            "width": 5,
            "height": 10,
        }
    }


def test_repr_mime_failure():
    class BadReprMime(object):
        def _repr_mimebundle_(self, include=None, exclude=None):
            raise RuntimeError

    f = get_ipython().display_formatter
    obj = BadReprMime()
    d, md = f.format(obj)
    assert "text/plain" in d
