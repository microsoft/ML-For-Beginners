"""Tests for pylab tools module.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


from binascii import a2b_base64
from io import BytesIO

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use('Agg')
from matplotlib.figure import Figure

from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import numpy as np

from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.display import _PNG, _JPEG
from .. import pylabtools as pt

from IPython.testing import decorators as dec


def test_figure_to_svg():
    # simple empty-figure test
    fig = plt.figure()
    assert pt.print_figure(fig, "svg") is None

    plt.close('all')

    # simple check for at least svg-looking output
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([1,2,3])
    plt.draw()
    svg = pt.print_figure(fig, "svg")[:100].lower()
    assert "doctype svg" in svg


def _check_pil_jpeg_bytes():
    """Skip if PIL can't write JPEGs to BytesIO objects"""
    # PIL's JPEG plugin can't write to BytesIO objects
    # Pillow fixes this
    from PIL import Image
    buf = BytesIO()
    img = Image.new("RGB", (4,4))
    try:
        img.save(buf, 'jpeg')
    except Exception as e:
        ename = e.__class__.__name__
        raise pytest.skip("PIL can't write JPEG to BytesIO: %s: %s" % (ename, e)) from e

@dec.skip_without("PIL.Image")
def test_figure_to_jpeg():
    _check_pil_jpeg_bytes()
    # simple check for at least jpeg-looking output
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([1,2,3])
    plt.draw()
    jpeg = pt.print_figure(fig, 'jpeg', pil_kwargs={'optimize': 50})[:100].lower()
    assert jpeg.startswith(_JPEG)

def test_retina_figure():
    # simple empty-figure test
    fig = plt.figure()
    assert pt.retina_figure(fig) == None
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([1,2,3])
    plt.draw()
    png, md = pt.retina_figure(fig)
    assert png.startswith(_PNG)
    assert "width" in md
    assert "height" in md


_fmt_mime_map = {
    'png': 'image/png',
    'jpeg': 'image/jpeg',
    'pdf': 'application/pdf',
    'retina': 'image/png',
    'svg': 'image/svg+xml',
}

def test_select_figure_formats_str():
    ip = get_ipython()
    for fmt, active_mime in _fmt_mime_map.items():
        pt.select_figure_formats(ip, fmt)
        for mime, f in ip.display_formatter.formatters.items():
            if mime == active_mime:
                assert Figure in f
            else:
                assert Figure not in f

def test_select_figure_formats_kwargs():
    ip = get_ipython()
    kwargs = dict(bbox_inches="tight")
    pt.select_figure_formats(ip, "png", **kwargs)
    formatter = ip.display_formatter.formatters["image/png"]
    f = formatter.lookup_by_type(Figure)
    cell = f.keywords
    expected = kwargs
    expected["base64"] = True
    expected["fmt"] = "png"
    assert cell == expected

    # check that the formatter doesn't raise
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([1,2,3])
    plt.draw()
    formatter.enabled = True
    png = formatter(fig)
    assert isinstance(png, str)
    png_bytes = a2b_base64(png)
    assert png_bytes.startswith(_PNG)

def test_select_figure_formats_set():
    ip = get_ipython()
    for fmts in [
        {'png', 'svg'},
        ['png'],
        ('jpeg', 'pdf', 'retina'),
        {'svg'},
    ]:
        active_mimes = {_fmt_mime_map[fmt] for fmt in fmts}
        pt.select_figure_formats(ip, fmts)
        for mime, f in ip.display_formatter.formatters.items():
            if mime in active_mimes:
                assert Figure in f
            else:
                assert Figure not in f

def test_select_figure_formats_bad():
    ip = get_ipython()
    with pytest.raises(ValueError):
        pt.select_figure_formats(ip, 'foo')
    with pytest.raises(ValueError):
        pt.select_figure_formats(ip, {'png', 'foo'})
    with pytest.raises(ValueError):
        pt.select_figure_formats(ip, ['retina', 'pdf', 'bar', 'bad'])

def test_import_pylab():
    ns = {}
    pt.import_pylab(ns, import_all=False)
    assert "plt" in ns
    assert ns["np"] == np


class TestPylabSwitch(object):
    class Shell(InteractiveShell):
        def init_history(self):
            """Sets up the command history, and starts regular autosaves."""
            self.config.HistoryManager.hist_file = ":memory:"
            super().init_history()

        def enable_gui(self, gui):
            pass

    def setup(self):
        import matplotlib
        def act_mpl(backend):
            matplotlib.rcParams['backend'] = backend

        # Save rcParams since they get modified
        self._saved_rcParams = matplotlib.rcParams
        self._saved_rcParamsOrig = matplotlib.rcParamsOrig
        matplotlib.rcParams = dict(backend="QtAgg")
        matplotlib.rcParamsOrig = dict(backend="QtAgg")

        # Mock out functions
        self._save_am = pt.activate_matplotlib
        pt.activate_matplotlib = act_mpl
        self._save_ip = pt.import_pylab
        pt.import_pylab = lambda *a,**kw:None
        self._save_cis = backend_inline.configure_inline_support
        backend_inline.configure_inline_support = lambda *a, **kw: None

    def teardown(self):
        pt.activate_matplotlib = self._save_am
        pt.import_pylab = self._save_ip
        backend_inline.configure_inline_support = self._save_cis
        import matplotlib
        matplotlib.rcParams = self._saved_rcParams
        matplotlib.rcParamsOrig = self._saved_rcParamsOrig

    def test_qt(self):
        s = self.Shell()
        gui, backend = s.enable_matplotlib(None)
        assert gui == "qt"
        assert s.pylab_gui_select == "qt"

        gui, backend = s.enable_matplotlib("inline")
        assert gui == "inline"
        assert s.pylab_gui_select == "qt"

        gui, backend = s.enable_matplotlib("qt")
        assert gui == "qt"
        assert s.pylab_gui_select == "qt"

        gui, backend = s.enable_matplotlib("inline")
        assert gui == "inline"
        assert s.pylab_gui_select == "qt"

        gui, backend = s.enable_matplotlib()
        assert gui == "qt"
        assert s.pylab_gui_select == "qt"

    def test_inline(self):
        s = self.Shell()
        gui, backend = s.enable_matplotlib("inline")
        assert gui == "inline"
        assert s.pylab_gui_select == None

        gui, backend = s.enable_matplotlib("inline")
        assert gui == "inline"
        assert s.pylab_gui_select == None

        gui, backend = s.enable_matplotlib("qt")
        assert gui == "qt"
        assert s.pylab_gui_select == "qt"

    def test_inline_twice(self):
        "Using '%matplotlib inline' twice should not reset formatters"

        ip = self.Shell()
        gui, backend = ip.enable_matplotlib("inline")
        assert gui == "inline"

        fmts =  {'png'}
        active_mimes = {_fmt_mime_map[fmt] for fmt in fmts}
        pt.select_figure_formats(ip, fmts)

        gui, backend = ip.enable_matplotlib("inline")
        assert gui == "inline"

        for mime, f in ip.display_formatter.formatters.items():
            if mime in active_mimes:
                assert Figure in f
            else:
                assert Figure not in f

    def test_qt_gtk(self):
        s = self.Shell()
        gui, backend = s.enable_matplotlib("qt")
        assert gui == "qt"
        assert s.pylab_gui_select == "qt"

        gui, backend = s.enable_matplotlib("gtk")
        assert gui == "qt"
        assert s.pylab_gui_select == "qt"


def test_no_gui_backends():
    for k in ['agg', 'svg', 'pdf', 'ps']:
        assert k not in pt.backend2gui


def test_figure_no_canvas():
    fig = Figure()
    fig.canvas = None
    pt.print_figure(fig)
