# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import json
import os
import warnings

from unittest import mock

import pytest

from IPython import display
from IPython.core.getipython import get_ipython
from IPython.utils.io import capture_output
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython import paths as ipath
from IPython.testing.tools import AssertNotPrints

import IPython.testing.decorators as dec

def test_image_size():
    """Simple test for display.Image(args, width=x,height=y)"""
    thisurl = 'http://www.google.fr/images/srpr/logo3w.png'
    img = display.Image(url=thisurl, width=200, height=200)
    assert '<img src="%s" width="200" height="200"/>' % (thisurl) == img._repr_html_()
    img = display.Image(url=thisurl, metadata={'width':200, 'height':200})
    assert '<img src="%s" width="200" height="200"/>' % (thisurl) == img._repr_html_()
    img = display.Image(url=thisurl, width=200)
    assert '<img src="%s" width="200"/>' % (thisurl) == img._repr_html_()
    img = display.Image(url=thisurl)
    assert '<img src="%s"/>' % (thisurl) == img._repr_html_()
    img = display.Image(url=thisurl, unconfined=True)
    assert '<img src="%s" class="unconfined"/>' % (thisurl) == img._repr_html_()


def test_image_mimes():
    fmt = get_ipython().display_formatter.format
    for format in display.Image._ACCEPTABLE_EMBEDDINGS:
        mime = display.Image._MIMETYPES[format]
        img = display.Image(b'garbage', format=format)
        data, metadata = fmt(img)
        assert sorted(data) == sorted([mime, "text/plain"])


def test_geojson():

    gj = display.GeoJSON(data={
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [-81.327, 296.038]
            },
            "properties": {
                "name": "Inca City"
            }
        },
        url_template="http://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/{basemap_id}/{z}/{x}/{y}.png",
        layer_options={
            "basemap_id": "celestia_mars-shaded-16k_global",
            "attribution": "Celestia/praesepe",
            "minZoom": 0,
            "maxZoom": 18,
        },
    )
    assert "<IPython.core.display.GeoJSON object>" == str(gj)


def test_retina_png():
    here = os.path.dirname(__file__)
    img = display.Image(os.path.join(here, "2x2.png"), retina=True)
    assert img.height == 1
    assert img.width == 1
    data, md = img._repr_png_()
    assert md["width"] == 1
    assert md["height"] == 1


def test_embed_svg_url():
    import gzip
    from io import BytesIO
    svg_data = b'<svg><circle x="0" y="0" r="1"/></svg>'
    url = 'http://test.com/circle.svg'

    gzip_svg = BytesIO()
    with gzip.open(gzip_svg, 'wb') as fp:
        fp.write(svg_data)
    gzip_svg = gzip_svg.getvalue()

    def mocked_urlopen(*args, **kwargs):
        class MockResponse:
            def __init__(self, svg):
                self._svg_data = svg
                self.headers = {'content-type': 'image/svg+xml'}

            def read(self):
                return self._svg_data

        if args[0] == url:
            return MockResponse(svg_data)
        elif args[0] == url + "z":
            ret = MockResponse(gzip_svg)
            ret.headers["content-encoding"] = "gzip"
            return ret
        return MockResponse(None)

    with mock.patch('urllib.request.urlopen', side_effect=mocked_urlopen):
        svg = display.SVG(url=url)
        assert svg._repr_svg_().startswith("<svg") is True
        svg = display.SVG(url=url + "z")
        assert svg._repr_svg_().startswith("<svg") is True


def test_retina_jpeg():
    here = os.path.dirname(__file__)
    img = display.Image(os.path.join(here, "2x2.jpg"), retina=True)
    assert img.height == 1
    assert img.width == 1
    data, md = img._repr_jpeg_()
    assert md["width"] == 1
    assert md["height"] == 1


def test_base64image():
    display.Image("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQMAAAAl21bKAAAAA1BMVEUAAACnej3aAAAAAWJLR0QAiAUdSAAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB94BCRQnOqNu0b4AAAAKSURBVAjXY2AAAAACAAHiIbwzAAAAAElFTkSuQmCC")

def test_image_filename_defaults():
    '''test format constraint, and validity of jpeg and png'''
    tpath = ipath.get_ipython_package_dir()
    pytest.raises(
        ValueError,
        display.Image,
        filename=os.path.join(tpath, "testing/tests/badformat.zip"),
        embed=True,
    )
    pytest.raises(ValueError, display.Image)
    pytest.raises(
        ValueError,
        display.Image,
        data="this is not an image",
        format="badformat",
        embed=True,
    )
    # check boths paths to allow packages to test at build and install time
    imgfile = os.path.join(tpath, 'core/tests/2x2.png')
    img = display.Image(filename=imgfile)
    assert "png" == img.format
    assert img._repr_png_() is not None
    img = display.Image(
        filename=os.path.join(tpath, "testing/tests/logo.jpg"), embed=False
    )
    assert "jpeg" == img.format
    assert img._repr_jpeg_() is None

def _get_inline_config():
    from matplotlib_inline.config import InlineBackend
    return InlineBackend.instance()


@dec.skip_without("matplotlib")
def test_set_matplotlib_close():
    cfg = _get_inline_config()
    cfg.close_figures = False
    with pytest.deprecated_call():
        display.set_matplotlib_close()
    assert cfg.close_figures
    with pytest.deprecated_call():
        display.set_matplotlib_close(False)
    assert not cfg.close_figures

_fmt_mime_map = {
    'png': 'image/png',
    'jpeg': 'image/jpeg',
    'pdf': 'application/pdf',
    'retina': 'image/png',
    'svg': 'image/svg+xml',
}

@dec.skip_without('matplotlib')
def test_set_matplotlib_formats():
    from matplotlib.figure import Figure
    formatters = get_ipython().display_formatter.formatters
    for formats in [
        ('png',),
        ('pdf', 'svg'),
        ('jpeg', 'retina', 'png'),
        (),
    ]:
        active_mimes = {_fmt_mime_map[fmt] for fmt in formats}
        with pytest.deprecated_call():
            display.set_matplotlib_formats(*formats)
        for mime, f in formatters.items():
            if mime in active_mimes:
                assert Figure in f
            else:
                assert Figure not in f


@dec.skip_without("matplotlib")
def test_set_matplotlib_formats_kwargs():
    from matplotlib.figure import Figure
    ip = get_ipython()
    cfg = _get_inline_config()
    cfg.print_figure_kwargs.update(dict(foo='bar'))
    kwargs = dict(dpi=150)
    with pytest.deprecated_call():
        display.set_matplotlib_formats("png", **kwargs)
    formatter = ip.display_formatter.formatters["image/png"]
    f = formatter.lookup_by_type(Figure)
    formatter_kwargs = f.keywords
    expected = kwargs
    expected["base64"] = True
    expected["fmt"] = "png"
    expected.update(cfg.print_figure_kwargs)
    assert formatter_kwargs == expected

def test_display_available():
    """
    Test that display is available without import

    We don't really care if it's in builtin or anything else, but it should
    always be available.
    """
    ip = get_ipython()
    with AssertNotPrints('NameError'):
        ip.run_cell('display')
    try:
        ip.run_cell('del display')
    except NameError:
        pass # it's ok, it might be in builtins
    # even if deleted it should be back
    with AssertNotPrints('NameError'):
        ip.run_cell('display')

def test_textdisplayobj_pretty_repr():
    p = display.Pretty("This is a simple test")
    assert repr(p) == "<IPython.core.display.Pretty object>"
    assert p.data == "This is a simple test"

    p._show_mem_addr = True
    assert repr(p) == object.__repr__(p)


def test_displayobject_repr():
    h = display.HTML("<br />")
    assert repr(h) == "<IPython.core.display.HTML object>"
    h._show_mem_addr = True
    assert repr(h) == object.__repr__(h)
    h._show_mem_addr = False
    assert repr(h) == "<IPython.core.display.HTML object>"

    j = display.Javascript("")
    assert repr(j) == "<IPython.core.display.Javascript object>"
    j._show_mem_addr = True
    assert repr(j) == object.__repr__(j)
    j._show_mem_addr = False
    assert repr(j) == "<IPython.core.display.Javascript object>"

@mock.patch('warnings.warn')
def test_encourage_iframe_over_html(m_warn):
    display.HTML()
    m_warn.assert_not_called()

    display.HTML('<br />')
    m_warn.assert_not_called()

    display.HTML('<html><p>Lots of content here</p><iframe src="http://a.com"></iframe>')
    m_warn.assert_not_called()

    display.HTML('<iframe src="http://a.com"></iframe>')
    m_warn.assert_called_with('Consider using IPython.display.IFrame instead')

    m_warn.reset_mock()
    display.HTML('<IFRAME SRC="http://a.com"></IFRAME>')
    m_warn.assert_called_with('Consider using IPython.display.IFrame instead')

def test_progress():
    p = display.ProgressBar(10)
    assert "0/10" in repr(p)
    p.html_width = "100%"
    p.progress = 5
    assert (
        p._repr_html_() == "<progress style='width:100%' max='10' value='5'></progress>"
    )


def test_progress_iter():
    with capture_output(display=False) as captured:
        for i in display.ProgressBar(5):
            out = captured.stdout
            assert "{0}/5".format(i) in out
    out = captured.stdout
    assert "5/5" in out


def test_json():
    d = {'a': 5}
    lis = [d]
    metadata = [
        {'expanded': False, 'root': 'root'},
        {'expanded': True,  'root': 'root'},
        {'expanded': False, 'root': 'custom'},
        {'expanded': True,  'root': 'custom'},
    ]
    json_objs = [
        display.JSON(d),
        display.JSON(d, expanded=True),
        display.JSON(d, root='custom'),
        display.JSON(d, expanded=True, root='custom'),
    ]
    for j, md in zip(json_objs, metadata):
        assert j._repr_json_() == (d, md)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        j = display.JSON(json.dumps(d))
        assert len(w) == 1
        assert j._repr_json_() == (d, metadata[0])

    json_objs = [
        display.JSON(lis),
        display.JSON(lis, expanded=True),
        display.JSON(lis, root='custom'),
        display.JSON(lis, expanded=True, root='custom'),
    ]
    for j, md in zip(json_objs, metadata):
        assert j._repr_json_() == (lis, md)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        j = display.JSON(json.dumps(lis))
        assert len(w) == 1
        assert j._repr_json_() == (lis, metadata[0])


def test_video_embedding():
    """use a tempfile, with dummy-data, to ensure that video embedding doesn't crash"""
    v = display.Video("http://ignored")
    assert not v.embed
    html = v._repr_html_()
    assert 'src="data:' not in html
    assert 'src="http://ignored"' in html

    with pytest.raises(ValueError):
        v = display.Video(b'abc')

    with NamedFileInTemporaryDirectory('test.mp4') as f:
        f.write(b'abc')
        f.close()

        v = display.Video(f.name)
        assert not v.embed
        html = v._repr_html_()
        assert 'src="data:' not in html

        v = display.Video(f.name, embed=True)
        html = v._repr_html_()
        assert 'src="data:video/mp4;base64,YWJj"' in html

        v = display.Video(f.name, embed=True, mimetype='video/other')
        html = v._repr_html_()
        assert 'src="data:video/other;base64,YWJj"' in html

        v = display.Video(b'abc', embed=True, mimetype='video/mp4')
        html = v._repr_html_()
        assert 'src="data:video/mp4;base64,YWJj"' in html

        v = display.Video(u'YWJj', embed=True, mimetype='video/xyz')
        html = v._repr_html_()
        assert 'src="data:video/xyz;base64,YWJj"' in html

def test_html_metadata():
    s = "<h1>Test</h1>"
    h = display.HTML(s, metadata={"isolated": True})
    assert h._repr_html_() == (s, {"isolated": True})


def test_display_id():
    ip = get_ipython()
    with mock.patch.object(ip.display_pub, 'publish') as pub:
        handle = display.display('x')
        assert handle is None
        handle = display.display('y', display_id='secret')
        assert isinstance(handle, display.DisplayHandle)
        handle2 = display.display('z', display_id=True)
        assert isinstance(handle2, display.DisplayHandle)
    assert handle.display_id != handle2.display_id

    assert pub.call_count == 3
    args, kwargs = pub.call_args_list[0]
    assert args == ()
    assert kwargs == {
        'data': {
            'text/plain': repr('x')
        },
        'metadata': {},
    }
    args, kwargs = pub.call_args_list[1]
    assert args == ()
    assert kwargs == {
        'data': {
            'text/plain': repr('y')
        },
        'metadata': {},
        'transient': {
            'display_id': handle.display_id,
        },
    }
    args, kwargs = pub.call_args_list[2]
    assert args == ()
    assert kwargs == {
        'data': {
            'text/plain': repr('z')
        },
        'metadata': {},
        'transient': {
            'display_id': handle2.display_id,
        },
    }


def test_update_display():
    ip = get_ipython()
    with mock.patch.object(ip.display_pub, 'publish') as pub:
        with pytest.raises(TypeError):
            display.update_display('x')
        display.update_display('x', display_id='1')
        display.update_display('y', display_id='2')
    args, kwargs = pub.call_args_list[0]
    assert args == ()
    assert kwargs == {
        'data': {
            'text/plain': repr('x')
        },
        'metadata': {},
        'transient': {
            'display_id': '1',
        },
        'update': True,
    }
    args, kwargs = pub.call_args_list[1]
    assert args == ()
    assert kwargs == {
        'data': {
            'text/plain': repr('y')
        },
        'metadata': {},
        'transient': {
            'display_id': '2',
        },
        'update': True,
    }


def test_display_handle():
    ip = get_ipython()
    handle = display.DisplayHandle()
    assert isinstance(handle.display_id, str)
    handle = display.DisplayHandle("my-id")
    assert handle.display_id == "my-id"
    with mock.patch.object(ip.display_pub, "publish") as pub:
        handle.display("x")
        handle.update("y")

    args, kwargs = pub.call_args_list[0]
    assert args == ()
    assert kwargs == {
        'data': {
            'text/plain': repr('x')
        },
        'metadata': {},
        'transient': {
            'display_id': handle.display_id,
        }
    }
    args, kwargs = pub.call_args_list[1]
    assert args == ()
    assert kwargs == {
        'data': {
            'text/plain': repr('y')
        },
        'metadata': {},
        'transient': {
            'display_id': handle.display_id,
        },
        'update': True,
    }


def test_image_alt_tag():
    """Simple test for display.Image(args, alt=x,)"""
    thisurl = "http://example.com/image.png"
    img = display.Image(url=thisurl, alt="an image")
    assert '<img src="%s" alt="an image"/>' % (thisurl) == img._repr_html_()
    img = display.Image(url=thisurl, unconfined=True, alt="an image")
    assert (
        '<img src="%s" class="unconfined" alt="an image"/>' % (thisurl)
        == img._repr_html_()
    )
    img = display.Image(url=thisurl, alt='>"& <')
    assert '<img src="%s" alt="&gt;&quot;&amp; &lt;"/>' % (thisurl) == img._repr_html_()

    img = display.Image(url=thisurl, metadata={"alt": "an image"})
    assert img.alt == "an image"
    here = os.path.dirname(__file__)
    img = display.Image(os.path.join(here, "2x2.png"), alt="an image")
    assert img.alt == "an image"
    _, md = img._repr_png_()
    assert md["alt"] == "an image"


def test_image_bad_filename_raises_proper_exception():
    with pytest.raises(FileNotFoundError):
        display.Image("/this/file/does/not/exist/")._repr_png_()
