from IPython.core.error import TryNext
from IPython.lib.clipboard import ClipboardEmpty
from IPython.testing.decorators import skip_if_no_x11


@skip_if_no_x11
def test_clipboard_get():
    # Smoketest for clipboard access - we can't easily guarantee that the
    # clipboard is accessible and has something on it, but this tries to
    # exercise the relevant code anyway.
    try:
        a = get_ipython().hooks.clipboard_get()
    except ClipboardEmpty:
        # Nothing in clipboard to get
        pass
    except TryNext:
        # No clipboard access API available
        pass
    else:
        assert isinstance(a, str)
