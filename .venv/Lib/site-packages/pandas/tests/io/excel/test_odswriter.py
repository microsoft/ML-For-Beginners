import re

import pytest

import pandas._testing as tm

from pandas.io.excel import ExcelWriter

odf = pytest.importorskip("odf")

pytestmark = pytest.mark.parametrize("ext", [".ods"])


def test_write_append_mode_raises(ext):
    msg = "Append mode is not supported with odf!"

    with tm.ensure_clean(ext) as f:
        with pytest.raises(ValueError, match=msg):
            ExcelWriter(f, engine="odf", mode="a")


@pytest.mark.parametrize("engine_kwargs", [None, {"kwarg": 1}])
def test_engine_kwargs(ext, engine_kwargs):
    # GH 42286
    # GH 43445
    # test for error: OpenDocumentSpreadsheet does not accept any arguments
    with tm.ensure_clean(ext) as f:
        if engine_kwargs is not None:
            error = re.escape(
                "OpenDocumentSpreadsheet() got an unexpected keyword argument 'kwarg'"
            )
            with pytest.raises(
                TypeError,
                match=error,
            ):
                ExcelWriter(f, engine="odf", engine_kwargs=engine_kwargs)
        else:
            with ExcelWriter(f, engine="odf", engine_kwargs=engine_kwargs) as _:
                pass


def test_book_and_sheets_consistent(ext):
    # GH#45687 - Ensure sheets is updated if user modifies book
    with tm.ensure_clean(ext) as f:
        with ExcelWriter(f) as writer:
            assert writer.sheets == {}
            table = odf.table.Table(name="test_name")
            writer.book.spreadsheet.addElement(table)
            assert writer.sheets == {"test_name": table}
