import pytest
import sys


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only test.')
def test_pydevd_api_breakpoints(tmpdir):
    from _pydevd_bundle.pydevd_api import PyDevdAPI
    from pydevd import PyDB
    import pydevd_file_utils
    api = PyDevdAPI()

    py_db = PyDB(set_as_global=False)

    dira = tmpdir.join('DirA')
    dira.mkdir()

    f = dira.join('filE.py')
    f.write_text('''
a = 1
b = 2
c = 3
''', 'utf-8')
    filename = str(f)

    result = api.add_breakpoint(
        py_db, filename, breakpoint_type='python-line', breakpoint_id=0, line=1, condition=None, func_name='None',
        expression=None, suspend_policy="NONE", hit_condition='', is_logpoint=False)
    assert not result.error_code

    result = api.add_breakpoint(
        py_db, filename, breakpoint_type='python-line', breakpoint_id=1, line=2, condition=None, func_name='None',
        expression=None, suspend_policy="NONE", hit_condition='', is_logpoint=False)
    assert not result.error_code

    canonical_path = pydevd_file_utils.canonical_normalized_path(filename)

    assert len(py_db.breakpoints[canonical_path]) == 2
    assert len(py_db.file_to_id_to_line_breakpoint[canonical_path]) == 2

    filename_replaced = filename.replace('DirA', 'dira')
    api.remove_all_breakpoints(py_db, filename_replaced)
    assert not py_db.breakpoints
    assert not py_db.file_to_id_to_line_breakpoint
