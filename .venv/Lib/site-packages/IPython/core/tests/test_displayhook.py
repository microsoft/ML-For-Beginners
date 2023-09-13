import sys
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.core.displayhook import CapturingDisplayHook
from IPython.utils.capture import CapturedIO

def test_output_displayed():
    """Checking to make sure that output is displayed"""
  
    with AssertPrints('2'):
       ip.run_cell('1+1', store_history=True)
      
    with AssertPrints('2'):
        ip.run_cell('1+1 # comment with a semicolon;', store_history=True)

    with AssertPrints('2'):
        ip.run_cell('1+1\n#commented_out_function();', store_history=True)

      
def test_output_quiet():
    """Checking to make sure that output is quiet"""
  
    with AssertNotPrints('2'):
        ip.run_cell('1+1;', store_history=True)
      
    with AssertNotPrints('2'):
        ip.run_cell('1+1; # comment with a semicolon', store_history=True)

    with AssertNotPrints('2'):
        ip.run_cell('1+1;\n#commented_out_function()', store_history=True)

def test_underscore_no_overwrite_user():
    ip.run_cell('_ = 42', store_history=True)
    ip.run_cell('1+1', store_history=True)

    with AssertPrints('42'):
        ip.run_cell('print(_)', store_history=True)

    ip.run_cell('del _', store_history=True)
    ip.run_cell('6+6', store_history=True)
    with AssertPrints('12'):
        ip.run_cell('_', store_history=True)


def test_underscore_no_overwrite_builtins():
    ip.run_cell("import gettext ; gettext.install('foo')", store_history=True)
    ip.run_cell('3+3', store_history=True)

    with AssertPrints('gettext'):
        ip.run_cell('print(_)', store_history=True)

    ip.run_cell('_ = "userset"', store_history=True)

    with AssertPrints('userset'):
        ip.run_cell('print(_)', store_history=True)
    ip.run_cell('import builtins; del builtins._')


def test_interactivehooks_ast_modes():
    """
    Test that ast nodes can be triggered with different modes
    """
    saved_mode = ip.ast_node_interactivity
    ip.ast_node_interactivity = 'last_expr_or_assign'

    try:
        with AssertPrints('2'):
            ip.run_cell('a = 1+1', store_history=True)

        with AssertPrints('9'):
            ip.run_cell('b = 1+8 # comment with a semicolon;', store_history=False)

        with AssertPrints('7'):
            ip.run_cell('c = 1+6\n#commented_out_function();', store_history=True)

        ip.run_cell('d = 11', store_history=True)
        with AssertPrints('12'):
            ip.run_cell('d += 1', store_history=True)

        with AssertNotPrints('42'):
            ip.run_cell('(u,v) = (41+1, 43-1)')

    finally:
        ip.ast_node_interactivity = saved_mode

def test_interactivehooks_ast_modes_semi_suppress():
    """
    Test that ast nodes can be triggered with different modes and suppressed
    by semicolon
    """
    saved_mode = ip.ast_node_interactivity
    ip.ast_node_interactivity = 'last_expr_or_assign'

    try:
        with AssertNotPrints('2'):
            ip.run_cell('x = 1+1;', store_history=True)

        with AssertNotPrints('7'):
            ip.run_cell('y = 1+6; # comment with a semicolon', store_history=True)

        with AssertNotPrints('9'):
            ip.run_cell('z = 1+8;\n#commented_out_function()', store_history=True)

    finally:
        ip.ast_node_interactivity = saved_mode

def test_capture_display_hook_format():
    """Tests that the capture display hook conforms to the CapturedIO output format"""
    hook = CapturingDisplayHook(ip)
    hook({"foo": "bar"})
    captured = CapturedIO(sys.stdout, sys.stderr, hook.outputs)
    # Should not raise with RichOutput transformation error
    captured.outputs
