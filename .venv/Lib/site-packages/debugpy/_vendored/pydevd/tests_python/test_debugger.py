# coding: utf-8
'''
    The idea is that we record the commands sent to the debugger and reproduce them from this script
    (so, this works as the client, which spawns the debugger as a separate process and communicates
    to it as if it was run from the outside)

    Note that it's a python script but it'll spawn a process to run as jython, ironpython and as python.
'''
import time

import pytest

from tests_python import debugger_unittest
from tests_python.debugger_unittest import (CMD_SET_PROPERTY_TRACE, REASON_CAUGHT_EXCEPTION,
    REASON_UNCAUGHT_EXCEPTION, REASON_STOP_ON_BREAKPOINT, REASON_THREAD_SUSPEND, overrides, CMD_THREAD_CREATE,
    CMD_GET_THREAD_STACK, REASON_STEP_INTO_MY_CODE, CMD_GET_EXCEPTION_DETAILS, IS_IRONPYTHON, IS_JYTHON, IS_CPYTHON,
    IS_APPVEYOR, wait_for_condition, CMD_GET_FRAME, CMD_GET_BREAKPOINT_EXCEPTION,
    CMD_THREAD_SUSPEND, CMD_STEP_OVER, REASON_STEP_OVER, CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION,
    CMD_THREAD_RESUME_SINGLE_NOTIFICATION, REASON_STEP_RETURN, REASON_STEP_RETURN_MY_CODE,
    REASON_STEP_OVER_MY_CODE, REASON_STEP_INTO, CMD_THREAD_KILL, IS_PYPY, REASON_STOP_ON_START,
    CMD_SMART_STEP_INTO, CMD_GET_VARIABLE)
from _pydevd_bundle.pydevd_constants import IS_WINDOWS, IS_PY38_OR_GREATER, \
    IS_MAC
from _pydevd_bundle.pydevd_comm_constants import CMD_RELOAD_CODE, CMD_INPUT_REQUESTED, \
    CMD_RUN_CUSTOM_OPERATION
import json
import pydevd_file_utils
import subprocess
import threading
from _pydev_bundle import pydev_log
from urllib.parse import unquote, unquote_plus

from tests_python.debug_constants import *  # noqa

pytest_plugins = [
    str('tests_python.debugger_fixtures'),
]

builtin_qualifier = "builtins"


@pytest.mark.skipif(not IS_CPYTHON, reason='Test needs gc.get_referrers/reference counting to really check anything.')
def test_case_referrers(case_setup):
    with case_setup.test_file('_debugger_case1.py') as writer:
        writer.log.append('writing add breakpoint')
        writer.write_add_breakpoint(6, 'set_up')

        writer.log.append('making initial run')
        writer.write_make_initial_run()

        writer.log.append('waiting for breakpoint hit')
        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.log.append('get frame')
        writer.write_get_frame(thread_id, frame_id)

        writer.log.append('step over')
        writer.write_step_over(thread_id)

        writer.log.append('get frame')
        writer.write_get_frame(thread_id, frame_id)

        writer.log.append('run thread')
        writer.write_run_thread(thread_id)

        writer.log.append('asserting')
        try:
            assert 13 == writer._sequence, 'Expected 13. Had: %s' % writer._sequence
        except:
            writer.log.append('assert failed!')
            raise
        writer.log.append('asserted')

        writer.finished_ok = True


def test_case_2(case_setup):
    with case_setup.test_file('_debugger_case2.py') as writer:
        writer.write_add_breakpoint(3, 'Call4')  # seq = 3
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_get_frame(thread_id, frame_id)  # Note: write get frame but not waiting for it to be gotten.

        writer.write_add_breakpoint(14, 'Call2')

        writer.write_run_thread(thread_id)

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_get_frame(thread_id, frame_id)  # Note: write get frame but not waiting for it to be gotten.

        writer.write_run_thread(thread_id)

        writer.log.append('Checking sequence. Found: %s' % (writer._sequence))
        assert 15 == writer._sequence, 'Expected 15. Had: %s' % writer._sequence

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.parametrize(
    'skip_suspend_on_breakpoint_exception, skip_print_breakpoint_exception',
    (
        [['NameError'], []],
        [['NameError'], ['NameError']],
        [[], []],  # Empty means it'll suspend/print in any exception
        [[], ['NameError']],
        [['ValueError'], ['Exception']],
        [['Exception'], ['ValueError']],  # ValueError will also suspend/print since we're dealing with a NameError
    )
)
def test_case_breakpoint_condition_exc(case_setup, skip_suspend_on_breakpoint_exception, skip_print_breakpoint_exception):

    msgs_in_stderr = (
        'Error while evaluating expression in conditional breakpoint: i > 5',
        'Traceback (most recent call last):',
        'File "<string>", line 1, in <module>',
    )

    # It could be one or the other in PyPy depending on the version.
    msgs_one_in_stderr = (
        "NameError: name 'i' is not defined",
        "global name 'i' is not defined",
    )

    def _ignore_stderr_line(line):
        if original_ignore_stderr_line(line):
            return True

        for msg in msgs_in_stderr + msgs_one_in_stderr:
            if msg in line:
                return True

        return False

    with case_setup.test_file('_debugger_case_breakpoint_condition_exc.py') as writer:

        original_ignore_stderr_line = writer._ignore_stderr_line
        writer._ignore_stderr_line = _ignore_stderr_line

        writer.write_suspend_on_breakpoint_exception(skip_suspend_on_breakpoint_exception, skip_print_breakpoint_exception)

        expect_print = skip_print_breakpoint_exception in ([], ['ValueError'])
        expect_suspend = skip_suspend_on_breakpoint_exception in ([], ['ValueError'])

        breakpoint_id = writer.write_add_breakpoint(
            writer.get_line_index_with_content('break here'), 'Call', condition='i > 5')

        if not expect_print:
            _original = writer.reader_thread.on_message_found

            def on_message_found(found_msg):
                for msg in msgs_in_stderr + msgs_one_in_stderr:
                    assert msg not in found_msg

            writer.reader_thread.on_message_found = on_message_found

        writer.write_make_initial_run()

        def check_error_msg(stderr):
            for msg in msgs_in_stderr:
                assert msg in stderr

            for msg in msgs_one_in_stderr:
                if msg in stderr:
                    break
            else:
                raise AssertionError('Did not find any of: %s in stderr: %s' % (
                    msgs_one_in_stderr, stderr))

        if expect_print:
            msg, ctx = writer.wait_for_output()
            check_error_msg(msg.replace('&gt;', '>'))

        if expect_suspend:
            writer.wait_for_message(CMD_GET_BREAKPOINT_EXCEPTION)
            hit = writer.wait_for_breakpoint_hit()
            writer.write_run_thread(hit.thread_id)

        if IS_JYTHON:
            # Jython will break twice.
            if expect_suspend:
                writer.wait_for_message(CMD_GET_BREAKPOINT_EXCEPTION)
                hit = writer.wait_for_breakpoint_hit()
                writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_get_frame(thread_id, frame_id)
        msg = writer.wait_for_message(CMD_GET_FRAME)
        name_to_value = {}
        for var in msg.var:
            name_to_value[var['name']] = var['value']
        assert name_to_value == {'i': 'int: 6', 'last_i': 'int: 6'}

        writer.write_remove_breakpoint(breakpoint_id)

        writer.write_run_thread(thread_id)

        writer.finished_ok = True


def test_case_remove_breakpoint(case_setup):
    with case_setup.test_file('_debugger_case_remove_breakpoint.py') as writer:
        breakpoint_id = writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_double_remove_breakpoint(case_setup):

    with case_setup.test_file('_debugger_case_remove_breakpoint.py') as writer:
        breakpoint_id = writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_remove_breakpoint(breakpoint_id)  # Double-remove (just check that we don't have an error).
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_IRONPYTHON, reason='This test fails once in a while due to timing issues on IronPython, so, skipping it.')
def test_case_3(case_setup):
    with case_setup.test_file('_debugger_case3.py') as writer:
        writer.write_make_initial_run()
        time.sleep(.5)
        breakpoint_id = writer.write_add_breakpoint(4, '')

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_get_frame(thread_id, frame_id)

        writer.write_run_thread(thread_id)

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_get_frame(thread_id, frame_id)

        writer.write_remove_breakpoint(breakpoint_id)

        writer.write_run_thread(thread_id)

        writer.finished_ok = True


def test_case_suspend_thread(case_setup):
    with case_setup.test_file('_debugger_case4.py') as writer:
        writer.write_make_initial_run()

        thread_id = writer.wait_for_new_thread()

        writer.write_suspend_thread(thread_id)

        while True:
            hit = writer.wait_for_breakpoint_hit((REASON_THREAD_SUSPEND, REASON_STOP_ON_BREAKPOINT))
            if hit.name == 'sleep':
                break  # Ok, broke on 'sleep'.
            else:
                # i.e.: if it doesn't hit on 'sleep', release and pause again.
                writer.write_run_thread(thread_id)
                time.sleep(.1)
                writer.write_suspend_thread(thread_id)

        assert hit.thread_id == thread_id

        writer.write_evaluate_expression('%s\t%s\t%s' % (hit.thread_id, hit.frame_id, 'LOCAL'), 'exit_while_loop()')
        writer.wait_for_evaluation([
            [
                '<var name="exit_while_loop()" type="str" qualifier="{0}" value="str: ok'.format(builtin_qualifier),
                '<var name="exit_while_loop()" type="str"  value="str: ok"',  # jython
             ]
        ])

        writer.write_run_thread(thread_id)

        writer.finished_ok = True


# Jython has a weird behavior: it seems it has fine-grained locking so that when
# we're inside the tracing other threads don't run (so, we can have only one
# thread paused in the debugger).
@pytest.mark.skipif(IS_JYTHON, reason='Jython can only have one thread stopped at each time.')
def test_case_suspend_all_thread(case_setup):
    with case_setup.test_file('_debugger_case_suspend_all.py') as writer:
        writer.write_make_initial_run()

        main_thread_id = writer.wait_for_new_thread()  # Main thread
        thread_id1 = writer.wait_for_new_thread()  # Thread 1
        thread_id2 = writer.wait_for_new_thread()  # Thread 2

        # Ok, all threads created, let's wait for the main thread to get to the join.
        writer.wait_for_thread_join(main_thread_id)

        writer.write_suspend_thread('*')

        # Wait for 2 threads to be suspended (the main thread is already in a join, so, it can't actually
        # break out of it while others don't proceed).
        hit0 = writer.wait_for_breakpoint_hit(REASON_THREAD_SUSPEND)
        hit1 = writer.wait_for_breakpoint_hit(REASON_THREAD_SUSPEND)

        writer.write_evaluate_expression('%s\t%s\t%s' % (hit0.thread_id, hit0.frame_id, 'LOCAL'), 'exit_while_loop(1)')
        writer.wait_for_evaluation([
            [
                '<var name="exit_while_loop(1)" type="str" qualifier="{0}" value="str: ok'.format(builtin_qualifier)
            ]
        ])

        writer.write_evaluate_expression('%s\t%s\t%s' % (hit1.thread_id, hit1.frame_id, 'LOCAL'), 'exit_while_loop(2)')
        writer.wait_for_evaluation('<var name="exit_while_loop(2)" type="str" qualifier="{0}" value="str: ok'.format(builtin_qualifier))

        writer.write_run_thread('*')

        writer.finished_ok = True


def test_case_5(case_setup):
    with case_setup.test_file('_debugger_case56.py') as writer:
        breakpoint_id = writer.write_add_breakpoint(2, 'Call2')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_get_frame(thread_id, frame_id)

        writer.write_remove_breakpoint(breakpoint_id)

        writer.write_step_return(thread_id)

        hit = writer.wait_for_breakpoint_hit('109')
        thread_id = hit.thread_id
        frame_id = hit.frame_id
        line = hit.line

        assert line == 8, 'Expecting it to go to line 8. Went to: %s' % line

        writer.write_step_in(thread_id)

        hit = writer.wait_for_breakpoint_hit('107')
        thread_id = hit.thread_id
        frame_id = hit.frame_id
        line = hit.line

        # goes to line 4 in jython (function declaration line)
        assert line in (4, 5), 'Expecting it to go to line 4 or 5. Went to: %s' % line

        writer.write_run_thread(thread_id)

        assert 15 == writer._sequence, 'Expected 15. Had: %s' % writer._sequence

        writer.finished_ok = True


def test_case_6(case_setup):
    with case_setup.test_file('_debugger_case56.py') as writer:
        writer.write_add_breakpoint(2, 'Call2')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_get_frame(thread_id, frame_id)

        writer.write_step_return(thread_id)

        hit = writer.wait_for_breakpoint_hit('109')
        thread_id = hit.thread_id
        frame_id = hit.frame_id
        line = hit.line

        assert line == 8, 'Expecting it to go to line 8. Went to: %s' % line

        writer.write_step_in(thread_id)

        hit = writer.wait_for_breakpoint_hit('107')
        thread_id = hit.thread_id
        frame_id = hit.frame_id
        line = hit.line

        # goes to line 4 in jython (function declaration line)
        assert line in (4, 5), 'Expecting it to go to line 4 or 5. Went to: %s' % line

        writer.write_run_thread(thread_id)

        assert 13 == writer._sequence, 'Expected 15. Had: %s' % writer._sequence

        writer.finished_ok = True


@pytest.mark.skipif(IS_IRONPYTHON, reason='This test is flaky on Jython, so, skipping it.')
def test_case_7(case_setup):
    # This test checks that we start without variables and at each step a new var is created, but on ironpython,
    # the variables exist all at once (with None values), so, we can't test it properly.
    with case_setup.test_file('_debugger_case_local_variables.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'), 'Call')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111')

        writer.write_get_frame(hit.thread_id, hit.frame_id)

        writer.wait_for_vars('<xml></xml>')  # no vars at this point

        writer.write_step_over(hit.thread_id)

        writer.wait_for_breakpoint_hit('108')

        writer.write_get_frame(hit.thread_id, hit.frame_id)

        writer.wait_for_vars([
            [
                '<xml><var name="variable_for_test_1" type="int" qualifier="{0}" value="int%253A 10" />%0A</xml>'.format(builtin_qualifier),
                '<var name="variable_for_test_1" type="int"  value="int',  # jython
            ]
        ])

        writer.write_step_over(hit.thread_id)

        writer.wait_for_breakpoint_hit('108')

        writer.write_get_frame(hit.thread_id, hit.frame_id)

        writer.wait_for_vars([
            [
                '<xml><var name="variable_for_test_1" type="int" qualifier="{0}" value="int%253A 10" />%0A<var name="variable_for_test_2" type="int" qualifier="{0}" value="int%253A 20" />%0A</xml>'.format(builtin_qualifier),
                '<var name="variable_for_test_1" type="int"  value="int%253A 10" />%0A<var name="variable_for_test_2" type="int"  value="int%253A 20" />%0A',  # jython
            ]
        ])

        writer.write_run_thread(hit.thread_id)

        assert 17 == writer._sequence, 'Expected 17. Had: %s' % writer._sequence

        writer.finished_ok = True


def test_case_8(case_setup):
    with case_setup.test_file('_debugger_case89.py') as writer:
        writer.write_add_breakpoint(10, 'Method3')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111')

        writer.write_step_return(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit('109', line=15)

        writer.write_run_thread(hit.thread_id)

        assert 9 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence

        writer.finished_ok = True


def test_case_9(case_setup):
    with case_setup.test_file('_debugger_case89.py') as writer:
        writer.write_add_breakpoint(10, 'Method3')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111')

        # Note: no active exception (should not give an error and should return no
        # exception details as there's no exception).
        writer.write_get_current_exception(hit.thread_id)

        msg = writer.wait_for_message(CMD_GET_EXCEPTION_DETAILS)
        assert msg.thread['id'] == hit.thread_id
        assert not hasattr(msg.thread, 'frames')  # No frames should be found.

        writer.write_step_over(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit('108', line=11)

        writer.write_step_over(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit('108', line=12)

        writer.write_run_thread(hit.thread_id)

        assert 13 == writer._sequence, 'Expected 13. Had: %s' % writer._sequence

        writer.finished_ok = True


def test_case_10(case_setup):
    with case_setup.test_file('_debugger_case_simple_calls.py') as writer:
        writer.write_add_breakpoint(2, 'None')  # None or Method should make hit.
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111')

        writer.write_step_return(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit('109', line=11)

        writer.write_step_over(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit('108', line=12)

        writer.write_run_thread(hit.thread_id)

        assert 11 == writer._sequence, 'Expected 11. Had: %s' % writer._sequence

        writer.finished_ok = True


def test_case_11(case_setup):
    with case_setup.test_file('_debugger_case_simple_calls.py') as writer:
        writer.write_add_breakpoint(2, 'Method1')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=2)
        assert hit.name == 'Method1'

        writer.write_step_over(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_STEP_OVER, line=3)
        assert hit.name == 'Method1'

        writer.write_step_over(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_STEP_OVER, line=12)  # Reverts to step in
        assert hit.name == 'Method2'

        writer.write_step_over(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_STEP_OVER, line=13)
        assert hit.name == 'Method2'

        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(REASON_STEP_OVER, line=18)  # Reverts to step in
        assert hit.name == '<module>'

        # Finish with a step over
        writer.write_step_over(hit.thread_id)

        if IS_JYTHON:
            writer.write_run_thread(hit.thread_id)
        else:
            # Finish with a step over
            writer.write_step_over(hit.thread_id)

        writer.finished_ok = True


def test_case_12(case_setup):
    # Note: In CPython we now ignore the function names, so, we'll stop at the breakpoint in line 2
    # regardless of the function name (we decide whether to stop in a line or not through the function
    # lines).
    with case_setup.test_file('_debugger_case_simple_calls.py') as writer:
        writer.write_add_breakpoint(2, '')  # Should not be hit: setting empty function (not None) should only hit global.
        writer.write_add_breakpoint(6, 'Method1a')
        writer.write_add_breakpoint(11, 'Method2')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111', line=11)

        writer.write_step_return(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit('111', line=6 if IS_JYTHON else 2)  # not a return (it stopped in the other breakpoint)

        writer.write_run_thread(hit.thread_id)

        if not IS_JYTHON:
            hit = writer.wait_for_breakpoint_hit('111', line=6)

            writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_IRONPYTHON, reason='Failing on IronPython (needs to be investigated).')
def test_case_13(case_setup):
    with case_setup.test_file('_debugger_case13.py') as writer:

        def _ignore_stderr_line(line):
            if original_ignore_stderr_line(line):
                return True

            if IS_JYTHON:
                for expected in (
                    "RuntimeWarning: Parent module '_pydevd_bundle' not found while handling absolute import",
                    "import __builtin__"):
                    if expected in line:
                        return True

            return False

        original_ignore_stderr_line = writer._ignore_stderr_line
        writer._ignore_stderr_line = _ignore_stderr_line

        writer.write_add_breakpoint(35, 'main')
        writer.write("%s\t%s\t%s" % (CMD_SET_PROPERTY_TRACE, writer.next_seq(), "true;false;false;true"))
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit('111')

        writer.write_get_frame(hit.thread_id, hit.frame_id)

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=25)
        # Should go inside setter method

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=36)

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=21)
        # Should go inside getter method

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107')

        # Disable property tracing
        writer.write("%s\t%s\t%s" % (CMD_SET_PROPERTY_TRACE, writer.next_seq(), "true;true;true;true"))
        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=39)
        # Should Skip step into properties setter

        # Enable property tracing
        writer.write("%s\t%s\t%s" % (CMD_SET_PROPERTY_TRACE, writer.next_seq(), "true;false;false;true"))
        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=8)
        # Should go inside getter method

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_14(case_setup):
    # Interactive Debug Console
    with case_setup.test_file('_debugger_case14.py') as writer:
        writer.write_add_breakpoint(22, 'main')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111')
        assert hit.thread_id, '%s not valid.' % hit.thread_id
        assert hit.frame_id, '%s not valid.' % hit.frame_id

        # Access some variable
        writer.write_debug_console_expression("%s\t%s\tEVALUATE\tcarObj.color" % (hit.thread_id, hit.frame_id))
        writer.wait_for_var(['<more>False</more>', '%27Black%27'])
        assert 7 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence

        # Change some variable
        writer.write_debug_console_expression("%s\t%s\tEVALUATE\tcarObj.color='Red'" % (hit.thread_id, hit.frame_id))
        writer.write_debug_console_expression("%s\t%s\tEVALUATE\tcarObj.color" % (hit.thread_id, hit.frame_id))
        writer.wait_for_var(['<more>False</more>', '%27Red%27'])
        assert 11 == writer._sequence, 'Expected 13. Had: %s' % writer._sequence

        # Iterate some loop
        writer.write_debug_console_expression("%s\t%s\tEVALUATE\tfor i in range(3):" % (hit.thread_id, hit.frame_id))
        writer.wait_for_var(['<xml><more>True</more></xml>'])
        writer.write_debug_console_expression("%s\t%s\tEVALUATE\t    print(i)" % (hit.thread_id, hit.frame_id))
        writer.wait_for_var(['<xml><more>True</more></xml>'])
        writer.write_debug_console_expression("%s\t%s\tEVALUATE\t" % (hit.thread_id, hit.frame_id))
        writer.wait_for_var(
            [
                '<xml><more>False</more><output message="0"></output><output message="1"></output><output message="2"></output></xml>'            ]
            )
        assert 17 == writer._sequence, 'Expected 19. Had: %s' % writer._sequence

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_case_15(case_setup):
    with case_setup.test_file('_debugger_case15.py') as writer:
        writer.write_add_breakpoint(22, 'main')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)

        # Access some variable
        writer.write_custom_operation("%s\t%s\tEXPRESSION\tcarObj.color" % (hit.thread_id, hit.frame_id), "EXEC", "f=lambda x: 'val=%s' % x", "f")
        writer.wait_for_custom_operation('val=Black')
        assert 7 == writer._sequence, 'Expected 7. Had: %s' % writer._sequence

        writer.write_custom_operation("%s\t%s\tEXPRESSION\tcarObj.color" % (hit.thread_id, hit.frame_id), "EXECFILE", debugger_unittest._get_debugger_test_file('_debugger_case15_execfile.py'), "f")
        writer.wait_for_custom_operation('val=Black')
        assert 9 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_case_16_resolve_numpy_array(case_setup):
    # numpy.ndarray resolver
    try:
        import numpy
    except ImportError:
        pytest.skip('numpy not available')
    with case_setup.test_file('_debugger_case16.py') as writer:
        writer.write_add_breakpoint(9, 'main')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)

        # In this test we check that the three arrays of different shapes, sizes and types
        # are all resolved properly as ndarrays.

        # First pass check is that we have all three expected variables defined
        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_multiple_vars((
            (
                '<var name="smallarray" type="ndarray" qualifier="numpy" value="ndarray%253A %255B 0.%252B1.j  1.%252B1.j  2.%252B1.j  3.%252B1.j  4.%252B1.j  5.%252B1.j  6.%252B1.j  7.%252B1.j  8.%252B1.j%250A  9.%252B1.j 10.%252B1.j 11.%252B1.j 12.%252B1.j 13.%252B1.j 14.%252B1.j 15.%252B1.j 16.%252B1.j 17.%252B1.j%250A 18.%252B1.j 19.%252B1.j 20.%252B1.j 21.%252B1.j 22.%252B1.j 23.%252B1.j 24.%252B1.j 25.%252B1.j 26.%252B1.j%250A 27.%252B1.j 28.%252B1.j 29.%252B1.j 30.%252B1.j 31.%252B1.j 32.%252B1.j 33.%252B1.j 34.%252B1.j 35.%252B1.j%250A 36.%252B1.j 37.%252B1.j 38.%252B1.j 39.%252B1.j 40.%252B1.j 41.%252B1.j 42.%252B1.j 43.%252B1.j 44.%252B1.j%250A 45.%252B1.j 46.%252B1.j 47.%252B1.j 48.%252B1.j 49.%252B1.j 50.%252B1.j 51.%252B1.j 52.%252B1.j 53.%252B1.j%250A 54.%252B1.j 55.%252B1.j 56.%252B1.j 57.%252B1.j 58.%252B1.j 59.%252B1.j 60.%252B1.j 61.%252B1.j 62.%252B1.j%250A 63.%252B1.j 64.%252B1.j 65.%252B1.j 66.%252B1.j 67.%252B1.j 68.%252B1.j 69.%252B1.j 70.%252B1.j 71.%252B1.j%250A 72.%252B1.j 73.%252B1.j 74.%252B1.j 75.%252B1.j 76.%252B1.j 77.%252B1.j 78.%252B1.j 79.%252B1.j 80.%252B1.j%250A 81.%252B1.j 82.%252B1.j 83.%252B1.j 84.%252B1.j 85.%252B1.j 86.%252B1.j 87.%252B1.j 88.%252B1.j 89.%252B1.j%250A 90.%252B1.j 91.%252B1.j 92.%252B1.j 93.%252B1.j 94.%252B1.j 95.%252B1.j 96.%252B1.j 97.%252B1.j 98.%252B1.j%250A 99.%252B1.j%255D" isContainer="True" />',
                '<var name="smallarray" type="ndarray" qualifier="numpy" value="ndarray%253A %255B  0.%252B1.j   1.%252B1.j   2.%252B1.j   3.%252B1.j   4.%252B1.j   5.%252B1.j   6.%252B1.j   7.%252B1.j%250A   8.%252B1.j   9.%252B1.j  10.%252B1.j  11.%252B1.j  12.%252B1.j  13.%252B1.j  14.%252B1.j  15.%252B1.j%250A  16.%252B1.j  17.%252B1.j  18.%252B1.j  19.%252B1.j  20.%252B1.j  21.%252B1.j  22.%252B1.j  23.%252B1.j%250A  24.%252B1.j  25.%252B1.j  26.%252B1.j  27.%252B1.j  28.%252B1.j  29.%252B1.j  30.%252B1.j  31.%252B1.j%250A  32.%252B1.j  33.%252B1.j  34.%252B1.j  35.%252B1.j  36.%252B1.j  37.%252B1.j  38.%252B1.j  39.%252B1.j%250A  40.%252B1.j  41.%252B1.j  42.%252B1.j  43.%252B1.j  44.%252B1.j  45.%252B1.j  46.%252B1.j  47.%252B1.j%250A  48.%252B1.j  49.%252B1.j  50.%252B1.j  51.%252B1.j  52.%252B1.j  53.%252B1.j  54.%252B1.j  55.%252B1.j%250A  56.%252B1.j  57.%252B1.j  58.%252B1.j  59.%252B1.j  60.%252B1.j  61.%252B1.j  62.%252B1.j  63.%252B1.j%250A  64.%252B1.j  65.%252B1.j  66.%252B1.j  67.%252B1.j  68.%252B1.j  69.%252B1.j  70.%252B1.j  71.%252B1.j%250A  72.%252B1.j  73.%252B1.j  74.%252B1.j  75.%252B1.j  76.%252B1.j  77.%252B1.j  78.%252B1.j  79.%252B1.j%250A  80.%252B1.j  81.%252B1.j  82.%252B1.j  83.%252B1.j  84.%252B1.j  85.%252B1.j  86.%252B1.j  87.%252B1.j%250A  88.%252B1.j  89.%252B1.j  90.%252B1.j  91.%252B1.j  92.%252B1.j  93.%252B1.j  94.%252B1.j  95.%252B1.j%250A  96.%252B1.j  97.%252B1.j  98.%252B1.j  99.%252B1.j%255D" isContainer="True" />'
            ),

            (
                '<var name="bigarray" type="ndarray" qualifier="numpy" value="ndarray%253A %255B%255B    0     1     2 ...  9997  9998  9999%255D%250A %255B10000 10001 10002 ... 19997 19998 19999%255D%250A %255B20000 20001 20002 ... 29997 29998 29999%255D%250A ...%250A %255B70000 70001 70002 ... 79997 79998 79999%255D%250A %255B80000 80001 80002 ... 89997 89998 89999%255D%250A %255B90000 90001 90002 ... 99997 99998 99999%255D%255D" isContainer="True" />',
                '<var name="bigarray" type="ndarray" qualifier="numpy" value="ndarray%253A %255B%255B    0     1     2 ...%252C  9997  9998  9999%255D%250A %255B10000 10001 10002 ...%252C 19997 19998 19999%255D%250A %255B20000 20001 20002 ...%252C 29997 29998 29999%255D%250A ...%252C %250A %255B70000 70001 70002 ...%252C 79997 79998 79999%255D%250A %255B80000 80001 80002 ...%252C 89997 89998 89999%255D%250A %255B90000 90001 90002 ...%252C 99997 99998 99999%255D%255D" isContainer="True" />'
            ),

            # Any of the ones below will do.
            (
                '<var name="hugearray" type="ndarray" qualifier="numpy" value="ndarray%253A %255B      0       1       2 ... 9999997 9999998 9999999%255D" isContainer="True" />',
                '<var name="hugearray" type="ndarray" qualifier="numpy" value="ndarray%253A %255B      0       1       2 ...%252C 9999997 9999998 9999999%255D" isContainer="True" />'
            )
        ))

        # For each variable, check each of the resolved (meta data) attributes...
        writer.write_get_variable(hit.thread_id, hit.frame_id, 'smallarray')
        writer.wait_for_multiple_vars((
            '<var name="min" type="complex128"',
            '<var name="max" type="complex128"',
            '<var name="shape" type="tuple"',
            '<var name="dtype" type="dtype',
            '<var name="size" type="int"',
        ))
        # ...and check that the internals are resolved properly
        writer.write_get_variable(hit.thread_id, hit.frame_id, 'smallarray\t__internals__')
        writer.wait_for_var('<var name="%27size%27')

        writer.write_get_variable(hit.thread_id, hit.frame_id, 'bigarray')
        # isContainer could be true on some numpy versions, so, we only check for the var begin.
        writer.wait_for_multiple_vars((
            [
                '<var name="min" type="int64" qualifier="numpy" value="int64%253A 0"',
                '<var name="min" type="int64" qualifier="numpy" value="int64%3A 0"',
                '<var name="size" type="int" qualifier="{0}" value="int%3A 100000"'.format(builtin_qualifier),
            ],
            [
                '<var name="max" type="int64" qualifier="numpy" value="int64%253A 99999"',
                '<var name="max" type="int32" qualifier="numpy" value="int32%253A 99999"',
                '<var name="max" type="int64" qualifier="numpy" value="int64%3A 99999"',
                '<var name="max" type="int32" qualifier="numpy" value="int32%253A 99999"',
            ],
            '<var name="shape" type="tuple"',
            '<var name="dtype" type="dtype',
            '<var name="size" type="int"'
        ))
        writer.write_get_variable(hit.thread_id, hit.frame_id, 'bigarray\t__internals__')
        writer.wait_for_var('<var name="%27size%27')

        # this one is different because it crosses the magic threshold where we don't calculate
        # the min/max
        writer.write_get_variable(hit.thread_id, hit.frame_id, 'hugearray')
        writer.wait_for_var((
            [
                '<var name="min" type="str" qualifier={0} value="str%253A ndarray too big%252C calculating min would slow down debugging" />'.format(builtin_qualifier),
                '<var name="min" type="str" qualifier={0} value="str%3A ndarray too big%252C calculating min would slow down debugging" />'.format(builtin_qualifier),
                '<var name="min" type="str" qualifier="{0}" value="str%253A ndarray too big%252C calculating min would slow down debugging" />'.format(builtin_qualifier),
                '<var name="min" type="str" qualifier="{0}" value="str%3A ndarray too big%252C calculating min would slow down debugging" />'.format(builtin_qualifier),
            ],
            [
                '<var name="max" type="str" qualifier={0} value="str%253A ndarray too big%252C calculating max would slow down debugging" />'.format(builtin_qualifier),
                '<var name="max" type="str" qualifier={0} value="str%3A ndarray too big%252C calculating max would slow down debugging" />'.format(builtin_qualifier),
                '<var name="max" type="str" qualifier="{0}" value="str%253A ndarray too big%252C calculating max would slow down debugging" />'.format(builtin_qualifier),
                '<var name="max" type="str" qualifier="{0}" value="str%3A ndarray too big%252C calculating max would slow down debugging" />'.format(builtin_qualifier),
            ],
            '<var name="shape" type="tuple"',
            '<var name="dtype" type="dtype',
            '<var name="size" type="int"',
        ))
        writer.write_get_variable(hit.thread_id, hit.frame_id, 'hugearray\t__internals__')
        writer.wait_for_var('<var name="%27size%27')

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_case_17(case_setup):
    # Check dont trace
    with case_setup.test_file('_debugger_case17.py') as writer:
        writer.write_enable_dont_trace(True)
        writer.write_add_breakpoint(writer.get_line_index_with_content('break1'), 'main')
        writer.write_add_breakpoint(writer.get_line_index_with_content('break2'), 'main')
        writer.write_add_breakpoint(writer.get_line_index_with_content('break3'), 'main')
        writer.write_add_breakpoint(writer.get_line_index_with_content('break4'), 'main')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)
        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=2)
        # Should Skip step into properties setter
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)
        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=2)
        # Should Skip step into properties setter
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)
        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=2)
        # Should Skip step into properties setter
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)
        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('107', line=2)
        # Should Skip step into properties setter
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_17a(case_setup):
    # Check dont trace return
    with case_setup.test_file('_debugger_case17a.py') as writer:
        writer.write_enable_dont_trace(True)
        break1_line = writer.get_line_index_with_content('break 1 here')
        writer.write_add_breakpoint(break1_line, 'm1')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=break1_line)

        writer.write_step_in(hit.thread_id)
        break2_line = writer.get_line_index_with_content('break 2 here')
        hit = writer.wait_for_breakpoint_hit('107', line=break2_line)

        # Should Skip step into properties setter
        assert hit.name == 'm3'
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_18(case_setup):
    # change local variable
    if IS_IRONPYTHON or IS_JYTHON:
        pytest.skip('Unsupported assign to local')

    with case_setup.test_file('_debugger_case18.py') as writer:
        writer.write_add_breakpoint(5, 'm2')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=5)

        writer.write_change_variable(hit.thread_id, hit.frame_id, 'a', '40')
        writer.wait_for_var('<xml><var name="" type="int" qualifier="{0}" value="int%253A 40" />%0A</xml>'.format(builtin_qualifier,))
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_19(case_setup):
    # Check evaluate '__' attributes
    with case_setup.test_file('_debugger_case19.py') as writer:
        writer.write_add_breakpoint(8, None)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=8)

        writer.write_evaluate_expression('%s\t%s\t%s' % (hit.thread_id, hit.frame_id, 'LOCAL'), 'a.__var')
        writer.wait_for_evaluation([
            [
                '<var name="a.__var" type="int" qualifier="{0}" value="int'.format(builtin_qualifier),
                '<var name="a.__var" type="int"  value="int',  # jython
            ]
        ])
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Monkey-patching related to starting threads not done on Jython.')
def test_case_20(case_setup):
    # Check that we were notified of threads creation before they started to run
    with case_setup.test_file('_debugger_case20.py') as writer:
        writer.write_make_initial_run()

        # We already check if it prints 'TEST SUCEEDED' by default, so, nothing
        # else should be needed in this test as it tests what's needed just by
        # running the module.
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_FLASK, reason='No flask available')
def test_case_flask(case_setup_flask):
    with case_setup_flask.test_file(EXPECTED_RETURNCODE='any') as writer:
        writer.write_multi_threads_single_notification(True)
        writer.write_add_breakpoint_jinja2(5, None, 'hello.html')
        writer.write_add_breakpoint_jinja2(8, None, 'hello.html')
        writer.write_make_initial_run()

        t = writer.create_request_thread()
        time.sleep(2)  # Give flask some time to get to startup before requesting the page
        t.start()

        hit = writer.wait_for_single_notification_as_hit(line=5)
        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_vars(['<var name="content" type="str"'])
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_single_notification_as_hit(line=8)
        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_vars(['<var name="content" type="str"'])
        writer.write_run_thread(hit.thread_id)

        contents = t.wait_for_contents()

        assert '<title>Hello</title>' in contents
        assert 'Flask-Jinja-Test' in contents

        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
def test_case_django_a(case_setup_django):

    def get_environ(writer):
        env = os.environ.copy()
        env.update({
            'PYDEVD_FILTER_LIBRARIES': '1',  # Global setting for in project or not
        })
        return env

    with case_setup_django.test_file(EXPECTED_RETURNCODE='any', get_environ=get_environ) as writer:
        writer.write_make_initial_run()

        # Wait for the first request that works...
        for i in range(4):
            try:
                t = writer.create_request_thread('my_app')
                t.start()
                contents = t.wait_for_contents()
                contents = contents.replace(' ', '').replace('\r', '').replace('\n', '')
                assert contents == '<ul><li>v1:v1</li><li>v2:v2</li></ul>'
                break
            except:
                if i == 3:
                    raise
                continue

        writer.write_add_breakpoint_django(5, None, 'index.html')
        t = writer.create_request_thread('my_app')
        t.start()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=5)
        writer.write_get_variable(hit.thread_id, hit.frame_id, 'entry')
        writer.wait_for_vars([
            '<var name="key" type="str"',
            'v1'
        ])

        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=5)
        writer.write_get_variable(hit.thread_id, hit.frame_id, 'entry')
        writer.wait_for_vars([
            '<var name="key" type="str"',
            'v2'
        ])

        writer.write_run_thread(hit.thread_id)

        contents = t.wait_for_contents()

        contents = contents.replace(' ', '').replace('\r', '').replace('\n', '')
        if contents != '<ul><li>v1:v1</li><li>v2:v2</li></ul>':
            raise AssertionError('%s != <ul><li>v1:v1</li><li>v2:v2</li></ul>' % (contents,))

        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
def test_case_django_b(case_setup_django):
    with case_setup_django.test_file(EXPECTED_RETURNCODE='any') as writer:
        writer.write_add_breakpoint_django(4, None, 'name.html')
        writer.write_add_exception_breakpoint_django()
        writer.write_remove_exception_breakpoint_django()
        writer.write_make_initial_run()

        t = writer.create_request_thread('my_app/name')
        time.sleep(5)  # Give django some time to get to startup before requesting the page
        t.start()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=4)

        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_var('<var name="form" type="NameForm" qualifier="my_app.forms" value="NameForm%253A')
        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
def test_case_django_template_inherits_no_exception(case_setup_django):
    with case_setup_django.test_file(EXPECTED_RETURNCODE='any') as writer:

        # Check that it doesn't have issues with inherits + django exception breakpoints.
        writer.write_add_exception_breakpoint_django()

        writer.write_make_initial_run()

        t = writer.create_request_thread('my_app/inherits')
        time.sleep(5)  # Give django some time to get to startup before requesting the page
        t.start()
        contents = t.wait_for_contents()

        contents = contents.replace(' ', '').replace('\r', '').replace('\n', '')
        assert contents == '''"chat_mode=True""chat_mode=False"'''

        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
def test_case_django_no_var_error(case_setup_django):
    with case_setup_django.test_file(EXPECTED_RETURNCODE='any') as writer:

        # Check that it doesn't have issues with inherits + django exception breakpoints.
        writer.write_add_exception_breakpoint_django()

        writer.write_make_initial_run()

        t = writer.create_request_thread('my_app/no_var_error')
        time.sleep(5)  # Give django some time to get to startup before requesting the page
        t.start()
        contents = t.wait_for_contents()

        contents = contents.replace(' ', '').replace('\r', '').replace('\n', '')
        assert contents == '''no_pat_name'''

        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
@pytest.mark.parametrize("jmc", [False, True])
def test_case_django_no_attribute_exception_breakpoint(case_setup_django, jmc):
    kwargs = {}
    if jmc:

        def get_environ(writer):
            env = os.environ.copy()
            env.update({
                'PYDEVD_FILTER_LIBRARIES': '1',  # Global setting for in project or not
            })
            return env

        kwargs['get_environ'] = get_environ

    with case_setup_django.test_file(EXPECTED_RETURNCODE='any', **kwargs) as writer:
        writer.write_add_exception_breakpoint_django()

        writer.write_make_initial_run()

        t = writer.create_request_thread('my_app/template_error')
        time.sleep(5)  # Give django some time to get to startup before requesting the page
        t.start()

        hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION, line=7, file='template_error.html')

        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_var('<var name="entry" type="Entry" qualifier="my_app.views" value="Entry: v1:v1" isContainer="True"')

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
def test_case_django_no_attribute_exception_breakpoint_and_regular_exceptions(case_setup_django):
    with case_setup_django.test_file(EXPECTED_RETURNCODE='any') as writer:
        writer.write_add_exception_breakpoint_django()

        # The django plugin has priority over the regular exception breakpoint.
        writer.write_add_exception_breakpoint_with_policy(
            'django.template.base.VariableDoesNotExist',
            notify_on_handled_exceptions=2,  # 2 means notify only on first raise.
            notify_on_unhandled_exceptions=0,
            ignore_libraries=0
        )
        writer.write_make_initial_run()

        t = writer.create_request_thread('my_app/template_error')
        time.sleep(5)  # Give django some time to get to startup before requesting the page
        t.start()

        hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION, line=7, file='template_error.html')

        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_var('<var name="entry" type="Entry" qualifier="my_app.views" value="Entry: v1:v1" isContainer="True"')

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_DJANGO, reason='No django available')
@pytest.mark.parametrize("jmc", [False, True])
def test_case_django_invalid_template_exception_breakpoint(case_setup_django, jmc):
    kwargs = {}
    if jmc:

        def get_environ(writer):
            env = os.environ.copy()
            env.update({
                'PYDEVD_FILTER_LIBRARIES': '1',  # Global setting for in project or not
            })
            return env

        kwargs['get_environ'] = get_environ

    with case_setup_django.test_file(EXPECTED_RETURNCODE='any', **kwargs) as writer:
        writer.write_add_exception_breakpoint_django()
        writer.write_make_initial_run()

        t = writer.create_request_thread('my_app/template_error2')
        time.sleep(5)  # Give django some time to get to startup before requesting the page
        t.start()

        hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION, line=4, file='template_error2.html')

        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_var('<var name="token" type="Token" qualifier="django.template.base" value="Token:')

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_CYTHON, reason='No cython available')
def test_cython(case_setup):
    from _pydevd_bundle import pydevd_cython
    assert pydevd_cython.trace_dispatch is not None


def _has_qt():
    try:
        try:
            from PySide import QtCore  # @UnresolvedImport
            return True
        except:
            from PySide2 import QtCore  # @UnresolvedImport
            return True
    except:
        try:
            from PyQt4 import QtCore  # @UnresolvedImport
            return True
        except:
            try:
                from PyQt5 import QtCore  # @UnresolvedImport
                return True
            except:
                pass
    return False


@pytest.mark.skipif(not _has_qt(), reason='No qt available')
def test_case_qthread1(case_setup):
    with case_setup.test_file('_debugger_case_qthread1.py') as writer:
        breakpoint_id = writer.write_add_breakpoint(writer.get_line_index_with_content('break here'), 'run')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_run_thread(hit.thread_id)

        writer.log.append('Checking sequence. Found: %s' % (writer._sequence))
        assert 9 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(not _has_qt(), reason='No qt available')
def test_case_qthread2(case_setup):
    with case_setup.test_file('_debugger_case_qthread2.py') as writer:
        breakpoint_id = writer.write_add_breakpoint(writer.get_line_index_with_content('break here'), 'long_running')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id

        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_run_thread(thread_id)

        writer.log.append('Checking sequence. Found: %s' % (writer._sequence))
        assert 9 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(not _has_qt(), reason='No qt available')
def test_case_qthread3(case_setup):
    with case_setup.test_file('_debugger_case_qthread3.py') as writer:
        breakpoint_id = writer.write_add_breakpoint(writer.get_line_index_with_content('break here'), 'run')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_run_thread(thread_id)

        writer.log.append('Checking sequence. Found: %s' % (writer._sequence))
        assert 9 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(not _has_qt(), reason='No qt available')
def test_case_qthread4(case_setup):
    with case_setup.test_file('_debugger_case_qthread4.py') as writer:
        original_additional_output_checks = writer.additional_output_checks

        def additional_output_checks(stdout, stderr):
            original_additional_output_checks(stdout, stderr)
            if 'On start called' not in stdout:
                raise AssertionError('Expected "On start called" to be in stdout:\n%s' % (stdout,))
            if 'Done sleeping' not in stdout:
                raise AssertionError('Expected "Done sleeping" to be in stdout:\n%s' % (stdout,))
            if 'native Qt signal is not callable' in stderr:
                raise AssertionError('Did not expect "native Qt signal is not callable" to be in stderr:\n%s' % (stderr,))

        breakpoint_id = writer.write_add_breakpoint(28, 'on_start')  # breakpoint on print('On start called2').
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_run_thread(hit.thread_id)

        writer.log.append('Checking sequence. Found: %s' % (writer._sequence))
        assert 9 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


def test_m_switch(case_setup_m_switch):
    with case_setup_m_switch.test_file() as writer:
        writer.log.append('writing add breakpoint')
        breakpoint_id = writer.write_add_breakpoint(1, None)

        writer.log.append('making initial run')
        writer.write_make_initial_run()

        writer.log.append('waiting for breakpoint hit')
        hit = writer.wait_for_breakpoint_hit()

        writer.write_remove_breakpoint(breakpoint_id)

        writer.log.append('run thread')
        writer.write_run_thread(hit.thread_id)

        writer.log.append('asserting')
        try:
            assert 9 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence
        except:
            writer.log.append('assert failed!')
            raise
        writer.log.append('asserted')

        writer.finished_ok = True


def test_module_entry_point(case_setup_m_switch_entry_point):
    with case_setup_m_switch_entry_point.test_file() as writer:
        writer.log.append('writing add breakpoint')
        breakpoint_id = writer.write_add_breakpoint(1, None)

        writer.log.append('making initial run')
        writer.write_make_initial_run()

        writer.log.append('waiting for breakpoint hit')
        hit = writer.wait_for_breakpoint_hit()

        writer.write_remove_breakpoint(breakpoint_id)

        writer.log.append('run thread')
        writer.write_run_thread(hit.thread_id)

        writer.log.append('asserting')
        try:
            assert 9 == writer._sequence, 'Expected 9. Had: %s' % writer._sequence
        except:
            writer.log.append('assert failed!')
            raise
        writer.log.append('asserted')

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_check_tracer_with_exceptions(case_setup):

    def get_environ(writer):
        env = os.environ.copy()
        # This test requires regular tracing (without cython).
        env['PYDEVD_USE_CYTHON'] = 'NO'
        env['PYDEVD_USE_FRAME_EVAL'] = 'NO'
        return env

    with case_setup.test_file('_debugger_case_check_tracer.py', get_environ=get_environ) as writer:
        writer.write_add_exception_breakpoint_with_policy('IndexError', "1", "1", "1")
        writer.write_make_initial_run()
        writer.finished_ok = True


@pytest.mark.parametrize('target_file', [
    '_debugger_case_unhandled_exceptions_generator.py',
    '_debugger_case_unhandled_exceptions_listcomp.py',
    ])
@pytest.mark.parametrize('unhandled', [False, True])
@pytest.mark.skipif(IS_JYTHON, reason='Not ok for Jython.')
def test_case_handled_and_unhandled_exception_generator(case_setup, target_file, unhandled):

    def check_test_suceeded_msg(writer, stdout, stderr):
        # Don't call super (we have an unhandled exception in the stack trace).
        return 'TEST SUCEEDED' in ''.join(stdout) and 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        if 'ZeroDivisionError' not in stderr:
            raise AssertionError('Expected test to have an unhandled exception.\nstdout:\n%s\n\nstderr:\n%s' % (
                stdout, stderr))

    with case_setup.test_file(
            target_file,
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks,
            EXPECTED_RETURNCODE=1,
        ) as writer:

        if unhandled:
            writer.write_add_exception_breakpoint_with_policy('Exception', "0", "1", "0")
        else:
            writer.write_add_exception_breakpoint_with_policy('Exception', "1", "0", "0")

        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION if unhandled else REASON_CAUGHT_EXCEPTION)
        assert hit.line == writer.get_line_index_with_content('# exc line')

        if 'generator' in target_file:
            expected_frame_names = ['<genexpr>', 'f', '<module>']
        else:
            expected_frame_names = ['<listcomp>', 'f', '<module>']

        writer.write_get_current_exception(hit.thread_id)
        msg = writer.wait_for_message(accept_message=lambda msg:'exc_type="' in msg and 'exc_desc="' in msg, unquote_msg=False)

        frame_names = [unquote(f['name']).replace('&lt;', '<').replace('&gt;', '>') for f in msg.thread.frame]
        assert frame_names == expected_frame_names

        writer.write_run_thread(hit.thread_id)

        if not unhandled:
            expected_lines = [
                writer.get_line_index_with_content('# exc line'),
                writer.get_line_index_with_content('# call exc'),
            ]

            for expected_line in expected_lines:
                hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION)
                assert hit.line == expected_line

                writer.write_get_current_exception(hit.thread_id)
                msg = writer.wait_for_message(accept_message=lambda msg:'exc_type="' in msg and 'exc_desc="' in msg, unquote_msg=False)

                frame_names = [unquote(f['name']).replace('&lt;', '<').replace('&gt;', '>') for f in msg.thread.frame]
                assert frame_names == expected_frame_names

                writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Failing on Jython -- needs to be investigated).')
def test_unhandled_exceptions_basic(case_setup):

    def check_test_suceeded_msg(writer, stdout, stderr):
        # Don't call super (we have an unhandled exception in the stack trace).
        return 'TEST SUCEEDED' in ''.join(stdout) and 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        if 'raise Exception' not in stderr:
            raise AssertionError('Expected test to have an unhandled exception.\nstdout:\n%s\n\nstderr:\n%s' % (
                stdout, stderr))

    with case_setup.test_file(
            '_debugger_case_unhandled_exceptions.py',
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks,
            EXPECTED_RETURNCODE=1,
        ) as writer:

        writer.write_add_exception_breakpoint_with_policy('Exception', "0", "1", "0")
        writer.write_make_initial_run()

        def check(hit, exc_type, exc_desc):
            writer.write_get_current_exception(hit.thread_id)
            msg = writer.wait_for_message(accept_message=lambda msg:exc_type in msg and 'exc_type="' in msg and 'exc_desc="' in msg, unquote_msg=False)
            assert unquote(msg.thread['exc_desc']) == exc_desc
            assert unquote(msg.thread['exc_type']) in (
                "&lt;type 'exceptions.%s'&gt;" % (exc_type,),  # py2
                "&lt;class '%s'&gt;" % (exc_type,)  # py3
            )
            if len(msg.thread.frame) == 0:
                assert unquote(unquote(msg.thread.frame['file'])).endswith('_debugger_case_unhandled_exceptions.py')
            else:
                assert unquote(unquote(msg.thread.frame[0]['file'])).endswith('_debugger_case_unhandled_exceptions.py')
            writer.write_run_thread(hit.thread_id)

        # Will stop in 2 background threads
        hit0 = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        thread_id1 = hit0.thread_id

        hit1 = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        thread_id2 = hit1.thread_id

        if hit0.name == 'thread_func2':
            check(hit0, 'ValueError', 'in thread 2')
            check(hit1, 'Exception', 'in thread 1')
        else:
            check(hit0, 'Exception', 'in thread 1')
            check(hit1, 'ValueError', 'in thread 2')

        writer.write_run_thread(thread_id1)
        writer.write_run_thread(thread_id2)

        # Will stop in main thread
        hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        assert hit.name == '<module>'
        thread_id3 = hit.thread_id

        # Requesting the stack in an unhandled exception should provide the stack of the exception,
        # not the current location of the program.
        writer.write_get_thread_stack(thread_id3)
        msg = writer.wait_for_message(CMD_GET_THREAD_STACK)
        assert len(msg.thread.frame) == 0  # In main thread (must have no back frames).
        assert msg.thread.frame['name'] == '<module>'
        check(hit, 'IndexError', 'in main')

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Failing on Jython -- needs to be investigated).')
def test_unhandled_exceptions_in_top_level1(case_setup_unhandled_exceptions):

    with case_setup_unhandled_exceptions.test_file(
            '_debugger_case_unhandled_exceptions_on_top_level.py',
            EXPECTED_RETURNCODE=1,
        ) as writer:

        writer.write_add_exception_breakpoint_with_policy('Exception', "0", "1", "0")
        writer.write_make_initial_run()

        # Will stop in main thread
        hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Failing on Jython -- needs to be investigated).')
def test_unhandled_exceptions_in_top_level2(case_setup_unhandled_exceptions):
    # Note: expecting unhandled exception to be printed to stderr.

    def get_environ(writer):
        env = os.environ.copy()
        curr_pythonpath = env.get('PYTHONPATH', '')

        pydevd_dirname = os.path.dirname(writer.get_pydevd_file())

        curr_pythonpath = pydevd_dirname + os.pathsep + curr_pythonpath
        env['PYTHONPATH'] = curr_pythonpath
        return env

    def update_command_line_args(writer, args):
        # Start pydevd with '-m' to see how it deal with being called with
        # runpy at the start.
        assert args[0].endswith('pydevd.py')
        args = ['-m', 'pydevd'] + args[1:]
        return args

    with case_setup_unhandled_exceptions.test_file(
            '_debugger_case_unhandled_exceptions_on_top_level.py',
            get_environ=get_environ,
            update_command_line_args=update_command_line_args,
            EXPECTED_RETURNCODE='any',
            ) as writer:

        writer.write_add_exception_breakpoint_with_policy('Exception', "0", "1", "0")
        writer.write_make_initial_run()

        # Should stop (only once) in the main thread.
        hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Failing on Jython -- needs to be investigated).')
def test_unhandled_exceptions_in_top_level3(case_setup_unhandled_exceptions):

    with case_setup_unhandled_exceptions.test_file(
            '_debugger_case_unhandled_exceptions_on_top_level.py',
            EXPECTED_RETURNCODE=1
        ) as writer:

        # Handled and unhandled
        # PySide2 has a bug in shibokensupport which will try to do: sys._getframe(1).
        # during the teardown (which will fail as there's no back frame in this case).
        # So, mark ignore libraries in this case.
        writer.write_add_exception_breakpoint_with_policy('Exception', "1", "1", ignore_libraries="1")
        writer.write_make_initial_run()

        # Will stop in main thread twice: once one we find that the exception is being
        # thrown and another in postmortem mode when we discover it's uncaught.
        hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Failing on Jython -- needs to be investigated).')
def test_unhandled_exceptions_in_top_level4(case_setup_unhandled_exceptions):

    # Note: expecting unhandled exception to be printed to stderr.
    with case_setup_unhandled_exceptions.test_file(
            '_debugger_case_unhandled_exceptions_on_top_level2.py',
            EXPECTED_RETURNCODE=1,
        ) as writer:

        # Handled and unhandled
        writer.write_add_exception_breakpoint_with_policy('Exception', "1", "1", "0")
        writer.write_make_initial_run()

        # We have an exception thrown and handled and another which is thrown and is then unhandled.
        hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='Only for Python.')
def test_case_set_next_statement(case_setup):

    with case_setup.test_file('_debugger_case_set_next_statement.py') as writer:
        breakpoint_id = writer.write_add_breakpoint(6, None)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=6)  # Stop in line a=3 (before setting it)

        writer.write_evaluate_expression('%s\t%s\t%s' % (hit.thread_id, hit.frame_id, 'LOCAL'), 'a')
        writer.wait_for_evaluation('<var name="a" type="int" qualifier="{0}" value="int: 2"'.format(builtin_qualifier))
        writer.write_set_next_statement(hit.thread_id, 2, 'method')
        hit = writer.wait_for_breakpoint_hit('127', line=2)

        # Check that it's still unchanged
        writer.write_evaluate_expression('%s\t%s\t%s' % (hit.thread_id, hit.frame_id, 'LOCAL'), 'a')
        writer.wait_for_evaluation('<var name="a" type="int" qualifier="{0}" value="int: 2"'.format(builtin_qualifier))

        # After a step over it should become 1 as we executed line which sets a = 1
        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit('108')

        writer.write_evaluate_expression('%s\t%s\t%s' % (hit.thread_id, hit.frame_id, 'LOCAL'), 'a')
        writer.wait_for_evaluation('<var name="a" type="int" qualifier="{0}" value="int: 1"'.format(builtin_qualifier))

        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_unhandled_exceptions_get_stack(case_setup_unhandled_exceptions):

    with case_setup_unhandled_exceptions.test_file(
            '_debugger_case_unhandled_exception_get_stack.py',
            EXPECTED_RETURNCODE='any',
            ) as writer:

        writer.write_add_exception_breakpoint_with_policy('Exception', "0", "1", "0")
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        writer.write_get_thread_stack(hit.thread_id)

        msg = writer.wait_for_get_thread_stack_message()
        files = [frame['file'] for frame in  msg.thread.frame]
        assert msg.thread['id'] == hit.thread_id
        if not files[0].endswith('_debugger_case_unhandled_exception_get_stack.py'):
            raise AssertionError('Expected to find _debugger_case_unhandled_exception_get_stack.py in files[0]. Found: %s' % ('\n'.join(files),))

        assert len(msg.thread.frame) == 0  # No back frames (stopped in main).
        assert msg.thread.frame['name'] == '<module>'
        assert msg.thread.frame['line'] == str(writer.get_line_index_with_content('break line on unhandled exception'))

        writer.write_run_thread(hit.thread_id)

        writer.log.append('Marking finished ok.')
        writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Requires Python 3.')
def test_case_throw_exc_reason_xml(case_setup):

    def check_test_suceeded_msg(self, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        assert "raise RuntimeError('TEST SUCEEDED')" in stderr
        assert "raise RuntimeError from e" in stderr
        assert "raise Exception('another while handling')" in stderr

    with case_setup.test_file(
            '_debugger_case_raise_with_cause.py',
            EXPECTED_RETURNCODE=1,
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks
        ) as writer:

        writer.write_add_exception_breakpoint_with_policy('Exception', "0", "1", "0")
        writer.write_make_initial_run()

        el = writer.wait_for_curr_exc_stack()
        name_and_lines = []
        for frame in el.thread.frame:
            name_and_lines.append((frame['name'], frame['line']))

        assert name_and_lines == [
            ('foobar', '20'),
            ('<module>', '23'),
            ('[Chained Exc: another while handling] foobar', '18'),
            ('[Chained Exc: another while handling] handle', '10'),
            ('[Chained Exc: TEST SUCEEDED] foobar', '16'),
            ('[Chained Exc: TEST SUCEEDED] method', '6'),
            ('[Chained Exc: TEST SUCEEDED] method2', '2'),
        ]

        hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
        writer.write_get_thread_stack(hit.thread_id)

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='Only for Python.')
def test_case_get_next_statement_targets(case_setup):
    with case_setup.test_file('_debugger_case_get_next_statement_targets.py') as writer:
        breakpoint_id = writer.write_add_breakpoint(21, None)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT, line=21)

        writer.write_get_next_statement_targets(hit.thread_id, hit.frame_id)
        targets = writer.wait_for_get_next_statement_targets()
        # Note: 20 may appear as a side-effect of the frame eval
        # mode (so, we have to ignore it here) -- this isn't ideal, but
        # it's also not that bad (that line has no code in the source and
        # executing it will just set the tracing for the method).
        targets.discard(20)
        # On Python 3.11 there's now a line 1 (which should be harmless).
        targets.discard(1)
        expected = set((2, 3, 5, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21))
        assert targets == expected, 'Expected targets to be %s, was: %s' % (expected, targets)

        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_IRONPYTHON or IS_JYTHON, reason='Failing on IronPython and Jython (needs to be investigated).')
def test_case_type_ext(case_setup):
    # Custom type presentation extensions

    def get_environ(self):
        env = os.environ.copy()

        python_path = env.get("PYTHONPATH", "")
        ext_base = debugger_unittest._get_debugger_test_file('my_extensions')
        env['PYTHONPATH'] = ext_base + os.pathsep + python_path  if python_path else ext_base
        return env

    with case_setup.test_file('_debugger_case_type_ext.py', get_environ=get_environ) as writer:
        writer.get_environ = get_environ

        writer.write_add_breakpoint(7, None)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111')
        writer.write_get_frame(hit.thread_id, hit.frame_id)
        assert writer.wait_for_var([
            [
                r'<var name="my_rect" type="Rect" qualifier="__main__" value="Rectangle%255BLength%253A 5%252C Width%253A 10 %252C Area%253A 50%255D" isContainer="True" />',
                r'<var name="my_rect" type="Rect"  value="Rect: <__main__.Rect object at',  # Jython
            ]
        ])
        writer.write_get_variable(hit.thread_id, hit.frame_id, 'my_rect')
        assert writer.wait_for_var(r'<var name="area" type="int" qualifier="{0}" value="int%253A 50" />'.format(builtin_qualifier))
        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_case_variable_access(case_setup, pyfile, data_regression):

    @pyfile
    def case_custom():
        obj = [
            tuple(range(9)),
            [
                tuple(range(5)),
            ]
        ]

        print('TEST SUCEEDED')

    with case_setup.test_file(case_custom) as writer:
        line = writer.get_line_index_with_content('TEST SUCEEDED')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111')
        writer.write_get_frame(hit.thread_id, hit.frame_id)

        frame_vars = writer.wait_for_untangled_message(
            accept_message=lambda cmd_id, untangled: cmd_id == CMD_GET_FRAME)

        obj_var = [v for v in frame_vars.var if v['name'] == 'obj'][0]
        assert obj_var['type'] == 'list'
        assert unquote_plus(obj_var['value']) == "<class 'list'>: [(0, 1, 2, 3, 4, 5, 6, 7, 8), [(0, 1, 2, 3, 4)]]"
        assert obj_var['isContainer'] == "True"

        def _skip_key_in_dict(key):
            try:
                int(key)
            except ValueError:
                if 'more' in key or '[' in key:
                    return False
                return True
            return False

        def collect_vars(locator, level=0):
            writer.write("%s\t%s\t%s\t%s" % (CMD_GET_VARIABLE, writer.next_seq(), hit.thread_id, locator))
            obj_vars = writer.wait_for_untangled_message(
                accept_message=lambda cmd_id, _untangled: cmd_id == CMD_GET_VARIABLE)

            for v in obj_vars.var:
                if _skip_key_in_dict(v['name']):
                    continue
                new_locator = locator + '\t' + v['name']
                yield level, v, new_locator
                if v['isContainer'] == 'True':
                    yield from collect_vars(new_locator, level + 1)

        found = []
        for level, val, _locator in collect_vars('%s\tFRAME\tobj' % hit.frame_id):
            found.append((('    ' * level) + val['name'] + ': ' + unquote_plus(val['value'])))

        data_regression.check(found)

        # Check referrers
        full_loc = '%s\t%s\t%s' % (hit.thread_id, hit.frame_id, 'FRAME\tobj\t1\t0')
        writer.write_custom_operation(full_loc, 'EXEC', "from _pydevd_bundle.pydevd_referrers import get_referrer_info", "get_referrer_info")
        msg = writer.wait_for_untangled_message(
            double_unquote=True,
            accept_message=lambda cmd_id, _untangled: cmd_id == CMD_RUN_CUSTOM_OPERATION)

        msg_vars = msg.var
        try:
            msg_vars['found_as']
            msg_vars = [msg_vars]
        except:
            pass  # it's a container.

        for v in msg_vars:
            if v['found_as'] == 'list[0]':
                # In pypy we may have more than one reference, find out the one
                referrer_id = v['id']
                assert int(referrer_id)
                assert unquote_plus(v['value']) == "<class 'list'>: [(0, 1, 2, 3, 4)]"
                break
        else:
            raise AssertionError("Unable to find ref with list[0]. Found: %s" % (msg_vars,))

        found = []
        by_id_locator = '%s\t%s' % (referrer_id, 'BY_ID')
        for level, val, _locator in collect_vars(by_id_locator):
            found.append((('    ' * level) + val['name'] + ': ' + unquote_plus(val['value'])))

        data_regression.check(found, basename='test_case_variable_access_by_id')

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(IS_IRONPYTHON or IS_JYTHON, reason='Failing on IronPython and Jython (needs to be investigated).')
def test_case_event_ext(case_setup):

    def get_environ(self):
        env = os.environ.copy()

        python_path = env.get("PYTHONPATH", "")
        ext_base = debugger_unittest._get_debugger_test_file('my_extensions')
        env['PYTHONPATH'] = ext_base + os.pathsep + python_path  if python_path else ext_base
        env["VERIFY_EVENT_TEST"] = "1"
        return env

    # Test initialize event for extensions
    with case_setup.test_file('_debugger_case_event_ext.py', get_environ=get_environ) as writer:

        original_additional_output_checks = writer.additional_output_checks

        @overrides(writer.additional_output_checks)
        def additional_output_checks(stdout, stderr):
            original_additional_output_checks(stdout, stderr)
            if 'INITIALIZE EVENT RECEIVED' not in stdout:
                raise AssertionError('No initialize event received')

        writer.additional_output_checks = additional_output_checks

        writer.write_make_initial_run()
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Jython does not seem to be creating thread started inside tracing (investigate).')
def test_case_writer_creation_deadlock(case_setup):
    # check case where there was a deadlock evaluating expressions
    with case_setup.test_file('_debugger_case_thread_creation_deadlock.py') as writer:
        writer.write_add_breakpoint(26, None)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111')

        assert hit.line == 26, 'Expected return to be in line 26, was: %s' % (hit.line,)

        writer.write_evaluate_expression('%s\t%s\t%s' % (hit.thread_id, hit.frame_id, 'LOCAL'), 'create_thread()')
        writer.wait_for_evaluation('<var name="create_thread()" type="str" qualifier="{0}" value="str: create_thread:ok'.format(builtin_qualifier))
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_skip_breakpoints_in_exceptions(case_setup):
    # Case where breakpoint is skipped after an exception is raised over it
    with case_setup.test_file('_debugger_case_skip_breakpoint_in_exceptions.py') as writer:
        writer.write_add_breakpoint(5, None)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('111', line=5)
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit('111', line=5)
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_handled_exceptions0(case_setup):
    # Stop only once per handled exception.
    with case_setup.test_file('_debugger_case_exceptions.py') as writer:
        writer.write_set_project_roots([os.path.dirname(writer.TEST_FILE)])
        writer.write_add_exception_breakpoint_with_policy(
            'IndexError',
            notify_on_handled_exceptions=2,  # Notify only once
            notify_on_unhandled_exceptions=0,
            ignore_libraries=1
        )
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(
            REASON_CAUGHT_EXCEPTION,
            line=writer.get_line_index_with_content('raise indexerror line')
        )

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not working on Jython (needs to be investigated).')
def test_case_handled_exceptions1(case_setup):

    # Stop multiple times for the same handled exception.
    def get_environ(self):
        env = os.environ.copy()

        env["IDE_PROJECT_ROOTS"] = os.path.dirname(self.TEST_FILE)
        return env

    with case_setup.test_file('_debugger_case_exceptions.py', get_environ=get_environ) as writer:
        writer.write_add_exception_breakpoint_with_policy(
            'IndexError',
            notify_on_handled_exceptions=1,  # Notify multiple times
            notify_on_unhandled_exceptions=0,
            ignore_libraries=1
        )
        writer.write_make_initial_run()

        def check(hit):
            writer.write_get_frame(hit.thread_id, hit.frame_id)
            writer.wait_for_message(accept_message=lambda msg:'__exception__' in msg and 'IndexError' in msg, unquote_msg=False)
            writer.write_get_current_exception(hit.thread_id)
            msg = writer.wait_for_message(accept_message=lambda msg:'IndexError' in msg and 'exc_type="' in msg and 'exc_desc="' in msg, unquote_msg=False)
            assert msg.thread['exc_desc'] == 'foo'
            assert unquote(msg.thread['exc_type']) in (
                "&lt;type 'exceptions.IndexError'&gt;",  # py2
                "&lt;class 'IndexError'&gt;"  # py3
            )

            assert unquote(unquote(msg.thread.frame[0]['file'])).endswith('_debugger_case_exceptions.py')
            writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(
            REASON_CAUGHT_EXCEPTION, line=writer.get_line_index_with_content('raise indexerror line'))
        check(hit)

        hit = writer.wait_for_breakpoint_hit(
            REASON_CAUGHT_EXCEPTION, line=writer.get_line_index_with_content('reraise on method2'))
        check(hit)

        hit = writer.wait_for_breakpoint_hit(
            REASON_CAUGHT_EXCEPTION, line=writer.get_line_index_with_content('handle on method1'))
        check(hit)

        writer.finished_ok = True


def test_case_handled_exceptions2(case_setup):

    # No IDE_PROJECT_ROOTS set.
    def get_environ(self):
        env = os.environ.copy()

        # Don't stop anywhere (note: having IDE_PROJECT_ROOTS = '' will consider
        # having anything not under site-packages as being in the project).
        env["IDE_PROJECT_ROOTS"] = '["empty"]'
        return env

    with case_setup.test_file('_debugger_case_exceptions.py', get_environ=get_environ) as writer:
        writer.write_add_exception_breakpoint_with_policy(
            'IndexError',
            notify_on_handled_exceptions=1,  # Notify multiple times
            notify_on_unhandled_exceptions=0,
            ignore_libraries=1
        )
        writer.write_make_initial_run()

        writer.finished_ok = True


def test_case_handled_exceptions3(case_setup):

    # Don't stop on exception thrown in the same context (only at caller).
    def get_environ(self):
        env = os.environ.copy()

        env["IDE_PROJECT_ROOTS"] = os.path.dirname(self.TEST_FILE)
        return env

    with case_setup.test_file('_debugger_case_exceptions.py', get_environ=get_environ) as writer:
        # Note: in this mode we'll only stop once.
        writer.write_set_py_exception_globals(
            break_on_uncaught=False,
            break_on_caught=True,
            skip_on_exceptions_thrown_in_same_context=False,
            ignore_exceptions_thrown_in_lines_with_ignore_exception=True,
            ignore_libraries=True,
            exceptions=('IndexError',)
        )

        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit(
            REASON_CAUGHT_EXCEPTION, line=writer.get_line_index_with_content('raise indexerror line'))
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_handled_exceptions4(case_setup):

    # Don't stop on exception thrown in the same context (only at caller).
    def get_environ(self):
        env = os.environ.copy()

        env["IDE_PROJECT_ROOTS"] = os.path.dirname(self.TEST_FILE)
        return env

    with case_setup.test_file('_debugger_case_exceptions.py', get_environ=get_environ) as writer:
        # Note: in this mode we'll only stop once.
        writer.write_set_py_exception_globals(
            break_on_uncaught=False,
            break_on_caught=True,
            skip_on_exceptions_thrown_in_same_context=True,
            ignore_exceptions_thrown_in_lines_with_ignore_exception=True,
            ignore_libraries=True,
            exceptions=('IndexError',)
        )

        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit(
            REASON_CAUGHT_EXCEPTION, line=writer.get_line_index_with_content('reraise on method2'))
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_settrace(case_setup):
    with case_setup.test_file('_debugger_case_settrace.py') as writer:
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit('108', line=12)
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit(line=7)
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(True, reason='This is *very* flaky.')
def test_case_scapy(case_setup):
    with case_setup.test_file('_debugger_case_scapy.py') as writer:
        writer.FORCE_KILL_PROCESS_WHEN_FINISHED_OK = True
        writer.reader_thread.set_messages_timeout(30)  # Starting scapy may be slow (timed out with 15 seconds on appveyor).
        writer.write_add_breakpoint(2, None)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_run_thread(thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(IS_APPVEYOR or IS_JYTHON, reason='Flaky on appveyor / Jython encoding issues (needs investigation).')
def test_redirect_output(case_setup):

    def get_environ(writer):
        env = os.environ.copy()

        env["PYTHONIOENCODING"] = 'utf-8'
        return env

    with case_setup.test_file('_debugger_case_redirect.py', get_environ=get_environ) as writer:
        original_ignore_stderr_line = writer._ignore_stderr_line

        @overrides(writer._ignore_stderr_line)
        def _ignore_stderr_line(line):
            if original_ignore_stderr_line(line):
                return True

            binary_junk = b'\xe8\xF0\x80\x80\x80'
            if sys.version_info[0] >= 3:
                binary_junk = binary_junk.decode('utf-8', 'replace')

            return line.startswith((
                'text',
                'binary',
                'a',
                binary_junk,
            ))

        writer._ignore_stderr_line = _ignore_stderr_line

        # Note: writes to stdout and stderr are now synchronous (so, the order
        # must always be consistent and there's a message for each write).
        expected = [
            'text\n',
            'binary or text\n',
            'ação1\n',
        ]

        if sys.version_info[0] >= 3:
            expected.extend((
                'binary\n',
                'ação2\n'.encode(encoding='latin1').decode('utf-8', 'replace'),
                'ação3\n',
            ))

        binary_junk = '\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd\n\n'
        if sys.version_info[0] >= 3:
            binary_junk = "\ufffd\ufffd\ufffd\ufffd\ufffd\n\n"
        expected.append(binary_junk)

        new_expected = [(x, 'stdout') for x in expected]
        new_expected.extend([(x, 'stderr') for x in expected])

        writer.write_start_redirect()

        writer.write_make_initial_run()
        msgs = []
        ignored = []
        while len(msgs) < len(new_expected):
            try:
                msg = writer.wait_for_output()
            except AssertionError:
                for msg in msgs:
                    sys.stderr.write('Found: %s\n' % (msg,))
                for msg in new_expected:
                    sys.stderr.write('Expected: %s\n' % (msg,))
                for msg in ignored:
                    sys.stderr.write('Ignored: %s\n' % (msg,))
                raise
            if msg not in new_expected:
                ignored.append(msg)
                continue
            msgs.append(msg)

        if msgs != new_expected:
            print(msgs)
            print(new_expected)
        assert msgs == new_expected
        writer.finished_ok = True


def _path_equals(path1, path2):
    path1 = pydevd_file_utils.normcase(path1)
    path2 = pydevd_file_utils.normcase(path2)
    return path1 == path2


@pytest.mark.parametrize('mixed_case', [True, False] if sys.platform == 'win32' else [False])
def test_path_translation(case_setup, mixed_case):

    def get_file_in_client(writer):
        # Instead of using: test_python/_debugger_case_path_translation.py
        # we'll set the breakpoints at foo/_debugger_case_path_translation.py
        file_in_client = os.path.dirname(os.path.dirname(writer.TEST_FILE))
        return os.path.join(os.path.dirname(file_in_client), 'foo', '_debugger_case_path_translation.py')

    def get_environ(writer):
        import json
        env = os.environ.copy()

        env["PYTHONIOENCODING"] = 'utf-8'

        assert writer.TEST_FILE.endswith('_debugger_case_path_translation.py')
        file_in_client = get_file_in_client(writer)
        if mixed_case:
            new_file_in_client = ''.join([file_in_client[i].upper() if i % 2 == 0 else file_in_client[i].lower() for i in range(len(file_in_client))])
            assert _path_equals(file_in_client, new_file_in_client)
        env["PATHS_FROM_ECLIPSE_TO_PYTHON"] = json.dumps([
            (
                os.path.dirname(file_in_client),
                os.path.dirname(writer.TEST_FILE)
            )
        ])
        return env

    with case_setup.test_file('_debugger_case_path_translation.py', get_environ=get_environ) as writer:
        from tests_python.debugger_unittest import CMD_LOAD_SOURCE
        writer.write_start_redirect()

        file_in_client = get_file_in_client(writer)
        assert 'tests_python' not in file_in_client
        writer.write_add_breakpoint(
            writer.get_line_index_with_content('break here'), 'call_this', filename=file_in_client)
        writer.write_make_initial_run()

        xml = writer.wait_for_message(lambda msg:'stop_reason="111"' in msg)
        assert xml.thread.frame[0]['file'] == file_in_client
        thread_id = xml.thread['id']

        # Request a file that exists
        files_to_match = [file_in_client]
        if IS_WINDOWS:
            files_to_match.append(file_in_client.upper())
        for f in files_to_match:
            writer.write_load_source(f)
            writer.wait_for_message(
                lambda msg:
                    '%s\t' % CMD_LOAD_SOURCE in msg and \
                    "def main():" in msg and \
                    "print('break here')" in msg and \
                    "print('TEST SUCEEDED!')" in msg
                , expect_xml=False)

        # Request a file that does not exist
        writer.write_load_source(file_in_client + 'not_existent.py')
        writer.wait_for_message(
            lambda msg:'901\t' in msg and ('FileNotFoundError' in msg or 'IOError' in msg),
            expect_xml=False)

        writer.write_run_thread(thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_linecache_xml(case_setup, tmpdir):
    from _pydevd_bundle.pydevd_comm_constants import CMD_LOAD_SOURCE_FROM_FRAME_ID

    with case_setup.test_file('_debugger_case_linecache.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('breakpoint'))
        writer.write_make_initial_run()

        # First hit is for breakpoint reached via a stack frame that doesn't have source.
        hit = writer.wait_for_breakpoint_hit()

        writer.write_get_thread_stack(hit.thread_id)
        msg = writer.wait_for_get_thread_stack_message()
        frame_ids = set()
        for frame in  msg.thread.frame:
            if frame['file'] == '<foo bar>':
                frame_ids.add(frame['id'])

        assert len(frame_ids) == 2

        for frame_id in frame_ids:
            writer.write_load_source_from_frame_id(frame_id)
            writer.wait_for_message(
                lambda msg:
                    '%s\t' % CMD_LOAD_SOURCE_FROM_FRAME_ID in msg and (
                        "[x for x in range(10)]" in msg and "def somemethod():" in msg
                    )
                , expect_xml=False)

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_show_bytecode_xml(case_setup, tmpdir):
    from _pydevd_bundle.pydevd_comm_constants import CMD_LOAD_SOURCE_FROM_FRAME_ID

    with case_setup.test_file('_debugger_case_show_bytecode.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('breakpoint'))
        writer.write_make_initial_run()

        # First hit is for breakpoint reached via a stack frame that doesn't have source.
        hit = writer.wait_for_breakpoint_hit()

        writer.write_get_thread_stack(hit.thread_id)
        msg = writer.wait_for_get_thread_stack_message()
        frame_ids = set()
        for frame in  msg.thread.frame:
            if frame['file'] == '<something>':
                frame_ids.add(frame['id'])

        assert len(frame_ids) == 2

        for frame_id in frame_ids:
            writer.write_load_source_from_frame_id(frame_id)
            writer.wait_for_message(
                lambda msg:
                    '%s\t' % CMD_LOAD_SOURCE_FROM_FRAME_ID in msg and (
                        "MyClass" in msg or "foo()" in msg
                    )
                , expect_xml=False)

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_evaluate_errors(case_setup):
    with case_setup.test_file('_debugger_case_local_variables.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'), 'Call')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_evaluate_expression('%s\t%s\t%s' % (thread_id, frame_id, 'LOCAL'), 'name_error')
        writer.wait_for_evaluation('<var name="name_error" type="NameError"')
        writer.write_run_thread(thread_id)
        writer.finished_ok = True


def test_list_threads(case_setup):
    with case_setup.test_file('_debugger_case_local_variables.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'), 'Call')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        seq = writer.write_list_threads()
        msg = writer.wait_for_list_threads(seq)
        assert msg.thread['name'] == 'MainThread'
        assert msg.thread['id'].startswith('pid')
        writer.write_run_thread(thread_id)
        writer.finished_ok = True


def test_case_print(case_setup):
    with case_setup.test_file('_debugger_case_print.py') as writer:
        writer.write_add_breakpoint(1, 'None')
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()
        thread_id = hit.thread_id
        frame_id = hit.frame_id

        writer.write_run_thread(thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not working on Jython (needs to be investigated).')
def test_case_lamdda(case_setup):
    with case_setup.test_file('_debugger_case_lamda.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'), 'None')
        writer.write_make_initial_run()

        for _ in range(3):  # We'll hit the same breakpoint 3 times.
            hit = writer.wait_for_breakpoint_hit()

            writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not working on Jython (needs to be investigated).')
def test_case_lamdda_multiline(case_setup):
    with case_setup.test_file('_debugger_case_lambda_multiline.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'), 'None')
        writer.write_make_initial_run()

        for _ in range(2):  # We'll hit the same breakpoint 2 times.
            hit = writer.wait_for_breakpoint_hit()

            writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not working on Jython (needs to be investigated).')
def test_case_method_single_line(case_setup):
    with case_setup.test_file('_debugger_case_method_single_line.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('Break here'), 'None')
        writer.write_make_initial_run()

        for _ in range(3):
            # We'll hit the same breakpoint 3 times (method creation,
            # method line for each call).
            hit = writer.wait_for_breakpoint_hit()

            writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not working properly on Jython (needs investigation).')
def test_case_suspension_policy(case_setup):
    with case_setup.test_file('_debugger_case_suspend_policy.py') as writer:
        writer.write_add_breakpoint(25, '', suspend_policy='ALL')
        writer.write_make_initial_run()

        thread_ids = []
        for i in range(3):
            writer.log.append('Waiting for thread %s of 3 to stop' % (i + 1,))
            # One thread is suspended with a breakpoint hit and the other 2 as thread suspended.
            hit = writer.wait_for_breakpoint_hit((REASON_STOP_ON_BREAKPOINT, REASON_THREAD_SUSPEND))
            thread_ids.append(hit.thread_id)

        for thread_id in thread_ids:
            writer.write_run_thread(thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Flaky on Jython (needs investigation).')
def test_case_get_thread_stack(case_setup):
    with case_setup.test_file('_debugger_case_get_thread_stack.py') as writer:

        original_ignore_stderr_line = writer._ignore_stderr_line

        @overrides(writer._ignore_stderr_line)
        def _ignore_stderr_line(line):
            if original_ignore_stderr_line(line):
                return True

            if IS_JYTHON:
                for expected in (
                    "RuntimeWarning: Parent module '_pydev_bundle' not found while handling absolute import",
                    "from java.lang import System"):
                    if expected in line:
                        return True

            return False

        writer._ignore_stderr_line = _ignore_stderr_line
        writer.write_add_breakpoint(18, None)
        writer.write_make_initial_run()

        thread_created_msgs = [writer.wait_for_message(CMD_THREAD_CREATE)]
        thread_created_msgs.append(writer.wait_for_message(CMD_THREAD_CREATE))
        thread_id_to_name = {}
        for msg in thread_created_msgs:
            thread_id_to_name[msg.thread['id']] = msg.thread['name']
        assert len(thread_id_to_name) == 2

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)
        assert hit.thread_id in thread_id_to_name

        for request_thread_id in thread_id_to_name:
            writer.write_get_thread_stack(request_thread_id)
            msg = writer.wait_for_get_thread_stack_message()
            files = [frame['file'] for frame in  msg.thread.frame]
            assert msg.thread['id'] == request_thread_id
            if not files[0].endswith('_debugger_case_get_thread_stack.py'):
                raise AssertionError('Expected to find _debugger_case_get_thread_stack.py in files[0]. Found: %s' % ('\n'.join(files),))

            if ([filename for filename in files if filename.endswith('pydevd.py')]):
                raise AssertionError('Did not expect to find pydevd.py. Found: %s' % ('\n'.join(files),))
            if request_thread_id == hit.thread_id:
                assert len(msg.thread.frame) == 0  # In main thread (must have no back frames).
                assert msg.thread.frame['name'] == '<module>'
            else:
                assert len(msg.thread.frame) > 1  # Stopped in threading (must have back frames).
                assert msg.thread.frame[0]['name'] == 'method'

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_case_dump_threads_to_stderr(case_setup):

    from tests_python.debugger_unittest import wait_for_condition

    def additional_output_checks(writer, stdout, stderr):
        assert is_stderr_ok(stderr), make_error_msg(stderr)

    def make_error_msg(stderr):
        return 'Did not find thread dump in stderr. stderr:\n%s' % (stderr,)

    def is_stderr_ok(stderr):
        return 'Thread Dump' in stderr and 'Thread pydevd.CommandThread  (daemon: True, pydevd thread: True)' in stderr

    with case_setup.test_file(
        '_debugger_case_get_thread_stack.py', additional_output_checks=additional_output_checks) as writer:
        writer.write_add_breakpoint(12, None)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)

        writer.write_dump_threads()
        wait_for_condition(
            lambda: is_stderr_ok(writer.get_stderr()),
            lambda: make_error_msg(writer.get_stderr())
            )
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_stop_on_start_regular(case_setup):

    with case_setup.test_file('_debugger_case_simple_calls.py') as writer:
        writer.write_stop_on_start()
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_START, file='_debugger_case_simple_calls.py', line=1)

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def _get_breakpoint_cases():
    if sys.version_info >= (3, 7):
        # Just check breakpoint()
        return ('_debugger_case_breakpoint.py',)
    else:
        # Check breakpoint() and sys.__breakpointhook__ replacement.
        return ('_debugger_case_breakpoint.py', '_debugger_case_breakpoint2.py')


@pytest.mark.parametrize("filename", _get_breakpoint_cases())
def test_py_37_breakpoint(case_setup, filename):
    with case_setup.test_file(filename) as writer:
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(file=filename, line=3)

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def _get_generator_cases():
    # On py3 we should check both versions.
    return (
        '_debugger_case_generator_py2.py',
        '_debugger_case_generator_py3.py',
    )


@pytest.mark.parametrize("filename", _get_generator_cases())
def test_generator_cases(case_setup, filename):
    with case_setup.test_file(filename) as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_stop_on_start_m_switch(case_setup_m_switch):

    with case_setup_m_switch.test_file() as writer:
        writer.write_stop_on_start()
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_START, file='_debugger_case_m_switch.py', line=1)

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


def test_stop_on_start_entry_point(case_setup_m_switch_entry_point):

    with case_setup_m_switch_entry_point.test_file() as writer:
        writer.write_stop_on_start()
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_START, file='_debugger_case_module_entry_point.py', line=1)

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not working properly on Jython (needs investigation).')
def test_debug_zip_files(case_setup, tmpdir):

    def get_environ(writer):
        env = os.environ.copy()
        curr_pythonpath = env.get('PYTHONPATH', '')

        curr_pythonpath = str(tmpdir.join('myzip.zip')) + os.pathsep + curr_pythonpath
        curr_pythonpath = str(tmpdir.join('myzip2.egg!')) + os.pathsep + curr_pythonpath
        env['PYTHONPATH'] = curr_pythonpath

        env["IDE_PROJECT_ROOTS"] = str(tmpdir.join('myzip.zip'))
        return env

    import zipfile
    zip_file = zipfile.ZipFile(
        str(tmpdir.join('myzip.zip')), 'w')
    zip_file.writestr('zipped/__init__.py', '')
    zip_file.writestr('zipped/zipped_contents.py', 'def call_in_zip():\n    return 1')
    zip_file.close()

    zip_file = zipfile.ZipFile(
        str(tmpdir.join('myzip2.egg!')), 'w')
    zip_file.writestr('zipped2/__init__.py', '')
    zip_file.writestr('zipped2/zipped_contents2.py', 'def call_in_zip2():\n    return 1')
    zip_file.close()

    with case_setup.test_file('_debugger_case_zip_files.py', get_environ=get_environ) as writer:
        writer.write_add_breakpoint(
            2,
            'None',
            filename=os.path.join(str(tmpdir.join('myzip.zip')), 'zipped', 'zipped_contents.py')
        )

        writer.write_add_breakpoint(
            2,
            'None',
            filename=os.path.join(str(tmpdir.join('myzip2.egg!')), 'zipped2', 'zipped_contents2.py')
        )

        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit()
        assert hit.name == 'call_in_zip'
        writer.write_run_thread(hit.thread_id)

        hit = writer.wait_for_breakpoint_hit()
        assert hit.name == 'call_in_zip2'
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
@pytest.mark.parametrize('file_to_check', [
    '_debugger_case_multiprocessing_2.py',
    '_debugger_case_multiprocessing.py',
    '_debugger_case_python_c.py',
    '_debugger_case_multiprocessing_pool.py'
])
def test_multiprocessing_simple(case_setup_multiprocessing, file_to_check):
    import threading
    from tests_python.debugger_unittest import AbstractWriterThread
    with case_setup_multiprocessing.test_file(file_to_check) as writer:
        break1_line = writer.get_line_index_with_content('break 1 here')
        break2_line = writer.get_line_index_with_content('break 2 here')

        writer.write_add_breakpoint(break1_line)
        writer.write_add_breakpoint(break2_line)

        server_socket = writer.server_socket

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread
                expected_connections = 1

                for _ in range(expected_connections):
                    server_socket.listen(1)
                    self.server_socket = server_socket
                    new_sock, addr = server_socket.accept()

                    reader_thread = ReaderThread(new_sock)
                    reader_thread.name = '  *** Multiprocess Reader Thread'
                    reader_thread.start()

                    writer2 = SecondaryProcessWriterThread()

                    writer2.reader_thread = reader_thread
                    writer2.sock = new_sock

                    writer2.write_version()
                    writer2.write_add_breakpoint(break1_line)
                    writer2.write_add_breakpoint(break2_line)
                    writer2.write_make_initial_run()

                hit = writer2.wait_for_breakpoint_hit()
                writer2.write_run_thread(hit.thread_id)

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()
        writer.write_make_initial_run()
        hit2 = writer.wait_for_breakpoint_hit()
        secondary_process_thread_communication.join(10)
        if secondary_process_thread_communication.is_alive():
            raise AssertionError('The SecondaryProcessThreadCommunication did not finish')
        writer.write_run_thread(hit2.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
@pytest.mark.parametrize('count', range(5))  # Call multiple times to exercise timing issues.
def test_multiprocessing_with_stopped_breakpoints(case_setup_multiprocessing, count, debugger_runner_simple):
    import threading
    from tests_python.debugger_unittest import AbstractWriterThread
    with case_setup_multiprocessing.test_file('_debugger_case_multiprocessing_stopped_threads.py') as writer:
        break_main_line = writer.get_line_index_with_content('break in main here')
        break_thread_line = writer.get_line_index_with_content('break in thread here')
        break_process_line = writer.get_line_index_with_content('break in process here')

        writer.write_add_breakpoint(break_main_line)
        writer.write_add_breakpoint(break_thread_line)
        writer.write_add_breakpoint(break_process_line)

        server_socket = writer.server_socket
        listening_event = threading.Event()

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread
                server_socket.listen(1)
                self.server_socket = server_socket
                listening_event.set()
                writer.log.append('  *** Multiprocess waiting on server_socket.accept()')
                new_sock, addr = server_socket.accept()

                reader_thread = ReaderThread(new_sock)
                reader_thread.name = '  *** Multiprocess Reader Thread'
                reader_thread.start()
                writer.log.append('  *** Multiprocess started ReaderThread')

                writer2 = SecondaryProcessWriterThread()
                writer2._WRITE_LOG_PREFIX = '  *** Multiprocess write: '
                writer2.log = writer.log

                writer2.reader_thread = reader_thread
                writer2.sock = new_sock

                writer2.write_version()
                writer2.write_add_breakpoint(break_main_line)
                writer2.write_add_breakpoint(break_thread_line)
                writer2.write_add_breakpoint(break_process_line)
                writer2.write_make_initial_run()
                hit = writer2.wait_for_breakpoint_hit()
                writer2.write_run_thread(hit.thread_id)

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()

        ok = listening_event.wait(timeout=10)
        assert ok
        writer.write_make_initial_run()
        hit2 = writer.wait_for_breakpoint_hit()  # Breaks in thread.
        writer.write_step_over(hit2.thread_id)

        hit2 = writer.wait_for_breakpoint_hit(REASON_STEP_OVER)  # line == event.set()

        # paused on breakpoint, will start process and pause on main thread
        # in the main process too.
        writer.write_step_over(hit2.thread_id)

        # Note: ignore the step over hit (go only for the breakpoint hit).
        main_hit = writer.wait_for_breakpoint_hit(REASON_STOP_ON_BREAKPOINT)

        secondary_process_thread_communication.join(10)
        if secondary_process_thread_communication.is_alive():
            raise AssertionError('The SecondaryProcessThreadCommunication did not finish')

        writer.write_run_thread(hit2.thread_id)
        writer.write_run_thread(main_hit.thread_id)

        # We must have found at least 2 debug files when doing multiprocessing (one for
        # each pid).
        assert len(pydev_log.list_log_files(debugger_runner_simple.pydevd_debug_file)) == 2
        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
@pytest.mark.parametrize('target', [
    '_debugger_case_quoting.py',
    '_debugger_case_subprocess_zip.py'
])
def test_subprocess_quoted_args(case_setup_multiprocessing, target):
    from tests_python.debugger_unittest import AbstractWriterThread
    with case_setup_multiprocessing.test_file(target) as writer:
        break_subprocess_line = writer.get_line_index_with_content('break here')

        writer.write_add_breakpoint(break_subprocess_line)

        server_socket = writer.server_socket

        class SecondaryProcessWriterThread(AbstractWriterThread):

            TEST_FILE = writer.get_main_filename()
            _sequence = -1

        class SecondaryProcessThreadCommunication(threading.Thread):

            def run(self):
                from tests_python.debugger_unittest import ReaderThread
                # Note: on linux on Python 2 because on Python 2 CPython subprocess.call will actually
                # create a fork first (at which point it'll connect) and then, later on it'll
                # call the main (as if it was a clean process as if PyDB wasn't created
                # the first time -- the debugger will still work, but it'll do an additional
                # connection.

                expected_connections = 1

                for _ in range(expected_connections):
                    server_socket.listen(1)
                    self.server_socket = server_socket
                    new_sock, addr = server_socket.accept()

                    reader_thread = ReaderThread(new_sock)
                    reader_thread.name = '  *** Multiprocess Reader Thread'
                    reader_thread.start()

                    writer2 = SecondaryProcessWriterThread()

                    writer2.reader_thread = reader_thread
                    writer2.sock = new_sock

                    writer2.write_version()
                    writer2.write_add_breakpoint(break_subprocess_line)
                    writer2.write_make_initial_run()
                hit = writer2.wait_for_breakpoint_hit()
                writer2.write_run_thread(hit.thread_id)

        secondary_process_thread_communication = SecondaryProcessThreadCommunication()
        secondary_process_thread_communication.start()
        writer.write_make_initial_run()

        secondary_process_thread_communication.join(10)
        if secondary_process_thread_communication.is_alive():
            raise AssertionError('The SecondaryProcessThreadCommunication did not finish')

        writer.finished_ok = True


def _attach_to_writer_pid(writer):
    import pydevd
    assert writer.process is not None

    def attach():
        attach_pydevd_file = os.path.join(os.path.dirname(pydevd.__file__), 'pydevd_attach_to_process', 'attach_pydevd.py')
        subprocess.call([sys.executable, attach_pydevd_file, '--pid', str(writer.process.pid), '--port', str(writer.port)])

    threading.Thread(target=attach).start()

    wait_for_condition(lambda: writer.finished_initialization)


@pytest.mark.skipif(not IS_CPYTHON or IS_MAC, reason='CPython only test (brittle on Mac).')
@pytest.mark.parametrize('reattach', [True, False])
def test_attach_to_pid_no_threads(case_setup_remote, reattach):
    with case_setup_remote.test_file('_debugger_case_attach_to_pid_simple.py', wait_for_port=False) as writer:
        time.sleep(1)  # Give it some time to initialize to get to the while loop.
        _attach_to_writer_pid(writer)

        bp_line = writer.get_line_index_with_content('break here')
        bp_id = writer.write_add_breakpoint(bp_line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(line=bp_line)

        if reattach:
            # This would be the same as a second attach to pid, so, the idea is closing the current
            # connection and then doing a new attach to pid.
            writer.write_remove_breakpoint(bp_id)
            writer.write_run_thread(hit.thread_id)

            writer.do_kill()  # This will simply close the open sockets without doing anything else.
            time.sleep(1)

            t = threading.Thread(target=writer.start_socket)
            t.start()
            wait_for_condition(lambda: hasattr(writer, 'port'))
            time.sleep(1)
            writer.process = writer.process
            _attach_to_writer_pid(writer)
            wait_for_condition(lambda: hasattr(writer, 'reader_thread'))
            time.sleep(1)

            bp_id = writer.write_add_breakpoint(bp_line)
            writer.write_make_initial_run()

            hit = writer.wait_for_breakpoint_hit(line=bp_line)

        writer.write_change_variable(hit.thread_id, hit.frame_id, 'wait', 'False')
        writer.wait_for_var('<xml><var name="" type="bool"')

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON or IS_MAC, reason='CPython only test (brittle on Mac).')
def test_attach_to_pid_halted(case_setup_remote):
    with case_setup_remote.test_file('_debugger_case_attach_to_pid_multiple_threads.py', wait_for_port=False) as writer:
        time.sleep(1)  # Give it some time to initialize and get to the proper halting condition
        _attach_to_writer_pid(writer)

        bp_line = writer.get_line_index_with_content('break thread here')
        writer.write_add_breakpoint(bp_line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(line=bp_line)

        writer.write_change_variable(hit.thread_id, hit.frame_id, 'wait', 'False')
        writer.wait_for_var('<xml><var name="" type="bool"')

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_remote_debugger_basic(case_setup_remote):
    with case_setup_remote.test_file('_debugger_case_remote.py') as writer:
        writer.log.append('making initial run')
        writer.write_make_initial_run()

        writer.log.append('waiting for breakpoint hit')
        hit = writer.wait_for_breakpoint_hit()

        writer.log.append('run thread')
        writer.write_run_thread(hit.thread_id)

        writer.log.append('asserting')
        try:
            assert 5 == writer._sequence, 'Expected 5. Had: %s' % writer._sequence
        except:
            writer.log.append('assert failed!')
            raise
        writer.log.append('asserted')

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_remote_debugger_threads(case_setup_remote):
    with case_setup_remote.test_file('_debugger_case_remote_threads.py') as writer:
        writer.write_make_initial_run()

        hit_in_main = writer.wait_for_breakpoint_hit()

        bp_line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(bp_line)

        # Break in the 2 threads.
        hit_in_thread1 = writer.wait_for_breakpoint_hit(line=bp_line)
        hit_in_thread2 = writer.wait_for_breakpoint_hit(line=bp_line)

        writer.write_change_variable(hit_in_thread1.thread_id, hit_in_thread1.frame_id, 'wait', 'False')
        writer.wait_for_var('<xml><var name="" type="bool"')
        writer.write_change_variable(hit_in_thread2.thread_id, hit_in_thread2.frame_id, 'wait', 'False')
        writer.wait_for_var('<xml><var name="" type="bool"')

        writer.write_run_thread(hit_in_main.thread_id)
        writer.write_run_thread(hit_in_thread1.thread_id)
        writer.write_run_thread(hit_in_thread2.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_py_37_breakpoint_remote(case_setup_remote):
    with case_setup_remote.test_file('_debugger_case_breakpoint_remote.py') as writer:
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(
            filename='_debugger_case_breakpoint_remote.py',
            line=13,
        )

        writer.write_run_thread(hit.thread_id)

        try:
            assert 5 == writer._sequence, 'Expected 5. Had: %s' % writer._sequence
        except:
            writer.log.append('assert failed!')
            raise
        writer.log.append('asserted')

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_py_37_breakpoint_remote_no_import(case_setup_remote):

    def get_environ(writer):
        env = os.environ.copy()
        curr_pythonpath = env.get('PYTHONPATH', '')

        pydevd_dirname = os.path.join(
            os.path.dirname(writer.get_pydevd_file()),
            'pydev_sitecustomize')

        curr_pythonpath = pydevd_dirname + os.pathsep + curr_pythonpath
        env['PYTHONPATH'] = curr_pythonpath
        return env

    with case_setup_remote.test_file(
        '_debugger_case_breakpoint_remote_no_import.py',
        get_environ=get_environ) as writer:
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(
            "108",
            filename='_debugger_case_breakpoint_remote_no_import.py',
            line=12,
        )

        writer.write_run_thread(hit.thread_id)

        try:
            assert 5 == writer._sequence, 'Expected 5. Had: %s' % writer._sequence
        except:
            writer.log.append('assert failed!')
            raise
        writer.log.append('asserted')

        writer.finished_ok = True


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
@pytest.mark.parametrize('authenticate', [True, False])
def test_remote_debugger_multi_proc(case_setup_remote, authenticate):

    access_token = None
    client_access_token = None
    if authenticate:
        access_token = 'tok123'
        client_access_token = 'tok456'

    class _SecondaryMultiProcProcessWriterThread(debugger_unittest.AbstractWriterThread):

        FORCE_KILL_PROCESS_WHEN_FINISHED_OK = True

        def __init__(self, server_socket):
            debugger_unittest.AbstractWriterThread.__init__(self)
            self.server_socket = server_socket

        def run(self):
            print('waiting for second process')
            self.sock, addr = self.server_socket.accept()
            print('accepted second process')

            from tests_python.debugger_unittest import ReaderThread
            self.reader_thread = ReaderThread(self.sock)
            self.reader_thread.name = 'Secondary Reader Thread'
            self.reader_thread.start()

            self._sequence = -1
            # initial command is always the version
            self.write_version()

            if authenticate:
                self.wait_for_message(lambda msg:'Client not authenticated.' in msg, expect_xml=False)
                self.write_authenticate(access_token=access_token, client_access_token=client_access_token)
                self.write_version()

            self.log.append('start_socket')
            self.write_make_initial_run()
            time.sleep(.5)
            self.finished_ok = True

    def do_kill(writer):
        debugger_unittest.AbstractWriterThread.do_kill(writer)
        if hasattr(writer, 'secondary_multi_proc_process_writer'):
            writer.secondary_multi_proc_process_writer.do_kill()

    with case_setup_remote.test_file(
            '_debugger_case_remote_1.py',
            do_kill=do_kill,
            EXPECTED_RETURNCODE='any',
            access_token=access_token,
            client_access_token=client_access_token,
        ) as writer:

        # It seems sometimes it becomes flaky on the ci because the process outlives the writer thread...
        # As we're only interested in knowing if a second connection was received, just kill the related
        # process.
        assert hasattr(writer, 'FORCE_KILL_PROCESS_WHEN_FINISHED_OK')
        writer.FORCE_KILL_PROCESS_WHEN_FINISHED_OK = True

        writer.log.append('making initial run')
        writer.write_make_initial_run()

        if authenticate:
            writer.wait_for_message(lambda msg:'Client not authenticated.' in msg, expect_xml=False)
            writer.write_authenticate(access_token=access_token, client_access_token=client_access_token)
            writer.write_make_initial_run()

        writer.log.append('waiting for breakpoint hit')
        hit = writer.wait_for_breakpoint_hit()

        writer.secondary_multi_proc_process_writer = secondary_multi_proc_process_writer = \
            _SecondaryMultiProcProcessWriterThread(writer.server_socket)
        secondary_multi_proc_process_writer.start()

        writer.log.append('run thread')
        writer.write_run_thread(hit.thread_id)

        for _i in range(400):
            if secondary_multi_proc_process_writer.finished_ok:
                break
            time.sleep(.1)
        else:
            writer.log.append('Secondary process not finished ok!')
            raise AssertionError('Secondary process not finished ok!')

        writer.log.append('Secondary process finished!')
        try:
            assert writer._sequence == 5 if not authenticate else 9, 'Found: %s' % writer._sequence
        except:
            writer.log.append('assert failed!')
            raise
        writer.log.append('asserted')

        writer.finished_ok = True


@pytest.mark.parametrize('handle', [True, False])
@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
def test_remote_unhandled_exceptions(case_setup_remote, handle):

    def check_test_suceeded_msg(writer, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        # Don't call super as we have an expected exception
        assert 'ValueError: TEST SUCEEDED' in stderr

    with case_setup_remote.test_file(
        '_debugger_case_remote_unhandled_exceptions.py',
        additional_output_checks=additional_output_checks,
        check_test_suceeded_msg=check_test_suceeded_msg,
        EXPECTED_RETURNCODE=1) as writer:

        writer.log.append('making initial run')
        writer.write_make_initial_run()

        writer.log.append('waiting for breakpoint hit')
        hit = writer.wait_for_breakpoint_hit()

        # Add, remove and add back
        writer.write_add_exception_breakpoint_with_policy('Exception', '0', '1', '0')
        writer.write_remove_exception_breakpoint('Exception')
        writer.write_add_exception_breakpoint_with_policy('Exception', '0', '1', '0')

        if not handle:
            writer.write_remove_exception_breakpoint('Exception')

        writer.log.append('run thread')
        writer.write_run_thread(hit.thread_id)

        if handle:
            writer.log.append('waiting for uncaught exception')
            hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
            writer.write_run_thread(hit.thread_id)

        writer.log.append('finished ok')
        writer.finished_ok = True


def test_trace_dispatch_correct(case_setup):

    def get_environ(writer):
        env = os.environ.copy()
        env['PYDEVD_USE_FRAME_EVAL'] = 'NO'  # This test checks trace dispatch (so, disable frame eval).
        return env

    with case_setup.test_file('_debugger_case_trace_dispatch.py', get_environ=get_environ) as writer:
        breakpoint_id = writer.write_add_breakpoint(5, 'method')
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit()
        writer.write_remove_breakpoint(breakpoint_id)
        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_case_single_notification_on_step(case_setup):
    from tests_python.debugger_unittest import REASON_STEP_INTO
    with case_setup.test_file('_debugger_case_import_main.py') as writer:
        writer.write_multi_threads_single_notification(True)
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'), '')
        writer.write_make_initial_run()

        hit = writer.wait_for_single_notification_as_hit()

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_single_notification_as_hit(reason=REASON_STEP_INTO)

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_single_notification_as_hit(reason=REASON_STEP_INTO)

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_single_notification_as_hit(reason=REASON_STEP_INTO)

        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not ok for Jython.')
def test_reload(case_setup, tmpdir):

    path = tmpdir.join('my_temp.py')
    path.write('''
import my_temp2
assert my_temp2.call() == 1
a = 10 # break here
assert my_temp2.call() == 2
print('TEST SUCEEDED!')
''')

    path2 = tmpdir.join('my_temp2.py')
    path2.write('''
def call():
    return 1
''')
    with case_setup.test_file(str(path)) as writer:
        break_line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(break_line, '')
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit()

        path2 = tmpdir.join('my_temp2.py')
        path2.write('''
def call():
    return 2
''')

        writer.write_reload('my_temp2')
        output = writer.wait_for_output()
        output2 = writer.wait_for_output()
        output3 = writer.wait_for_output()

        assert output[0].startswith('code reload: Start reloading module: "my_temp2"')
        assert output2[0].startswith('code reload: Updated function code:')
        assert output3[0].startswith('code reload: reload finished')
        assert output[1] == 'stderr'
        assert output2[1] == 'stderr'
        assert output3[1] == 'stderr'

        writer.wait_for_message(CMD_RELOAD_CODE)
        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Not working with Jython on ci (needs investigation).')
def test_custom_frames(case_setup):
    with case_setup.test_file('_debugger_case_custom_frames.py') as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        for i in range(3):
            writer.write_step_over(hit.thread_id)

            # Check that the frame-related threads have been killed.
            for _ in range(i):
                writer.wait_for_message(CMD_THREAD_KILL, expect_xml=False)

            # Main thread stopped
            writer.wait_for_breakpoint_hit(REASON_STEP_OVER)

            # At each time we have an additional custom frame (which is shown as if it
            # was a thread which is created and then suspended).
            for _ in range(i):
                writer.wait_for_message(CMD_THREAD_CREATE)
                writer.wait_for_breakpoint_hit(REASON_THREAD_SUSPEND)

        writer.write_run_thread(hit.thread_id)

        # Check that the frame-related threads have been killed.
        for _ in range(i):
            try:
                writer.wait_for_message(CMD_THREAD_KILL, expect_xml=False, timeout=1)
            except debugger_unittest.TimeoutError:
                # Flaky: sometimes the thread kill is not received because
                # the process exists before the message is sent.
                break

        writer.finished_ok = True


@pytest.mark.skipif(not TEST_GEVENT, reason='Gevent not installed.')
def test_gevent(case_setup):

    def get_environ(writer):
        env = os.environ.copy()
        env['GEVENT_SUPPORT'] = 'True'
        return env

    with case_setup.test_file('_debugger_case_gevent.py', get_environ=get_environ) as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        for _i in range(10):
            hit = writer.wait_for_breakpoint_hit(name='run')
            writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(not TEST_GEVENT, reason='Gevent not installed.')
@pytest.mark.parametrize('show', [True, False])
def test_gevent_show_paused_greenlets(case_setup, show):

    def get_environ(writer):
        env = os.environ.copy()
        env['GEVENT_SUPPORT'] = 'True'
        if show:
            env['GEVENT_SHOW_PAUSED_GREENLETS'] = 'True'
        else:
            env['GEVENT_SHOW_PAUSED_GREENLETS'] = 'False'
        return env

    with case_setup.test_file('_debugger_case_gevent_simple.py', get_environ=get_environ) as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit(name='bar')
        writer.write_run_thread(hit.thread_id)

        seq = writer.write_list_threads()
        msg = writer.wait_for_list_threads(seq)

        if show:
            assert len(msg) > 1
        else:
            assert len(msg) == 1

        writer.finished_ok = True


@pytest.mark.skipif(not TEST_GEVENT, reason='Gevent not installed.')
def test_gevent_remote(case_setup_remote):

    def get_environ(writer):
        env = os.environ.copy()
        env['GEVENT_SUPPORT'] = 'True'
        return env

    with case_setup_remote.test_file('_debugger_case_gevent.py', get_environ=get_environ, append_command_line_args=['remote']) as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        for _i in range(10):
            hit = writer.wait_for_breakpoint_hit(name='run')
            writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_return_value(case_setup):
    with case_setup.test_file('_debugger_case_return_value.py') as writer:
        break_line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(break_line)
        writer.write_show_return_vars()
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit(name='main', line=break_line)

        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(REASON_STEP_OVER, name='main', line=break_line + 1)
        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_vars([
            [
                '<var name="method1" type="int" qualifier="%s" value="int: 1" isRetVal="True"' % (builtin_qualifier,),
                '<var name="method1" type="int"  value="int%253A 1" isRetVal="True"',
            ],
        ])

        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(REASON_STEP_OVER, name='main', line=break_line + 2)
        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_vars([
            [
                '<var name="method2" type="int" qualifier="%s" value="int: 2" isRetVal="True"' % (builtin_qualifier,),
                '<var name="method2" type="int"  value="int%253A 2" isRetVal="True"',
            ],
        ])

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_gettr_warning(case_setup):
    with case_setup.test_file('_debugger_case_warnings.py') as writer:
        break_line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(break_line)
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit(line=break_line)

        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_vars([
            [
                '<var name="obj'
            ],
        ])

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Jython can only have one thread stopped at each time.')
@pytest.mark.parametrize('check_single_notification', [True, False])
def test_run_pause_all_threads_single_notification(case_setup, check_single_notification):
    from tests_python.debugger_unittest import TimeoutError
    with case_setup.test_file('_debugger_case_multiple_threads.py') as writer:
        # : :type writer: AbstractWriterThread
        writer.write_multi_threads_single_notification(True)
        writer.write_make_initial_run()

        main_thread_id = writer.wait_for_new_thread()
        thread_id1 = writer.wait_for_new_thread()
        thread_id2 = writer.wait_for_new_thread()

        # Ok, all threads created, let's wait for the main thread to get to the join.
        writer.wait_for_thread_join(main_thread_id)

        writer.write_suspend_thread('*')

        if check_single_notification:
            dct = writer.wait_for_json_message(CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION)
            assert dct['thread_id'] in (thread_id1, thread_id2)
            assert dct['stop_reason'] == REASON_THREAD_SUSPEND
        else:
            # We should have a single thread suspended event for both threads.
            hit0 = writer.wait_for_breakpoint_hit(REASON_THREAD_SUSPEND)
            assert hit0.thread_id in (thread_id1, thread_id2)

            hit1 = writer.wait_for_breakpoint_hit(REASON_THREAD_SUSPEND)
            assert hit1.thread_id in (thread_id1, thread_id2)

            with pytest.raises(TimeoutError):
                # The main thread should not receive a hit as it's effectively deadlocked until other
                # threads finish.
                writer.wait_for_breakpoint_hit(REASON_THREAD_SUSPEND, timeout=1)

        # Doing a step in in one thread, when paused should notify on both threads.
        writer.write_step_over(thread_id1)

        if check_single_notification:
            dct = writer.wait_for_json_message(CMD_THREAD_RESUME_SINGLE_NOTIFICATION)  # Note: prefer wait_for_single_notification_as_hit
            assert dct['thread_id'] == thread_id1

            dct = writer.wait_for_json_message(CMD_THREAD_SUSPEND_SINGLE_NOTIFICATION)  # Note: prefer wait_for_single_notification_as_hit
            assert dct['thread_id'] == thread_id1
            assert dct['stop_reason'] == REASON_STEP_OVER

            hit = writer.get_current_stack_hit(thread_id1)

        else:
            hit = writer.wait_for_breakpoint_hit(CMD_STEP_OVER)

        writer.write_evaluate_expression('%s\t%s\t%s' % (hit.thread_id, hit.frame_id, 'LOCAL'), 'stop_loop()')
        writer.wait_for_evaluation('<var name="stop_loop()" type="str" qualifier="{0}" value="str: stopped_loop'.format(builtin_qualifier))

        writer.write_run_thread('*')
        writer.finished_ok = True


def scenario_uncaught(writer):
    hit = writer.wait_for_breakpoint_hit()
    writer.write_add_exception_breakpoint_with_policy('ValueError', '0', '1', '0')
    writer.write_run_thread(hit.thread_id)

    hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
    writer.write_run_thread(hit.thread_id)


def scenario_caught(writer):
    hit = writer.wait_for_breakpoint_hit()
    writer.write_add_exception_breakpoint_with_policy('ValueError', '1', '0', '0')
    writer.write_run_thread(hit.thread_id)

    for _ in range(2):
        hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

    # Note: the one in the top-level will be hit once as caught (but not another time
    # in postmortem mode).
    hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION)
    writer.write_run_thread(hit.thread_id)


def scenario_caught_and_uncaught(writer):
    hit = writer.wait_for_breakpoint_hit()
    writer.write_add_exception_breakpoint_with_policy('ValueError', '1', '1', '0')
    writer.write_run_thread(hit.thread_id)

    for _ in range(2):
        hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION)
        writer.write_run_thread(hit.thread_id)

    # Note: the one in the top-level will be hit once as caught and another in postmortem mode.
    hit = writer.wait_for_breakpoint_hit(REASON_CAUGHT_EXCEPTION)
    writer.write_run_thread(hit.thread_id)

    hit = writer.wait_for_breakpoint_hit(REASON_UNCAUGHT_EXCEPTION)
    writer.write_run_thread(hit.thread_id)


@pytest.mark.skipif(not IS_CPYTHON, reason='CPython only test.')
@pytest.mark.parametrize(
    'check_scenario',
    [
        scenario_uncaught,
        scenario_caught,
        scenario_caught_and_uncaught,
    ]
)
def test_top_level_exceptions_on_attach(case_setup_remote, check_scenario):

    def check_test_suceeded_msg(writer, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        # Don't call super as we have an expected exception
        assert 'ValueError: TEST SUCEEDED' in stderr

    with case_setup_remote.test_file(
        '_debugger_case_remote_unhandled_exceptions2.py',
        additional_output_checks=additional_output_checks,
        check_test_suceeded_msg=check_test_suceeded_msg,
        EXPECTED_RETURNCODE=1) as writer:

        writer.log.append('making initial run')
        writer.write_make_initial_run()

        check_scenario(writer)

        writer.log.append('finished ok')
        writer.finished_ok = True


@pytest.mark.parametrize('filename, break_at_lines', [
    ('_debugger_case_tracing.py', {2: 'frame_eval'}),

    ('_debugger_case_tracing.py', {3: 'frame_eval'}),
    ('_debugger_case_tracing.py', {4: 'frame_eval'}),
    ('_debugger_case_tracing.py', {2: 'frame_eval', 4: 'trace'}),

    ('_debugger_case_tracing.py', {8: 'frame_eval'}),
    ('_debugger_case_tracing.py', {9: 'frame_eval'}),
    ('_debugger_case_tracing.py', {10: 'frame_eval'}),

    # Note: second frame eval hit is actually a trace because after we
    # hit the first frame eval we don't actually stop tracing a given
    # frame (known limitation to be fixed in the future).
    # -- needs a better test
    ('_debugger_case_tracing.py', {8: 'frame_eval', 10: 'trace'}),
])
def test_frame_eval_limitations(case_setup, filename, break_at_lines):
    '''
    Test with limitations to be addressed in the future.
    '''
    with case_setup.test_file(filename) as writer:
        for break_at_line in break_at_lines:
            writer.write_add_breakpoint(break_at_line)

        writer.log.append('making initial run')
        writer.write_make_initial_run()

        for break_at_line, break_mode in break_at_lines.items():
            writer.log.append('waiting for breakpoint hit')
            hit = writer.wait_for_breakpoint_hit()
            thread_id = hit.thread_id

            if (IS_PY36_OR_GREATER and TEST_CYTHON) and not TODO_PY311:
                assert hit.suspend_type == break_mode
            else:
                # Before 3.6 frame eval is not available.
                assert hit.suspend_type == 'trace'

            writer.log.append('run thread')
            writer.write_run_thread(thread_id)

        writer.finished_ok = True


def test_step_return_my_code(case_setup):
    with case_setup.test_file('my_code/my_code.py') as writer:
        writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('my_code')])
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_in_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_INTO_MY_CODE)
        assert hit.name == 'callback1'

        writer.write_step_in_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_INTO_MY_CODE)
        assert hit.name == 'callback2'

        writer.write_step_return_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_RETURN_MY_CODE)
        assert hit.name == 'callback1'

        writer.write_step_return_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_RETURN_MY_CODE)
        assert hit.name == '<module>'

        writer.write_step_return_my_code(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(TODO_PY311, reason='Needs bytecode support in Python 3.11')
def test_smart_step_into_case1(case_setup):
    with case_setup.test_file('_debugger_case_smart_step_into.py') as writer:
        line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit(line=line)

        found = writer.get_step_into_variants(hit.thread_id, hit.frame_id, line, line)

        # Remove the offset/childOffset to compare (as it changes for each python version)
        assert [x[:-2] for x in found] == [
            ('bar', 'false', '14', '1'), ('foo', 'false', '14', '1'), ('call_outer', 'false', '14', '1')]

        # Note: this is just using the name, not really taking using the context.
        writer.write_smart_step_into(hit.thread_id, line, 'foo')
        hit = writer.wait_for_breakpoint_hit(reason=CMD_SMART_STEP_INTO)
        assert hit.line == writer.get_line_index_with_content('on foo mark')

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(TODO_PY311, reason='Needs bytecode support in Python 3.11')
def test_smart_step_into_case2(case_setup):
    with case_setup.test_file('_debugger_case_smart_step_into2.py') as writer:
        line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit(line=line)

        found = writer.get_step_into_variants(hit.thread_id, hit.frame_id, line, line)

        # Note: we have multiple 'foo' calls, so, we have to differentiate to
        # know in which one we want to stop.
        OFFSET_POS = 4
        writer.write_smart_step_into(hit.thread_id, 'offset=' + found[2][OFFSET_POS], 'foo')
        hit = writer.wait_for_breakpoint_hit(reason=CMD_SMART_STEP_INTO)
        assert hit.line == writer.get_line_index_with_content('on foo mark')

        writer.write_get_frame(hit.thread_id, hit.frame_id)
        writer.wait_for_var([
            (
                '<var name="arg" type="int" qualifier="__builtin__" value="int: 3"',
                '<var name="arg" type="int" qualifier="builtins" value="int: 3"',
            )
        ])

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(TODO_PY311, reason='Needs bytecode support in Python 3.11')
def test_smart_step_into_case3(case_setup):
    with case_setup.test_file('_debugger_case_smart_step_into3.py') as writer:
        line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit(line=line)

        found = writer.get_step_into_variants(hit.thread_id, hit.frame_id, 0, 9999)

        # Note: we have multiple 'foo' calls, so, we have to differentiate to
        # know in which one we want to stop.
        NAME_POS = 0
        OFFSET_POS = 4
        CHILD_OFFSET_POS = 5

        f = [x for x in found if x[NAME_POS] == 'foo']
        assert len(f) == 1

        writer.write_smart_step_into(hit.thread_id, 'offset=' + f[0][OFFSET_POS] + ';' + f[0][CHILD_OFFSET_POS], 'foo')
        hit = writer.wait_for_breakpoint_hit(reason=CMD_SMART_STEP_INTO)
        assert hit.line == writer.get_line_index_with_content('on foo mark')

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_step_over_my_code(case_setup):
    with case_setup.test_file('my_code/my_code.py') as writer:
        writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('my_code')])
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_in_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_INTO_MY_CODE)
        assert hit.name == 'callback1'

        writer.write_step_in_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_INTO_MY_CODE)
        assert hit.name == 'callback2'

        writer.write_step_over_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_OVER_MY_CODE)  # Note: goes from step over to step into
        assert hit.name == 'callback1'

        writer.write_step_over_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_OVER_MY_CODE)  # Note: goes from step over to step into
        assert hit.name == '<module>'

        writer.write_step_over_my_code(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_OVER_MY_CODE)
        assert hit.name == '<module>'

        writer.write_step_over_my_code(hit.thread_id)
        writer.finished_ok = True


@pytest.fixture(
    params=[
        'step_over',
        'step_return',
        'step_in',
    ]
)
def step_method(request):
    return request.param


def test_sysexit_on_filtered_file(case_setup):

    def get_environ(writer):
        env = os.environ.copy()
        env.update({'PYDEVD_FILTERS': json.dumps({'**/_debugger_case_sysexit.py': True})})
        return env

    with case_setup.test_file('_debugger_case_sysexit.py', get_environ=get_environ, EXPECTED_RETURNCODE=1) as writer:
        writer.write_add_exception_breakpoint_with_policy(
            'SystemExit',
            notify_on_handled_exceptions=1,  # Notify multiple times
            notify_on_unhandled_exceptions=1,
            ignore_libraries=0
        )

        writer.write_make_initial_run()
        writer.finished_ok = True


@pytest.mark.parametrize("scenario", [
    'handled_once',
    'handled_multiple',
    'unhandled',
])
def test_exception_not_on_filtered_file(case_setup, scenario):

    def get_environ(writer):
        env = os.environ.copy()
        env.update({'PYDEVD_FILTERS': json.dumps({'**/other.py': True})})
        return env

    def check_test_suceeded_msg(writer, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        if 'raise RuntimeError' not in stderr:
            raise AssertionError('Expected test to have an unhandled exception.\nstdout:\n%s\n\nstderr:\n%s' % (
                stdout, stderr))

    with case_setup.test_file(
            'my_code/my_code_exception.py',
            get_environ=get_environ,
            EXPECTED_RETURNCODE='any',
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks,
        ) as writer:

        if scenario == 'handled_once':
            writer.write_add_exception_breakpoint_with_policy(
                'RuntimeError',
                notify_on_handled_exceptions=2,  # Notify only once
                notify_on_unhandled_exceptions=0,
                ignore_libraries=0
            )
        elif scenario == 'handled_multiple':
            writer.write_add_exception_breakpoint_with_policy(
                'RuntimeError',
                notify_on_handled_exceptions=1,  # Notify multiple times
                notify_on_unhandled_exceptions=0,
                ignore_libraries=0
            )
        elif scenario == 'unhandled':
            writer.write_add_exception_breakpoint_with_policy(
                'RuntimeError',
                notify_on_handled_exceptions=0,
                notify_on_unhandled_exceptions=1,
                ignore_libraries=0
            )

        writer.write_make_initial_run()
        for _i in range(3 if scenario == 'handled_multiple' else 1):
            hit = writer.wait_for_breakpoint_hit(
                REASON_UNCAUGHT_EXCEPTION if scenario == 'unhandled' else REASON_CAUGHT_EXCEPTION)
            writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_exception_on_filtered_file(case_setup):

    def get_environ(writer):
        env = os.environ.copy()
        env.update({'PYDEVD_FILTERS': json.dumps({'**/other.py': True})})
        return env

    def check_test_suceeded_msg(writer, stdout, stderr):
        return 'TEST SUCEEDED' in ''.join(stderr)

    def additional_output_checks(writer, stdout, stderr):
        if 'raise RuntimeError' not in stderr:
            raise AssertionError('Expected test to have an unhandled exception.\nstdout:\n%s\n\nstderr:\n%s' % (
                stdout, stderr))

    with case_setup.test_file(
            'my_code/my_code_exception_on_other.py',
            get_environ=get_environ,
            EXPECTED_RETURNCODE='any',
            check_test_suceeded_msg=check_test_suceeded_msg,
            additional_output_checks=additional_output_checks,
        ) as writer:
        writer.write_add_exception_breakpoint_with_policy(
            'RuntimeError',
            notify_on_handled_exceptions=2,  # Notify only once
            notify_on_unhandled_exceptions=1,
            ignore_libraries=0
        )

        writer.write_make_initial_run()

        # Note: the unhandled exception was initially raised in a file which is filtered out, but we
        # should be able to see the frames which are part of the project.
        hit = writer.wait_for_breakpoint_hit(
            REASON_UNCAUGHT_EXCEPTION,
            file='my_code_exception_on_other.py',
            line=writer.get_line_index_with_content('other.raise_exception()')
        )
        writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


@pytest.mark.parametrize("environ", [
    {'PYDEVD_FILTER_LIBRARIES': '1'},  # Global setting for step over
    {'PYDEVD_FILTERS': json.dumps({'**/other.py': True})},  # specify as json
    {'PYDEVD_FILTERS': '**/other.py'},  # specify ';' separated list
])
@pytest.mark.skipif(IS_JYTHON, reason='Flaky on Jython.')
def test_step_over_my_code_global_settings(case_setup, environ, step_method):

    def get_environ(writer):
        env = os.environ.copy()
        env.update(environ)
        return env

    def do_step():
        if step_method == 'step_over':
            writer.write_step_over(hit.thread_id)
            return REASON_STEP_OVER  # Note: goes from step over to step into
        elif step_method == 'step_return':
            writer.write_step_return(hit.thread_id)
            return REASON_STEP_RETURN
        else:
            assert step_method == 'step_in'
            writer.write_step_in(hit.thread_id)
            return REASON_STEP_INTO

    with case_setup.test_file('my_code/my_code.py', get_environ=get_environ) as writer:
        writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('my_code')])
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_INTO)
        assert hit.name == 'callback1'

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_INTO)
        assert hit.name == 'callback2'

        stop_reason = do_step()
        hit = writer.wait_for_breakpoint_hit(reason=stop_reason)
        assert hit.name == 'callback1'

        stop_reason = do_step()
        hit = writer.wait_for_breakpoint_hit(reason=stop_reason)
        assert hit.name == '<module>'

        if IS_JYTHON:
            # Jython may get to exit functions, so, just resume the thread.
            writer.write_run_thread(hit.thread_id)

        else:
            stop_reason = do_step()

            if step_method != 'step_return':
                stop_reason = do_step()
                if step_method == 'step_over':
                    stop_reason = REASON_STEP_OVER

                hit = writer.wait_for_breakpoint_hit(reason=stop_reason)
                assert hit.name == '<module>'

                writer.write_step_over(hit.thread_id)

        writer.finished_ok = True


def test_step_over_my_code_global_setting_and_explicit_include(case_setup):

    def get_environ(writer):
        env = os.environ.copy()
        env.update({
            'PYDEVD_FILTER_LIBRARIES': '1',  # Global setting for in project or not
            # specify as json (force include).
            'PYDEVD_FILTERS': json.dumps({'**/other.py': False})
        })
        return env

    with case_setup.test_file('my_code/my_code.py', get_environ=get_environ) as writer:
        writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('my_code')])
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_INTO)

        # Although we filtered out non-project files, other.py is explicitly included.
        assert hit.name == 'call_me_back1'

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_access_token(case_setup):

    def update_command_line_args(self, args):
        i = args.index('--client')
        assert i > 0
        args.insert(i, '--access-token')
        args.insert(i + 1, 'bar123')
        args.insert(i, '--client-access-token')
        args.insert(i + 1, 'foo234')
        return args

    with case_setup.test_file('_debugger_case_print.py', update_command_line_args=update_command_line_args) as writer:
        writer.write_add_breakpoint(1, 'None')  # I.e.: should not work (not authenticated).

        writer.wait_for_message(lambda msg:'Client not authenticated.' in msg, expect_xml=False)

        writer.write_authenticate(access_token='bar123', client_access_token='foo234')

        writer.write_version()

        writer.write_make_initial_run()

        writer.finished_ok = True


def test_namedtuple(case_setup):
    '''
    Check that we don't step into <string> in the namedtuple constructor.
    '''
    with case_setup.test_file('_debugger_case_namedtuple.py') as writer:
        line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        expected_line = line

        for _ in range(2):
            expected_line += 1
            writer.write_step_in(hit.thread_id)
            hit = writer.wait_for_breakpoint_hit(
                reason=REASON_STEP_INTO, file='_debugger_case_namedtuple.py', line=expected_line)

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_matplotlib_activation(case_setup):
    try:
        import matplotlib
    except ImportError:
        return

    def get_environ(writer):
        env = os.environ.copy()
        env.update({
            'IPYTHONENABLE': 'True',
        })
        return env

    with case_setup.test_file('_debugger_case_matplotlib.py', get_environ=get_environ) as writer:
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        for _ in range(3):
            hit = writer.wait_for_breakpoint_hit()
            writer.write_run_thread(hit.thread_id)

        writer.finished_ok = True


_GENERATOR_FILES = [
    '_debugger_case_generator3.py',
    '_debugger_case_generator.py',
    '_debugger_case_generator2.py',
]


@pytest.mark.parametrize('target_filename', _GENERATOR_FILES)
@pytest.mark.skipif(IS_JYTHON, reason='We do not detect generator returns on Jython.')
def test_generator_step_over_basic(case_setup, target_filename):
    with case_setup.test_file(target_filename) as writer:
        line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        # Note: not using for so that we know which step failed in the ci if it fails.
        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_OVER,
            file=target_filename,
            line=writer.get_line_index_with_content('step 1')
        )

        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_OVER,
            file=target_filename,
            line=writer.get_line_index_with_content('step 2')
        )

        if IS_PY38_OR_GREATER and target_filename == '_debugger_case_generator2.py':
            # On py 3.8 it goes back to the return line.
            writer.write_step_over(hit.thread_id)
            hit = writer.wait_for_breakpoint_hit(
                reason=REASON_STEP_OVER,
                file=target_filename,
                line=writer.get_line_index_with_content('return \\')
            )

        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_OVER,
            file=target_filename,
            line=writer.get_line_index_with_content('step 3')
        )

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.parametrize('target_filename', _GENERATOR_FILES)
@pytest.mark.skipif(IS_JYTHON, reason='We do not detect generator returns on Jython.')
def test_generator_step_return(case_setup, target_filename):
    with case_setup.test_file(target_filename) as writer:
        line = writer.get_line_index_with_content('break here')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        # Note: not using for so that we know which step failed in the ci if it fails.
        writer.write_step_return(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_RETURN,
            file=target_filename,
            line=writer.get_line_index_with_content('generator return')
        )

        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_OVER,
            file=target_filename,
            line=writer.get_line_index_with_content('step 3')
        )

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
def test_stepin_not_my_code_coroutine(case_setup):

    def get_environ(writer):
        environ = {'PYDEVD_FILTERS': '{"**/not_my_coroutine.py": true}'}
        env = os.environ.copy()
        env.update(environ)
        return env

    with case_setup.test_file('my_code/my_code_coroutine.py', get_environ=get_environ) as writer:
        writer.write_set_project_roots([debugger_unittest._get_debugger_test_file('my_code')])
        writer.write_add_breakpoint(writer.get_line_index_with_content('break here'))
        writer.write_make_initial_run()
        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(reason=REASON_STEP_INTO)
        assert hit.name == 'main'

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.skipif(IS_JYTHON, reason='Flaky on Jython')
def test_generator_step_in(case_setup):
    with case_setup.test_file('_debugger_case_generator_step_in.py') as writer:
        line = writer.get_line_index_with_content('stop 1')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        for i in range(2, 5):
            writer.write_step_in(hit.thread_id)
            kwargs = {}
            if not IS_JYTHON:
                kwargs['line'] = writer.get_line_index_with_content('stop %s' % (i,))
            hit = writer.wait_for_breakpoint_hit(
                reason=REASON_STEP_INTO,
                file='_debugger_case_generator_step_in.py',
                **kwargs
            )

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.parametrize(
    'target_filename',
    [
        '_debugger_case_asyncio.py',
        '_debugger_case_trio.py',
    ]
)
@pytest.mark.skipif(not IS_CPYTHON or not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
def test_asyncio_step_over_basic(case_setup, target_filename):
    with case_setup.test_file(target_filename) as writer:
        line = writer.get_line_index_with_content('break main')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_OVER,
            file=target_filename,
            line=writer.get_line_index_with_content('step main')
        )

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.parametrize(
    'target_filename',
    [
        '_debugger_case_asyncio.py',
        '_debugger_case_trio.py',
    ]
)
@pytest.mark.skipif(not IS_CPYTHON or not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
def test_asyncio_step_over_end_of_function(case_setup, target_filename):
    with case_setup.test_file(target_filename) as writer:
        line = writer.get_line_index_with_content('break count 2')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_over(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_OVER,
            name=('sleep', 'wait_task_rescheduled'),
        )
        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.parametrize(
    'target_filename',
    [
        '_debugger_case_asyncio.py',
        '_debugger_case_trio.py',
    ]
)
@pytest.mark.skipif(not IS_CPYTHON or not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
def test_asyncio_step_in(case_setup, target_filename):
    with case_setup.test_file(target_filename) as writer:
        line = writer.get_line_index_with_content('break count 1')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_return(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_RETURN,
            file=target_filename,
            line=writer.get_line_index_with_content('break main')
        )

        writer.write_step_in(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_INTO,
            name=('sleep', 'wait_task_rescheduled'),
        )

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


@pytest.mark.parametrize(
    'target_filename',
    [
        '_debugger_case_asyncio.py',
        '_debugger_case_trio.py',
    ]
)
@pytest.mark.skipif(not IS_CPYTHON or not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
def test_asyncio_step_return(case_setup, target_filename):
    with case_setup.test_file(target_filename) as writer:
        line = writer.get_line_index_with_content('break count 1')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit()

        writer.write_step_return(hit.thread_id)
        hit = writer.wait_for_breakpoint_hit(
            reason=REASON_STEP_RETURN,
            file=target_filename,
            line=writer.get_line_index_with_content('break main')
        )

        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True


def test_notify_stdin(case_setup, pyfile):

    @pyfile
    def case_stdin():
        import sys
        print('Write something:')
        contents = sys.stdin.readline()
        print('Found: ' + contents)

        print('TEST SUCEEDED')

    def additional_output_checks(writer, stdout, stderr):
        assert 'Found: foo' in stdout

    with case_setup.test_file(
            case_stdin,
            additional_output_checks=additional_output_checks,
        ) as writer:
            writer.write_make_initial_run()
            msg = writer.wait_for_message(CMD_INPUT_REQUESTED, expect_xml=False)
            assert msg.split('\t')[-1] == 'True'
            process = writer.process
            process.stdin.write(b'foo\n')
            process.stdin.flush()
            msg = writer.wait_for_message(CMD_INPUT_REQUESTED, expect_xml=False)
            assert msg.split('\t')[-1] == 'False'

            writer.finished_ok = True


def test_frame_eval_mode_corner_case_01(case_setup):

    with case_setup.test_file(
            'wrong_bytecode/_debugger_case_wrong_bytecode.py',
        ) as writer:
            line = writer.get_line_index_with_content('break here')
            writer.write_add_breakpoint(line)
            writer.write_make_initial_run()
            hit = writer.wait_for_breakpoint_hit(line=writer.get_line_index_with_content('break here'), file='_debugger_case_wrong_bytecode.py')
            writer.write_step_over(hit.thread_id)

            hit = writer.wait_for_breakpoint_hit(line=writer.get_line_index_with_content('step 1'), file='_debugger_case_wrong_bytecode.py', reason=REASON_STEP_OVER)
            writer.write_step_over(hit.thread_id)

            hit = writer.wait_for_breakpoint_hit(line=writer.get_line_index_with_content('step 2'), file='_debugger_case_wrong_bytecode.py', reason=REASON_STEP_OVER)
            writer.write_step_over(hit.thread_id)

            hit = writer.wait_for_breakpoint_hit(line=writer.get_line_index_with_content('step 3'), file='_debugger_case_wrong_bytecode.py', reason=REASON_STEP_OVER)
            writer.write_step_over(hit.thread_id)

            hit = writer.wait_for_breakpoint_hit(line=writer.get_line_index_with_content('step 4'), file='_debugger_case_wrong_bytecode.py', reason=REASON_STEP_OVER)
            writer.write_step_over(hit.thread_id)

            hit = writer.wait_for_breakpoint_hit(line=writer.get_line_index_with_content('step 5'), file='_debugger_case_wrong_bytecode.py', reason=REASON_STEP_OVER)
            writer.write_run_thread(hit.thread_id)

            writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
def test_frame_eval_mode_corner_case_02(case_setup):

    with case_setup.test_file(
            '_bytecode_super.py',
        ) as writer:
            line = writer.get_line_index_with_content('break here')
            writer.write_add_breakpoint(line)
            writer.write_make_initial_run()

            hit = writer.wait_for_breakpoint_hit(line=line, file='_bytecode_super.py')

            writer.write_run_thread(hit.thread_id)

            writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
def test_frame_eval_mode_corner_case_03(case_setup):

    with case_setup.test_file(
            '_bytecode_constructs.py',
        ) as writer:
            line = writer.get_line_index_with_content('break while')
            writer.write_add_breakpoint(line)
            writer.write_make_initial_run()

            hit = writer.wait_for_breakpoint_hit(line=line)

            writer.write_step_over(hit.thread_id)
            hit = writer.wait_for_breakpoint_hit(line=line + 1, reason=REASON_STEP_OVER)

            writer.write_step_over(hit.thread_id)  # i.e.: check that the jump target is still ok.
            hit = writer.wait_for_breakpoint_hit(line=line, reason=REASON_STOP_ON_BREAKPOINT)

            writer.write_step_over(hit.thread_id)
            hit = writer.wait_for_breakpoint_hit(line=line + 1, reason=REASON_STEP_OVER)

            writer.write_step_over(hit.thread_id)
            hit = writer.wait_for_breakpoint_hit(line=line, reason=REASON_STOP_ON_BREAKPOINT)

            writer.write_run_thread(hit.thread_id)

            writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
def test_frame_eval_mode_corner_case_04(case_setup):

    with case_setup.test_file(
            '_bytecode_constructs.py',
        ) as writer:
            line = writer.get_line_index_with_content('break for')
            writer.write_add_breakpoint(line)
            writer.write_make_initial_run()

            hit = writer.wait_for_breakpoint_hit(line=line)
            writer.write_run_thread(hit.thread_id)

            hit = writer.wait_for_breakpoint_hit(line=line)
            writer.write_run_thread(hit.thread_id)

            hit = writer.wait_for_breakpoint_hit(line=line)
            writer.write_run_thread(hit.thread_id)

            writer.finished_ok = True


@pytest.mark.skipif(not IS_PY36_OR_GREATER, reason='Only CPython 3.6 onwards')
@pytest.mark.parametrize(
    'break_name',
    [
        'break except',
        'break with',
        'break try 1',
        'break try 2',
        'break finally 1',
        'break except 2',
        'break finally 2',
        'break finally 3',
        'break finally 4',
        'break in dict',
        'break else',
    ]
)
def test_frame_eval_mode_corner_case_many(case_setup, break_name):
    if break_name == 'break finally 4' and sys.version_info[:2] == (3, 9):
        # This case is currently failing in Python 3.9
        return

    # Check the constructs where we stop only once and proceed.
    with case_setup.test_file(
            '_bytecode_constructs.py',
        ) as writer:
            line = writer.get_line_index_with_content(break_name)
            writer.write_add_breakpoint(line)
            writer.write_make_initial_run()

            hit = writer.wait_for_breakpoint_hit(line=line)
            writer.write_run_thread(hit.thread_id)

            if break_name == 'break with':
                if sys.version_info[:2] >= (3, 10):
                    # On Python 3.10 it'll actually backtrack for the
                    # with and thus will execute the line where the
                    # 'with' statement was started again.
                    hit = writer.wait_for_breakpoint_hit(line=line)
                    writer.write_run_thread(hit.thread_id)

            writer.finished_ok = True


check_shadowed = [
    (
        u'''
if __name__ == '__main__':
    import queue
    print(queue)
''',
        'queue.py',
        u'shadowed = True\n'
    ),

    (
        u'''
if __name__ == '__main__':
    import queue
    print(queue)
''',
        'queue.py',
        u'raise AssertionError("error on import")'
    )
]


@pytest.mark.parametrize('module_name_and_content', check_shadowed)
def test_debugger_shadowed_imports(case_setup, tmpdir, module_name_and_content):
    main_content, module_name, content = module_name_and_content
    target = tmpdir.join('main.py')
    shadowed = tmpdir.join(module_name)

    target.write_text(main_content, encoding='utf-8')

    shadowed.write_text(content, encoding='utf-8')

    def get_environ(writer):
        env = os.environ.copy()
        env.update({
            'PYTHONPATH': str(tmpdir),
        })
        return env

    try:
        with case_setup.test_file(
                str(target),
                get_environ=get_environ,
                wait_for_initialization=False,
            ) as writer:
            writer.write_make_initial_run()
    except AssertionError:
        pass  # This is expected as pydevd didn't start-up.

    assert ('the module "%s" could not be imported because it is shadowed by:' % (module_name.split('.')[0])) in writer.get_stderr()


def test_debugger_hide_pydevd_threads(case_setup, pyfile):

    @pyfile
    def target_file():
        import threading
        from _pydevd_bundle import pydevd_constants
        found_pydevd_thread = False
        for t in threading.enumerate():
            if getattr(t, 'is_pydev_daemon_thread', False):
                found_pydevd_thread = True

        if pydevd_constants.IS_CPYTHON:
            assert not found_pydevd_thread
        else:
            assert found_pydevd_thread
        print('TEST SUCEEDED')

    with case_setup.test_file(target_file) as writer:
        line = writer.get_line_index_with_content('TEST SUCEEDED')
        writer.write_add_breakpoint(line)
        writer.write_make_initial_run()

        hit = writer.wait_for_breakpoint_hit(line=line)
        writer.write_run_thread(hit.thread_id)
        writer.finished_ok = True

# Jython needs some vars to be set locally.
# set JAVA_HOME=c:\bin\jdk1.8.0_172
# set PATH=%PATH%;C:\bin\jython2.7.0\bin
# set PATH=%PATH%;%JAVA_HOME%\bin
# c:\bin\jython2.7.0\bin\jython.exe -m py.test tests_python


if __name__ == '__main__':
    pytest.main(['-k', 'test_case_12'])
