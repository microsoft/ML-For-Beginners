from tests_python import debugger_unittest
import sys
import re
import os

CHECK_BASELINE, CHECK_REGULAR, CHECK_CYTHON, CHECK_FRAME_EVAL = 'baseline', 'regular', 'cython', 'frame_eval'

pytest_plugins = [
    str('tests_python.debugger_fixtures'),
]

RUNS = 5


class PerformanceWriterThread(debugger_unittest.AbstractWriterThread):

    CHECK = None

    debugger_unittest.AbstractWriterThread.get_environ  # overrides

    def get_environ(self):
        env = os.environ.copy()
        if self.CHECK == CHECK_BASELINE:
            env['PYTHONPATH'] = r'X:\PyDev.Debugger.baseline'

        elif self.CHECK == CHECK_CYTHON:
            env['PYDEVD_USE_CYTHON'] = 'YES'
            env['PYDEVD_USE_FRAME_EVAL'] = 'NO'

        elif self.CHECK == CHECK_FRAME_EVAL:
            env['PYDEVD_USE_CYTHON'] = 'YES'
            env['PYDEVD_USE_FRAME_EVAL'] = 'YES'

        elif self.CHECK == CHECK_REGULAR:
            env['PYDEVD_USE_CYTHON'] = 'NO'
            env['PYDEVD_USE_FRAME_EVAL'] = 'NO'

        else:
            raise AssertionError("Don't know what to check.")
        return env

    debugger_unittest.AbstractWriterThread.get_pydevd_file  # overrides

    def get_pydevd_file(self):
        if self.CHECK == CHECK_BASELINE:
            return os.path.abspath(os.path.join(r'X:\PyDev.Debugger.baseline', 'pydevd.py'))
        dirname = os.path.dirname(__file__)
        dirname = os.path.dirname(dirname)
        return os.path.abspath(os.path.join(dirname, 'pydevd.py'))


class CheckDebuggerPerformance(debugger_unittest.DebuggerRunner):

    def get_command_line(self):
        return [sys.executable]

    def _get_time_from_result(self, stdout):
        match = re.search(r'TotalTime>>((\d|\.)+)<<', stdout)
        time_taken = match.group(1)
        return float(time_taken)

    def obtain_results(self, benchmark_name, filename):

        class PerformanceCheck(PerformanceWriterThread):
            TEST_FILE = debugger_unittest._get_debugger_test_file(filename)
            BENCHMARK_NAME = benchmark_name

        writer_thread_class = PerformanceCheck

        runs = RUNS
        all_times = []
        for _ in range(runs):
            stdout_ref = []

            def store_stdout(stdout, stderr):
                stdout_ref.append(stdout)

            with self.check_case(writer_thread_class) as writer:
                writer.additional_output_checks = store_stdout
                yield writer

            assert len(stdout_ref) == 1
            all_times.append(self._get_time_from_result(stdout_ref[0]))
            print('partial for: %s: %.3fs' % (writer_thread_class.BENCHMARK_NAME, all_times[-1]))
        if len(all_times) > 3:
            all_times.remove(min(all_times))
            all_times.remove(max(all_times))
        time_when_debugged = sum(all_times) / float(len(all_times))

        args = self.get_command_line()
        args.append(writer_thread_class.TEST_FILE)
        # regular_time = self._get_time_from_result(self.run_process(args, writer_thread=None))
        # simple_trace_time = self._get_time_from_result(self.run_process(args+['--regular-trace'], writer_thread=None))

        if 'SPEEDTIN_AUTHORIZATION_KEY' in os.environ:

            SPEEDTIN_AUTHORIZATION_KEY = os.environ['SPEEDTIN_AUTHORIZATION_KEY']

            # sys.path.append(r'X:\speedtin\pyspeedtin')
            import pyspeedtin  # If the authorization key is there, pyspeedtin must be available
            import pydevd
            pydevd_cython_project_id, pydevd_pure_python_project_id = 6, 7
            if writer_thread_class.CHECK == CHECK_BASELINE:
                project_ids = (pydevd_cython_project_id, pydevd_pure_python_project_id)
            elif writer_thread_class.CHECK == CHECK_REGULAR:
                project_ids = (pydevd_pure_python_project_id,)
            elif writer_thread_class.CHECK == CHECK_CYTHON:
                project_ids = (pydevd_cython_project_id,)
            else:
                raise AssertionError('Wrong check: %s' % (writer_thread_class.CHECK))
            for project_id in project_ids:
                api = pyspeedtin.PySpeedTinApi(authorization_key=SPEEDTIN_AUTHORIZATION_KEY, project_id=project_id)

                benchmark_name = writer_thread_class.BENCHMARK_NAME

                if writer_thread_class.CHECK == CHECK_BASELINE:
                    version = '0.0.1_baseline'
                    return  # No longer commit the baseline (it's immutable right now).
                else:
                    version = pydevd.__version__,

                commit_id, branch, commit_date = api.git_commit_id_branch_and_date_from_path(pydevd.__file__)
                api.add_benchmark(benchmark_name)
                api.add_measurement(
                    benchmark_name,
                    value=time_when_debugged,
                    version=version,
                    released=False,
                    branch=branch,
                    commit_id=commit_id,
                    commit_date=commit_date,
                )
                api.commit()

        self.performance_msg = '%s: %.3fs ' % (writer_thread_class.BENCHMARK_NAME, time_when_debugged)

    def method_calls_with_breakpoint(self):
        for writer in self.obtain_results('method_calls_with_breakpoint', '_performance_1.py'):
            writer.write_add_breakpoint(17, 'method')
            writer.write_make_initial_run()
            writer.finished_ok = True

        return self.performance_msg

    def method_calls_without_breakpoint(self):
        for writer in self.obtain_results('method_calls_without_breakpoint', '_performance_1.py'):
            writer.write_make_initial_run()
            writer.finished_ok = True

        return self.performance_msg

    def method_calls_with_step_over(self):
        for writer in self.obtain_results('method_calls_with_step_over', '_performance_1.py'):
            writer.write_add_breakpoint(26, None)

            writer.write_make_initial_run()
            hit = writer.wait_for_breakpoint_hit('111')

            writer.write_step_over(hit.thread_id)
            hit = writer.wait_for_breakpoint_hit('108')

            writer.write_run_thread(hit.thread_id)
            writer.finished_ok = True

        return self.performance_msg

    def method_calls_with_exception_breakpoint(self):
        for writer in self.obtain_results('method_calls_with_exception_breakpoint', '_performance_1.py'):
            writer.write_add_exception_breakpoint('ValueError')
            writer.write_make_initial_run()
            writer.finished_ok = True

        return self.performance_msg

    def global_scope_1_with_breakpoint(self):
        for writer in self.obtain_results('global_scope_1_with_breakpoint', '_performance_2.py'):
            writer.write_add_breakpoint(writer.get_line_index_with_content('Breakpoint here'), None)
            writer.write_make_initial_run()
            writer.finished_ok = True

        return self.performance_msg

    def global_scope_2_with_breakpoint(self):
        for writer in self.obtain_results('global_scope_2_with_breakpoint', '_performance_3.py'):
            writer.write_add_breakpoint(17, None)
            writer.write_make_initial_run()
            writer.finished_ok = True

        return self.performance_msg


if __name__ == '__main__':
    # Local times gotten (python 3.6)
    # method_calls_with_breakpoint: 1.150s
    # method_calls_without_breakpoint: 0.240s
    # method_calls_with_step_over: 2.680s
    # method_calls_with_exception_breakpoint: 0.235s
    # global_scope_1_with_breakpoint: 4.249s
    # global_scope_2_with_breakpoint: 2.807s
    # Checking: cython
    # method_calls_with_breakpoint: 0.526s
    # method_calls_without_breakpoint: 0.130s
    # method_calls_with_step_over: 1.133s
    # method_calls_with_exception_breakpoint: 0.136s
    # global_scope_1_with_breakpoint: 1.827s
    # global_scope_2_with_breakpoint: 1.405s
    # Checking: frame_eval
    # method_calls_with_breakpoint: 0.133s
    # method_calls_without_breakpoint: 0.128s
    # method_calls_with_step_over: 0.130s
    # method_calls_with_exception_breakpoint: 0.125s
    # global_scope_1_with_breakpoint: 0.281s
    # global_scope_2_with_breakpoint: 0.169s
    # TotalTime for profile: 209.01s

    debugger_unittest.SHOW_WRITES_AND_READS = False
    debugger_unittest.SHOW_OTHER_DEBUG_INFO = False
    debugger_unittest.SHOW_STDOUT = False

    import time
    start_time = time.time()

    tmpdir = None

    msgs = []
    for check in (
            # CHECK_BASELINE, -- Checks against the version checked out at X:\PyDev.Debugger.baseline.
            CHECK_REGULAR,
            CHECK_CYTHON,
            CHECK_FRAME_EVAL,
        ):
        PerformanceWriterThread.CHECK = check
        msgs.append('Checking: %s' % (check,))
        check_debugger_performance = CheckDebuggerPerformance(tmpdir)
        msgs.append(check_debugger_performance.method_calls_with_breakpoint())
        msgs.append(check_debugger_performance.method_calls_without_breakpoint())
        msgs.append(check_debugger_performance.method_calls_with_step_over())
        msgs.append(check_debugger_performance.method_calls_with_exception_breakpoint())
        msgs.append(check_debugger_performance.global_scope_1_with_breakpoint())
        msgs.append(check_debugger_performance.global_scope_2_with_breakpoint())

    for msg in msgs:
        print(msg)

    print('TotalTime for profile: %.2fs' % (time.time() - start_time,))
