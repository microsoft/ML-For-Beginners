#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for psutil.Process class."""

import collections
import errno
import getpass
import itertools
import os
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import time
import types
import unittest

import psutil
from psutil import AIX
from psutil import BSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import open_text
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil._compat import super
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_THREADS
from psutil.tests import MACOS_11PLUS
from psutil.tests import PYPY
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import PsutilTestCase
from psutil.tests import ThreadTask
from psutil.tests import call_until
from psutil.tests import copyload_shared_lib
from psutil.tests import create_exe
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import reap_children
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import skip_on_access_denied
from psutil.tests import skip_on_not_implemented
from psutil.tests import wait_for_pid


# ===================================================================
# --- psutil.Process class tests
# ===================================================================


class TestProcess(PsutilTestCase):
    """Tests for psutil.Process class."""

    def spawn_psproc(self, *args, **kwargs):
        sproc = self.spawn_testproc(*args, **kwargs)
        return psutil.Process(sproc.pid)

    # ---

    def test_pid(self):
        p = psutil.Process()
        self.assertEqual(p.pid, os.getpid())
        with self.assertRaises(AttributeError):
            p.pid = 33

    def test_kill(self):
        p = self.spawn_psproc()
        p.kill()
        code = p.wait()
        if WINDOWS:
            self.assertEqual(code, signal.SIGTERM)
        else:
            self.assertEqual(code, -signal.SIGKILL)
        self.assertProcessGone(p)

    def test_terminate(self):
        p = self.spawn_psproc()
        p.terminate()
        code = p.wait()
        if WINDOWS:
            self.assertEqual(code, signal.SIGTERM)
        else:
            self.assertEqual(code, -signal.SIGTERM)
        self.assertProcessGone(p)

    def test_send_signal(self):
        sig = signal.SIGKILL if POSIX else signal.SIGTERM
        p = self.spawn_psproc()
        p.send_signal(sig)
        code = p.wait()
        if WINDOWS:
            self.assertEqual(code, sig)
        else:
            self.assertEqual(code, -sig)
        self.assertProcessGone(p)

    @unittest.skipIf(not POSIX, "not POSIX")
    def test_send_signal_mocked(self):
        sig = signal.SIGTERM
        p = self.spawn_psproc()
        with mock.patch('psutil.os.kill',
                        side_effect=OSError(errno.ESRCH, "")):
            self.assertRaises(psutil.NoSuchProcess, p.send_signal, sig)

        p = self.spawn_psproc()
        with mock.patch('psutil.os.kill',
                        side_effect=OSError(errno.EPERM, "")):
            self.assertRaises(psutil.AccessDenied, p.send_signal, sig)

    def test_wait_exited(self):
        # Test waitpid() + WIFEXITED -> WEXITSTATUS.
        # normal return, same as exit(0)
        cmd = [PYTHON_EXE, "-c", "pass"]
        p = self.spawn_psproc(cmd)
        code = p.wait()
        self.assertEqual(code, 0)
        self.assertProcessGone(p)
        # exit(1), implicit in case of error
        cmd = [PYTHON_EXE, "-c", "1 / 0"]
        p = self.spawn_psproc(cmd, stderr=subprocess.PIPE)
        code = p.wait()
        self.assertEqual(code, 1)
        self.assertProcessGone(p)
        # via sys.exit()
        cmd = [PYTHON_EXE, "-c", "import sys; sys.exit(5);"]
        p = self.spawn_psproc(cmd)
        code = p.wait()
        self.assertEqual(code, 5)
        self.assertProcessGone(p)
        # via os._exit()
        cmd = [PYTHON_EXE, "-c", "import os; os._exit(5);"]
        p = self.spawn_psproc(cmd)
        code = p.wait()
        self.assertEqual(code, 5)
        self.assertProcessGone(p)

    @unittest.skipIf(NETBSD, "fails on NETBSD")
    def test_wait_stopped(self):
        p = self.spawn_psproc()
        if POSIX:
            # Test waitpid() + WIFSTOPPED and WIFCONTINUED.
            # Note: if a process is stopped it ignores SIGTERM.
            p.send_signal(signal.SIGSTOP)
            self.assertRaises(psutil.TimeoutExpired, p.wait, timeout=0.001)
            p.send_signal(signal.SIGCONT)
            self.assertRaises(psutil.TimeoutExpired, p.wait, timeout=0.001)
            p.send_signal(signal.SIGTERM)
            self.assertEqual(p.wait(), -signal.SIGTERM)
            self.assertEqual(p.wait(), -signal.SIGTERM)
        else:
            p.suspend()
            self.assertRaises(psutil.TimeoutExpired, p.wait, timeout=0.001)
            p.resume()
            self.assertRaises(psutil.TimeoutExpired, p.wait, timeout=0.001)
            p.terminate()
            self.assertEqual(p.wait(), signal.SIGTERM)
            self.assertEqual(p.wait(), signal.SIGTERM)

    def test_wait_non_children(self):
        # Test wait() against a process which is not our direct
        # child.
        child, grandchild = self.spawn_children_pair()
        self.assertRaises(psutil.TimeoutExpired, child.wait, 0.01)
        self.assertRaises(psutil.TimeoutExpired, grandchild.wait, 0.01)
        # We also terminate the direct child otherwise the
        # grandchild will hang until the parent is gone.
        child.terminate()
        grandchild.terminate()
        child_ret = child.wait()
        grandchild_ret = grandchild.wait()
        if POSIX:
            self.assertEqual(child_ret, -signal.SIGTERM)
            # For processes which are not our children we're supposed
            # to get None.
            self.assertEqual(grandchild_ret, None)
        else:
            self.assertEqual(child_ret, signal.SIGTERM)
            self.assertEqual(child_ret, signal.SIGTERM)

    def test_wait_timeout(self):
        p = self.spawn_psproc()
        p.name()
        self.assertRaises(psutil.TimeoutExpired, p.wait, 0.01)
        self.assertRaises(psutil.TimeoutExpired, p.wait, 0)
        self.assertRaises(ValueError, p.wait, -1)

    def test_wait_timeout_nonblocking(self):
        p = self.spawn_psproc()
        self.assertRaises(psutil.TimeoutExpired, p.wait, 0)
        p.kill()
        stop_at = time.time() + GLOBAL_TIMEOUT
        while time.time() < stop_at:
            try:
                code = p.wait(0)
                break
            except psutil.TimeoutExpired:
                pass
        else:
            raise self.fail('timeout')
        if POSIX:
            self.assertEqual(code, -signal.SIGKILL)
        else:
            self.assertEqual(code, signal.SIGTERM)
        self.assertProcessGone(p)

    def test_cpu_percent(self):
        p = psutil.Process()
        p.cpu_percent(interval=0.001)
        p.cpu_percent(interval=0.001)
        for _ in range(100):
            percent = p.cpu_percent(interval=None)
            self.assertIsInstance(percent, float)
            self.assertGreaterEqual(percent, 0.0)
        with self.assertRaises(ValueError):
            p.cpu_percent(interval=-1)

    def test_cpu_percent_numcpus_none(self):
        # See: https://github.com/giampaolo/psutil/issues/1087
        with mock.patch('psutil.cpu_count', return_value=None) as m:
            psutil.Process().cpu_percent()
            assert m.called

    def test_cpu_times(self):
        times = psutil.Process().cpu_times()
        assert (times.user > 0.0) or (times.system > 0.0), times
        assert (times.children_user >= 0.0), times
        assert (times.children_system >= 0.0), times
        if LINUX:
            assert times.iowait >= 0.0, times
        # make sure returned values can be pretty printed with strftime
        for name in times._fields:
            time.strftime("%H:%M:%S", time.localtime(getattr(times, name)))

    def test_cpu_times_2(self):
        user_time, kernel_time = psutil.Process().cpu_times()[:2]
        utime, ktime = os.times()[:2]

        # Use os.times()[:2] as base values to compare our results
        # using a tolerance  of +/- 0.1 seconds.
        # It will fail if the difference between the values is > 0.1s.
        if (max([user_time, utime]) - min([user_time, utime])) > 0.1:
            raise self.fail("expected: %s, found: %s" % (utime, user_time))

        if (max([kernel_time, ktime]) - min([kernel_time, ktime])) > 0.1:
            raise self.fail("expected: %s, found: %s" % (ktime, kernel_time))

    @unittest.skipIf(not HAS_PROC_CPU_NUM, "not supported")
    def test_cpu_num(self):
        p = psutil.Process()
        num = p.cpu_num()
        self.assertGreaterEqual(num, 0)
        if psutil.cpu_count() == 1:
            self.assertEqual(num, 0)
        self.assertIn(p.cpu_num(), range(psutil.cpu_count()))

    def test_create_time(self):
        p = self.spawn_psproc()
        now = time.time()
        create_time = p.create_time()

        # Use time.time() as base value to compare our result using a
        # tolerance of +/- 1 second.
        # It will fail if the difference between the values is > 2s.
        difference = abs(create_time - now)
        if difference > 2:
            raise self.fail("expected: %s, found: %s, difference: %s"
                            % (now, create_time, difference))

        # make sure returned value can be pretty printed with strftime
        time.strftime("%Y %m %d %H:%M:%S", time.localtime(p.create_time()))

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_terminal(self):
        terminal = psutil.Process().terminal()
        if terminal is not None:
            tty = os.path.realpath(sh('tty'))
            self.assertEqual(terminal, tty)

    @unittest.skipIf(not HAS_PROC_IO_COUNTERS, 'not supported')
    @skip_on_not_implemented(only_if=LINUX)
    def test_io_counters(self):
        p = psutil.Process()
        # test reads
        io1 = p.io_counters()
        with open(PYTHON_EXE, 'rb') as f:
            f.read()
        io2 = p.io_counters()
        if not BSD and not AIX:
            self.assertGreater(io2.read_count, io1.read_count)
            self.assertEqual(io2.write_count, io1.write_count)
            if LINUX:
                self.assertGreater(io2.read_chars, io1.read_chars)
                self.assertEqual(io2.write_chars, io1.write_chars)
        else:
            self.assertGreaterEqual(io2.read_bytes, io1.read_bytes)
            self.assertGreaterEqual(io2.write_bytes, io1.write_bytes)

        # test writes
        io1 = p.io_counters()
        with open(self.get_testfn(), 'wb') as f:
            if PY3:
                f.write(bytes("x" * 1000000, 'ascii'))
            else:
                f.write("x" * 1000000)
        io2 = p.io_counters()
        self.assertGreaterEqual(io2.write_count, io1.write_count)
        self.assertGreaterEqual(io2.write_bytes, io1.write_bytes)
        self.assertGreaterEqual(io2.read_count, io1.read_count)
        self.assertGreaterEqual(io2.read_bytes, io1.read_bytes)
        if LINUX:
            self.assertGreater(io2.write_chars, io1.write_chars)
            self.assertGreaterEqual(io2.read_chars, io1.read_chars)

        # sanity check
        for i in range(len(io2)):
            if BSD and i >= 2:
                # On BSD read_bytes and write_bytes are always set to -1.
                continue
            self.assertGreaterEqual(io2[i], 0)
            self.assertGreaterEqual(io2[i], 0)

    @unittest.skipIf(not HAS_IONICE, "not supported")
    @unittest.skipIf(not LINUX, "linux only")
    def test_ionice_linux(self):
        p = psutil.Process()
        if not CI_TESTING:
            self.assertEqual(p.ionice()[0], psutil.IOPRIO_CLASS_NONE)
        self.assertEqual(psutil.IOPRIO_CLASS_NONE, 0)
        self.assertEqual(psutil.IOPRIO_CLASS_RT, 1)  # high
        self.assertEqual(psutil.IOPRIO_CLASS_BE, 2)  # normal
        self.assertEqual(psutil.IOPRIO_CLASS_IDLE, 3)  # low
        init = p.ionice()
        try:
            # low
            p.ionice(psutil.IOPRIO_CLASS_IDLE)
            self.assertEqual(tuple(p.ionice()), (psutil.IOPRIO_CLASS_IDLE, 0))
            with self.assertRaises(ValueError):  # accepts no value
                p.ionice(psutil.IOPRIO_CLASS_IDLE, value=7)
            # normal
            p.ionice(psutil.IOPRIO_CLASS_BE)
            self.assertEqual(tuple(p.ionice()), (psutil.IOPRIO_CLASS_BE, 0))
            p.ionice(psutil.IOPRIO_CLASS_BE, value=7)
            self.assertEqual(tuple(p.ionice()), (psutil.IOPRIO_CLASS_BE, 7))
            with self.assertRaises(ValueError):
                p.ionice(psutil.IOPRIO_CLASS_BE, value=8)
            try:
                p.ionice(psutil.IOPRIO_CLASS_RT, value=7)
            except psutil.AccessDenied:
                pass
            # errs
            self.assertRaisesRegex(
                ValueError, "ioclass accepts no value",
                p.ionice, psutil.IOPRIO_CLASS_NONE, 1)
            self.assertRaisesRegex(
                ValueError, "ioclass accepts no value",
                p.ionice, psutil.IOPRIO_CLASS_IDLE, 1)
            self.assertRaisesRegex(
                ValueError, "'ioclass' argument must be specified",
                p.ionice, value=1)
        finally:
            ioclass, value = init
            if ioclass == psutil.IOPRIO_CLASS_NONE:
                value = 0
            p.ionice(ioclass, value)

    @unittest.skipIf(not HAS_IONICE, "not supported")
    @unittest.skipIf(not WINDOWS, 'not supported on this win version')
    def test_ionice_win(self):
        p = psutil.Process()
        if not CI_TESTING:
            self.assertEqual(p.ionice(), psutil.IOPRIO_NORMAL)
        init = p.ionice()
        try:
            # base
            p.ionice(psutil.IOPRIO_VERYLOW)
            self.assertEqual(p.ionice(), psutil.IOPRIO_VERYLOW)
            p.ionice(psutil.IOPRIO_LOW)
            self.assertEqual(p.ionice(), psutil.IOPRIO_LOW)
            try:
                p.ionice(psutil.IOPRIO_HIGH)
            except psutil.AccessDenied:
                pass
            else:
                self.assertEqual(p.ionice(), psutil.IOPRIO_HIGH)
            # errs
            self.assertRaisesRegex(
                TypeError, "value argument not accepted on Windows",
                p.ionice, psutil.IOPRIO_NORMAL, value=1)
            self.assertRaisesRegex(
                ValueError, "is not a valid priority",
                p.ionice, psutil.IOPRIO_HIGH + 1)
        finally:
            p.ionice(init)

    @unittest.skipIf(not HAS_RLIMIT, "not supported")
    def test_rlimit_get(self):
        import resource
        p = psutil.Process(os.getpid())
        names = [x for x in dir(psutil) if x.startswith('RLIMIT')]
        assert names, names
        for name in names:
            value = getattr(psutil, name)
            self.assertGreaterEqual(value, 0)
            if name in dir(resource):
                self.assertEqual(value, getattr(resource, name))
                # XXX - On PyPy RLIMIT_INFINITY returned by
                # resource.getrlimit() is reported as a very big long
                # number instead of -1. It looks like a bug with PyPy.
                if PYPY:
                    continue
                self.assertEqual(p.rlimit(value), resource.getrlimit(value))
            else:
                ret = p.rlimit(value)
                self.assertEqual(len(ret), 2)
                self.assertGreaterEqual(ret[0], -1)
                self.assertGreaterEqual(ret[1], -1)

    @unittest.skipIf(not HAS_RLIMIT, "not supported")
    def test_rlimit_set(self):
        p = self.spawn_psproc()
        p.rlimit(psutil.RLIMIT_NOFILE, (5, 5))
        self.assertEqual(p.rlimit(psutil.RLIMIT_NOFILE), (5, 5))
        # If pid is 0 prlimit() applies to the calling process and
        # we don't want that.
        if LINUX:
            with self.assertRaisesRegex(ValueError, "can't use prlimit"):
                psutil._psplatform.Process(0).rlimit(0)
        with self.assertRaises(ValueError):
            p.rlimit(psutil.RLIMIT_NOFILE, (5, 5, 5))

    @unittest.skipIf(not HAS_RLIMIT, "not supported")
    def test_rlimit(self):
        p = psutil.Process()
        testfn = self.get_testfn()
        soft, hard = p.rlimit(psutil.RLIMIT_FSIZE)
        try:
            p.rlimit(psutil.RLIMIT_FSIZE, (1024, hard))
            with open(testfn, "wb") as f:
                f.write(b"X" * 1024)
            # write() or flush() doesn't always cause the exception
            # but close() will.
            with self.assertRaises(IOError) as exc:
                with open(testfn, "wb") as f:
                    f.write(b"X" * 1025)
            self.assertEqual(exc.exception.errno if PY3 else exc.exception[0],
                             errno.EFBIG)
        finally:
            p.rlimit(psutil.RLIMIT_FSIZE, (soft, hard))
            self.assertEqual(p.rlimit(psutil.RLIMIT_FSIZE), (soft, hard))

    @unittest.skipIf(not HAS_RLIMIT, "not supported")
    def test_rlimit_infinity(self):
        # First set a limit, then re-set it by specifying INFINITY
        # and assume we overridden the previous limit.
        p = psutil.Process()
        soft, hard = p.rlimit(psutil.RLIMIT_FSIZE)
        try:
            p.rlimit(psutil.RLIMIT_FSIZE, (1024, hard))
            p.rlimit(psutil.RLIMIT_FSIZE, (psutil.RLIM_INFINITY, hard))
            with open(self.get_testfn(), "wb") as f:
                f.write(b"X" * 2048)
        finally:
            p.rlimit(psutil.RLIMIT_FSIZE, (soft, hard))
            self.assertEqual(p.rlimit(psutil.RLIMIT_FSIZE), (soft, hard))

    @unittest.skipIf(not HAS_RLIMIT, "not supported")
    def test_rlimit_infinity_value(self):
        # RLIMIT_FSIZE should be RLIM_INFINITY, which will be a really
        # big number on a platform with large file support.  On these
        # platforms we need to test that the get/setrlimit functions
        # properly convert the number to a C long long and that the
        # conversion doesn't raise an error.
        p = psutil.Process()
        soft, hard = p.rlimit(psutil.RLIMIT_FSIZE)
        self.assertEqual(psutil.RLIM_INFINITY, hard)
        p.rlimit(psutil.RLIMIT_FSIZE, (soft, hard))

    def test_num_threads(self):
        # on certain platforms such as Linux we might test for exact
        # thread number, since we always have with 1 thread per process,
        # but this does not apply across all platforms (MACOS, Windows)
        p = psutil.Process()
        if OPENBSD:
            try:
                step1 = p.num_threads()
            except psutil.AccessDenied:
                raise unittest.SkipTest("on OpenBSD this requires root access")
        else:
            step1 = p.num_threads()

        with ThreadTask():
            step2 = p.num_threads()
            self.assertEqual(step2, step1 + 1)

    @unittest.skipIf(not WINDOWS, 'WINDOWS only')
    def test_num_handles(self):
        # a better test is done later into test/_windows.py
        p = psutil.Process()
        self.assertGreater(p.num_handles(), 0)

    @unittest.skipIf(not HAS_THREADS, 'not supported')
    def test_threads(self):
        p = psutil.Process()
        if OPENBSD:
            try:
                step1 = p.threads()
            except psutil.AccessDenied:
                raise unittest.SkipTest("on OpenBSD this requires root access")
        else:
            step1 = p.threads()

        with ThreadTask():
            step2 = p.threads()
            self.assertEqual(len(step2), len(step1) + 1)
            athread = step2[0]
            # test named tuple
            self.assertEqual(athread.id, athread[0])
            self.assertEqual(athread.user_time, athread[1])
            self.assertEqual(athread.system_time, athread[2])

    @retry_on_failure()
    @skip_on_access_denied(only_if=MACOS)
    @unittest.skipIf(not HAS_THREADS, 'not supported')
    def test_threads_2(self):
        p = self.spawn_psproc()
        if OPENBSD:
            try:
                p.threads()
            except psutil.AccessDenied:
                raise unittest.SkipTest(
                    "on OpenBSD this requires root access")
        self.assertAlmostEqual(
            p.cpu_times().user,
            sum([x.user_time for x in p.threads()]), delta=0.1)
        self.assertAlmostEqual(
            p.cpu_times().system,
            sum([x.system_time for x in p.threads()]), delta=0.1)

    @retry_on_failure()
    def test_memory_info(self):
        p = psutil.Process()

        # step 1 - get a base value to compare our results
        rss1, vms1 = p.memory_info()[:2]
        percent1 = p.memory_percent()
        self.assertGreater(rss1, 0)
        self.assertGreater(vms1, 0)

        # step 2 - allocate some memory
        memarr = [None] * 1500000

        rss2, vms2 = p.memory_info()[:2]
        percent2 = p.memory_percent()

        # step 3 - make sure that the memory usage bumped up
        self.assertGreater(rss2, rss1)
        self.assertGreaterEqual(vms2, vms1)  # vms might be equal
        self.assertGreater(percent2, percent1)
        del memarr

        if WINDOWS:
            mem = p.memory_info()
            self.assertEqual(mem.rss, mem.wset)
            self.assertEqual(mem.vms, mem.pagefile)

        mem = p.memory_info()
        for name in mem._fields:
            self.assertGreaterEqual(getattr(mem, name), 0)

    def test_memory_full_info(self):
        p = psutil.Process()
        total = psutil.virtual_memory().total
        mem = p.memory_full_info()
        for name in mem._fields:
            value = getattr(mem, name)
            self.assertGreaterEqual(value, 0, msg=(name, value))
            if name == 'vms' and OSX or LINUX:
                continue
            self.assertLessEqual(value, total, msg=(name, value, total))
        if LINUX or WINDOWS or MACOS:
            self.assertGreaterEqual(mem.uss, 0)
        if LINUX:
            self.assertGreaterEqual(mem.pss, 0)
            self.assertGreaterEqual(mem.swap, 0)

    @unittest.skipIf(not HAS_MEMORY_MAPS, "not supported")
    def test_memory_maps(self):
        p = psutil.Process()
        maps = p.memory_maps()
        self.assertEqual(len(maps), len(set(maps)))
        ext_maps = p.memory_maps(grouped=False)

        for nt in maps:
            if not nt.path.startswith('['):
                assert os.path.isabs(nt.path), nt.path
                if POSIX:
                    try:
                        assert os.path.exists(nt.path) or \
                            os.path.islink(nt.path), nt.path
                    except AssertionError:
                        if not LINUX:
                            raise
                        else:
                            # https://github.com/giampaolo/psutil/issues/759
                            with open_text('/proc/self/smaps') as f:
                                data = f.read()
                            if "%s (deleted)" % nt.path not in data:
                                raise
                else:
                    # XXX - On Windows we have this strange behavior with
                    # 64 bit dlls: they are visible via explorer but cannot
                    # be accessed via os.stat() (wtf?).
                    if '64' not in os.path.basename(nt.path):
                        try:
                            st = os.stat(nt.path)
                        except FileNotFoundError:
                            pass
                        else:
                            assert stat.S_ISREG(st.st_mode), nt.path
        for nt in ext_maps:
            for fname in nt._fields:
                value = getattr(nt, fname)
                if fname == 'path':
                    continue
                elif fname in ('addr', 'perms'):
                    assert value, value
                else:
                    self.assertIsInstance(value, (int, long))
                    assert value >= 0, value

    @unittest.skipIf(not HAS_MEMORY_MAPS, "not supported")
    def test_memory_maps_lists_lib(self):
        # Make sure a newly loaded shared lib is listed.
        p = psutil.Process()
        with copyload_shared_lib() as path:
            def normpath(p):
                return os.path.realpath(os.path.normcase(p))
            libpaths = [normpath(x.path)
                        for x in p.memory_maps()]
            self.assertIn(normpath(path), libpaths)

    def test_memory_percent(self):
        p = psutil.Process()
        p.memory_percent()
        self.assertRaises(ValueError, p.memory_percent, memtype="?!?")
        if LINUX or MACOS or WINDOWS:
            p.memory_percent(memtype='uss')

    def test_is_running(self):
        p = self.spawn_psproc()
        assert p.is_running()
        assert p.is_running()
        p.kill()
        p.wait()
        assert not p.is_running()
        assert not p.is_running()

    def test_exe(self):
        p = self.spawn_psproc()
        exe = p.exe()
        try:
            self.assertEqual(exe, PYTHON_EXE)
        except AssertionError:
            if WINDOWS and len(exe) == len(PYTHON_EXE):
                # on Windows we don't care about case sensitivity
                normcase = os.path.normcase
                self.assertEqual(normcase(exe), normcase(PYTHON_EXE))
            else:
                # certain platforms such as BSD are more accurate returning:
                # "/usr/local/bin/python2.7"
                # ...instead of:
                # "/usr/local/bin/python"
                # We do not want to consider this difference in accuracy
                # an error.
                ver = "%s.%s" % (sys.version_info[0], sys.version_info[1])
                try:
                    self.assertEqual(exe.replace(ver, ''),
                                     PYTHON_EXE.replace(ver, ''))
                except AssertionError:
                    # Typically MACOS. Really not sure what to do here.
                    pass

        out = sh([exe, "-c", "import os; print('hey')"])
        self.assertEqual(out, 'hey')

    def test_cmdline(self):
        cmdline = [PYTHON_EXE, "-c", "import time; time.sleep(60)"]
        p = self.spawn_psproc(cmdline)
        # XXX - most of the times the underlying sysctl() call on Net
        # and Open BSD returns a truncated string.
        # Also /proc/pid/cmdline behaves the same so it looks
        # like this is a kernel bug.
        # XXX - AIX truncates long arguments in /proc/pid/cmdline
        if NETBSD or OPENBSD or AIX:
            self.assertEqual(p.cmdline()[0], PYTHON_EXE)
        else:
            if MACOS and CI_TESTING:
                pyexe = p.cmdline()[0]
                if pyexe != PYTHON_EXE:
                    self.assertEqual(' '.join(p.cmdline()[1:]),
                                     ' '.join(cmdline[1:]))
                    return
            self.assertEqual(' '.join(p.cmdline()), ' '.join(cmdline))

    @unittest.skipIf(PYPY, "broken on PYPY")
    def test_long_cmdline(self):
        testfn = self.get_testfn()
        create_exe(testfn)
        cmdline = [testfn] + (["0123456789"] * 20)
        p = self.spawn_psproc(cmdline)
        if OPENBSD:
            # XXX: for some reason the test process may turn into a
            # zombie (don't know why).
            try:
                self.assertEqual(p.cmdline(), cmdline)
            except psutil.ZombieProcess:
                raise self.skipTest("OPENBSD: process turned into zombie")
        else:
            self.assertEqual(p.cmdline(), cmdline)

    def test_name(self):
        p = self.spawn_psproc(PYTHON_EXE)
        name = p.name().lower()
        pyexe = os.path.basename(os.path.realpath(sys.executable)).lower()
        assert pyexe.startswith(name), (pyexe, name)

    @unittest.skipIf(PYPY, "unreliable on PYPY")
    def test_long_name(self):
        testfn = self.get_testfn(suffix="0123456789" * 2)
        create_exe(testfn)
        p = self.spawn_psproc(testfn)
        if OPENBSD:
            # XXX: for some reason the test process may turn into a
            # zombie (don't know why). Because the name() is long, all
            # UNIX kernels truncate it to 15 chars, so internally psutil
            # tries to guess the full name() from the cmdline(). But the
            # cmdline() of a zombie on OpenBSD fails (internally), so we
            # just compare the first 15 chars. Full explanation:
            # https://github.com/giampaolo/psutil/issues/2239
            try:
                self.assertEqual(p.name(), os.path.basename(testfn))
            except AssertionError:
                if p.status() == psutil.STATUS_ZOMBIE:
                    assert os.path.basename(testfn).startswith(p.name())
                else:
                    raise
        else:
            self.assertEqual(p.name(), os.path.basename(testfn))

    # XXX
    @unittest.skipIf(SUNOS, "broken on SUNOS")
    @unittest.skipIf(AIX, "broken on AIX")
    @unittest.skipIf(PYPY, "broken on PYPY")
    def test_prog_w_funky_name(self):
        # Test that name(), exe() and cmdline() correctly handle programs
        # with funky chars such as spaces and ")", see:
        # https://github.com/giampaolo/psutil/issues/628
        funky_path = self.get_testfn(suffix='foo bar )')
        create_exe(funky_path)
        cmdline = [funky_path, "-c",
                   "import time; [time.sleep(0.01) for x in range(3000)];"
                   "arg1", "arg2", "", "arg3", ""]
        p = self.spawn_psproc(cmdline)
        self.assertEqual(p.cmdline(), cmdline)
        self.assertEqual(p.name(), os.path.basename(funky_path))
        self.assertEqual(os.path.normcase(p.exe()),
                         os.path.normcase(funky_path))

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_uids(self):
        p = psutil.Process()
        real, effective, saved = p.uids()
        # os.getuid() refers to "real" uid
        self.assertEqual(real, os.getuid())
        # os.geteuid() refers to "effective" uid
        self.assertEqual(effective, os.geteuid())
        # No such thing as os.getsuid() ("saved" uid), but starting
        # from python 2.7 we have os.getresuid() which returns all
        # of them.
        if hasattr(os, "getresuid"):
            self.assertEqual(os.getresuid(), p.uids())

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_gids(self):
        p = psutil.Process()
        real, effective, saved = p.gids()
        # os.getuid() refers to "real" uid
        self.assertEqual(real, os.getgid())
        # os.geteuid() refers to "effective" uid
        self.assertEqual(effective, os.getegid())
        # No such thing as os.getsgid() ("saved" gid), but starting
        # from python 2.7 we have os.getresgid() which returns all
        # of them.
        if hasattr(os, "getresuid"):
            self.assertEqual(os.getresgid(), p.gids())

    def test_nice(self):
        p = psutil.Process()
        self.assertRaises(TypeError, p.nice, "str")
        init = p.nice()
        try:
            if WINDOWS:
                # A CI runner may limit our maximum priority, which will break
                # this test. Instead, we test in order of increasing priority,
                # and match either the expected value or the highest so far.
                highest_prio = None
                for prio in [psutil.IDLE_PRIORITY_CLASS,
                             psutil.BELOW_NORMAL_PRIORITY_CLASS,
                             psutil.NORMAL_PRIORITY_CLASS,
                             psutil.ABOVE_NORMAL_PRIORITY_CLASS,
                             psutil.HIGH_PRIORITY_CLASS,
                             psutil.REALTIME_PRIORITY_CLASS]:
                    with self.subTest(prio=prio):
                        try:
                            p.nice(prio)
                        except psutil.AccessDenied:
                            pass
                        else:
                            new_prio = p.nice()
                            if CI_TESTING:
                                if new_prio == prio or highest_prio is None:
                                    highest_prio = prio
                                self.assertEqual(new_prio, highest_prio)
                            else:
                                self.assertEqual(new_prio, prio)
            else:
                try:
                    if hasattr(os, "getpriority"):
                        self.assertEqual(
                            os.getpriority(os.PRIO_PROCESS, os.getpid()),
                            p.nice())
                    p.nice(1)
                    self.assertEqual(p.nice(), 1)
                    if hasattr(os, "getpriority"):
                        self.assertEqual(
                            os.getpriority(os.PRIO_PROCESS, os.getpid()),
                            p.nice())
                    # XXX - going back to previous nice value raises
                    # AccessDenied on MACOS
                    if not MACOS:
                        p.nice(0)
                        self.assertEqual(p.nice(), 0)
                except psutil.AccessDenied:
                    pass
        finally:
            try:
                p.nice(init)
            except psutil.AccessDenied:
                pass

    def test_status(self):
        p = psutil.Process()
        self.assertEqual(p.status(), psutil.STATUS_RUNNING)

    def test_username(self):
        p = self.spawn_psproc()
        username = p.username()
        if WINDOWS:
            domain, username = username.split('\\')
            getpass_user = getpass.getuser()
            if getpass_user.endswith('$'):
                # When running as a service account (most likely to be
                # NetworkService), these user name calculations don't produce
                # the same result, causing the test to fail.
                raise unittest.SkipTest('running as service account')
            self.assertEqual(username, getpass_user)
            if 'USERDOMAIN' in os.environ:
                self.assertEqual(domain, os.environ['USERDOMAIN'])
        else:
            self.assertEqual(username, getpass.getuser())

    def test_cwd(self):
        p = self.spawn_psproc()
        self.assertEqual(p.cwd(), os.getcwd())

    def test_cwd_2(self):
        cmd = [PYTHON_EXE, "-c",
               "import os, time; os.chdir('..'); time.sleep(60)"]
        p = self.spawn_psproc(cmd)
        call_until(p.cwd, "ret == os.path.dirname(os.getcwd())")

    @unittest.skipIf(not HAS_CPU_AFFINITY, 'not supported')
    def test_cpu_affinity(self):
        p = psutil.Process()
        initial = p.cpu_affinity()
        assert initial, initial
        self.addCleanup(p.cpu_affinity, initial)

        if hasattr(os, "sched_getaffinity"):
            self.assertEqual(initial, list(os.sched_getaffinity(p.pid)))
        self.assertEqual(len(initial), len(set(initial)))

        all_cpus = list(range(len(psutil.cpu_percent(percpu=True))))
        for n in all_cpus:
            p.cpu_affinity([n])
            self.assertEqual(p.cpu_affinity(), [n])
            if hasattr(os, "sched_getaffinity"):
                self.assertEqual(p.cpu_affinity(),
                                 list(os.sched_getaffinity(p.pid)))
            # also test num_cpu()
            if hasattr(p, "num_cpu"):
                self.assertEqual(p.cpu_affinity()[0], p.num_cpu())

        # [] is an alias for "all eligible CPUs"; on Linux this may
        # not be equal to all available CPUs, see:
        # https://github.com/giampaolo/psutil/issues/956
        p.cpu_affinity([])
        if LINUX:
            self.assertEqual(p.cpu_affinity(), p._proc._get_eligible_cpus())
        else:
            self.assertEqual(p.cpu_affinity(), all_cpus)
        if hasattr(os, "sched_getaffinity"):
            self.assertEqual(p.cpu_affinity(),
                             list(os.sched_getaffinity(p.pid)))
        #
        self.assertRaises(TypeError, p.cpu_affinity, 1)
        p.cpu_affinity(initial)
        # it should work with all iterables, not only lists
        p.cpu_affinity(set(all_cpus))
        p.cpu_affinity(tuple(all_cpus))

    @unittest.skipIf(not HAS_CPU_AFFINITY, 'not supported')
    def test_cpu_affinity_errs(self):
        p = self.spawn_psproc()
        invalid_cpu = [len(psutil.cpu_times(percpu=True)) + 10]
        self.assertRaises(ValueError, p.cpu_affinity, invalid_cpu)
        self.assertRaises(ValueError, p.cpu_affinity, range(10000, 11000))
        self.assertRaises(TypeError, p.cpu_affinity, [0, "1"])
        self.assertRaises(ValueError, p.cpu_affinity, [0, -1])

    @unittest.skipIf(not HAS_CPU_AFFINITY, 'not supported')
    def test_cpu_affinity_all_combinations(self):
        p = psutil.Process()
        initial = p.cpu_affinity()
        assert initial, initial
        self.addCleanup(p.cpu_affinity, initial)

        # All possible CPU set combinations.
        if len(initial) > 12:
            initial = initial[:12]  # ...otherwise it will take forever
        combos = []
        for i in range(0, len(initial) + 1):
            for subset in itertools.combinations(initial, i):
                if subset:
                    combos.append(list(subset))

        for combo in combos:
            p.cpu_affinity(combo)
            self.assertEqual(sorted(p.cpu_affinity()), sorted(combo))

    # TODO: #595
    @unittest.skipIf(BSD, "broken on BSD")
    # can't find any process file on Appveyor
    @unittest.skipIf(APPVEYOR, "unreliable on APPVEYOR")
    def test_open_files(self):
        p = psutil.Process()
        testfn = self.get_testfn()
        files = p.open_files()
        self.assertNotIn(testfn, files)
        with open(testfn, 'wb') as f:
            f.write(b'x' * 1024)
            f.flush()
            # give the kernel some time to see the new file
            files = call_until(p.open_files, "len(ret) != %i" % len(files))
            filenames = [os.path.normcase(x.path) for x in files]
            self.assertIn(os.path.normcase(testfn), filenames)
            if LINUX:
                for file in files:
                    if file.path == testfn:
                        self.assertEqual(file.position, 1024)
        for file in files:
            assert os.path.isfile(file.path), file

        # another process
        cmdline = "import time; f = open(r'%s', 'r'); time.sleep(60);" % testfn
        p = self.spawn_psproc([PYTHON_EXE, "-c", cmdline])

        for x in range(100):
            filenames = [os.path.normcase(x.path) for x in p.open_files()]
            if testfn in filenames:
                break
            time.sleep(.01)
        else:
            self.assertIn(os.path.normcase(testfn), filenames)
        for file in filenames:
            assert os.path.isfile(file), file

    # TODO: #595
    @unittest.skipIf(BSD, "broken on BSD")
    # can't find any process file on Appveyor
    @unittest.skipIf(APPVEYOR, "unreliable on APPVEYOR")
    def test_open_files_2(self):
        # test fd and path fields
        p = psutil.Process()
        normcase = os.path.normcase
        testfn = self.get_testfn()
        with open(testfn, 'w') as fileobj:
            for file in p.open_files():
                if normcase(file.path) == normcase(fileobj.name) or \
                        file.fd == fileobj.fileno():
                    break
            else:
                raise self.fail("no file found; files=%s" % (
                                repr(p.open_files())))
            self.assertEqual(normcase(file.path), normcase(fileobj.name))
            if WINDOWS:
                self.assertEqual(file.fd, -1)
            else:
                self.assertEqual(file.fd, fileobj.fileno())
            # test positions
            ntuple = p.open_files()[0]
            self.assertEqual(ntuple[0], ntuple.path)
            self.assertEqual(ntuple[1], ntuple.fd)
            # test file is gone
            self.assertNotIn(fileobj.name, p.open_files())

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_num_fds(self):
        p = psutil.Process()
        testfn = self.get_testfn()
        start = p.num_fds()
        file = open(testfn, 'w')
        self.addCleanup(file.close)
        self.assertEqual(p.num_fds(), start + 1)
        sock = socket.socket()
        self.addCleanup(sock.close)
        self.assertEqual(p.num_fds(), start + 2)
        file.close()
        sock.close()
        self.assertEqual(p.num_fds(), start)

    @skip_on_not_implemented(only_if=LINUX)
    @unittest.skipIf(OPENBSD or NETBSD, "not reliable on OPENBSD & NETBSD")
    def test_num_ctx_switches(self):
        p = psutil.Process()
        before = sum(p.num_ctx_switches())
        for _ in range(500000):
            after = sum(p.num_ctx_switches())
            if after > before:
                return
        raise self.fail(
            "num ctx switches still the same after 50.000 iterations")

    def test_ppid(self):
        p = psutil.Process()
        if hasattr(os, 'getppid'):
            self.assertEqual(p.ppid(), os.getppid())
        p = self.spawn_psproc()
        self.assertEqual(p.ppid(), os.getpid())
        if APPVEYOR:
            # Occasional failures, see:
            # https://ci.appveyor.com/project/giampaolo/psutil/build/
            #     job/0hs623nenj7w4m33
            return

    def test_parent(self):
        p = self.spawn_psproc()
        self.assertEqual(p.parent().pid, os.getpid())

        lowest_pid = psutil.pids()[0]
        self.assertIsNone(psutil.Process(lowest_pid).parent())

    def test_parent_multi(self):
        parent = psutil.Process()
        child, grandchild = self.spawn_children_pair()
        self.assertEqual(grandchild.parent(), child)
        self.assertEqual(child.parent(), parent)

    def test_parent_disappeared(self):
        # Emulate a case where the parent process disappeared.
        p = self.spawn_psproc()
        with mock.patch("psutil.Process",
                        side_effect=psutil.NoSuchProcess(0, 'foo')):
            self.assertIsNone(p.parent())

    @retry_on_failure()
    def test_parents(self):
        parent = psutil.Process()
        assert parent.parents()
        child, grandchild = self.spawn_children_pair()
        self.assertEqual(child.parents()[0], parent)
        self.assertEqual(grandchild.parents()[0], child)
        self.assertEqual(grandchild.parents()[1], parent)

    def test_children(self):
        parent = psutil.Process()
        self.assertEqual(parent.children(), [])
        self.assertEqual(parent.children(recursive=True), [])
        # On Windows we set the flag to 0 in order to cancel out the
        # CREATE_NO_WINDOW flag (enabled by default) which creates
        # an extra "conhost.exe" child.
        child = self.spawn_psproc(creationflags=0)
        children1 = parent.children()
        children2 = parent.children(recursive=True)
        for children in (children1, children2):
            self.assertEqual(len(children), 1)
            self.assertEqual(children[0].pid, child.pid)
            self.assertEqual(children[0].ppid(), parent.pid)

    def test_children_recursive(self):
        # Test children() against two sub processes, p1 and p2, where
        # p1 (our child) spawned p2 (our grandchild).
        parent = psutil.Process()
        child, grandchild = self.spawn_children_pair()
        self.assertEqual(parent.children(), [child])
        self.assertEqual(parent.children(recursive=True), [child, grandchild])
        # If the intermediate process is gone there's no way for
        # children() to recursively find it.
        child.terminate()
        child.wait()
        self.assertEqual(parent.children(recursive=True), [])

    def test_children_duplicates(self):
        # find the process which has the highest number of children
        table = collections.defaultdict(int)
        for p in psutil.process_iter():
            try:
                table[p.ppid()] += 1
            except psutil.Error:
                pass
        # this is the one, now let's make sure there are no duplicates
        pid = sorted(table.items(), key=lambda x: x[1])[-1][0]
        if LINUX and pid == 0:
            raise self.skipTest("PID 0")
        p = psutil.Process(pid)
        try:
            c = p.children(recursive=True)
        except psutil.AccessDenied:  # windows
            pass
        else:
            self.assertEqual(len(c), len(set(c)))

    def test_parents_and_children(self):
        parent = psutil.Process()
        child, grandchild = self.spawn_children_pair()
        # forward
        children = parent.children(recursive=True)
        self.assertEqual(len(children), 2)
        self.assertEqual(children[0], child)
        self.assertEqual(children[1], grandchild)
        # backward
        parents = grandchild.parents()
        self.assertEqual(parents[0], child)
        self.assertEqual(parents[1], parent)

    def test_suspend_resume(self):
        p = self.spawn_psproc()
        p.suspend()
        for _ in range(100):
            if p.status() == psutil.STATUS_STOPPED:
                break
            time.sleep(0.01)
        p.resume()
        self.assertNotEqual(p.status(), psutil.STATUS_STOPPED)

    def test_invalid_pid(self):
        self.assertRaises(TypeError, psutil.Process, "1")
        self.assertRaises(ValueError, psutil.Process, -1)

    def test_as_dict(self):
        p = psutil.Process()
        d = p.as_dict(attrs=['exe', 'name'])
        self.assertEqual(sorted(d.keys()), ['exe', 'name'])

        p = psutil.Process(min(psutil.pids()))
        d = p.as_dict(attrs=['connections'], ad_value='foo')
        if not isinstance(d['connections'], list):
            self.assertEqual(d['connections'], 'foo')

        # Test ad_value is set on AccessDenied.
        with mock.patch('psutil.Process.nice', create=True,
                        side_effect=psutil.AccessDenied):
            self.assertEqual(
                p.as_dict(attrs=["nice"], ad_value=1), {"nice": 1})

        # Test that NoSuchProcess bubbles up.
        with mock.patch('psutil.Process.nice', create=True,
                        side_effect=psutil.NoSuchProcess(p.pid, "name")):
            self.assertRaises(
                psutil.NoSuchProcess, p.as_dict, attrs=["nice"])

        # Test that ZombieProcess is swallowed.
        with mock.patch('psutil.Process.nice', create=True,
                        side_effect=psutil.ZombieProcess(p.pid, "name")):
            self.assertEqual(
                p.as_dict(attrs=["nice"], ad_value="foo"), {"nice": "foo"})

        # By default APIs raising NotImplementedError are
        # supposed to be skipped.
        with mock.patch('psutil.Process.nice', create=True,
                        side_effect=NotImplementedError):
            d = p.as_dict()
            self.assertNotIn('nice', list(d.keys()))
            # ...unless the user explicitly asked for some attr.
            with self.assertRaises(NotImplementedError):
                p.as_dict(attrs=["nice"])

        # errors
        with self.assertRaises(TypeError):
            p.as_dict('name')
        with self.assertRaises(ValueError):
            p.as_dict(['foo'])
        with self.assertRaises(ValueError):
            p.as_dict(['foo', 'bar'])

    def test_oneshot(self):
        p = psutil.Process()
        with mock.patch("psutil._psplatform.Process.cpu_times") as m:
            with p.oneshot():
                p.cpu_times()
                p.cpu_times()
            self.assertEqual(m.call_count, 1)

        with mock.patch("psutil._psplatform.Process.cpu_times") as m:
            p.cpu_times()
            p.cpu_times()
        self.assertEqual(m.call_count, 2)

    def test_oneshot_twice(self):
        # Test the case where the ctx manager is __enter__ed twice.
        # The second __enter__ is supposed to resut in a NOOP.
        p = psutil.Process()
        with mock.patch("psutil._psplatform.Process.cpu_times") as m1:
            with mock.patch("psutil._psplatform.Process.oneshot_enter") as m2:
                with p.oneshot():
                    p.cpu_times()
                    p.cpu_times()
                    with p.oneshot():
                        p.cpu_times()
                        p.cpu_times()
                self.assertEqual(m1.call_count, 1)
                self.assertEqual(m2.call_count, 1)

        with mock.patch("psutil._psplatform.Process.cpu_times") as m:
            p.cpu_times()
            p.cpu_times()
        self.assertEqual(m.call_count, 2)

    def test_oneshot_cache(self):
        # Make sure oneshot() cache is nonglobal. Instead it's
        # supposed to be bound to the Process instance, see:
        # https://github.com/giampaolo/psutil/issues/1373
        p1, p2 = self.spawn_children_pair()
        p1_ppid = p1.ppid()
        p2_ppid = p2.ppid()
        self.assertNotEqual(p1_ppid, p2_ppid)
        with p1.oneshot():
            self.assertEqual(p1.ppid(), p1_ppid)
            self.assertEqual(p2.ppid(), p2_ppid)
        with p2.oneshot():
            self.assertEqual(p1.ppid(), p1_ppid)
            self.assertEqual(p2.ppid(), p2_ppid)

    def test_halfway_terminated_process(self):
        # Test that NoSuchProcess exception gets raised in case the
        # process dies after we create the Process object.
        # Example:
        # >>> proc = Process(1234)
        # >>> time.sleep(2)  # time-consuming task, process dies in meantime
        # >>> proc.name()
        # Refers to Issue #15
        def assert_raises_nsp(fun, fun_name):
            try:
                ret = fun()
            except psutil.ZombieProcess:  # differentiate from NSP
                raise
            except psutil.NoSuchProcess:
                pass
            except psutil.AccessDenied:
                if OPENBSD and fun_name in ('threads', 'num_threads'):
                    return
                raise
            else:
                # NtQuerySystemInformation succeeds even if process is gone.
                if WINDOWS and fun_name in ('exe', 'name'):
                    return
                raise self.fail("%r didn't raise NSP and returned %r "
                                "instead" % (fun, ret))

        p = self.spawn_psproc()
        p.terminate()
        p.wait()
        if WINDOWS:  # XXX
            call_until(psutil.pids, "%s not in ret" % p.pid)
        self.assertProcessGone(p)

        ns = process_namespace(p)
        for fun, name in ns.iter(ns.all):
            assert_raises_nsp(fun, name)

        # NtQuerySystemInformation succeeds even if process is gone.
        if WINDOWS and not GITHUB_ACTIONS:
            normcase = os.path.normcase
            self.assertEqual(normcase(p.exe()), normcase(PYTHON_EXE))

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_zombie_process(self):
        def succeed_or_zombie_p_exc(fun):
            try:
                return fun()
            except (psutil.ZombieProcess, psutil.AccessDenied):
                pass

        parent, zombie = self.spawn_zombie()
        # A zombie process should always be instantiable
        zproc = psutil.Process(zombie.pid)
        # ...and at least its status always be querable
        self.assertEqual(zproc.status(), psutil.STATUS_ZOMBIE)
        # ...and it should be considered 'running'
        assert zproc.is_running()
        # ...and as_dict() shouldn't crash
        zproc.as_dict()
        # ...its parent should 'see' it (edit: not true on BSD and MACOS
        # descendants = [x.pid for x in psutil.Process().children(
        #                recursive=True)]
        # self.assertIn(zpid, descendants)
        # XXX should we also assume ppid be usable?  Note: this
        # would be an important use case as the only way to get
        # rid of a zombie is to kill its parent.
        # self.assertEqual(zpid.ppid(), os.getpid())
        # ...and all other APIs should be able to deal with it

        ns = process_namespace(zproc)
        for fun, name in ns.iter(ns.all):
            succeed_or_zombie_p_exc(fun)

        assert psutil.pid_exists(zproc.pid)
        self.assertIn(zproc.pid, psutil.pids())
        self.assertIn(zproc.pid, [x.pid for x in psutil.process_iter()])
        psutil._pmap = {}
        self.assertIn(zproc.pid, [x.pid for x in psutil.process_iter()])

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_zombie_process_is_running_w_exc(self):
        # Emulate a case where internally is_running() raises
        # ZombieProcess.
        p = psutil.Process()
        with mock.patch("psutil.Process",
                        side_effect=psutil.ZombieProcess(0)) as m:
            assert p.is_running()
            assert m.called

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_zombie_process_status_w_exc(self):
        # Emulate a case where internally status() raises
        # ZombieProcess.
        p = psutil.Process()
        with mock.patch("psutil._psplatform.Process.status",
                        side_effect=psutil.ZombieProcess(0)) as m:
            self.assertEqual(p.status(), psutil.STATUS_ZOMBIE)
            assert m.called

    def test_reused_pid(self):
        # Emulate a case where PID has been reused by another process.
        subp = self.spawn_testproc()
        p = psutil.Process(subp.pid)
        p._ident = (p.pid, p.create_time() + 100)
        assert not p.is_running()
        assert p != psutil.Process(subp.pid)
        msg = "process no longer exists and its PID has been reused"
        self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.suspend)
        self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.resume)
        self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.terminate)
        self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.kill)
        self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.children)

    def test_pid_0(self):
        # Process(0) is supposed to work on all platforms except Linux
        if 0 not in psutil.pids():
            self.assertRaises(psutil.NoSuchProcess, psutil.Process, 0)
            # These 2 are a contradiction, but "ps" says PID 1's parent
            # is PID 0.
            assert not psutil.pid_exists(0)
            self.assertEqual(psutil.Process(1).ppid(), 0)
            return

        p = psutil.Process(0)
        exc = psutil.AccessDenied if WINDOWS else ValueError
        self.assertRaises(exc, p.wait)
        self.assertRaises(exc, p.terminate)
        self.assertRaises(exc, p.suspend)
        self.assertRaises(exc, p.resume)
        self.assertRaises(exc, p.kill)
        self.assertRaises(exc, p.send_signal, signal.SIGTERM)

        # test all methods
        ns = process_namespace(p)
        for fun, name in ns.iter(ns.getters + ns.setters):
            try:
                ret = fun()
            except psutil.AccessDenied:
                pass
            else:
                if name in ("uids", "gids"):
                    self.assertEqual(ret.real, 0)
                elif name == "username":
                    user = 'NT AUTHORITY\\SYSTEM' if WINDOWS else 'root'
                    self.assertEqual(p.username(), user)
                elif name == "name":
                    assert name, name

        if not OPENBSD:
            self.assertIn(0, psutil.pids())
            assert psutil.pid_exists(0)

    @unittest.skipIf(not HAS_ENVIRON, "not supported")
    def test_environ(self):
        def clean_dict(d):
            # Most of these are problematic on Travis.
            d.pop("PLAT", None)
            d.pop("HOME", None)
            if MACOS:
                d.pop("__CF_USER_TEXT_ENCODING", None)
                d.pop("VERSIONER_PYTHON_PREFER_32_BIT", None)
                d.pop("VERSIONER_PYTHON_VERSION", None)
            return dict(
                [(k.replace("\r", "").replace("\n", ""),
                  v.replace("\r", "").replace("\n", ""))
                 for k, v in d.items()])

        self.maxDiff = None
        p = psutil.Process()
        d1 = clean_dict(p.environ())
        d2 = clean_dict(os.environ.copy())
        if not OSX and GITHUB_ACTIONS:
            self.assertEqual(d1, d2)

    @unittest.skipIf(not HAS_ENVIRON, "not supported")
    @unittest.skipIf(not POSIX, "POSIX only")
    @unittest.skipIf(
        MACOS_11PLUS,
        "macOS 11+ can't get another process environment, issue #2084"
    )
    def test_weird_environ(self):
        # environment variables can contain values without an equals sign
        code = textwrap.dedent("""
            #include <unistd.h>
            #include <fcntl.h>

            char * const argv[] = {"cat", 0};
            char * const envp[] = {"A=1", "X", "C=3", 0};

            int main(void) {
                // Close stderr on exec so parent can wait for the
                // execve to finish.
                if (fcntl(2, F_SETFD, FD_CLOEXEC) != 0)
                    return 0;
                return execve("/bin/cat", argv, envp);
            }
            """)
        path = self.get_testfn()
        create_exe(path, c_code=code)
        sproc = self.spawn_testproc(
            [path], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        p = psutil.Process(sproc.pid)
        wait_for_pid(p.pid)
        assert p.is_running()
        # Wait for process to exec or exit.
        self.assertEqual(sproc.stderr.read(), b"")
        if MACOS and CI_TESTING:
            try:
                env = p.environ()
            except psutil.AccessDenied:
                # XXX: fails sometimes with:
                # PermissionError from 'sysctl(KERN_PROCARGS2) -> EIO'
                return
        else:
            env = p.environ()
        self.assertEqual(env, {"A": "1", "C": "3"})
        sproc.communicate()
        self.assertEqual(sproc.returncode, 0)


# ===================================================================
# --- Limited user tests
# ===================================================================


if POSIX and os.getuid() == 0:

    class LimitedUserTestCase(TestProcess):
        """Repeat the previous tests by using a limited user.
        Executed only on UNIX and only if the user who run the test script
        is root.
        """
        # the uid/gid the test suite runs under
        if hasattr(os, 'getuid'):
            PROCESS_UID = os.getuid()
            PROCESS_GID = os.getgid()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # re-define all existent test methods in order to
            # ignore AccessDenied exceptions
            for attr in [x for x in dir(self) if x.startswith('test')]:
                meth = getattr(self, attr)

                def test_(self):
                    try:
                        meth()  # noqa
                    except psutil.AccessDenied:
                        pass
                setattr(self, attr, types.MethodType(test_, self))

        def setUp(self):
            super().setUp()
            os.setegid(1000)
            os.seteuid(1000)

        def tearDown(self):
            os.setegid(self.PROCESS_UID)
            os.seteuid(self.PROCESS_GID)
            super().tearDown()

        def test_nice(self):
            try:
                psutil.Process().nice(-1)
            except psutil.AccessDenied:
                pass
            else:
                raise self.fail("exception not raised")

        @unittest.skipIf(1, "causes problem as root")
        def test_zombie_process(self):
            pass


# ===================================================================
# --- psutil.Popen tests
# ===================================================================


class TestPopen(PsutilTestCase):
    """Tests for psutil.Popen class."""

    @classmethod
    def tearDownClass(cls):
        reap_children()

    def test_misc(self):
        # XXX this test causes a ResourceWarning on Python 3 because
        # psutil.__subproc instance doesn't get properly freed.
        # Not sure what to do though.
        cmd = [PYTHON_EXE, "-c", "import time; time.sleep(60);"]
        with psutil.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=PYTHON_EXE_ENV) as proc:
            proc.name()
            proc.cpu_times()
            proc.stdin
            self.assertTrue(dir(proc))
            self.assertRaises(AttributeError, getattr, proc, 'foo')
            proc.terminate()
        if POSIX:
            self.assertEqual(proc.wait(5), -signal.SIGTERM)
        else:
            self.assertEqual(proc.wait(5), signal.SIGTERM)

    def test_ctx_manager(self):
        with psutil.Popen([PYTHON_EXE, "-V"],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          stdin=subprocess.PIPE, env=PYTHON_EXE_ENV) as proc:
            proc.communicate()
        assert proc.stdout.closed
        assert proc.stderr.closed
        assert proc.stdin.closed
        self.assertEqual(proc.returncode, 0)

    def test_kill_terminate(self):
        # subprocess.Popen()'s terminate(), kill() and send_signal() do
        # not raise exception after the process is gone. psutil.Popen
        # diverges from that.
        cmd = [PYTHON_EXE, "-c", "import time; time.sleep(60);"]
        with psutil.Popen(cmd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=PYTHON_EXE_ENV) as proc:
            proc.terminate()
            proc.wait()
            self.assertRaises(psutil.NoSuchProcess, proc.terminate)
            self.assertRaises(psutil.NoSuchProcess, proc.kill)
            self.assertRaises(psutil.NoSuchProcess, proc.send_signal,
                              signal.SIGTERM)
            if WINDOWS:
                self.assertRaises(psutil.NoSuchProcess, proc.send_signal,
                                  signal.CTRL_C_EVENT)
                self.assertRaises(psutil.NoSuchProcess, proc.send_signal,
                                  signal.CTRL_BREAK_EVENT)


if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)
