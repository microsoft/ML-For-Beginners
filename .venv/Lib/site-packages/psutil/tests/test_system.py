#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for system APIS."""

import contextlib
import datetime
import errno
import os
import platform
import pprint
import shutil
import signal
import socket
import sys
import time
import unittest

import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import DEVNULL
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import IS_64BIT
from psutil.tests import MACOS_12PLUS
from psutil.tests import PYPY
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import check_net_address
from psutil.tests import enum
from psutil.tests import mock
from psutil.tests import retry_on_failure


# ===================================================================
# --- System-related API tests
# ===================================================================


class TestProcessAPIs(PsutilTestCase):

    def test_process_iter(self):
        self.assertIn(os.getpid(), [x.pid for x in psutil.process_iter()])
        sproc = self.spawn_testproc()
        self.assertIn(sproc.pid, [x.pid for x in psutil.process_iter()])
        p = psutil.Process(sproc.pid)
        p.kill()
        p.wait()
        self.assertNotIn(sproc.pid, [x.pid for x in psutil.process_iter()])

        with mock.patch('psutil.Process',
                        side_effect=psutil.NoSuchProcess(os.getpid())):
            self.assertEqual(list(psutil.process_iter()), [])
        with mock.patch('psutil.Process',
                        side_effect=psutil.AccessDenied(os.getpid())):
            with self.assertRaises(psutil.AccessDenied):
                list(psutil.process_iter())

    def test_prcess_iter_w_attrs(self):
        for p in psutil.process_iter(attrs=['pid']):
            self.assertEqual(list(p.info.keys()), ['pid'])
        with self.assertRaises(ValueError):
            list(psutil.process_iter(attrs=['foo']))
        with mock.patch("psutil._psplatform.Process.cpu_times",
                        side_effect=psutil.AccessDenied(0, "")) as m:
            for p in psutil.process_iter(attrs=["pid", "cpu_times"]):
                self.assertIsNone(p.info['cpu_times'])
                self.assertGreaterEqual(p.info['pid'], 0)
            assert m.called
        with mock.patch("psutil._psplatform.Process.cpu_times",
                        side_effect=psutil.AccessDenied(0, "")) as m:
            flag = object()
            for p in psutil.process_iter(
                    attrs=["pid", "cpu_times"], ad_value=flag):
                self.assertIs(p.info['cpu_times'], flag)
                self.assertGreaterEqual(p.info['pid'], 0)
            assert m.called

    @unittest.skipIf(PYPY and WINDOWS,
                     "spawn_testproc() unreliable on PYPY + WINDOWS")
    def test_wait_procs(self):
        def callback(p):
            pids.append(p.pid)

        pids = []
        sproc1 = self.spawn_testproc()
        sproc2 = self.spawn_testproc()
        sproc3 = self.spawn_testproc()
        procs = [psutil.Process(x.pid) for x in (sproc1, sproc2, sproc3)]
        self.assertRaises(ValueError, psutil.wait_procs, procs, timeout=-1)
        self.assertRaises(TypeError, psutil.wait_procs, procs, callback=1)
        t = time.time()
        gone, alive = psutil.wait_procs(procs, timeout=0.01, callback=callback)

        self.assertLess(time.time() - t, 0.5)
        self.assertEqual(gone, [])
        self.assertEqual(len(alive), 3)
        self.assertEqual(pids, [])
        for p in alive:
            self.assertFalse(hasattr(p, 'returncode'))

        @retry_on_failure(30)
        def test_1(procs, callback):
            gone, alive = psutil.wait_procs(procs, timeout=0.03,
                                            callback=callback)
            self.assertEqual(len(gone), 1)
            self.assertEqual(len(alive), 2)
            return gone, alive

        sproc3.terminate()
        gone, alive = test_1(procs, callback)
        self.assertIn(sproc3.pid, [x.pid for x in gone])
        if POSIX:
            self.assertEqual(gone.pop().returncode, -signal.SIGTERM)
        else:
            self.assertEqual(gone.pop().returncode, 1)
        self.assertEqual(pids, [sproc3.pid])
        for p in alive:
            self.assertFalse(hasattr(p, 'returncode'))

        @retry_on_failure(30)
        def test_2(procs, callback):
            gone, alive = psutil.wait_procs(procs, timeout=0.03,
                                            callback=callback)
            self.assertEqual(len(gone), 3)
            self.assertEqual(len(alive), 0)
            return gone, alive

        sproc1.terminate()
        sproc2.terminate()
        gone, alive = test_2(procs, callback)
        self.assertEqual(set(pids), set([sproc1.pid, sproc2.pid, sproc3.pid]))
        for p in gone:
            self.assertTrue(hasattr(p, 'returncode'))

    @unittest.skipIf(PYPY and WINDOWS,
                     "spawn_testproc() unreliable on PYPY + WINDOWS")
    def test_wait_procs_no_timeout(self):
        sproc1 = self.spawn_testproc()
        sproc2 = self.spawn_testproc()
        sproc3 = self.spawn_testproc()
        procs = [psutil.Process(x.pid) for x in (sproc1, sproc2, sproc3)]
        for p in procs:
            p.terminate()
        psutil.wait_procs(procs)

    def test_pid_exists(self):
        sproc = self.spawn_testproc()
        self.assertTrue(psutil.pid_exists(sproc.pid))
        p = psutil.Process(sproc.pid)
        p.kill()
        p.wait()
        self.assertFalse(psutil.pid_exists(sproc.pid))
        self.assertFalse(psutil.pid_exists(-1))
        self.assertEqual(psutil.pid_exists(0), 0 in psutil.pids())

    def test_pid_exists_2(self):
        pids = psutil.pids()
        for pid in pids:
            try:
                assert psutil.pid_exists(pid)
            except AssertionError:
                # in case the process disappeared in meantime fail only
                # if it is no longer in psutil.pids()
                time.sleep(.1)
                self.assertNotIn(pid, psutil.pids())
        pids = range(max(pids) + 15000, max(pids) + 16000)
        for pid in pids:
            self.assertFalse(psutil.pid_exists(pid), msg=pid)


class TestMiscAPIs(PsutilTestCase):

    def test_boot_time(self):
        bt = psutil.boot_time()
        self.assertIsInstance(bt, float)
        self.assertGreater(bt, 0)
        self.assertLess(bt, time.time())

    @unittest.skipIf(CI_TESTING and not psutil.users(), "unreliable on CI")
    def test_users(self):
        users = psutil.users()
        self.assertNotEqual(users, [])
        for user in users:
            assert user.name, user
            self.assertIsInstance(user.name, str)
            self.assertIsInstance(user.terminal, (str, type(None)))
            if user.host is not None:
                self.assertIsInstance(user.host, (str, type(None)))
            user.terminal
            user.host
            assert user.started > 0.0, user
            datetime.datetime.fromtimestamp(user.started)
            if WINDOWS or OPENBSD:
                self.assertIsNone(user.pid)
            else:
                psutil.Process(user.pid)

    def test_test(self):
        # test for psutil.test() function
        stdout = sys.stdout
        sys.stdout = DEVNULL
        try:
            psutil.test()
        finally:
            sys.stdout = stdout

    def test_os_constants(self):
        names = ["POSIX", "WINDOWS", "LINUX", "MACOS", "FREEBSD", "OPENBSD",
                 "NETBSD", "BSD", "SUNOS"]
        for name in names:
            self.assertIsInstance(getattr(psutil, name), bool, msg=name)

        if os.name == 'posix':
            assert psutil.POSIX
            assert not psutil.WINDOWS
            names.remove("POSIX")
            if "linux" in sys.platform.lower():
                assert psutil.LINUX
                names.remove("LINUX")
            elif "bsd" in sys.platform.lower():
                assert psutil.BSD
                self.assertEqual([psutil.FREEBSD, psutil.OPENBSD,
                                  psutil.NETBSD].count(True), 1)
                names.remove("BSD")
                names.remove("FREEBSD")
                names.remove("OPENBSD")
                names.remove("NETBSD")
            elif "sunos" in sys.platform.lower() or \
                    "solaris" in sys.platform.lower():
                assert psutil.SUNOS
                names.remove("SUNOS")
            elif "darwin" in sys.platform.lower():
                assert psutil.MACOS
                names.remove("MACOS")
        else:
            assert psutil.WINDOWS
            assert not psutil.POSIX
            names.remove("WINDOWS")

        # assert all other constants are set to False
        for name in names:
            self.assertIs(getattr(psutil, name), False, msg=name)


class TestMemoryAPIs(PsutilTestCase):

    def test_virtual_memory(self):
        mem = psutil.virtual_memory()
        assert mem.total > 0, mem
        assert mem.available > 0, mem
        assert 0 <= mem.percent <= 100, mem
        assert mem.used > 0, mem
        assert mem.free >= 0, mem
        for name in mem._fields:
            value = getattr(mem, name)
            if name != 'percent':
                self.assertIsInstance(value, (int, long))
            if name != 'total':
                if not value >= 0:
                    raise self.fail("%r < 0 (%s)" % (name, value))
                if value > mem.total:
                    raise self.fail("%r > total (total=%s, %s=%s)"
                                    % (name, mem.total, name, value))

    def test_swap_memory(self):
        mem = psutil.swap_memory()
        self.assertEqual(
            mem._fields, ('total', 'used', 'free', 'percent', 'sin', 'sout'))

        assert mem.total >= 0, mem
        assert mem.used >= 0, mem
        if mem.total > 0:
            # likely a system with no swap partition
            assert mem.free > 0, mem
        else:
            assert mem.free == 0, mem
        assert 0 <= mem.percent <= 100, mem
        assert mem.sin >= 0, mem
        assert mem.sout >= 0, mem


class TestCpuAPIs(PsutilTestCase):

    def test_cpu_count_logical(self):
        logical = psutil.cpu_count()
        self.assertIsNotNone(logical)
        self.assertEqual(logical, len(psutil.cpu_times(percpu=True)))
        self.assertGreaterEqual(logical, 1)
        #
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as fd:
                cpuinfo_data = fd.read()
            if "physical id" not in cpuinfo_data:
                raise unittest.SkipTest("cpuinfo doesn't include physical id")

    def test_cpu_count_cores(self):
        logical = psutil.cpu_count()
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            raise self.skipTest("cpu_count_cores() is None")
        if WINDOWS and sys.getwindowsversion()[:2] <= (6, 1):  # <= Vista
            self.assertIsNone(cores)
        else:
            self.assertGreaterEqual(cores, 1)
            self.assertGreaterEqual(logical, cores)

    def test_cpu_count_none(self):
        # https://github.com/giampaolo/psutil/issues/1085
        for val in (-1, 0, None):
            with mock.patch('psutil._psplatform.cpu_count_logical',
                            return_value=val) as m:
                self.assertIsNone(psutil.cpu_count())
                assert m.called
            with mock.patch('psutil._psplatform.cpu_count_cores',
                            return_value=val) as m:
                self.assertIsNone(psutil.cpu_count(logical=False))
                assert m.called

    def test_cpu_times(self):
        # Check type, value >= 0, str().
        total = 0
        times = psutil.cpu_times()
        sum(times)
        for cp_time in times:
            self.assertIsInstance(cp_time, float)
            self.assertGreaterEqual(cp_time, 0.0)
            total += cp_time
        self.assertEqual(total, sum(times))
        str(times)
        # CPU times are always supposed to increase over time
        # or at least remain the same and that's because time
        # cannot go backwards.
        # Surprisingly sometimes this might not be the case (at
        # least on Windows and Linux), see:
        # https://github.com/giampaolo/psutil/issues/392
        # https://github.com/giampaolo/psutil/issues/645
        # if not WINDOWS:
        #     last = psutil.cpu_times()
        #     for x in range(100):
        #         new = psutil.cpu_times()
        #         for field in new._fields:
        #             new_t = getattr(new, field)
        #             last_t = getattr(last, field)
        #             self.assertGreaterEqual(new_t, last_t,
        #                                     msg="%s %s" % (new_t, last_t))
        #         last = new

    def test_cpu_times_time_increases(self):
        # Make sure time increases between calls.
        t1 = sum(psutil.cpu_times())
        stop_at = time.time() + GLOBAL_TIMEOUT
        while time.time() < stop_at:
            t2 = sum(psutil.cpu_times())
            if t2 > t1:
                return
        raise self.fail("time remained the same")

    def test_per_cpu_times(self):
        # Check type, value >= 0, str().
        for times in psutil.cpu_times(percpu=True):
            total = 0
            sum(times)
            for cp_time in times:
                self.assertIsInstance(cp_time, float)
                self.assertGreaterEqual(cp_time, 0.0)
                total += cp_time
            self.assertEqual(total, sum(times))
            str(times)
        self.assertEqual(len(psutil.cpu_times(percpu=True)[0]),
                         len(psutil.cpu_times(percpu=False)))

        # Note: in theory CPU times are always supposed to increase over
        # time or remain the same but never go backwards. In practice
        # sometimes this is not the case.
        # This issue seemd to be afflict Windows:
        # https://github.com/giampaolo/psutil/issues/392
        # ...but it turns out also Linux (rarely) behaves the same.
        # last = psutil.cpu_times(percpu=True)
        # for x in range(100):
        #     new = psutil.cpu_times(percpu=True)
        #     for index in range(len(new)):
        #         newcpu = new[index]
        #         lastcpu = last[index]
        #         for field in newcpu._fields:
        #             new_t = getattr(newcpu, field)
        #             last_t = getattr(lastcpu, field)
        #             self.assertGreaterEqual(
        #                 new_t, last_t, msg="%s %s" % (lastcpu, newcpu))
        #     last = new

    def test_per_cpu_times_2(self):
        # Simulate some work load then make sure time have increased
        # between calls.
        tot1 = psutil.cpu_times(percpu=True)
        giveup_at = time.time() + GLOBAL_TIMEOUT
        while True:
            if time.time() >= giveup_at:
                return self.fail("timeout")
            tot2 = psutil.cpu_times(percpu=True)
            for t1, t2 in zip(tot1, tot2):
                t1, t2 = psutil._cpu_busy_time(t1), psutil._cpu_busy_time(t2)
                difference = t2 - t1
                if difference >= 0.05:
                    return

    def test_cpu_times_comparison(self):
        # Make sure the sum of all per cpu times is almost equal to
        # base "one cpu" times.
        base = psutil.cpu_times()
        per_cpu = psutil.cpu_times(percpu=True)
        summed_values = base._make([sum(num) for num in zip(*per_cpu)])
        for field in base._fields:
            self.assertAlmostEqual(
                getattr(base, field), getattr(summed_values, field), delta=1)

    def _test_cpu_percent(self, percent, last_ret, new_ret):
        try:
            self.assertIsInstance(percent, float)
            self.assertGreaterEqual(percent, 0.0)
            self.assertIsNot(percent, -0.0)
            self.assertLessEqual(percent, 100.0 * psutil.cpu_count())
        except AssertionError as err:
            raise AssertionError("\n%s\nlast=%s\nnew=%s" % (
                err, pprint.pformat(last_ret), pprint.pformat(new_ret)))

    def test_cpu_percent(self):
        last = psutil.cpu_percent(interval=0.001)
        for _ in range(100):
            new = psutil.cpu_percent(interval=None)
            self._test_cpu_percent(new, last, new)
            last = new
        with self.assertRaises(ValueError):
            psutil.cpu_percent(interval=-1)

    def test_per_cpu_percent(self):
        last = psutil.cpu_percent(interval=0.001, percpu=True)
        self.assertEqual(len(last), psutil.cpu_count())
        for _ in range(100):
            new = psutil.cpu_percent(interval=None, percpu=True)
            for percent in new:
                self._test_cpu_percent(percent, last, new)
            last = new
        with self.assertRaises(ValueError):
            psutil.cpu_percent(interval=-1, percpu=True)

    def test_cpu_times_percent(self):
        last = psutil.cpu_times_percent(interval=0.001)
        for _ in range(100):
            new = psutil.cpu_times_percent(interval=None)
            for percent in new:
                self._test_cpu_percent(percent, last, new)
            self._test_cpu_percent(sum(new), last, new)
            last = new
        with self.assertRaises(ValueError):
            psutil.cpu_times_percent(interval=-1)

    def test_per_cpu_times_percent(self):
        last = psutil.cpu_times_percent(interval=0.001, percpu=True)
        self.assertEqual(len(last), psutil.cpu_count())
        for _ in range(100):
            new = psutil.cpu_times_percent(interval=None, percpu=True)
            for cpu in new:
                for percent in cpu:
                    self._test_cpu_percent(percent, last, new)
                self._test_cpu_percent(sum(cpu), last, new)
            last = new

    def test_per_cpu_times_percent_negative(self):
        # see: https://github.com/giampaolo/psutil/issues/645
        psutil.cpu_times_percent(percpu=True)
        zero_times = [x._make([0 for x in range(len(x._fields))])
                      for x in psutil.cpu_times(percpu=True)]
        with mock.patch('psutil.cpu_times', return_value=zero_times):
            for cpu in psutil.cpu_times_percent(percpu=True):
                for percent in cpu:
                    self._test_cpu_percent(percent, None, None)

    def test_cpu_stats(self):
        # Tested more extensively in per-platform test modules.
        infos = psutil.cpu_stats()
        self.assertEqual(
            infos._fields,
            ('ctx_switches', 'interrupts', 'soft_interrupts', 'syscalls'))
        for name in infos._fields:
            value = getattr(infos, name)
            self.assertGreaterEqual(value, 0)
            # on AIX, ctx_switches is always 0
            if not AIX and name in ('ctx_switches', 'interrupts'):
                self.assertGreater(value, 0)

    # TODO: remove this once 1892 is fixed
    @unittest.skipIf(MACOS and platform.machine() == 'arm64',
                     "skipped due to #1892")
    @unittest.skipIf(not HAS_CPU_FREQ, "not supported")
    def test_cpu_freq(self):
        def check_ls(ls):
            for nt in ls:
                self.assertEqual(nt._fields, ('current', 'min', 'max'))
                if nt.max != 0.0:
                    self.assertLessEqual(nt.current, nt.max)
                for name in nt._fields:
                    value = getattr(nt, name)
                    self.assertIsInstance(value, (int, long, float))
                    self.assertGreaterEqual(value, 0)

        ls = psutil.cpu_freq(percpu=True)
        if FREEBSD and not ls:
            raise self.skipTest("returns empty list on FreeBSD")

        assert ls, ls
        check_ls([psutil.cpu_freq(percpu=False)])

        if LINUX:
            self.assertEqual(len(ls), psutil.cpu_count())

    @unittest.skipIf(not HAS_GETLOADAVG, "not supported")
    def test_getloadavg(self):
        loadavg = psutil.getloadavg()
        self.assertEqual(len(loadavg), 3)
        for load in loadavg:
            self.assertIsInstance(load, float)
            self.assertGreaterEqual(load, 0.0)


class TestDiskAPIs(PsutilTestCase):

    @unittest.skipIf(PYPY and not IS_64BIT, "unreliable on PYPY32 + 32BIT")
    def test_disk_usage(self):
        usage = psutil.disk_usage(os.getcwd())
        self.assertEqual(usage._fields, ('total', 'used', 'free', 'percent'))

        assert usage.total > 0, usage
        assert usage.used > 0, usage
        assert usage.free > 0, usage
        assert usage.total > usage.used, usage
        assert usage.total > usage.free, usage
        assert 0 <= usage.percent <= 100, usage.percent
        if hasattr(shutil, 'disk_usage'):
            # py >= 3.3, see: http://bugs.python.org/issue12442
            shutil_usage = shutil.disk_usage(os.getcwd())
            tolerance = 5 * 1024 * 1024  # 5MB
            self.assertEqual(usage.total, shutil_usage.total)
            self.assertAlmostEqual(usage.free, shutil_usage.free,
                                   delta=tolerance)
            if not MACOS_12PLUS:
                # see https://github.com/giampaolo/psutil/issues/2147
                self.assertAlmostEqual(usage.used, shutil_usage.used,
                                       delta=tolerance)

        # if path does not exist OSError ENOENT is expected across
        # all platforms
        fname = self.get_testfn()
        with self.assertRaises(FileNotFoundError):
            psutil.disk_usage(fname)

    @unittest.skipIf(not ASCII_FS, "not an ASCII fs")
    def test_disk_usage_unicode(self):
        # See: https://github.com/giampaolo/psutil/issues/416
        with self.assertRaises(UnicodeEncodeError):
            psutil.disk_usage(UNICODE_SUFFIX)

    def test_disk_usage_bytes(self):
        psutil.disk_usage(b'.')

    def test_disk_partitions(self):
        def check_ntuple(nt):
            self.assertIsInstance(nt.device, str)
            self.assertIsInstance(nt.mountpoint, str)
            self.assertIsInstance(nt.fstype, str)
            self.assertIsInstance(nt.opts, str)
            self.assertIsInstance(nt.maxfile, (int, type(None)))
            self.assertIsInstance(nt.maxpath, (int, type(None)))
            if nt.maxfile is not None and not GITHUB_ACTIONS:
                self.assertGreater(nt.maxfile, 0)
            if nt.maxpath is not None:
                self.assertGreater(nt.maxpath, 0)

        # all = False
        ls = psutil.disk_partitions(all=False)
        self.assertTrue(ls, msg=ls)
        for disk in ls:
            check_ntuple(disk)
            if WINDOWS and 'cdrom' in disk.opts:
                continue
            if not POSIX:
                assert os.path.exists(disk.device), disk
            else:
                # we cannot make any assumption about this, see:
                # http://goo.gl/p9c43
                disk.device
            # on modern systems mount points can also be files
            assert os.path.exists(disk.mountpoint), disk
            assert disk.fstype, disk

        # all = True
        ls = psutil.disk_partitions(all=True)
        self.assertTrue(ls, msg=ls)
        for disk in psutil.disk_partitions(all=True):
            check_ntuple(disk)
            if not WINDOWS and disk.mountpoint:
                try:
                    os.stat(disk.mountpoint)
                except OSError as err:
                    if GITHUB_ACTIONS and MACOS and err.errno == errno.EIO:
                        continue
                    # http://mail.python.org/pipermail/python-dev/
                    #     2012-June/120787.html
                    if err.errno not in (errno.EPERM, errno.EACCES):
                        raise
                else:
                    assert os.path.exists(disk.mountpoint), disk

        # ---

        def find_mount_point(path):
            path = os.path.abspath(path)
            while not os.path.ismount(path):
                path = os.path.dirname(path)
            return path.lower()

        mount = find_mount_point(__file__)
        mounts = [x.mountpoint.lower() for x in
                  psutil.disk_partitions(all=True) if x.mountpoint]
        self.assertIn(mount, mounts)

    @unittest.skipIf(LINUX and not os.path.exists('/proc/diskstats'),
                     '/proc/diskstats not available on this linux version')
    @unittest.skipIf(CI_TESTING and not psutil.disk_io_counters(),
                     "unreliable on CI")  # no visible disks
    def test_disk_io_counters(self):
        def check_ntuple(nt):
            self.assertEqual(nt[0], nt.read_count)
            self.assertEqual(nt[1], nt.write_count)
            self.assertEqual(nt[2], nt.read_bytes)
            self.assertEqual(nt[3], nt.write_bytes)
            if not (OPENBSD or NETBSD):
                self.assertEqual(nt[4], nt.read_time)
                self.assertEqual(nt[5], nt.write_time)
                if LINUX:
                    self.assertEqual(nt[6], nt.read_merged_count)
                    self.assertEqual(nt[7], nt.write_merged_count)
                    self.assertEqual(nt[8], nt.busy_time)
                elif FREEBSD:
                    self.assertEqual(nt[6], nt.busy_time)
            for name in nt._fields:
                assert getattr(nt, name) >= 0, nt

        ret = psutil.disk_io_counters(perdisk=False)
        assert ret is not None, "no disks on this system?"
        check_ntuple(ret)
        ret = psutil.disk_io_counters(perdisk=True)
        # make sure there are no duplicates
        self.assertEqual(len(ret), len(set(ret)))
        for key in ret:
            assert key, key
            check_ntuple(ret[key])

    def test_disk_io_counters_no_disks(self):
        # Emulate a case where no disks are installed, see:
        # https://github.com/giampaolo/psutil/issues/1062
        with mock.patch('psutil._psplatform.disk_io_counters',
                        return_value={}) as m:
            self.assertIsNone(psutil.disk_io_counters(perdisk=False))
            self.assertEqual(psutil.disk_io_counters(perdisk=True), {})
            assert m.called


class TestNetAPIs(PsutilTestCase):

    @unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
    def test_net_io_counters(self):
        def check_ntuple(nt):
            self.assertEqual(nt[0], nt.bytes_sent)
            self.assertEqual(nt[1], nt.bytes_recv)
            self.assertEqual(nt[2], nt.packets_sent)
            self.assertEqual(nt[3], nt.packets_recv)
            self.assertEqual(nt[4], nt.errin)
            self.assertEqual(nt[5], nt.errout)
            self.assertEqual(nt[6], nt.dropin)
            self.assertEqual(nt[7], nt.dropout)
            assert nt.bytes_sent >= 0, nt
            assert nt.bytes_recv >= 0, nt
            assert nt.packets_sent >= 0, nt
            assert nt.packets_recv >= 0, nt
            assert nt.errin >= 0, nt
            assert nt.errout >= 0, nt
            assert nt.dropin >= 0, nt
            assert nt.dropout >= 0, nt

        ret = psutil.net_io_counters(pernic=False)
        check_ntuple(ret)
        ret = psutil.net_io_counters(pernic=True)
        self.assertNotEqual(ret, [])
        for key in ret:
            self.assertTrue(key)
            self.assertIsInstance(key, str)
            check_ntuple(ret[key])

    @unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
    def test_net_io_counters_no_nics(self):
        # Emulate a case where no NICs are installed, see:
        # https://github.com/giampaolo/psutil/issues/1062
        with mock.patch('psutil._psplatform.net_io_counters',
                        return_value={}) as m:
            self.assertIsNone(psutil.net_io_counters(pernic=False))
            self.assertEqual(psutil.net_io_counters(pernic=True), {})
            assert m.called

    def test_net_if_addrs(self):
        nics = psutil.net_if_addrs()
        assert nics, nics

        nic_stats = psutil.net_if_stats()

        # Not reliable on all platforms (net_if_addrs() reports more
        # interfaces).
        # self.assertEqual(sorted(nics.keys()),
        #                  sorted(psutil.net_io_counters(pernic=True).keys()))

        families = set([socket.AF_INET, socket.AF_INET6, psutil.AF_LINK])
        for nic, addrs in nics.items():
            self.assertIsInstance(nic, str)
            self.assertEqual(len(set(addrs)), len(addrs))
            for addr in addrs:
                self.assertIsInstance(addr.family, int)
                self.assertIsInstance(addr.address, str)
                self.assertIsInstance(addr.netmask, (str, type(None)))
                self.assertIsInstance(addr.broadcast, (str, type(None)))
                self.assertIn(addr.family, families)
                if sys.version_info >= (3, 4) and not PYPY:
                    self.assertIsInstance(addr.family, enum.IntEnum)
                if nic_stats[nic].isup:
                    # Do not test binding to addresses of interfaces
                    # that are down
                    if addr.family == socket.AF_INET:
                        s = socket.socket(addr.family)
                        with contextlib.closing(s):
                            s.bind((addr.address, 0))
                    elif addr.family == socket.AF_INET6:
                        info = socket.getaddrinfo(
                            addr.address, 0, socket.AF_INET6,
                            socket.SOCK_STREAM, 0, socket.AI_PASSIVE)[0]
                        af, socktype, proto, canonname, sa = info
                        s = socket.socket(af, socktype, proto)
                        with contextlib.closing(s):
                            s.bind(sa)
                for ip in (addr.address, addr.netmask, addr.broadcast,
                           addr.ptp):
                    if ip is not None:
                        # TODO: skip AF_INET6 for now because I get:
                        # AddressValueError: Only hex digits permitted in
                        # u'c6f3%lxcbr0' in u'fe80::c8e0:fff:fe54:c6f3%lxcbr0'
                        if addr.family != socket.AF_INET6:
                            check_net_address(ip, addr.family)
                # broadcast and ptp addresses are mutually exclusive
                if addr.broadcast:
                    self.assertIsNone(addr.ptp)
                elif addr.ptp:
                    self.assertIsNone(addr.broadcast)

        if BSD or MACOS or SUNOS:
            if hasattr(socket, "AF_LINK"):
                self.assertEqual(psutil.AF_LINK, socket.AF_LINK)
        elif LINUX:
            self.assertEqual(psutil.AF_LINK, socket.AF_PACKET)
        elif WINDOWS:
            self.assertEqual(psutil.AF_LINK, -1)

    def test_net_if_addrs_mac_null_bytes(self):
        # Simulate that the underlying C function returns an incomplete
        # MAC address. psutil is supposed to fill it with null bytes.
        # https://github.com/giampaolo/psutil/issues/786
        if POSIX:
            ret = [('em1', psutil.AF_LINK, '06:3d:29', None, None, None)]
        else:
            ret = [('em1', -1, '06-3d-29', None, None, None)]
        with mock.patch('psutil._psplatform.net_if_addrs',
                        return_value=ret) as m:
            addr = psutil.net_if_addrs()['em1'][0]
            assert m.called
            if POSIX:
                self.assertEqual(addr.address, '06:3d:29:00:00:00')
            else:
                self.assertEqual(addr.address, '06-3d-29-00-00-00')

    def test_net_if_stats(self):
        nics = psutil.net_if_stats()
        assert nics, nics
        all_duplexes = (psutil.NIC_DUPLEX_FULL,
                        psutil.NIC_DUPLEX_HALF,
                        psutil.NIC_DUPLEX_UNKNOWN)
        for name, stats in nics.items():
            self.assertIsInstance(name, str)
            isup, duplex, speed, mtu, flags = stats
            self.assertIsInstance(isup, bool)
            self.assertIn(duplex, all_duplexes)
            self.assertIn(duplex, all_duplexes)
            self.assertGreaterEqual(speed, 0)
            self.assertGreaterEqual(mtu, 0)
            self.assertIsInstance(flags, str)

    @unittest.skipIf(not (LINUX or BSD or MACOS),
                     "LINUX or BSD or MACOS specific")
    def test_net_if_stats_enodev(self):
        # See: https://github.com/giampaolo/psutil/issues/1279
        with mock.patch('psutil._psutil_posix.net_if_mtu',
                        side_effect=OSError(errno.ENODEV, "")) as m:
            ret = psutil.net_if_stats()
            self.assertEqual(ret, {})
            assert m.called


class TestSensorsAPIs(PsutilTestCase):

    @unittest.skipIf(not HAS_SENSORS_TEMPERATURES, "not supported")
    def test_sensors_temperatures(self):
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            self.assertIsInstance(name, str)
            for entry in entries:
                self.assertIsInstance(entry.label, str)
                if entry.current is not None:
                    self.assertGreaterEqual(entry.current, 0)
                if entry.high is not None:
                    self.assertGreaterEqual(entry.high, 0)
                if entry.critical is not None:
                    self.assertGreaterEqual(entry.critical, 0)

    @unittest.skipIf(not HAS_SENSORS_TEMPERATURES, "not supported")
    def test_sensors_temperatures_fahreneit(self):
        d = {'coretemp': [('label', 50.0, 60.0, 70.0)]}
        with mock.patch("psutil._psplatform.sensors_temperatures",
                        return_value=d) as m:
            temps = psutil.sensors_temperatures(
                fahrenheit=True)['coretemp'][0]
            assert m.called
            self.assertEqual(temps.current, 122.0)
            self.assertEqual(temps.high, 140.0)
            self.assertEqual(temps.critical, 158.0)

    @unittest.skipIf(not HAS_SENSORS_BATTERY, "not supported")
    @unittest.skipIf(not HAS_BATTERY, "no battery")
    def test_sensors_battery(self):
        ret = psutil.sensors_battery()
        self.assertGreaterEqual(ret.percent, 0)
        self.assertLessEqual(ret.percent, 100)
        if ret.secsleft not in (psutil.POWER_TIME_UNKNOWN,
                                psutil.POWER_TIME_UNLIMITED):
            self.assertGreaterEqual(ret.secsleft, 0)
        else:
            if ret.secsleft == psutil.POWER_TIME_UNLIMITED:
                self.assertTrue(ret.power_plugged)
        self.assertIsInstance(ret.power_plugged, bool)

    @unittest.skipIf(not HAS_SENSORS_FANS, "not supported")
    def test_sensors_fans(self):
        fans = psutil.sensors_fans()
        for name, entries in fans.items():
            self.assertIsInstance(name, str)
            for entry in entries:
                self.assertIsInstance(entry.label, str)
                self.assertIsInstance(entry.current, (int, long))
                self.assertGreaterEqual(entry.current, 0)


if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)
