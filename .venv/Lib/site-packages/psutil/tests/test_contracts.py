#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Contracts tests. These tests mainly check API sanity in terms of
returned types and APIs availability.
Some of these are duplicates of tests test_system.py and test_process.py
"""

import errno
import multiprocessing
import os
import platform
import signal
import stat
import sys
import time
import traceback
import unittest

import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil._compat import range
from psutil._compat import unicode
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYPY
from psutil.tests import SKIP_SYSCONS
from psutil.tests import VALID_PROC_STATUSES
from psutil.tests import PsutilTestCase
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import enum
from psutil.tests import is_namedtuple
from psutil.tests import kernel_version
from psutil.tests import process_namespace
from psutil.tests import serialrun


# ===================================================================
# --- APIs availability
# ===================================================================

# Make sure code reflects what doc promises in terms of APIs
# availability.

class TestAvailConstantsAPIs(PsutilTestCase):

    def test_PROCFS_PATH(self):
        self.assertEqual(hasattr(psutil, "PROCFS_PATH"),
                         LINUX or SUNOS or AIX)

    def test_win_priority(self):
        ae = self.assertEqual
        ae(hasattr(psutil, "ABOVE_NORMAL_PRIORITY_CLASS"), WINDOWS)
        ae(hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"), WINDOWS)
        ae(hasattr(psutil, "HIGH_PRIORITY_CLASS"), WINDOWS)
        ae(hasattr(psutil, "IDLE_PRIORITY_CLASS"), WINDOWS)
        ae(hasattr(psutil, "NORMAL_PRIORITY_CLASS"), WINDOWS)
        ae(hasattr(psutil, "REALTIME_PRIORITY_CLASS"), WINDOWS)

    def test_linux_ioprio_linux(self):
        ae = self.assertEqual
        ae(hasattr(psutil, "IOPRIO_CLASS_NONE"), LINUX)
        ae(hasattr(psutil, "IOPRIO_CLASS_RT"), LINUX)
        ae(hasattr(psutil, "IOPRIO_CLASS_BE"), LINUX)
        ae(hasattr(psutil, "IOPRIO_CLASS_IDLE"), LINUX)

    def test_linux_ioprio_windows(self):
        ae = self.assertEqual
        ae(hasattr(psutil, "IOPRIO_HIGH"), WINDOWS)
        ae(hasattr(psutil, "IOPRIO_NORMAL"), WINDOWS)
        ae(hasattr(psutil, "IOPRIO_LOW"), WINDOWS)
        ae(hasattr(psutil, "IOPRIO_VERYLOW"), WINDOWS)

    @unittest.skipIf(GITHUB_ACTIONS and LINUX,
                     "unsupported on GITHUB_ACTIONS + LINUX")
    def test_rlimit(self):
        ae = self.assertEqual
        ae(hasattr(psutil, "RLIM_INFINITY"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_AS"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_CORE"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_CPU"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_DATA"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_FSIZE"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_MEMLOCK"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_NOFILE"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_NPROC"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_RSS"), LINUX or FREEBSD)
        ae(hasattr(psutil, "RLIMIT_STACK"), LINUX or FREEBSD)

        ae(hasattr(psutil, "RLIMIT_LOCKS"), LINUX)
        if POSIX:
            if kernel_version() >= (2, 6, 8):
                ae(hasattr(psutil, "RLIMIT_MSGQUEUE"), LINUX)
            if kernel_version() >= (2, 6, 12):
                ae(hasattr(psutil, "RLIMIT_NICE"), LINUX)
            if kernel_version() >= (2, 6, 12):
                ae(hasattr(psutil, "RLIMIT_RTPRIO"), LINUX)
            if kernel_version() >= (2, 6, 25):
                ae(hasattr(psutil, "RLIMIT_RTTIME"), LINUX)
            if kernel_version() >= (2, 6, 8):
                ae(hasattr(psutil, "RLIMIT_SIGPENDING"), LINUX)

        ae(hasattr(psutil, "RLIMIT_SWAP"), FREEBSD)
        ae(hasattr(psutil, "RLIMIT_SBSIZE"), FREEBSD)
        ae(hasattr(psutil, "RLIMIT_NPTS"), FREEBSD)


class TestAvailSystemAPIs(PsutilTestCase):

    def test_win_service_iter(self):
        self.assertEqual(hasattr(psutil, "win_service_iter"), WINDOWS)

    def test_win_service_get(self):
        self.assertEqual(hasattr(psutil, "win_service_get"), WINDOWS)

    def test_cpu_freq(self):
        self.assertEqual(hasattr(psutil, "cpu_freq"),
                         LINUX or MACOS or WINDOWS or FREEBSD or OPENBSD)

    def test_sensors_temperatures(self):
        self.assertEqual(
            hasattr(psutil, "sensors_temperatures"), LINUX or FREEBSD)

    def test_sensors_fans(self):
        self.assertEqual(hasattr(psutil, "sensors_fans"), LINUX)

    def test_battery(self):
        self.assertEqual(hasattr(psutil, "sensors_battery"),
                         LINUX or WINDOWS or FREEBSD or MACOS)


class TestAvailProcessAPIs(PsutilTestCase):

    def test_environ(self):
        self.assertEqual(hasattr(psutil.Process, "environ"),
                         LINUX or MACOS or WINDOWS or AIX or SUNOS or
                         FREEBSD or OPENBSD or NETBSD)

    def test_uids(self):
        self.assertEqual(hasattr(psutil.Process, "uids"), POSIX)

    def test_gids(self):
        self.assertEqual(hasattr(psutil.Process, "uids"), POSIX)

    def test_terminal(self):
        self.assertEqual(hasattr(psutil.Process, "terminal"), POSIX)

    def test_ionice(self):
        self.assertEqual(hasattr(psutil.Process, "ionice"), LINUX or WINDOWS)

    @unittest.skipIf(GITHUB_ACTIONS and LINUX,
                     "unsupported on GITHUB_ACTIONS + LINUX")
    def test_rlimit(self):
        self.assertEqual(hasattr(psutil.Process, "rlimit"), LINUX or FREEBSD)

    def test_io_counters(self):
        hasit = hasattr(psutil.Process, "io_counters")
        self.assertEqual(hasit, not (MACOS or SUNOS))

    def test_num_fds(self):
        self.assertEqual(hasattr(psutil.Process, "num_fds"), POSIX)

    def test_num_handles(self):
        self.assertEqual(hasattr(psutil.Process, "num_handles"), WINDOWS)

    def test_cpu_affinity(self):
        self.assertEqual(hasattr(psutil.Process, "cpu_affinity"),
                         LINUX or WINDOWS or FREEBSD)

    def test_cpu_num(self):
        self.assertEqual(hasattr(psutil.Process, "cpu_num"),
                         LINUX or FREEBSD or SUNOS)

    def test_memory_maps(self):
        hasit = hasattr(psutil.Process, "memory_maps")
        self.assertEqual(
            hasit, False if OPENBSD or NETBSD or AIX or MACOS else True)


# ===================================================================
# --- API types
# ===================================================================


class TestSystemAPITypes(PsutilTestCase):
    """Check the return types of system related APIs.
    Mainly we want to test we never return unicode on Python 2, see:
    https://github.com/giampaolo/psutil/issues/1039
    """

    @classmethod
    def setUpClass(cls):
        cls.proc = psutil.Process()

    def assert_ntuple_of_nums(self, nt, type_=float, gezero=True):
        assert is_namedtuple(nt)
        for n in nt:
            self.assertIsInstance(n, type_)
            if gezero:
                self.assertGreaterEqual(n, 0)

    def test_cpu_times(self):
        self.assert_ntuple_of_nums(psutil.cpu_times())
        for nt in psutil.cpu_times(percpu=True):
            self.assert_ntuple_of_nums(nt)

    def test_cpu_percent(self):
        self.assertIsInstance(psutil.cpu_percent(interval=None), float)
        self.assertIsInstance(psutil.cpu_percent(interval=0.00001), float)

    def test_cpu_times_percent(self):
        self.assert_ntuple_of_nums(psutil.cpu_times_percent(interval=None))
        self.assert_ntuple_of_nums(psutil.cpu_times_percent(interval=0.0001))

    def test_cpu_count(self):
        self.assertIsInstance(psutil.cpu_count(), int)

    # TODO: remove this once 1892 is fixed
    @unittest.skipIf(MACOS and platform.machine() == 'arm64',
                     "skipped due to #1892")
    @unittest.skipIf(not HAS_CPU_FREQ, "not supported")
    def test_cpu_freq(self):
        if psutil.cpu_freq() is None:
            raise self.skipTest("cpu_freq() returns None")
        self.assert_ntuple_of_nums(psutil.cpu_freq(), type_=(float, int, long))

    def test_disk_io_counters(self):
        # Duplicate of test_system.py. Keep it anyway.
        for k, v in psutil.disk_io_counters(perdisk=True).items():
            self.assertIsInstance(k, str)
            self.assert_ntuple_of_nums(v, type_=(int, long))

    def test_disk_partitions(self):
        # Duplicate of test_system.py. Keep it anyway.
        for disk in psutil.disk_partitions():
            self.assertIsInstance(disk.device, str)
            self.assertIsInstance(disk.mountpoint, str)
            self.assertIsInstance(disk.fstype, str)
            self.assertIsInstance(disk.opts, str)
            self.assertIsInstance(disk.maxfile, (int, type(None)))
            self.assertIsInstance(disk.maxpath, (int, type(None)))

    @unittest.skipIf(SKIP_SYSCONS, "requires root")
    def test_net_connections(self):
        with create_sockets():
            ret = psutil.net_connections('all')
            self.assertEqual(len(ret), len(set(ret)))
            for conn in ret:
                assert is_namedtuple(conn)

    def test_net_if_addrs(self):
        # Duplicate of test_system.py. Keep it anyway.
        for ifname, addrs in psutil.net_if_addrs().items():
            self.assertIsInstance(ifname, str)
            for addr in addrs:
                if enum is not None and not PYPY:
                    self.assertIsInstance(addr.family, enum.IntEnum)
                else:
                    self.assertIsInstance(addr.family, int)
                self.assertIsInstance(addr.address, str)
                self.assertIsInstance(addr.netmask, (str, type(None)))
                self.assertIsInstance(addr.broadcast, (str, type(None)))

    def test_net_if_stats(self):
        # Duplicate of test_system.py. Keep it anyway.
        for ifname, info in psutil.net_if_stats().items():
            self.assertIsInstance(ifname, str)
            self.assertIsInstance(info.isup, bool)
            if enum is not None:
                self.assertIsInstance(info.duplex, enum.IntEnum)
            else:
                self.assertIsInstance(info.duplex, int)
            self.assertIsInstance(info.speed, int)
            self.assertIsInstance(info.mtu, int)

    @unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
    def test_net_io_counters(self):
        # Duplicate of test_system.py. Keep it anyway.
        for ifname, _ in psutil.net_io_counters(pernic=True).items():
            self.assertIsInstance(ifname, str)

    @unittest.skipIf(not HAS_SENSORS_FANS, "not supported")
    def test_sensors_fans(self):
        # Duplicate of test_system.py. Keep it anyway.
        for name, units in psutil.sensors_fans().items():
            self.assertIsInstance(name, str)
            for unit in units:
                self.assertIsInstance(unit.label, str)
                self.assertIsInstance(unit.current, (float, int, type(None)))

    @unittest.skipIf(not HAS_SENSORS_TEMPERATURES, "not supported")
    def test_sensors_temperatures(self):
        # Duplicate of test_system.py. Keep it anyway.
        for name, units in psutil.sensors_temperatures().items():
            self.assertIsInstance(name, str)
            for unit in units:
                self.assertIsInstance(unit.label, str)
                self.assertIsInstance(unit.current, (float, int, type(None)))
                self.assertIsInstance(unit.high, (float, int, type(None)))
                self.assertIsInstance(unit.critical, (float, int, type(None)))

    def test_boot_time(self):
        # Duplicate of test_system.py. Keep it anyway.
        self.assertIsInstance(psutil.boot_time(), float)

    def test_users(self):
        # Duplicate of test_system.py. Keep it anyway.
        for user in psutil.users():
            self.assertIsInstance(user.name, str)
            self.assertIsInstance(user.terminal, (str, type(None)))
            self.assertIsInstance(user.host, (str, type(None)))
            self.assertIsInstance(user.pid, (int, type(None)))


class TestProcessWaitType(PsutilTestCase):

    @unittest.skipIf(not POSIX, "not POSIX")
    def test_negative_signal(self):
        p = psutil.Process(self.spawn_testproc().pid)
        p.terminate()
        code = p.wait()
        self.assertEqual(code, -signal.SIGTERM)
        if enum is not None:
            self.assertIsInstance(code, enum.IntEnum)
        else:
            self.assertIsInstance(code, int)


# ===================================================================
# --- Featch all processes test
# ===================================================================


def proc_info(pid):
    tcase = PsutilTestCase()

    def check_exception(exc, proc, name, ppid):
        tcase.assertEqual(exc.pid, pid)
        tcase.assertEqual(exc.name, name)
        if isinstance(exc, psutil.ZombieProcess):
            if exc.ppid is not None:
                tcase.assertGreaterEqual(exc.ppid, 0)
                tcase.assertEqual(exc.ppid, ppid)
        elif isinstance(exc, psutil.NoSuchProcess):
            tcase.assertProcessGone(proc)
        str(exc)

    def do_wait():
        if pid != 0:
            try:
                proc.wait(0)
            except psutil.Error as exc:
                check_exception(exc, proc, name, ppid)

    try:
        proc = psutil.Process(pid)
        d = proc.as_dict(['ppid', 'name'])
    except psutil.NoSuchProcess:
        return {}

    name, ppid = d['name'], d['ppid']
    info = {'pid': proc.pid}
    ns = process_namespace(proc)
    # We don't use oneshot() because in order not to fool
    # check_exception() in case of NSP.
    for fun, fun_name in ns.iter(ns.getters, clear_cache=False):
        try:
            info[fun_name] = fun()
        except psutil.Error as exc:
            check_exception(exc, proc, name, ppid)
            continue
    do_wait()
    return info


@serialrun
class TestFetchAllProcesses(PsutilTestCase):
    """Test which iterates over all running processes and performs
    some sanity checks against Process API's returned values.
    Uses a process pool to get info about all processes.
    """

    def setUp(self):
        # Using a pool in a CI env may result in deadlock, see:
        # https://github.com/giampaolo/psutil/issues/2104
        if not CI_TESTING:
            self.pool = multiprocessing.Pool()

    def tearDown(self):
        if not CI_TESTING:
            self.pool.terminate()
            self.pool.join()

    def iter_proc_info(self):
        # Fixes "can't pickle <function proc_info>: it's not the
        # same object as test_contracts.proc_info".
        from psutil.tests.test_contracts import proc_info

        if not CI_TESTING:
            return self.pool.imap_unordered(proc_info, psutil.pids())
        else:
            ls = []
            for pid in psutil.pids():
                ls.append(proc_info(pid))
            return ls

    def test_all(self):
        failures = []
        for info in self.iter_proc_info():
            for name, value in info.items():
                meth = getattr(self, name)
                try:
                    meth(value, info)
                except AssertionError:
                    s = '\n' + '=' * 70 + '\n'
                    s += "FAIL: test_%s pid=%s, ret=%s\n" % (
                        name, info['pid'], repr(value))
                    s += '-' * 70
                    s += "\n%s" % traceback.format_exc()
                    s = "\n".join((" " * 4) + i for i in s.splitlines()) + "\n"
                    failures.append(s)
                else:
                    if value not in (0, 0.0, [], None, '', {}):
                        assert value, value
        if failures:
            raise self.fail(''.join(failures))

    def cmdline(self, ret, info):
        self.assertIsInstance(ret, list)
        for part in ret:
            self.assertIsInstance(part, str)

    def exe(self, ret, info):
        self.assertIsInstance(ret, (str, unicode))
        self.assertEqual(ret.strip(), ret)
        if ret:
            if WINDOWS and not ret.endswith('.exe'):
                return  # May be "Registry", "MemCompression", ...
            assert os.path.isabs(ret), ret
            # Note: os.stat() may return False even if the file is there
            # hence we skip the test, see:
            # http://stackoverflow.com/questions/3112546/os-path-exists-lies
            if POSIX and os.path.isfile(ret):
                if hasattr(os, 'access') and hasattr(os, "X_OK"):
                    # XXX: may fail on MACOS
                    try:
                        assert os.access(ret, os.X_OK)
                    except AssertionError:
                        if os.path.exists(ret) and not CI_TESTING:
                            raise

    def pid(self, ret, info):
        self.assertIsInstance(ret, int)
        self.assertGreaterEqual(ret, 0)

    def ppid(self, ret, info):
        self.assertIsInstance(ret, (int, long))
        self.assertGreaterEqual(ret, 0)

    def name(self, ret, info):
        self.assertIsInstance(ret, (str, unicode))
        if APPVEYOR and not ret and info['status'] == 'stopped':
            return
        # on AIX, "<exiting>" processes don't have names
        if not AIX:
            assert ret

    def create_time(self, ret, info):
        self.assertIsInstance(ret, float)
        try:
            self.assertGreaterEqual(ret, 0)
        except AssertionError:
            # XXX
            if OPENBSD and info['status'] == psutil.STATUS_ZOMBIE:
                pass
            else:
                raise
        # this can't be taken for granted on all platforms
        # self.assertGreaterEqual(ret, psutil.boot_time())
        # make sure returned value can be pretty printed
        # with strftime
        time.strftime("%Y %m %d %H:%M:%S", time.localtime(ret))

    def uids(self, ret, info):
        assert is_namedtuple(ret)
        for uid in ret:
            self.assertIsInstance(uid, int)
            self.assertGreaterEqual(uid, 0)

    def gids(self, ret, info):
        assert is_namedtuple(ret)
        # note: testing all gids as above seems not to be reliable for
        # gid == 30 (nodoby); not sure why.
        for gid in ret:
            self.assertIsInstance(gid, int)
            if not MACOS and not NETBSD:
                self.assertGreaterEqual(gid, 0)

    def username(self, ret, info):
        self.assertIsInstance(ret, str)
        self.assertEqual(ret.strip(), ret)
        assert ret.strip()

    def status(self, ret, info):
        self.assertIsInstance(ret, str)
        assert ret
        self.assertNotEqual(ret, '?')  # XXX
        self.assertIn(ret, VALID_PROC_STATUSES)

    def io_counters(self, ret, info):
        assert is_namedtuple(ret)
        for field in ret:
            self.assertIsInstance(field, (int, long))
            if field != -1:
                self.assertGreaterEqual(field, 0)

    def ionice(self, ret, info):
        if LINUX:
            self.assertIsInstance(ret.ioclass, int)
            self.assertIsInstance(ret.value, int)
            self.assertGreaterEqual(ret.ioclass, 0)
            self.assertGreaterEqual(ret.value, 0)
        else:  # Windows, Cygwin
            choices = [
                psutil.IOPRIO_VERYLOW,
                psutil.IOPRIO_LOW,
                psutil.IOPRIO_NORMAL,
                psutil.IOPRIO_HIGH]
            self.assertIsInstance(ret, int)
            self.assertGreaterEqual(ret, 0)
            self.assertIn(ret, choices)

    def num_threads(self, ret, info):
        self.assertIsInstance(ret, int)
        if APPVEYOR and not ret and info['status'] == 'stopped':
            return
        self.assertGreaterEqual(ret, 1)

    def threads(self, ret, info):
        self.assertIsInstance(ret, list)
        for t in ret:
            assert is_namedtuple(t)
            self.assertGreaterEqual(t.id, 0)
            self.assertGreaterEqual(t.user_time, 0)
            self.assertGreaterEqual(t.system_time, 0)
            for field in t:
                self.assertIsInstance(field, (int, float))

    def cpu_times(self, ret, info):
        assert is_namedtuple(ret)
        for n in ret:
            self.assertIsInstance(n, float)
            self.assertGreaterEqual(n, 0)
        # TODO: check ntuple fields

    def cpu_percent(self, ret, info):
        self.assertIsInstance(ret, float)
        assert 0.0 <= ret <= 100.0, ret

    def cpu_num(self, ret, info):
        self.assertIsInstance(ret, int)
        if FREEBSD and ret == -1:
            return
        self.assertGreaterEqual(ret, 0)
        if psutil.cpu_count() == 1:
            self.assertEqual(ret, 0)
        self.assertIn(ret, list(range(psutil.cpu_count())))

    def memory_info(self, ret, info):
        assert is_namedtuple(ret)
        for value in ret:
            self.assertIsInstance(value, (int, long))
            self.assertGreaterEqual(value, 0)
        if WINDOWS:
            self.assertGreaterEqual(ret.peak_wset, ret.wset)
            self.assertGreaterEqual(ret.peak_paged_pool, ret.paged_pool)
            self.assertGreaterEqual(ret.peak_nonpaged_pool, ret.nonpaged_pool)
            self.assertGreaterEqual(ret.peak_pagefile, ret.pagefile)

    def memory_full_info(self, ret, info):
        assert is_namedtuple(ret)
        total = psutil.virtual_memory().total
        for name in ret._fields:
            value = getattr(ret, name)
            self.assertIsInstance(value, (int, long))
            self.assertGreaterEqual(value, 0, msg=(name, value))
            if LINUX or OSX and name in ('vms', 'data'):
                # On Linux there are processes (e.g. 'goa-daemon') whose
                # VMS is incredibly high for some reason.
                continue
            self.assertLessEqual(value, total, msg=(name, value, total))

        if LINUX:
            self.assertGreaterEqual(ret.pss, ret.uss)

    def open_files(self, ret, info):
        self.assertIsInstance(ret, list)
        for f in ret:
            self.assertIsInstance(f.fd, int)
            self.assertIsInstance(f.path, str)
            self.assertEqual(f.path.strip(), f.path)
            if WINDOWS:
                self.assertEqual(f.fd, -1)
            elif LINUX:
                self.assertIsInstance(f.position, int)
                self.assertIsInstance(f.mode, str)
                self.assertIsInstance(f.flags, int)
                self.assertGreaterEqual(f.position, 0)
                self.assertIn(f.mode, ('r', 'w', 'a', 'r+', 'a+'))
                self.assertGreater(f.flags, 0)
            elif BSD and not f.path:
                # XXX see: https://github.com/giampaolo/psutil/issues/595
                continue
            assert os.path.isabs(f.path), f
            try:
                st = os.stat(f.path)
            except FileNotFoundError:
                pass
            else:
                assert stat.S_ISREG(st.st_mode), f

    def num_fds(self, ret, info):
        self.assertIsInstance(ret, int)
        self.assertGreaterEqual(ret, 0)

    def connections(self, ret, info):
        with create_sockets():
            self.assertEqual(len(ret), len(set(ret)))
            for conn in ret:
                assert is_namedtuple(conn)
                check_connection_ntuple(conn)

    def cwd(self, ret, info):
        self.assertIsInstance(ret, (str, unicode))
        self.assertEqual(ret.strip(), ret)
        if ret:
            assert os.path.isabs(ret), ret
            try:
                st = os.stat(ret)
            except OSError as err:
                if WINDOWS and err.errno in \
                        psutil._psplatform.ACCESS_DENIED_SET:
                    pass
                # directory has been removed in mean time
                elif err.errno != errno.ENOENT:
                    raise
            else:
                assert stat.S_ISDIR(st.st_mode)

    def memory_percent(self, ret, info):
        self.assertIsInstance(ret, float)
        assert 0 <= ret <= 100, ret

    def is_running(self, ret, info):
        self.assertIsInstance(ret, bool)

    def cpu_affinity(self, ret, info):
        self.assertIsInstance(ret, list)
        assert ret != [], ret
        cpus = list(range(psutil.cpu_count()))
        for n in ret:
            self.assertIsInstance(n, int)
            self.assertIn(n, cpus)

    def terminal(self, ret, info):
        self.assertIsInstance(ret, (str, type(None)))
        if ret is not None:
            assert os.path.isabs(ret), ret
            assert os.path.exists(ret), ret

    def memory_maps(self, ret, info):
        for nt in ret:
            self.assertIsInstance(nt.addr, str)
            self.assertIsInstance(nt.perms, str)
            self.assertIsInstance(nt.path, str)
            for fname in nt._fields:
                value = getattr(nt, fname)
                if fname == 'path':
                    if not value.startswith(("[", "anon_inode:")):
                        assert os.path.isabs(nt.path), nt.path
                        # commented as on Linux we might get
                        # '/foo/bar (deleted)'
                        # assert os.path.exists(nt.path), nt.path
                elif fname == 'addr':
                    assert value, repr(value)
                elif fname == 'perms':
                    if not WINDOWS:
                        assert value, repr(value)
                else:
                    self.assertIsInstance(value, (int, long))
                    self.assertGreaterEqual(value, 0)

    def num_handles(self, ret, info):
        self.assertIsInstance(ret, int)
        self.assertGreaterEqual(ret, 0)

    def nice(self, ret, info):
        self.assertIsInstance(ret, int)
        if POSIX:
            assert -20 <= ret <= 20, ret
        else:
            priorities = [getattr(psutil, x) for x in dir(psutil)
                          if x.endswith('_PRIORITY_CLASS')]
            self.assertIn(ret, priorities)
            if sys.version_info > (3, 4):
                self.assertIsInstance(ret, enum.IntEnum)
            else:
                self.assertIsInstance(ret, int)

    def num_ctx_switches(self, ret, info):
        assert is_namedtuple(ret)
        for value in ret:
            self.assertIsInstance(value, (int, long))
            self.assertGreaterEqual(value, 0)

    def rlimit(self, ret, info):
        self.assertIsInstance(ret, tuple)
        self.assertEqual(len(ret), 2)
        self.assertGreaterEqual(ret[0], -1)
        self.assertGreaterEqual(ret[1], -1)

    def environ(self, ret, info):
        self.assertIsInstance(ret, dict)
        for k, v in ret.items():
            self.assertIsInstance(k, str)
            self.assertIsInstance(v, str)


if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)
