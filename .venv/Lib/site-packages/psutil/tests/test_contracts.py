#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Contracts tests. These tests mainly check API sanity in terms of
returned types and APIs availability.
Some of these are duplicates of tests test_system.py and test_process.py.
"""

import platform
import signal
import unittest

import psutil
from psutil import AIX
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import long
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYPY
from psutil.tests import SKIP_SYSCONS
from psutil.tests import PsutilTestCase
from psutil.tests import create_sockets
from psutil.tests import enum
from psutil.tests import is_namedtuple
from psutil.tests import kernel_version


# ===================================================================
# --- APIs availability
# ===================================================================

# Make sure code reflects what doc promises in terms of APIs
# availability.


class TestAvailConstantsAPIs(PsutilTestCase):
    def test_PROCFS_PATH(self):
        self.assertEqual(hasattr(psutil, "PROCFS_PATH"), LINUX or SUNOS or AIX)

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

    @unittest.skipIf(
        GITHUB_ACTIONS and LINUX, "unsupported on GITHUB_ACTIONS + LINUX"
    )
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
        self.assertEqual(
            hasattr(psutil, "cpu_freq"),
            LINUX or MACOS or WINDOWS or FREEBSD or OPENBSD,
        )

    def test_sensors_temperatures(self):
        self.assertEqual(
            hasattr(psutil, "sensors_temperatures"), LINUX or FREEBSD
        )

    def test_sensors_fans(self):
        self.assertEqual(hasattr(psutil, "sensors_fans"), LINUX)

    def test_battery(self):
        self.assertEqual(
            hasattr(psutil, "sensors_battery"),
            LINUX or WINDOWS or FREEBSD or MACOS,
        )


class TestAvailProcessAPIs(PsutilTestCase):
    def test_environ(self):
        self.assertEqual(
            hasattr(psutil.Process, "environ"),
            LINUX
            or MACOS
            or WINDOWS
            or AIX
            or SUNOS
            or FREEBSD
            or OPENBSD
            or NETBSD,
        )

    def test_uids(self):
        self.assertEqual(hasattr(psutil.Process, "uids"), POSIX)

    def test_gids(self):
        self.assertEqual(hasattr(psutil.Process, "uids"), POSIX)

    def test_terminal(self):
        self.assertEqual(hasattr(psutil.Process, "terminal"), POSIX)

    def test_ionice(self):
        self.assertEqual(hasattr(psutil.Process, "ionice"), LINUX or WINDOWS)

    @unittest.skipIf(
        GITHUB_ACTIONS and LINUX, "unsupported on GITHUB_ACTIONS + LINUX"
    )
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
        self.assertEqual(
            hasattr(psutil.Process, "cpu_affinity"),
            LINUX or WINDOWS or FREEBSD,
        )

    def test_cpu_num(self):
        self.assertEqual(
            hasattr(psutil.Process, "cpu_num"), LINUX or FREEBSD or SUNOS
        )

    def test_memory_maps(self):
        hasit = hasattr(psutil.Process, "memory_maps")
        self.assertEqual(hasit, not (OPENBSD or NETBSD or AIX or MACOS))


# ===================================================================
# --- API types
# ===================================================================


class TestSystemAPITypes(PsutilTestCase):
    """Check the return types of system related APIs.
    Mainly we want to test we never return unicode on Python 2, see:
    https://github.com/giampaolo/psutil/issues/1039.
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
    @unittest.skipIf(
        MACOS and platform.machine() == 'arm64', "skipped due to #1892"
    )
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
        for ifname in psutil.net_io_counters(pernic=True):
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


if __name__ == '__main__':
    from psutil.tests.runner import run_from_name

    run_from_name(__file__)
