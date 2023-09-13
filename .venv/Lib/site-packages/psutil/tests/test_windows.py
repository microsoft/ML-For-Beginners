#!/usr/bin/env python3
# -*- coding: UTF-8 -*

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Windows specific tests."""

import datetime
import errno
import glob
import os
import platform
import re
import signal
import subprocess
import sys
import time
import unittest
import warnings

import psutil
from psutil import WINDOWS
from psutil._compat import FileNotFoundError
from psutil._compat import super
from psutil._compat import which
from psutil.tests import APPVEYOR
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_BATTERY
from psutil.tests import IS_64BIT
from psutil.tests import PY3
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate


if WINDOWS and not PYPY:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import win32api  # requires "pip install pywin32"
        import win32con
        import win32process
        import wmi  # requires "pip install wmi" / "make setup-dev-env"

if WINDOWS:
    from psutil._pswindows import convert_oserror


cext = psutil._psplatform.cext


@unittest.skipIf(not WINDOWS, "WINDOWS only")
@unittest.skipIf(PYPY, "pywin32 not available on PYPY")
# https://github.com/giampaolo/psutil/pull/1762#issuecomment-632892692
@unittest.skipIf(GITHUB_ACTIONS and not PY3, "pywin32 broken on GITHUB + PY2")
class WindowsTestCase(PsutilTestCase):
    pass


def powershell(cmd):
    """Currently not used, but avalable just in case. Usage:

    >>> powershell(
        "Get-CIMInstance Win32_PageFileUsage | Select AllocatedBaseSize")
    """
    if not which("powershell.exe"):
        raise unittest.SkipTest("powershell.exe not available")
    cmdline = \
        'powershell.exe -ExecutionPolicy Bypass -NoLogo -NonInteractive ' + \
        '-NoProfile -WindowStyle Hidden -Command "%s"' % cmd
    return sh(cmdline)


def wmic(path, what, converter=int):
    """Currently not used, but avalable just in case. Usage:

    >>> wmic("Win32_OperatingSystem", "FreePhysicalMemory")
    2134124534
    """
    out = sh("wmic path %s get %s" % (path, what)).strip()
    data = "".join(out.splitlines()[1:]).strip()  # get rid of the header
    if converter is not None:
        if "," in what:
            return tuple([converter(x) for x in data.split()])
        else:
            return converter(data)
    else:
        return data


# ===================================================================
# System APIs
# ===================================================================


class TestCpuAPIs(WindowsTestCase):

    @unittest.skipIf('NUMBER_OF_PROCESSORS' not in os.environ,
                     'NUMBER_OF_PROCESSORS env var is not available')
    def test_cpu_count_vs_NUMBER_OF_PROCESSORS(self):
        # Will likely fail on many-cores systems:
        # https://stackoverflow.com/questions/31209256
        num_cpus = int(os.environ['NUMBER_OF_PROCESSORS'])
        self.assertEqual(num_cpus, psutil.cpu_count())

    def test_cpu_count_vs_GetSystemInfo(self):
        # Will likely fail on many-cores systems:
        # https://stackoverflow.com/questions/31209256
        sys_value = win32api.GetSystemInfo()[5]
        psutil_value = psutil.cpu_count()
        self.assertEqual(sys_value, psutil_value)

    def test_cpu_count_logical_vs_wmi(self):
        w = wmi.WMI()
        procs = sum(proc.NumberOfLogicalProcessors
                    for proc in w.Win32_Processor())
        self.assertEqual(psutil.cpu_count(), procs)

    def test_cpu_count_cores_vs_wmi(self):
        w = wmi.WMI()
        cores = sum(proc.NumberOfCores for proc in w.Win32_Processor())
        self.assertEqual(psutil.cpu_count(logical=False), cores)

    def test_cpu_count_vs_cpu_times(self):
        self.assertEqual(psutil.cpu_count(),
                         len(psutil.cpu_times(percpu=True)))

    def test_cpu_freq(self):
        w = wmi.WMI()
        proc = w.Win32_Processor()[0]
        self.assertEqual(proc.CurrentClockSpeed, psutil.cpu_freq().current)
        self.assertEqual(proc.MaxClockSpeed, psutil.cpu_freq().max)


class TestSystemAPIs(WindowsTestCase):

    def test_nic_names(self):
        out = sh('ipconfig /all')
        nics = psutil.net_io_counters(pernic=True).keys()
        for nic in nics:
            if "pseudo-interface" in nic.replace(' ', '-').lower():
                continue
            if nic not in out:
                raise self.fail(
                    "%r nic wasn't found in 'ipconfig /all' output" % nic)

    def test_total_phymem(self):
        w = wmi.WMI().Win32_ComputerSystem()[0]
        self.assertEqual(int(w.TotalPhysicalMemory),
                         psutil.virtual_memory().total)

    def test_free_phymem(self):
        w = wmi.WMI().Win32_PerfRawData_PerfOS_Memory()[0]
        self.assertAlmostEqual(
            int(w.AvailableBytes), psutil.virtual_memory().free,
            delta=TOLERANCE_SYS_MEM)

    def test_total_swapmem(self):
        w = wmi.WMI().Win32_PerfRawData_PerfOS_Memory()[0]
        self.assertEqual(int(w.CommitLimit) - psutil.virtual_memory().total,
                         psutil.swap_memory().total)
        if (psutil.swap_memory().total == 0):
            self.assertEqual(0, psutil.swap_memory().free)
            self.assertEqual(0, psutil.swap_memory().used)

    def test_percent_swapmem(self):
        if (psutil.swap_memory().total > 0):
            w = wmi.WMI().Win32_PerfRawData_PerfOS_PagingFile(
                Name="_Total")[0]
            # calculate swap usage to percent
            percentSwap = int(w.PercentUsage) * 100 / int(w.PercentUsage_Base)
            # exact percent may change but should be reasonable
            # assert within +/- 5% and between 0 and 100%
            self.assertGreaterEqual(psutil.swap_memory().percent, 0)
            self.assertAlmostEqual(psutil.swap_memory().percent, percentSwap,
                                   delta=5)
            self.assertLessEqual(psutil.swap_memory().percent, 100)

    # @unittest.skipIf(wmi is None, "wmi module is not installed")
    # def test__UPTIME(self):
    #     # _UPTIME constant is not public but it is used internally
    #     # as value to return for pid 0 creation time.
    #     # WMI behaves the same.
    #     w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
    #     p = psutil.Process(0)
    #     wmic_create = str(w.CreationDate.split('.')[0])
    #     psutil_create = time.strftime("%Y%m%d%H%M%S",
    #                                   time.localtime(p.create_time()))

    # Note: this test is not very reliable
    @unittest.skipIf(APPVEYOR, "test not relieable on appveyor")
    @retry_on_failure()
    def test_pids(self):
        # Note: this test might fail if the OS is starting/killing
        # other processes in the meantime
        w = wmi.WMI().Win32_Process()
        wmi_pids = set([x.ProcessId for x in w])
        psutil_pids = set(psutil.pids())
        self.assertEqual(wmi_pids, psutil_pids)

    @retry_on_failure()
    def test_disks(self):
        ps_parts = psutil.disk_partitions(all=True)
        wmi_parts = wmi.WMI().Win32_LogicalDisk()
        for ps_part in ps_parts:
            for wmi_part in wmi_parts:
                if ps_part.device.replace('\\', '') == wmi_part.DeviceID:
                    if not ps_part.mountpoint:
                        # this is usually a CD-ROM with no disk inserted
                        break
                    if 'cdrom' in ps_part.opts:
                        break
                    if ps_part.mountpoint.startswith('A:'):
                        break  # floppy
                    try:
                        usage = psutil.disk_usage(ps_part.mountpoint)
                    except FileNotFoundError:
                        # usually this is the floppy
                        break
                    self.assertEqual(usage.total, int(wmi_part.Size))
                    wmi_free = int(wmi_part.FreeSpace)
                    self.assertEqual(usage.free, wmi_free)
                    # 10 MB tolerance
                    if abs(usage.free - wmi_free) > 10 * 1024 * 1024:
                        raise self.fail("psutil=%s, wmi=%s" % (
                            usage.free, wmi_free))
                    break
            else:
                raise self.fail("can't find partition %s" % repr(ps_part))

    @retry_on_failure()
    def test_disk_usage(self):
        for disk in psutil.disk_partitions():
            if 'cdrom' in disk.opts:
                continue
            sys_value = win32api.GetDiskFreeSpaceEx(disk.mountpoint)
            psutil_value = psutil.disk_usage(disk.mountpoint)
            self.assertAlmostEqual(sys_value[0], psutil_value.free,
                                   delta=TOLERANCE_DISK_USAGE)
            self.assertAlmostEqual(sys_value[1], psutil_value.total,
                                   delta=TOLERANCE_DISK_USAGE)
            self.assertEqual(psutil_value.used,
                             psutil_value.total - psutil_value.free)

    def test_disk_partitions(self):
        sys_value = [
            x + '\\' for x in win32api.GetLogicalDriveStrings().split("\\\x00")
            if x and not x.startswith('A:')]
        psutil_value = [x.mountpoint for x in psutil.disk_partitions(all=True)
                        if not x.mountpoint.startswith('A:')]
        self.assertEqual(sys_value, psutil_value)

    def test_net_if_stats(self):
        ps_names = set(cext.net_if_stats())
        wmi_adapters = wmi.WMI().Win32_NetworkAdapter()
        wmi_names = set()
        for wmi_adapter in wmi_adapters:
            wmi_names.add(wmi_adapter.Name)
            wmi_names.add(wmi_adapter.NetConnectionID)
        self.assertTrue(ps_names & wmi_names,
                        "no common entries in %s, %s" % (ps_names, wmi_names))

    def test_boot_time(self):
        wmi_os = wmi.WMI().Win32_OperatingSystem()
        wmi_btime_str = wmi_os[0].LastBootUpTime.split('.')[0]
        wmi_btime_dt = datetime.datetime.strptime(
            wmi_btime_str, "%Y%m%d%H%M%S")
        psutil_dt = datetime.datetime.fromtimestamp(psutil.boot_time())
        diff = abs((wmi_btime_dt - psutil_dt).total_seconds())
        self.assertLessEqual(diff, 5)

    def test_boot_time_fluctuation(self):
        # https://github.com/giampaolo/psutil/issues/1007
        with mock.patch('psutil._pswindows.cext.boot_time', return_value=5):
            self.assertEqual(psutil.boot_time(), 5)
        with mock.patch('psutil._pswindows.cext.boot_time', return_value=4):
            self.assertEqual(psutil.boot_time(), 5)
        with mock.patch('psutil._pswindows.cext.boot_time', return_value=6):
            self.assertEqual(psutil.boot_time(), 5)
        with mock.patch('psutil._pswindows.cext.boot_time', return_value=333):
            self.assertEqual(psutil.boot_time(), 333)


# ===================================================================
# sensors_battery()
# ===================================================================


class TestSensorsBattery(WindowsTestCase):

    def test_has_battery(self):
        if win32api.GetPwrCapabilities()['SystemBatteriesPresent']:
            self.assertIsNotNone(psutil.sensors_battery())
        else:
            self.assertIsNone(psutil.sensors_battery())

    @unittest.skipIf(not HAS_BATTERY, "no battery")
    def test_percent(self):
        w = wmi.WMI()
        battery_wmi = w.query('select * from Win32_Battery')[0]
        battery_psutil = psutil.sensors_battery()
        self.assertAlmostEqual(
            battery_psutil.percent, battery_wmi.EstimatedChargeRemaining,
            delta=1)

    @unittest.skipIf(not HAS_BATTERY, "no battery")
    def test_power_plugged(self):
        w = wmi.WMI()
        battery_wmi = w.query('select * from Win32_Battery')[0]
        battery_psutil = psutil.sensors_battery()
        # Status codes:
        # https://msdn.microsoft.com/en-us/library/aa394074(v=vs.85).aspx
        self.assertEqual(battery_psutil.power_plugged,
                         battery_wmi.BatteryStatus == 2)

    def test_emulate_no_battery(self):
        with mock.patch("psutil._pswindows.cext.sensors_battery",
                        return_value=(0, 128, 0, 0)) as m:
            self.assertIsNone(psutil.sensors_battery())
            assert m.called

    def test_emulate_power_connected(self):
        with mock.patch("psutil._pswindows.cext.sensors_battery",
                        return_value=(1, 0, 0, 0)) as m:
            self.assertEqual(psutil.sensors_battery().secsleft,
                             psutil.POWER_TIME_UNLIMITED)
            assert m.called

    def test_emulate_power_charging(self):
        with mock.patch("psutil._pswindows.cext.sensors_battery",
                        return_value=(0, 8, 0, 0)) as m:
            self.assertEqual(psutil.sensors_battery().secsleft,
                             psutil.POWER_TIME_UNLIMITED)
            assert m.called

    def test_emulate_secs_left_unknown(self):
        with mock.patch("psutil._pswindows.cext.sensors_battery",
                        return_value=(0, 0, 0, -1)) as m:
            self.assertEqual(psutil.sensors_battery().secsleft,
                             psutil.POWER_TIME_UNKNOWN)
            assert m.called


# ===================================================================
# Process APIs
# ===================================================================


class TestProcess(WindowsTestCase):

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_testproc().pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    def test_issue_24(self):
        p = psutil.Process(0)
        self.assertRaises(psutil.AccessDenied, p.kill)

    def test_special_pid(self):
        p = psutil.Process(4)
        self.assertEqual(p.name(), 'System')
        # use __str__ to access all common Process properties to check
        # that nothing strange happens
        str(p)
        p.username()
        self.assertTrue(p.create_time() >= 0.0)
        try:
            rss, vms = p.memory_info()[:2]
        except psutil.AccessDenied:
            # expected on Windows Vista and Windows 7
            if not platform.uname()[1] in ('vista', 'win-7', 'win7'):
                raise
        else:
            self.assertTrue(rss > 0)

    def test_send_signal(self):
        p = psutil.Process(self.pid)
        self.assertRaises(ValueError, p.send_signal, signal.SIGINT)

    def test_num_handles_increment(self):
        p = psutil.Process(os.getpid())
        before = p.num_handles()
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION,
                                      win32con.FALSE, os.getpid())
        after = p.num_handles()
        self.assertEqual(after, before + 1)
        win32api.CloseHandle(handle)
        self.assertEqual(p.num_handles(), before)

    def test_ctrl_signals(self):
        p = psutil.Process(self.spawn_testproc().pid)
        p.send_signal(signal.CTRL_C_EVENT)
        p.send_signal(signal.CTRL_BREAK_EVENT)
        p.kill()
        p.wait()
        self.assertRaises(psutil.NoSuchProcess,
                          p.send_signal, signal.CTRL_C_EVENT)
        self.assertRaises(psutil.NoSuchProcess,
                          p.send_signal, signal.CTRL_BREAK_EVENT)

    def test_username(self):
        name = win32api.GetUserNameEx(win32con.NameSamCompatible)
        if name.endswith('$'):
            # When running as a service account (most likely to be
            # NetworkService), these user name calculations don't produce the
            # same result, causing the test to fail.
            raise unittest.SkipTest('running as service account')
        self.assertEqual(psutil.Process().username(), name)

    def test_cmdline(self):
        sys_value = re.sub('[ ]+', ' ', win32api.GetCommandLine()).strip()
        psutil_value = ' '.join(psutil.Process().cmdline())
        if sys_value[0] == '"' != psutil_value[0]:
            # The PyWin32 command line may retain quotes around argv[0] if they
            # were used unnecessarily, while psutil will omit them. So remove
            # the first 2 quotes from sys_value if not in psutil_value.
            # A path to an executable will not contain quotes, so this is safe.
            sys_value = sys_value.replace('"', '', 2)
        self.assertEqual(sys_value, psutil_value)

    # XXX - occasional failures

    # def test_cpu_times(self):
    #     handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION,
    #                                   win32con.FALSE, os.getpid())
    #     self.addCleanup(win32api.CloseHandle, handle)
    #     sys_value = win32process.GetProcessTimes(handle)
    #     psutil_value = psutil.Process().cpu_times()
    #     self.assertAlmostEqual(
    #         psutil_value.user, sys_value['UserTime'] / 10000000.0,
    #         delta=0.2)
    #     self.assertAlmostEqual(
    #         psutil_value.user, sys_value['KernelTime'] / 10000000.0,
    #         delta=0.2)

    def test_nice(self):
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION,
                                      win32con.FALSE, os.getpid())
        self.addCleanup(win32api.CloseHandle, handle)
        sys_value = win32process.GetPriorityClass(handle)
        psutil_value = psutil.Process().nice()
        self.assertEqual(psutil_value, sys_value)

    def test_memory_info(self):
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION,
                                      win32con.FALSE, self.pid)
        self.addCleanup(win32api.CloseHandle, handle)
        sys_value = win32process.GetProcessMemoryInfo(handle)
        psutil_value = psutil.Process(self.pid).memory_info()
        self.assertEqual(
            sys_value['PeakWorkingSetSize'], psutil_value.peak_wset)
        self.assertEqual(
            sys_value['WorkingSetSize'], psutil_value.wset)
        self.assertEqual(
            sys_value['QuotaPeakPagedPoolUsage'], psutil_value.peak_paged_pool)
        self.assertEqual(
            sys_value['QuotaPagedPoolUsage'], psutil_value.paged_pool)
        self.assertEqual(
            sys_value['QuotaPeakNonPagedPoolUsage'],
            psutil_value.peak_nonpaged_pool)
        self.assertEqual(
            sys_value['QuotaNonPagedPoolUsage'], psutil_value.nonpaged_pool)
        self.assertEqual(
            sys_value['PagefileUsage'], psutil_value.pagefile)
        self.assertEqual(
            sys_value['PeakPagefileUsage'], psutil_value.peak_pagefile)

        self.assertEqual(psutil_value.rss, psutil_value.wset)
        self.assertEqual(psutil_value.vms, psutil_value.pagefile)

    def test_wait(self):
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION,
                                      win32con.FALSE, self.pid)
        self.addCleanup(win32api.CloseHandle, handle)
        p = psutil.Process(self.pid)
        p.terminate()
        psutil_value = p.wait()
        sys_value = win32process.GetExitCodeProcess(handle)
        self.assertEqual(psutil_value, sys_value)

    def test_cpu_affinity(self):
        def from_bitmask(x):
            return [i for i in range(64) if (1 << i) & x]

        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION,
                                      win32con.FALSE, self.pid)
        self.addCleanup(win32api.CloseHandle, handle)
        sys_value = from_bitmask(
            win32process.GetProcessAffinityMask(handle)[0])
        psutil_value = psutil.Process(self.pid).cpu_affinity()
        self.assertEqual(psutil_value, sys_value)

    def test_io_counters(self):
        handle = win32api.OpenProcess(win32con.PROCESS_QUERY_INFORMATION,
                                      win32con.FALSE, os.getpid())
        self.addCleanup(win32api.CloseHandle, handle)
        sys_value = win32process.GetProcessIoCounters(handle)
        psutil_value = psutil.Process().io_counters()
        self.assertEqual(
            psutil_value.read_count, sys_value['ReadOperationCount'])
        self.assertEqual(
            psutil_value.write_count, sys_value['WriteOperationCount'])
        self.assertEqual(
            psutil_value.read_bytes, sys_value['ReadTransferCount'])
        self.assertEqual(
            psutil_value.write_bytes, sys_value['WriteTransferCount'])
        self.assertEqual(
            psutil_value.other_count, sys_value['OtherOperationCount'])
        self.assertEqual(
            psutil_value.other_bytes, sys_value['OtherTransferCount'])

    def test_num_handles(self):
        import ctypes
        import ctypes.wintypes
        PROCESS_QUERY_INFORMATION = 0x400
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION, 0, self.pid)
        self.addCleanup(ctypes.windll.kernel32.CloseHandle, handle)

        hndcnt = ctypes.wintypes.DWORD()
        ctypes.windll.kernel32.GetProcessHandleCount(
            handle, ctypes.byref(hndcnt))
        sys_value = hndcnt.value
        psutil_value = psutil.Process(self.pid).num_handles()
        self.assertEqual(psutil_value, sys_value)

    def test_error_partial_copy(self):
        # https://github.com/giampaolo/psutil/issues/875
        exc = WindowsError()
        exc.winerror = 299
        with mock.patch("psutil._psplatform.cext.proc_cwd", side_effect=exc):
            with mock.patch("time.sleep") as m:
                p = psutil.Process()
                self.assertRaises(psutil.AccessDenied, p.cwd)
        self.assertGreaterEqual(m.call_count, 5)

    def test_exe(self):
        # NtQuerySystemInformation succeeds if process is gone. Make sure
        # it raises NSP for a non existent pid.
        pid = psutil.pids()[-1] + 99999
        proc = psutil._psplatform.Process(pid)
        self.assertRaises(psutil.NoSuchProcess, proc.exe)


class TestProcessWMI(WindowsTestCase):
    """Compare Process API results with WMI."""

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_testproc().pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    def test_name(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        self.assertEqual(p.name(), w.Caption)

    # This fail on github because using virtualenv for test environment
    @unittest.skipIf(GITHUB_ACTIONS, "unreliable path on GITHUB_ACTIONS")
    def test_exe(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        # Note: wmi reports the exe as a lower case string.
        # Being Windows paths case-insensitive we ignore that.
        self.assertEqual(p.exe().lower(), w.ExecutablePath.lower())

    def test_cmdline(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        self.assertEqual(' '.join(p.cmdline()),
                         w.CommandLine.replace('"', ''))

    def test_username(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        domain, _, username = w.GetOwner()
        username = "%s\\%s" % (domain, username)
        self.assertEqual(p.username(), username)

    @retry_on_failure()
    def test_memory_rss(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        rss = p.memory_info().rss
        self.assertEqual(rss, int(w.WorkingSetSize))

    @retry_on_failure()
    def test_memory_vms(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        vms = p.memory_info().vms
        # http://msdn.microsoft.com/en-us/library/aa394372(VS.85).aspx
        # ...claims that PageFileUsage is represented in Kilo
        # bytes but funnily enough on certain platforms bytes are
        # returned instead.
        wmi_usage = int(w.PageFileUsage)
        if (vms != wmi_usage) and (vms != wmi_usage * 1024):
            raise self.fail("wmi=%s, psutil=%s" % (wmi_usage, vms))

    def test_create_time(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        wmic_create = str(w.CreationDate.split('.')[0])
        psutil_create = time.strftime("%Y%m%d%H%M%S",
                                      time.localtime(p.create_time()))
        self.assertEqual(wmic_create, psutil_create)


# ---


@unittest.skipIf(not WINDOWS, "WINDOWS only")
class TestDualProcessImplementation(PsutilTestCase):
    """
    Certain APIs on Windows have 2 internal implementations, one
    based on documented Windows APIs, another one based
    NtQuerySystemInformation() which gets called as fallback in
    case the first fails because of limited permission error.
    Here we test that the two methods return the exact same value,
    see:
    https://github.com/giampaolo/psutil/issues/304
    """

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_testproc().pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    def test_memory_info(self):
        mem_1 = psutil.Process(self.pid).memory_info()
        with mock.patch("psutil._psplatform.cext.proc_memory_info",
                        side_effect=OSError(errno.EPERM, "msg")) as fun:
            mem_2 = psutil.Process(self.pid).memory_info()
            self.assertEqual(len(mem_1), len(mem_2))
            for i in range(len(mem_1)):
                self.assertGreaterEqual(mem_1[i], 0)
                self.assertGreaterEqual(mem_2[i], 0)
                self.assertAlmostEqual(mem_1[i], mem_2[i], delta=512)
            assert fun.called

    def test_create_time(self):
        ctime = psutil.Process(self.pid).create_time()
        with mock.patch("psutil._psplatform.cext.proc_times",
                        side_effect=OSError(errno.EPERM, "msg")) as fun:
            self.assertEqual(psutil.Process(self.pid).create_time(), ctime)
            assert fun.called

    def test_cpu_times(self):
        cpu_times_1 = psutil.Process(self.pid).cpu_times()
        with mock.patch("psutil._psplatform.cext.proc_times",
                        side_effect=OSError(errno.EPERM, "msg")) as fun:
            cpu_times_2 = psutil.Process(self.pid).cpu_times()
            assert fun.called
            self.assertAlmostEqual(
                cpu_times_1.user, cpu_times_2.user, delta=0.01)
            self.assertAlmostEqual(
                cpu_times_1.system, cpu_times_2.system, delta=0.01)

    def test_io_counters(self):
        io_counters_1 = psutil.Process(self.pid).io_counters()
        with mock.patch("psutil._psplatform.cext.proc_io_counters",
                        side_effect=OSError(errno.EPERM, "msg")) as fun:
            io_counters_2 = psutil.Process(self.pid).io_counters()
            for i in range(len(io_counters_1)):
                self.assertAlmostEqual(
                    io_counters_1[i], io_counters_2[i], delta=5)
            assert fun.called

    def test_num_handles(self):
        num_handles = psutil.Process(self.pid).num_handles()
        with mock.patch("psutil._psplatform.cext.proc_num_handles",
                        side_effect=OSError(errno.EPERM, "msg")) as fun:
            self.assertEqual(psutil.Process(self.pid).num_handles(),
                             num_handles)
            assert fun.called

    def test_cmdline(self):
        for pid in psutil.pids():
            try:
                a = cext.proc_cmdline(pid, use_peb=True)
                b = cext.proc_cmdline(pid, use_peb=False)
            except OSError as err:
                err = convert_oserror(err)
                if not isinstance(err, (psutil.AccessDenied,
                                        psutil.NoSuchProcess)):
                    raise
            else:
                self.assertEqual(a, b)


@unittest.skipIf(not WINDOWS, "WINDOWS only")
class RemoteProcessTestCase(PsutilTestCase):
    """Certain functions require calling ReadProcessMemory.
    This trivially works when called on the current process.
    Check that this works on other processes, especially when they
    have a different bitness.
    """

    @staticmethod
    def find_other_interpreter():
        # find a python interpreter that is of the opposite bitness from us
        code = "import sys; sys.stdout.write(str(sys.maxsize > 2**32))"

        # XXX: a different and probably more stable approach might be to access
        # the registry but accessing 64 bit paths from a 32 bit process
        for filename in glob.glob(r"C:\Python*\python.exe"):
            proc = subprocess.Popen(args=[filename, "-c", code],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            output, _ = proc.communicate()
            proc.wait()
            if output == str(not IS_64BIT):
                return filename

    test_args = ["-c", "import sys; sys.stdin.read()"]

    def setUp(self):
        super().setUp()

        other_python = self.find_other_interpreter()
        if other_python is None:
            raise unittest.SkipTest(
                "could not find interpreter with opposite bitness")
        if IS_64BIT:
            self.python64 = sys.executable
            self.python32 = other_python
        else:
            self.python64 = other_python
            self.python32 = sys.executable

        env = os.environ.copy()
        env["THINK_OF_A_NUMBER"] = str(os.getpid())
        self.proc32 = self.spawn_testproc(
            [self.python32] + self.test_args,
            env=env,
            stdin=subprocess.PIPE)
        self.proc64 = self.spawn_testproc(
            [self.python64] + self.test_args,
            env=env,
            stdin=subprocess.PIPE)

    def tearDown(self):
        super().tearDown()
        self.proc32.communicate()
        self.proc64.communicate()

    def test_cmdline_32(self):
        p = psutil.Process(self.proc32.pid)
        self.assertEqual(len(p.cmdline()), 3)
        self.assertEqual(p.cmdline()[1:], self.test_args)

    def test_cmdline_64(self):
        p = psutil.Process(self.proc64.pid)
        self.assertEqual(len(p.cmdline()), 3)
        self.assertEqual(p.cmdline()[1:], self.test_args)

    def test_cwd_32(self):
        p = psutil.Process(self.proc32.pid)
        self.assertEqual(p.cwd(), os.getcwd())

    def test_cwd_64(self):
        p = psutil.Process(self.proc64.pid)
        self.assertEqual(p.cwd(), os.getcwd())

    def test_environ_32(self):
        p = psutil.Process(self.proc32.pid)
        e = p.environ()
        self.assertIn("THINK_OF_A_NUMBER", e)
        self.assertEqual(e["THINK_OF_A_NUMBER"], str(os.getpid()))

    def test_environ_64(self):
        p = psutil.Process(self.proc64.pid)
        try:
            p.environ()
        except psutil.AccessDenied:
            pass


# ===================================================================
# Windows services
# ===================================================================


@unittest.skipIf(not WINDOWS, "WINDOWS only")
class TestServices(PsutilTestCase):

    def test_win_service_iter(self):
        valid_statuses = set([
            "running",
            "paused",
            "start",
            "pause",
            "continue",
            "stop",
            "stopped",
        ])
        valid_start_types = set([
            "automatic",
            "manual",
            "disabled",
        ])
        valid_statuses = set([
            "running",
            "paused",
            "start_pending",
            "pause_pending",
            "continue_pending",
            "stop_pending",
            "stopped"
        ])
        for serv in psutil.win_service_iter():
            data = serv.as_dict()
            self.assertIsInstance(data['name'], str)
            self.assertNotEqual(data['name'].strip(), "")
            self.assertIsInstance(data['display_name'], str)
            self.assertIsInstance(data['username'], str)
            self.assertIn(data['status'], valid_statuses)
            if data['pid'] is not None:
                psutil.Process(data['pid'])
            self.assertIsInstance(data['binpath'], str)
            self.assertIsInstance(data['username'], str)
            self.assertIsInstance(data['start_type'], str)
            self.assertIn(data['start_type'], valid_start_types)
            self.assertIn(data['status'], valid_statuses)
            self.assertIsInstance(data['description'], str)
            pid = serv.pid()
            if pid is not None:
                p = psutil.Process(pid)
                self.assertTrue(p.is_running())
            # win_service_get
            s = psutil.win_service_get(serv.name())
            # test __eq__
            self.assertEqual(serv, s)

    def test_win_service_get(self):
        ERROR_SERVICE_DOES_NOT_EXIST = \
            psutil._psplatform.cext.ERROR_SERVICE_DOES_NOT_EXIST
        ERROR_ACCESS_DENIED = psutil._psplatform.cext.ERROR_ACCESS_DENIED

        name = next(psutil.win_service_iter()).name()
        with self.assertRaises(psutil.NoSuchProcess) as cm:
            psutil.win_service_get(name + '???')
        self.assertEqual(cm.exception.name, name + '???')

        # test NoSuchProcess
        service = psutil.win_service_get(name)
        if PY3:
            args = (0, "msg", 0, ERROR_SERVICE_DOES_NOT_EXIST)
        else:
            args = (ERROR_SERVICE_DOES_NOT_EXIST, "msg")
        exc = WindowsError(*args)
        with mock.patch("psutil._psplatform.cext.winservice_query_status",
                        side_effect=exc):
            self.assertRaises(psutil.NoSuchProcess, service.status)
        with mock.patch("psutil._psplatform.cext.winservice_query_config",
                        side_effect=exc):
            self.assertRaises(psutil.NoSuchProcess, service.username)

        # test AccessDenied
        if PY3:
            args = (0, "msg", 0, ERROR_ACCESS_DENIED)
        else:
            args = (ERROR_ACCESS_DENIED, "msg")
        exc = WindowsError(*args)
        with mock.patch("psutil._psplatform.cext.winservice_query_status",
                        side_effect=exc):
            self.assertRaises(psutil.AccessDenied, service.status)
        with mock.patch("psutil._psplatform.cext.winservice_query_config",
                        side_effect=exc):
            self.assertRaises(psutil.AccessDenied, service.username)

        # test __str__ and __repr__
        self.assertIn(service.name(), str(service))
        self.assertIn(service.display_name(), str(service))
        self.assertIn(service.name(), repr(service))
        self.assertIn(service.display_name(), repr(service))


if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)
