#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""macOS specific tests."""

import platform
import re
import time
import unittest

import psutil
from psutil import MACOS
from psutil import POSIX
from psutil.tests import HAS_BATTERY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate


if POSIX:
    from psutil._psutil_posix import getpagesize


def sysctl(cmdline):
    """Expects a sysctl command with an argument and parse the result
    returning only the value of interest.
    """
    out = sh(cmdline)
    result = out.split()[1]
    try:
        return int(result)
    except ValueError:
        return result


def vm_stat(field):
    """Wrapper around 'vm_stat' cmdline utility."""
    out = sh('vm_stat')
    for line in out.split('\n'):
        if field in line:
            break
    else:
        raise ValueError("line not found")
    return int(re.search(r'\d+', line).group(0)) * getpagesize()


@unittest.skipIf(not MACOS, "MACOS only")
class TestProcess(PsutilTestCase):

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_testproc().pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    def test_process_create_time(self):
        output = sh("ps -o lstart -p %s" % self.pid)
        start_ps = output.replace('STARTED', '').strip()
        hhmmss = start_ps.split(' ')[-2]
        year = start_ps.split(' ')[-1]
        start_psutil = psutil.Process(self.pid).create_time()
        self.assertEqual(
            hhmmss,
            time.strftime("%H:%M:%S", time.localtime(start_psutil)))
        self.assertEqual(
            year,
            time.strftime("%Y", time.localtime(start_psutil)))


@unittest.skipIf(not MACOS, "MACOS only")
class TestSystemAPIs(PsutilTestCase):

    # --- disk

    @retry_on_failure()
    def test_disks(self):
        # test psutil.disk_usage() and psutil.disk_partitions()
        # against "df -a"
        def df(path):
            out = sh('df -k "%s"' % path).strip()
            lines = out.split('\n')
            lines.pop(0)
            line = lines.pop(0)
            dev, total, used, free = line.split()[:4]
            if dev == 'none':
                dev = ''
            total = int(total) * 1024
            used = int(used) * 1024
            free = int(free) * 1024
            return dev, total, used, free

        for part in psutil.disk_partitions(all=False):
            usage = psutil.disk_usage(part.mountpoint)
            dev, total, used, free = df(part.mountpoint)
            self.assertEqual(part.device, dev)
            self.assertEqual(usage.total, total)
            self.assertAlmostEqual(usage.free, free,
                                   delta=TOLERANCE_DISK_USAGE)
            self.assertAlmostEqual(usage.used, used,
                                   delta=TOLERANCE_DISK_USAGE)

    # --- cpu

    def test_cpu_count_logical(self):
        num = sysctl("sysctl hw.logicalcpu")
        self.assertEqual(num, psutil.cpu_count(logical=True))

    def test_cpu_count_cores(self):
        num = sysctl("sysctl hw.physicalcpu")
        self.assertEqual(num, psutil.cpu_count(logical=False))

    # TODO: remove this once 1892 is fixed
    @unittest.skipIf(platform.machine() == 'arm64', "skipped due to #1892")
    def test_cpu_freq(self):
        freq = psutil.cpu_freq()
        self.assertEqual(
            freq.current * 1000 * 1000, sysctl("sysctl hw.cpufrequency"))
        self.assertEqual(
            freq.min * 1000 * 1000, sysctl("sysctl hw.cpufrequency_min"))
        self.assertEqual(
            freq.max * 1000 * 1000, sysctl("sysctl hw.cpufrequency_max"))

    # --- virtual mem

    def test_vmem_total(self):
        sysctl_hwphymem = sysctl('sysctl hw.memsize')
        self.assertEqual(sysctl_hwphymem, psutil.virtual_memory().total)

    @retry_on_failure()
    def test_vmem_free(self):
        vmstat_val = vm_stat("free")
        psutil_val = psutil.virtual_memory().free
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_active(self):
        vmstat_val = vm_stat("active")
        psutil_val = psutil.virtual_memory().active
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_inactive(self):
        vmstat_val = vm_stat("inactive")
        psutil_val = psutil.virtual_memory().inactive
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_wired(self):
        vmstat_val = vm_stat("wired")
        psutil_val = psutil.virtual_memory().wired
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    # --- swap mem

    @retry_on_failure()
    def test_swapmem_sin(self):
        vmstat_val = vm_stat("Pageins")
        psutil_val = psutil.swap_memory().sin
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_swapmem_sout(self):
        vmstat_val = vm_stat("Pageout")
        psutil_val = psutil.swap_memory().sout
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    # --- network

    def test_net_if_stats(self):
        for name, stats in psutil.net_if_stats().items():
            try:
                out = sh("ifconfig %s" % name)
            except RuntimeError:
                pass
            else:
                self.assertEqual(stats.isup, 'RUNNING' in out, msg=out)
                self.assertEqual(stats.mtu,
                                 int(re.findall(r'mtu (\d+)', out)[0]))

    # --- sensors_battery

    @unittest.skipIf(not HAS_BATTERY, "no battery")
    def test_sensors_battery(self):
        out = sh("pmset -g batt")
        percent = re.search(r"(\d+)%", out).group(1)
        drawing_from = re.search("Now drawing from '([^']+)'", out).group(1)
        power_plugged = drawing_from == "AC Power"
        psutil_result = psutil.sensors_battery()
        self.assertEqual(psutil_result.power_plugged, power_plugged)
        self.assertEqual(psutil_result.percent, int(percent))


if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)
