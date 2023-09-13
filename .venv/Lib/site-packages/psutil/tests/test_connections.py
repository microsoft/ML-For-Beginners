#!/usr/bin/env python3

# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Tests for net_connections() and Process.connections() APIs."""

import os
import socket
import textwrap
import unittest
from contextlib import closing
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_DGRAM
from socket import SOCK_STREAM

import psutil
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil.tests import AF_UNIX
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import SKIP_SYSCONS
from psutil.tests import PsutilTestCase
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import reap_children
from psutil.tests import retry_on_failure
from psutil.tests import serialrun
from psutil.tests import skip_on_access_denied
from psutil.tests import tcp_socketpair
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file


thisproc = psutil.Process()
SOCK_SEQPACKET = getattr(socket, "SOCK_SEQPACKET", object())


@serialrun
class ConnectionTestCase(PsutilTestCase):

    def setUp(self):
        if not (NETBSD or FREEBSD):
            # process opens a UNIX socket to /var/log/run.
            cons = thisproc.connections(kind='all')
            assert not cons, cons

    def tearDown(self):
        if not (FREEBSD or NETBSD):
            # Make sure we closed all resources.
            # NetBSD opens a UNIX socket to /var/log/run.
            cons = thisproc.connections(kind='all')
            assert not cons, cons

    def compare_procsys_connections(self, pid, proc_cons, kind='all'):
        """Given a process PID and its list of connections compare
        those against system-wide connections retrieved via
        psutil.net_connections.
        """
        try:
            sys_cons = psutil.net_connections(kind=kind)
        except psutil.AccessDenied:
            # On MACOS, system-wide connections are retrieved by iterating
            # over all processes
            if MACOS:
                return
            else:
                raise
        # Filter for this proc PID and exlucde PIDs from the tuple.
        sys_cons = [c[:-1] for c in sys_cons if c.pid == pid]
        sys_cons.sort()
        proc_cons.sort()
        self.assertEqual(proc_cons, sys_cons)


class TestBasicOperations(ConnectionTestCase):

    @unittest.skipIf(SKIP_SYSCONS, "requires root")
    def test_system(self):
        with create_sockets():
            for conn in psutil.net_connections(kind='all'):
                check_connection_ntuple(conn)

    def test_process(self):
        with create_sockets():
            for conn in psutil.Process().connections(kind='all'):
                check_connection_ntuple(conn)

    def test_invalid_kind(self):
        self.assertRaises(ValueError, thisproc.connections, kind='???')
        self.assertRaises(ValueError, psutil.net_connections, kind='???')


@serialrun
class TestUnconnectedSockets(ConnectionTestCase):
    """Tests sockets which are open but not connected to anything."""

    def get_conn_from_sock(self, sock):
        cons = thisproc.connections(kind='all')
        smap = dict([(c.fd, c) for c in cons])
        if NETBSD or FREEBSD:
            # NetBSD opens a UNIX socket to /var/log/run
            # so there may be more connections.
            return smap[sock.fileno()]
        else:
            self.assertEqual(len(cons), 1)
            if cons[0].fd != -1:
                self.assertEqual(smap[sock.fileno()].fd, sock.fileno())
            return cons[0]

    def check_socket(self, sock):
        """Given a socket, makes sure it matches the one obtained
        via psutil. It assumes this process created one connection
        only (the one supposed to be checked).
        """
        conn = self.get_conn_from_sock(sock)
        check_connection_ntuple(conn)

        # fd, family, type
        if conn.fd != -1:
            self.assertEqual(conn.fd, sock.fileno())
        self.assertEqual(conn.family, sock.family)
        # see: http://bugs.python.org/issue30204
        self.assertEqual(
            conn.type, sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE))

        # local address
        laddr = sock.getsockname()
        if not laddr and PY3 and isinstance(laddr, bytes):
            # See: http://bugs.python.org/issue30205
            laddr = laddr.decode()
        if sock.family == AF_INET6:
            laddr = laddr[:2]
        self.assertEqual(conn.laddr, laddr)

        # XXX Solaris can't retrieve system-wide UNIX sockets
        if sock.family == AF_UNIX and HAS_CONNECTIONS_UNIX:
            cons = thisproc.connections(kind='all')
            self.compare_procsys_connections(os.getpid(), cons, kind='all')
        return conn

    def test_tcp_v4(self):
        addr = ("127.0.0.1", 0)
        with closing(bind_socket(AF_INET, SOCK_STREAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            assert not conn.raddr
            self.assertEqual(conn.status, psutil.CONN_LISTEN)

    @unittest.skipIf(not supports_ipv6(), "IPv6 not supported")
    def test_tcp_v6(self):
        addr = ("::1", 0)
        with closing(bind_socket(AF_INET6, SOCK_STREAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            assert not conn.raddr
            self.assertEqual(conn.status, psutil.CONN_LISTEN)

    def test_udp_v4(self):
        addr = ("127.0.0.1", 0)
        with closing(bind_socket(AF_INET, SOCK_DGRAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            assert not conn.raddr
            self.assertEqual(conn.status, psutil.CONN_NONE)

    @unittest.skipIf(not supports_ipv6(), "IPv6 not supported")
    def test_udp_v6(self):
        addr = ("::1", 0)
        with closing(bind_socket(AF_INET6, SOCK_DGRAM, addr=addr)) as sock:
            conn = self.check_socket(sock)
            assert not conn.raddr
            self.assertEqual(conn.status, psutil.CONN_NONE)

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_unix_tcp(self):
        testfn = self.get_testfn()
        with closing(bind_unix_socket(testfn, type=SOCK_STREAM)) as sock:
            conn = self.check_socket(sock)
            assert not conn.raddr
            self.assertEqual(conn.status, psutil.CONN_NONE)

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_unix_udp(self):
        testfn = self.get_testfn()
        with closing(bind_unix_socket(testfn, type=SOCK_STREAM)) as sock:
            conn = self.check_socket(sock)
            assert not conn.raddr
            self.assertEqual(conn.status, psutil.CONN_NONE)


@serialrun
class TestConnectedSocket(ConnectionTestCase):
    """Test socket pairs which are are actually connected to
    each other.
    """

    # On SunOS, even after we close() it, the server socket stays around
    # in TIME_WAIT state.
    @unittest.skipIf(SUNOS, "unreliable on SUONS")
    def test_tcp(self):
        addr = ("127.0.0.1", 0)
        assert not thisproc.connections(kind='tcp4')
        server, client = tcp_socketpair(AF_INET, addr=addr)
        try:
            cons = thisproc.connections(kind='tcp4')
            self.assertEqual(len(cons), 2)
            self.assertEqual(cons[0].status, psutil.CONN_ESTABLISHED)
            self.assertEqual(cons[1].status, psutil.CONN_ESTABLISHED)
            # May not be fast enough to change state so it stays
            # commenteed.
            # client.close()
            # cons = thisproc.connections(kind='all')
            # self.assertEqual(len(cons), 1)
            # self.assertEqual(cons[0].status, psutil.CONN_CLOSE_WAIT)
        finally:
            server.close()
            client.close()

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_unix(self):
        testfn = self.get_testfn()
        server, client = unix_socketpair(testfn)
        try:
            cons = thisproc.connections(kind='unix')
            assert not (cons[0].laddr and cons[0].raddr)
            assert not (cons[1].laddr and cons[1].raddr)
            if NETBSD or FREEBSD:
                # On NetBSD creating a UNIX socket will cause
                # a UNIX connection to  /var/run/log.
                cons = [c for c in cons if c.raddr != '/var/run/log']
            self.assertEqual(len(cons), 2, msg=cons)
            if LINUX or FREEBSD or SUNOS or OPENBSD:
                # remote path is never set
                self.assertEqual(cons[0].raddr, "")
                self.assertEqual(cons[1].raddr, "")
                # one local address should though
                self.assertEqual(testfn, cons[0].laddr or cons[1].laddr)
            else:
                # On other systems either the laddr or raddr
                # of both peers are set.
                self.assertEqual(cons[0].laddr or cons[1].laddr, testfn)
        finally:
            server.close()
            client.close()


class TestFilters(ConnectionTestCase):

    def test_filters(self):
        def check(kind, families, types):
            for conn in thisproc.connections(kind=kind):
                self.assertIn(conn.family, families)
                self.assertIn(conn.type, types)
            if not SKIP_SYSCONS:
                for conn in psutil.net_connections(kind=kind):
                    self.assertIn(conn.family, families)
                    self.assertIn(conn.type, types)

        with create_sockets():
            check('all',
                  [AF_INET, AF_INET6, AF_UNIX],
                  [SOCK_STREAM, SOCK_DGRAM, SOCK_SEQPACKET])
            check('inet',
                  [AF_INET, AF_INET6],
                  [SOCK_STREAM, SOCK_DGRAM])
            check('inet4',
                  [AF_INET],
                  [SOCK_STREAM, SOCK_DGRAM])
            check('tcp',
                  [AF_INET, AF_INET6],
                  [SOCK_STREAM])
            check('tcp4',
                  [AF_INET],
                  [SOCK_STREAM])
            check('tcp6',
                  [AF_INET6],
                  [SOCK_STREAM])
            check('udp',
                  [AF_INET, AF_INET6],
                  [SOCK_DGRAM])
            check('udp4',
                  [AF_INET],
                  [SOCK_DGRAM])
            check('udp6',
                  [AF_INET6],
                  [SOCK_DGRAM])
            if HAS_CONNECTIONS_UNIX:
                check('unix',
                      [AF_UNIX],
                      [SOCK_STREAM, SOCK_DGRAM, SOCK_SEQPACKET])

    @skip_on_access_denied(only_if=MACOS)
    def test_combos(self):
        reap_children()

        def check_conn(proc, conn, family, type, laddr, raddr, status, kinds):
            all_kinds = ("all", "inet", "inet4", "inet6", "tcp", "tcp4",
                         "tcp6", "udp", "udp4", "udp6")
            check_connection_ntuple(conn)
            self.assertEqual(conn.family, family)
            self.assertEqual(conn.type, type)
            self.assertEqual(conn.laddr, laddr)
            self.assertEqual(conn.raddr, raddr)
            self.assertEqual(conn.status, status)
            for kind in all_kinds:
                cons = proc.connections(kind=kind)
                if kind in kinds:
                    assert cons
                else:
                    assert not cons, cons
            # compare against system-wide connections
            # XXX Solaris can't retrieve system-wide UNIX
            # sockets.
            if HAS_CONNECTIONS_UNIX:
                self.compare_procsys_connections(proc.pid, [conn])

        tcp_template = textwrap.dedent("""
            import socket, time
            s = socket.socket({family}, socket.SOCK_STREAM)
            s.bind(('{addr}', 0))
            s.listen(5)
            with open('{testfn}', 'w') as f:
                f.write(str(s.getsockname()[:2]))
            time.sleep(60)
            """)

        udp_template = textwrap.dedent("""
            import socket, time
            s = socket.socket({family}, socket.SOCK_DGRAM)
            s.bind(('{addr}', 0))
            with open('{testfn}', 'w') as f:
                f.write(str(s.getsockname()[:2]))
            time.sleep(60)
            """)

        # must be relative on Windows
        testfile = os.path.basename(self.get_testfn(dir=os.getcwd()))
        tcp4_template = tcp_template.format(
            family=int(AF_INET), addr="127.0.0.1", testfn=testfile)
        udp4_template = udp_template.format(
            family=int(AF_INET), addr="127.0.0.1", testfn=testfile)
        tcp6_template = tcp_template.format(
            family=int(AF_INET6), addr="::1", testfn=testfile)
        udp6_template = udp_template.format(
            family=int(AF_INET6), addr="::1", testfn=testfile)

        # launch various subprocess instantiating a socket of various
        # families and types to enrich psutil results
        tcp4_proc = self.pyrun(tcp4_template)
        tcp4_addr = eval(wait_for_file(testfile, delete=True))
        udp4_proc = self.pyrun(udp4_template)
        udp4_addr = eval(wait_for_file(testfile, delete=True))
        if supports_ipv6():
            tcp6_proc = self.pyrun(tcp6_template)
            tcp6_addr = eval(wait_for_file(testfile, delete=True))
            udp6_proc = self.pyrun(udp6_template)
            udp6_addr = eval(wait_for_file(testfile, delete=True))
        else:
            tcp6_proc = None
            udp6_proc = None
            tcp6_addr = None
            udp6_addr = None

        for p in thisproc.children():
            cons = p.connections()
            self.assertEqual(len(cons), 1)
            for conn in cons:
                # TCP v4
                if p.pid == tcp4_proc.pid:
                    check_conn(p, conn, AF_INET, SOCK_STREAM, tcp4_addr, (),
                               psutil.CONN_LISTEN,
                               ("all", "inet", "inet4", "tcp", "tcp4"))
                # UDP v4
                elif p.pid == udp4_proc.pid:
                    check_conn(p, conn, AF_INET, SOCK_DGRAM, udp4_addr, (),
                               psutil.CONN_NONE,
                               ("all", "inet", "inet4", "udp", "udp4"))
                # TCP v6
                elif p.pid == getattr(tcp6_proc, "pid", None):
                    check_conn(p, conn, AF_INET6, SOCK_STREAM, tcp6_addr, (),
                               psutil.CONN_LISTEN,
                               ("all", "inet", "inet6", "tcp", "tcp6"))
                # UDP v6
                elif p.pid == getattr(udp6_proc, "pid", None):
                    check_conn(p, conn, AF_INET6, SOCK_DGRAM, udp6_addr, (),
                               psutil.CONN_NONE,
                               ("all", "inet", "inet6", "udp", "udp6"))

    def test_count(self):
        with create_sockets():
            # tcp
            cons = thisproc.connections(kind='tcp')
            self.assertEqual(len(cons), 2 if supports_ipv6() else 1)
            for conn in cons:
                self.assertIn(conn.family, (AF_INET, AF_INET6))
                self.assertEqual(conn.type, SOCK_STREAM)
            # tcp4
            cons = thisproc.connections(kind='tcp4')
            self.assertEqual(len(cons), 1)
            self.assertEqual(cons[0].family, AF_INET)
            self.assertEqual(cons[0].type, SOCK_STREAM)
            # tcp6
            if supports_ipv6():
                cons = thisproc.connections(kind='tcp6')
                self.assertEqual(len(cons), 1)
                self.assertEqual(cons[0].family, AF_INET6)
                self.assertEqual(cons[0].type, SOCK_STREAM)
            # udp
            cons = thisproc.connections(kind='udp')
            self.assertEqual(len(cons), 2 if supports_ipv6() else 1)
            for conn in cons:
                self.assertIn(conn.family, (AF_INET, AF_INET6))
                self.assertEqual(conn.type, SOCK_DGRAM)
            # udp4
            cons = thisproc.connections(kind='udp4')
            self.assertEqual(len(cons), 1)
            self.assertEqual(cons[0].family, AF_INET)
            self.assertEqual(cons[0].type, SOCK_DGRAM)
            # udp6
            if supports_ipv6():
                cons = thisproc.connections(kind='udp6')
                self.assertEqual(len(cons), 1)
                self.assertEqual(cons[0].family, AF_INET6)
                self.assertEqual(cons[0].type, SOCK_DGRAM)
            # inet
            cons = thisproc.connections(kind='inet')
            self.assertEqual(len(cons), 4 if supports_ipv6() else 2)
            for conn in cons:
                self.assertIn(conn.family, (AF_INET, AF_INET6))
                self.assertIn(conn.type, (SOCK_STREAM, SOCK_DGRAM))
            # inet6
            if supports_ipv6():
                cons = thisproc.connections(kind='inet6')
                self.assertEqual(len(cons), 2)
                for conn in cons:
                    self.assertEqual(conn.family, AF_INET6)
                    self.assertIn(conn.type, (SOCK_STREAM, SOCK_DGRAM))
            # Skipped on BSD becayse by default the Python process
            # creates a UNIX socket to '/var/run/log'.
            if HAS_CONNECTIONS_UNIX and not (FREEBSD or NETBSD):
                cons = thisproc.connections(kind='unix')
                self.assertEqual(len(cons), 3)
                for conn in cons:
                    self.assertEqual(conn.family, AF_UNIX)
                    self.assertIn(conn.type, (SOCK_STREAM, SOCK_DGRAM))


@unittest.skipIf(SKIP_SYSCONS, "requires root")
class TestSystemWideConnections(ConnectionTestCase):
    """Tests for net_connections()."""

    def test_it(self):
        def check(cons, families, types_):
            for conn in cons:
                self.assertIn(conn.family, families, msg=conn)
                if conn.family != AF_UNIX:
                    self.assertIn(conn.type, types_, msg=conn)
                check_connection_ntuple(conn)

        with create_sockets():
            from psutil._common import conn_tmap
            for kind, groups in conn_tmap.items():
                # XXX: SunOS does not retrieve UNIX sockets.
                if kind == 'unix' and not HAS_CONNECTIONS_UNIX:
                    continue
                families, types_ = groups
                cons = psutil.net_connections(kind)
                self.assertEqual(len(cons), len(set(cons)))
                check(cons, families, types_)

    @retry_on_failure()
    def test_multi_sockets_procs(self):
        # Creates multiple sub processes, each creating different
        # sockets. For each process check that proc.connections()
        # and net_connections() return the same results.
        # This is done mainly to check whether net_connections()'s
        # pid is properly set, see:
        # https://github.com/giampaolo/psutil/issues/1013
        with create_sockets() as socks:
            expected = len(socks)
        pids = []
        times = 10
        fnames = []
        for _ in range(times):
            fname = self.get_testfn()
            fnames.append(fname)
            src = textwrap.dedent("""\
                import time, os
                from psutil.tests import create_sockets
                with create_sockets():
                    with open(r'%s', 'w') as f:
                        f.write("hello")
                    time.sleep(60)
                """ % fname)
            sproc = self.pyrun(src)
            pids.append(sproc.pid)

        # sync
        for fname in fnames:
            wait_for_file(fname)

        syscons = [x for x in psutil.net_connections(kind='all') if x.pid
                   in pids]
        for pid in pids:
            self.assertEqual(len([x for x in syscons if x.pid == pid]),
                             expected)
            p = psutil.Process(pid)
            self.assertEqual(len(p.connections('all')), expected)


class TestMisc(PsutilTestCase):

    def test_connection_constants(self):
        ints = []
        strs = []
        for name in dir(psutil):
            if name.startswith('CONN_'):
                num = getattr(psutil, name)
                str_ = str(num)
                assert str_.isupper(), str_
                self.assertNotIn(str, strs)
                self.assertNotIn(num, ints)
                ints.append(num)
                strs.append(str_)
        if SUNOS:
            psutil.CONN_IDLE
            psutil.CONN_BOUND
        if WINDOWS:
            psutil.CONN_DELETE_TCB


if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)
