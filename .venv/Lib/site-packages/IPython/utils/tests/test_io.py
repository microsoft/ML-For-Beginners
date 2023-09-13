# encoding: utf-8
"""Tests for io.py"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import sys
from io import StringIO

import unittest

from IPython.utils.io import Tee, capture_output


def test_tee_simple():
    "Very simple check with stdout only"
    chan = StringIO()
    text = 'Hello'
    tee = Tee(chan, channel='stdout')
    print(text, file=chan)
    assert chan.getvalue() == text + "\n"


class TeeTestCase(unittest.TestCase):

    def tchan(self, channel):
        trap = StringIO()
        chan = StringIO()
        text = 'Hello'
        
        std_ori = getattr(sys, channel)
        setattr(sys, channel, trap)

        tee = Tee(chan, channel=channel)

        print(text, end='', file=chan)
        trap_val = trap.getvalue()
        self.assertEqual(chan.getvalue(), text)

        tee.close()

        setattr(sys, channel, std_ori)
        assert getattr(sys, channel) == std_ori

    def test(self):
        for chan in ['stdout', 'stderr']:
            self.tchan(chan)

class TestIOStream(unittest.TestCase):

    def test_capture_output(self):
        """capture_output() context works"""
        
        with capture_output() as io:
            print("hi, stdout")
            print("hi, stderr", file=sys.stderr)

        self.assertEqual(io.stdout, "hi, stdout\n")
        self.assertEqual(io.stderr, "hi, stderr\n")
