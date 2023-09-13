# -*- coding: utf-8
"""Tests for prompt generation."""

import unittest

from IPython.core.prompts import  LazyEvaluate

class PromptTests(unittest.TestCase):
    def test_lazy_eval_unicode(self):
        u = u'ünicødé'
        lz = LazyEvaluate(lambda : u)
        self.assertEqual(str(lz), u)
        self.assertEqual(format(lz), u)
    
    def test_lazy_eval_nonascii_bytes(self):
        u = u'ünicødé'
        b = u.encode('utf8')
        lz = LazyEvaluate(lambda : b)
        # unicode(lz) would fail
        self.assertEqual(str(lz), str(b))
        self.assertEqual(format(lz), str(b))
    
    def test_lazy_eval_float(self):
        f = 0.503
        lz = LazyEvaluate(lambda : f)
        
        self.assertEqual(str(lz), str(f))
        self.assertEqual(format(lz), str(f))
        self.assertEqual(format(lz, '.1'), '0.5')

