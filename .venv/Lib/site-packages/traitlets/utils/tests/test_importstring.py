# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
#
# Adapted from enthought.traits, Copyright (c) Enthought, Inc.,
# also under the terms of the Modified BSD License.
"""Tests for traitlets.utils.importstring."""

import os
from unittest import TestCase

from ..importstring import import_item


class TestImportItem(TestCase):
    def test_import_unicode(self):
        self.assertIs(os, import_item("os"))
        self.assertIs(os.path, import_item("os.path"))
        self.assertIs(os.path.join, import_item("os.path.join"))

    def test_bad_input(self):
        class NotAString:
            pass

        msg = "import_item accepts strings, not '%s'." % NotAString
        with self.assertRaisesRegex(TypeError, msg):
            import_item(NotAString())
