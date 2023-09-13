"""Shim to allow python -m tornado.test.

This only works in python 2.7+.
"""
from tornado.test.runtests import all, main

# tornado.testing.main autodiscovery relies on 'all' being present in
# the main module, so import it here even though it is not used directly.
# The following line prevents a pyflakes warning.
all = all

main()
