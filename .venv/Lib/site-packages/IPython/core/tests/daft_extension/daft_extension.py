# -*- coding: utf-8 -*-
"""
Useless IPython extension to test installing and loading extensions.
"""
some_vars = {'arq': 185}

def load_ipython_extension(ip):
    # set up simplified quantity input
    ip.push(some_vars)

def unload_ipython_extension(ip):
    ip.drop_by_id(some_vars)
