# coding: utf-8
"""Test suite for our sysinfo utilities."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import json
import pytest

from IPython.utils import sysinfo


def test_json_getsysinfo():
    """
    test that it is easily jsonable and don't return bytes somewhere.
    """
    json.dumps(sysinfo.get_sys_info())


def test_num_cpus():
    with pytest.deprecated_call():
        sysinfo.num_cpus()
