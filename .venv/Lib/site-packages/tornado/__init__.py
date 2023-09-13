#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""The Tornado web server and tools."""

# version is a human-readable version number.

# version_info is a four-tuple for programmatic comparison. The first
# three numbers are the components of the version number.  The fourth
# is zero for an official release, positive for a development branch,
# or negative for a release candidate or beta (after the base version
# number has been incremented)
version = "6.3.3"
version_info = (6, 3, 3, 0)

import importlib
import typing

__all__ = [
    "auth",
    "autoreload",
    "concurrent",
    "curl_httpclient",
    "escape",
    "gen",
    "http1connection",
    "httpclient",
    "httpserver",
    "httputil",
    "ioloop",
    "iostream",
    "locale",
    "locks",
    "log",
    "netutil",
    "options",
    "platform",
    "process",
    "queues",
    "routing",
    "simple_httpclient",
    "tcpclient",
    "tcpserver",
    "template",
    "testing",
    "util",
    "web",
]


# Copied from https://peps.python.org/pep-0562/
def __getattr__(name: str) -> typing.Any:
    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
