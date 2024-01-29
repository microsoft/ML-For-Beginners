"""Replacements for sys.displayhook that publish over ZMQ."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.
from __future__ import annotations

import builtins
import sys
import typing as t

from IPython.core.displayhook import DisplayHook
from jupyter_client.session import Session, extract_header
from traitlets import Any, Dict, Instance

from ipykernel.jsonutil import encode_images, json_clean


class ZMQDisplayHook:
    """A simple displayhook that publishes the object's repr over a ZeroMQ
    socket."""

    topic = b"execute_result"

    def __init__(self, session, pub_socket):
        """Initialize the hook."""
        self.session = session
        self.pub_socket = pub_socket
        self.parent_header = {}

    def get_execution_count(self):
        """This method is replaced in kernelapp"""
        return 0

    def __call__(self, obj):
        """Handle a hook call."""
        if obj is None:
            return

        builtins._ = obj  # type:ignore[attr-defined]
        sys.stdout.flush()
        sys.stderr.flush()
        contents = {
            "execution_count": self.get_execution_count(),
            "data": {"text/plain": repr(obj)},
            "metadata": {},
        }
        self.session.send(
            self.pub_socket, "execute_result", contents, parent=self.parent_header, ident=self.topic
        )

    def set_parent(self, parent):
        """Set the parent header."""
        self.parent_header = extract_header(parent)


class ZMQShellDisplayHook(DisplayHook):
    """A displayhook subclass that publishes data using ZeroMQ. This is intended
    to work with an InteractiveShell instance. It sends a dict of different
    representations of the object."""

    topic = None

    session = Instance(Session, allow_none=True)
    pub_socket = Any(allow_none=True)
    parent_header = Dict({})
    msg: dict[str, t.Any] | None

    def set_parent(self, parent):
        """Set the parent for outbound messages."""
        self.parent_header = extract_header(parent)

    def start_displayhook(self):
        """Start the display hook."""
        if self.session:
            self.msg = self.session.msg(
                "execute_result",
                {
                    "data": {},
                    "metadata": {},
                },
                parent=self.parent_header,
            )

    def write_output_prompt(self):
        """Write the output prompt."""
        if self.msg:
            self.msg["content"]["execution_count"] = self.prompt_count

    def write_format_data(self, format_dict, md_dict=None):
        """Write format data to the message."""
        if self.msg:
            self.msg["content"]["data"] = json_clean(encode_images(format_dict))
            self.msg["content"]["metadata"] = md_dict

    def finish_displayhook(self):
        """Finish up all displayhook activities."""
        sys.stdout.flush()
        sys.stderr.flush()
        if self.msg and self.msg["content"]["data"] and self.session:
            self.session.send(self.pub_socket, self.msg, ident=self.topic)
        self.msg = None
