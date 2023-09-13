"""Adapters for Jupyter msg spec versions."""
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import json
import re
from typing import Any, Dict, List, Tuple

from ._version import protocol_version_info


def code_to_line(code: str, cursor_pos: int) -> Tuple[str, int]:
    """Turn a multiline code block and cursor position into a single line
    and new cursor position.

    For adapting ``complete_`` and ``object_info_request``.
    """
    if not code:
        return "", 0
    for line in code.splitlines(True):
        n = len(line)
        if cursor_pos > n:
            cursor_pos -= n
        else:
            break
    return line, cursor_pos


_match_bracket = re.compile(r"\([^\(\)]+\)", re.UNICODE)
_end_bracket = re.compile(r"\([^\(]*$", re.UNICODE)
_identifier = re.compile(r"[a-z_][0-9a-z._]*", re.I | re.UNICODE)


def extract_oname_v4(code: str, cursor_pos: int) -> str:
    """Reimplement token-finding logic from IPython 2.x javascript

    for adapting object_info_request from v5 to v4
    """

    line, _ = code_to_line(code, cursor_pos)

    oldline = line
    line = _match_bracket.sub("", line)
    while oldline != line:
        oldline = line
        line = _match_bracket.sub("", line)

    # remove everything after last open bracket
    line = _end_bracket.sub("", line)
    matches = _identifier.findall(line)
    if matches:
        return matches[-1]
    else:
        return ""


class Adapter:
    """Base class for adapting messages

    Override message_type(msg) methods to create adapters.
    """

    msg_type_map: Dict[str, str] = {}

    def update_header(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Update the header."""
        return msg

    def update_metadata(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Update the metadata."""
        return msg

    def update_msg_type(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Update the message type."""
        header = msg["header"]
        msg_type = header["msg_type"]
        if msg_type in self.msg_type_map:
            msg["msg_type"] = header["msg_type"] = self.msg_type_map[msg_type]
        return msg

    def handle_reply_status_error(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """This will be called *instead of* the regular handler

        on any reply with status != ok
        """
        return msg

    def __call__(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        msg = self.update_header(msg)
        msg = self.update_metadata(msg)
        msg = self.update_msg_type(msg)
        header = msg["header"]

        handler = getattr(self, header["msg_type"], None)
        if handler is None:
            return msg

        # handle status=error replies separately (no change, at present)
        if msg["content"].get("status", None) in {"error", "aborted"}:
            return self.handle_reply_status_error(msg)
        return handler(msg)


def _version_str_to_list(version: str) -> List[int]:
    """convert a version string to a list of ints

    non-int segments are excluded
    """
    v = []
    for part in version.split("."):
        try:
            v.append(int(part))
        except ValueError:
            pass
    return v


class V5toV4(Adapter):
    """Adapt msg protocol v5 to v4"""

    version = "4.1"

    msg_type_map = {
        "execute_result": "pyout",
        "execute_input": "pyin",
        "error": "pyerr",
        "inspect_request": "object_info_request",
        "inspect_reply": "object_info_reply",
    }

    def update_header(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Update the header."""
        msg["header"].pop("version", None)
        msg["parent_header"].pop("version", None)
        return msg

    # shell channel

    def kernel_info_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a kernel info reply."""
        v4c = {}
        content = msg["content"]
        for key in ("language_version", "protocol_version"):
            if key in content:
                v4c[key] = _version_str_to_list(content[key])
        if content.get("implementation", "") == "ipython" and "implementation_version" in content:
            v4c["ipython_version"] = _version_str_to_list(content["implementation_version"])
        language_info = content.get("language_info", {})
        language = language_info.get("name", "")
        v4c.setdefault("language", language)
        if "version" in language_info:
            v4c.setdefault("language_version", _version_str_to_list(language_info["version"]))
        msg["content"] = v4c
        return msg

    def execute_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an execute request."""
        content = msg["content"]
        content.setdefault("user_variables", [])
        return msg

    def execute_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an execute reply."""
        content = msg["content"]
        content.setdefault("user_variables", {})
        # TODO: handle payloads
        return msg

    def complete_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complete request."""
        content = msg["content"]
        code = content["code"]
        cursor_pos = content["cursor_pos"]
        line, cursor_pos = code_to_line(code, cursor_pos)

        new_content = msg["content"] = {}
        new_content["text"] = ""
        new_content["line"] = line
        new_content["block"] = None
        new_content["cursor_pos"] = cursor_pos
        return msg

    def complete_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complete reply."""
        content = msg["content"]
        cursor_start = content.pop("cursor_start")
        cursor_end = content.pop("cursor_end")
        match_len = cursor_end - cursor_start
        content["matched_text"] = content["matches"][0][:match_len]
        content.pop("metadata", None)
        return msg

    def object_info_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an object info request."""
        content = msg["content"]
        code = content["code"]
        cursor_pos = content["cursor_pos"]
        line, _ = code_to_line(code, cursor_pos)

        new_content = msg["content"] = {}
        new_content["oname"] = extract_oname_v4(code, cursor_pos)
        new_content["detail_level"] = content["detail_level"]
        return msg

    def object_info_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """inspect_reply can't be easily backward compatible"""
        msg["content"] = {"found": False, "oname": "unknown"}
        return msg

    # iopub channel

    def stream(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a stream message."""
        content = msg["content"]
        content["data"] = content.pop("text")
        return msg

    def display_data(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a display data message."""
        content = msg["content"]
        content.setdefault("source", "display")
        data = content["data"]
        if "application/json" in data:
            try:
                data["application/json"] = json.dumps(data["application/json"])
            except Exception:
                # warn?
                pass
        return msg

    # stdin channel

    def input_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an input request."""
        msg["content"].pop("password", None)
        return msg


class V4toV5(Adapter):
    """Convert msg spec V4 to V5"""

    version = "5.0"

    # invert message renames above
    msg_type_map = {v: k for k, v in V5toV4.msg_type_map.items()}

    def update_header(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Update the header."""
        msg["header"]["version"] = self.version
        if msg["parent_header"]:
            msg["parent_header"]["version"] = self.version
        return msg

    # shell channel

    def kernel_info_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a kernel info reply."""
        content = msg["content"]
        for key in ("protocol_version", "ipython_version"):
            if key in content:
                content[key] = ".".join(map(str, content[key]))

        content.setdefault("protocol_version", "4.1")

        if content["language"].startswith("python") and "ipython_version" in content:
            content["implementation"] = "ipython"
            content["implementation_version"] = content.pop("ipython_version")

        language = content.pop("language")
        language_info = content.setdefault("language_info", {})
        language_info.setdefault("name", language)
        if "language_version" in content:
            language_version = ".".join(map(str, content.pop("language_version")))
            language_info.setdefault("version", language_version)

        content["banner"] = ""
        return msg

    def execute_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an execute request."""
        content = msg["content"]
        user_variables = content.pop("user_variables", [])
        user_expressions = content.setdefault("user_expressions", {})
        for v in user_variables:
            user_expressions[v] = v
        return msg

    def execute_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an execute reply."""
        content = msg["content"]
        user_expressions = content.setdefault("user_expressions", {})
        user_variables = content.pop("user_variables", {})
        if user_variables:
            user_expressions.update(user_variables)

        # Pager payloads became a mime bundle
        for payload in content.get("payload", []):
            if payload.get("source", None) == "page" and ("text" in payload):
                if "data" not in payload:
                    payload["data"] = {}
                payload["data"]["text/plain"] = payload.pop("text")

        return msg

    def complete_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complete request."""
        old_content = msg["content"]

        new_content = msg["content"] = {}
        new_content["code"] = old_content["line"]
        new_content["cursor_pos"] = old_content["cursor_pos"]
        return msg

    def complete_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a complete reply."""
        # complete_reply needs more context than we have to get cursor_start and end.
        # use special end=null to indicate current cursor position and negative offset
        # for start relative to the cursor.
        # start=None indicates that start == end (accounts for no -0).
        content = msg["content"]
        new_content = msg["content"] = {"status": "ok"}
        new_content["matches"] = content["matches"]
        if content["matched_text"]:
            new_content["cursor_start"] = -len(content["matched_text"])
        else:
            # no -0, use None to indicate that start == end
            new_content["cursor_start"] = None
        new_content["cursor_end"] = None
        new_content["metadata"] = {}
        return msg

    def inspect_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an inspect request."""
        content = msg["content"]
        name = content["oname"]

        new_content = msg["content"] = {}
        new_content["code"] = name
        new_content["cursor_pos"] = len(name)
        new_content["detail_level"] = content["detail_level"]
        return msg

    def inspect_reply(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """inspect_reply can't be easily backward compatible"""
        content = msg["content"]
        new_content = msg["content"] = {"status": "ok"}
        found = new_content["found"] = content["found"]
        new_content["data"] = data = {}
        new_content["metadata"] = {}
        if found:
            lines = []
            for key in ("call_def", "init_definition", "definition"):
                if content.get(key, False):
                    lines.append(content[key])
                    break
            for key in ("call_docstring", "init_docstring", "docstring"):
                if content.get(key, False):
                    lines.append(content[key])
                    break
            if not lines:
                lines.append("<empty docstring>")
            data["text/plain"] = "\n".join(lines)
        return msg

    # iopub channel

    def stream(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a stream message."""
        content = msg["content"]
        content["text"] = content.pop("data")
        return msg

    def display_data(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle display data."""
        content = msg["content"]
        content.pop("source", None)
        data = content["data"]
        if "application/json" in data:
            try:
                data["application/json"] = json.loads(data["application/json"])
            except Exception:
                # warn?
                pass
        return msg

    # stdin channel

    def input_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an input request."""
        msg["content"].setdefault("password", False)
        return msg


def adapt(msg: Dict[str, Any], to_version: int = protocol_version_info[0]) -> Dict[str, Any]:
    """Adapt a single message to a target version

    Parameters
    ----------

    msg : dict
        A Jupyter message.
    to_version : int, optional
        The target major version.
        If unspecified, adapt to the current version.

    Returns
    -------

    msg : dict
        A Jupyter message appropriate in the new version.
    """
    from .session import utcnow

    header = msg["header"]
    if "date" not in header:
        header["date"] = utcnow()
    if "version" in header:
        from_version = int(header["version"].split(".")[0])
    else:
        # assume last version before adding the key to the header
        from_version = 4
    adapter = adapters.get((from_version, to_version), None)
    if adapter is None:
        return msg
    return adapter(msg)


# one adapter per major version from,to
adapters = {
    (5, 4): V5toV4(),
    (4, 5): V4toV5(),
}
