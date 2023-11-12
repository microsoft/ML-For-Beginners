"""
Substantially copied from NumpyDoc 1.0pre.
"""
from collections import namedtuple
from collections.abc import Mapping
import copy
import inspect
import re
import textwrap

from statsmodels.tools.sm_exceptions import ParseError


def dedent_lines(lines):
    """Deindent a list of lines maximally"""
    return textwrap.dedent("\n".join(lines)).split("\n")


def strip_blank_lines(line):
    """Remove leading and trailing blank lines from a list of lines"""
    while line and not line[0].strip():
        del line[0]
    while line and not line[-1].strip():
        del line[-1]
    return line


class Reader:
    """
    A line-based string reader.
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           String with lines separated by '\n'.
        """
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split("\n")  # store string as list of lines

        self.reset()

    def __getitem__(self, n):
        return self._str[n]

    def reset(self):
        self._line_num = 0  # current line nr

    def read(self):
        if not self.eof():
            out = self[self._line_num]
            self._line_num += 1
            return out
        else:
            return ""

    def seek_next_non_empty_line(self):
        for line in self[self._line_num :]:
            if line.strip():
                break
            else:
                self._line_num += 1

    def eof(self):
        return self._line_num >= len(self._str)

    def read_to_condition(self, condition_func):
        start = self._line_num
        for line in self[start:]:
            if condition_func(line):
                return self[start : self._line_num]
            self._line_num += 1
            if self.eof():
                return self[start : self._line_num + 1]
        return []

    def read_to_next_empty_line(self):
        self.seek_next_non_empty_line()

        def is_empty(line):
            return not line.strip()

        return self.read_to_condition(is_empty)

    def read_to_next_unindented_line(self):
        def is_unindented(line):
            return line.strip() and (len(line.lstrip()) == len(line))

        return self.read_to_condition(is_unindented)

    def peek(self, n=0):
        if self._line_num + n < len(self._str):
            return self[self._line_num + n]
        else:
            return ""

    def is_empty(self):
        return not "".join(self._str).strip()


Parameter = namedtuple("Parameter", ["name", "type", "desc"])


class NumpyDocString(Mapping):
    """Parses a numpydoc string to an abstract representation

    Instances define a mapping from section title to structured data.
    """

    sections = {
        "Signature": "",
        "Summary": [""],
        "Extended Summary": [],
        "Parameters": [],
        "Returns": [],
        "Yields": [],
        "Receives": [],
        "Raises": [],
        "Warns": [],
        "Other Parameters": [],
        "Attributes": [],
        "Methods": [],
        "See Also": [],
        "Notes": [],
        "Warnings": [],
        "References": "",
        "Examples": "",
        "index": {},
    }

    def __init__(self, docstring):
        orig_docstring = docstring
        docstring = textwrap.dedent(docstring).split("\n")

        self._doc = Reader(docstring)
        self._parsed_data = copy.deepcopy(self.sections)

        try:
            self._parse()
        except ParseError as e:
            e.docstring = orig_docstring
            raise

    def __getitem__(self, key):
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        if key not in self._parsed_data:
            self._error_location("Unknown section %s" % key)
        else:
            self._parsed_data[key] = val

    def __iter__(self):
        return iter(self._parsed_data)

    def __len__(self):
        return len(self._parsed_data)

    def _is_at_section(self):
        self._doc.seek_next_non_empty_line()

        if self._doc.eof():
            return False

        l1 = self._doc.peek().strip()  # e.g. Parameters

        if l1.startswith(".. index::"):
            return True

        l2 = self._doc.peek(1).strip()  # ---------- or ==========
        return l2.startswith("-" * len(l1)) or l2.startswith("=" * len(l1))

    def _strip(self, doc):
        i = 0
        j = 0
        for i, line in enumerate(doc):
            if line.strip():
                break

        for j, line in enumerate(doc[::-1]):
            if line.strip():
                break

        return doc[i : len(doc) - j]

    def _read_to_next_section(self):
        section = self._doc.read_to_next_empty_line()

        while not self._is_at_section() and not self._doc.eof():
            if not self._doc.peek(-1).strip():  # previous line was empty
                section += [""]

            section += self._doc.read_to_next_empty_line()

        return section

    def _read_sections(self):
        while not self._doc.eof():
            data = self._read_to_next_section()
            name = data[0].strip()

            if name.startswith(".."):  # index section
                yield name, data[1:]
            elif len(data) < 2:
                yield StopIteration
            else:
                yield name, self._strip(data[2:])

    def _parse_param_list(self, content, single_element_is_type=False):
        r = Reader(content)
        params = []
        while not r.eof():
            header = r.read().strip()
            if " : " in header:
                arg_name, arg_type = header.split(" : ")[:2]
            else:
                if single_element_is_type:
                    arg_name, arg_type = "", header
                else:
                    arg_name, arg_type = header, ""

            desc = r.read_to_next_unindented_line()
            desc = dedent_lines(desc)
            desc = strip_blank_lines(desc)

            params.append(Parameter(arg_name, arg_type, desc))

        return params

    # See also supports the following formats.
    #
    # <FUNCNAME>
    # <FUNCNAME> SPACE* COLON SPACE+ <DESC> SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)+ (COMMA | PERIOD)? SPACE*
    # <FUNCNAME> ( COMMA SPACE+ <FUNCNAME>)* SPACE* COLON SPACE+ <DESC> SPACE*

    # <FUNCNAME> is one of
    #   <PLAIN_FUNCNAME>
    #   COLON <ROLE> COLON BACKTICK <PLAIN_FUNCNAME> BACKTICK
    # where
    #   <PLAIN_FUNCNAME> is a legal function name, and
    #   <ROLE> is any nonempty sequence of word characters.
    # Examples: func_f1  :meth:`func_h1` :obj:`~baz.obj_r` :class:`class_j`
    # <DESC> is a string describing the function.

    _role = r":(?P<role>\w+):"
    _funcbacktick = r"`(?P<name>(?:~\w+\.)?[a-zA-Z0-9_\.-]+)`"
    _funcplain = r"(?P<name2>[a-zA-Z0-9_\.-]+)"
    _funcname = r"(" + _role + _funcbacktick + r"|" + _funcplain + r")"
    _funcnamenext = _funcname.replace("role", "rolenext")
    _funcnamenext = _funcnamenext.replace("name", "namenext")
    _description = r"(?P<description>\s*:(\s+(?P<desc>\S+.*))?)?\s*$"
    _func_rgx = re.compile(r"^\s*" + _funcname + r"\s*")
    _line_rgx = re.compile(
        r"^\s*"
        + r"(?P<allfuncs>"
        + _funcname  # group for all function names
        + r"(?P<morefuncs>([,]\s+"
        + _funcnamenext
        + r")*)"
        + r")"
        +  # end of "allfuncs"
        # Some function lists have a trailing comma (or period)
        r"(?P<trailing>[,\.])?"
        + _description
    )

    # Empty <DESC> elements are replaced with '..'
    empty_description = ".."

    def _parse_see_also(self, content):
        """
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, :meth:`func_name`, func_name3
        """

        items = []

        def parse_item_name(text):
            """Match ':role:`name`' or 'name'."""
            m = self._func_rgx.match(text)
            if not m:
                raise ParseError(f"{text} is not a item name")
            role = m.group("role")
            name = m.group("name") if role else m.group("name2")
            return name, role, m.end()

        rest = []
        for line in content:
            if not line.strip():
                continue

            line_match = self._line_rgx.match(line)
            description = None
            if line_match:
                description = line_match.group("desc")
                if line_match.group("trailing") and description:
                    self._error_location(
                        "Unexpected comma or period after function list at "
                        "index %d of line "
                        '"%s"' % (line_match.end("trailing"), line)
                    )
            if not description and line.startswith(" "):
                rest.append(line.strip())
            elif line_match:
                funcs = []
                text = line_match.group("allfuncs")
                while True:
                    if not text.strip():
                        break
                    name, role, match_end = parse_item_name(text)
                    funcs.append((name, role))
                    text = text[match_end:].strip()
                    if text and text[0] == ",":
                        text = text[1:].strip()
                rest = list(filter(None, [description]))
                items.append((funcs, rest))
            else:
                raise ParseError(f"{line} is not a item name")
        return items

    def _parse_index(self, section, content):
        """
        .. index: default
           :refguide: something, else, and more
        """

        def strip_each_in(lst):
            return [s.strip() for s in lst]

        out = {}
        section = section.split("::")
        if len(section) > 1:
            out["default"] = strip_each_in(section[1].split(","))[0]
        for line in content:
            line = line.split(":")
            if len(line) > 2:
                out[line[1]] = strip_each_in(line[2].split(","))
        return out

    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        if self._is_at_section():
            return

        # If several signatures present, take the last one
        while True:
            summary = self._doc.read_to_next_empty_line()
            summary_str = " ".join([s.strip() for s in summary]).strip()
            compiled = re.compile(r"^([\w., ]+=)?\s*[\w\.]+\(.*\)$")
            if compiled.match(summary_str):
                self["Signature"] = summary_str
                if not self._is_at_section():
                    continue
            break

        if summary is not None:
            self["Summary"] = summary

        if not self._is_at_section():
            self["Extended Summary"] = self._read_to_next_section()

    def _parse(self):
        self._doc.reset()
        self._parse_summary()

        sections = list(self._read_sections())
        section_names = set([section for section, content in sections])

        has_returns = "Returns" in section_names
        has_yields = "Yields" in section_names
        # We could do more tests, but we are not. Arbitrarily.
        if has_returns and has_yields:
            msg = "Docstring contains both a Returns and Yields section."
            raise ValueError(msg)
        if not has_yields and "Receives" in section_names:
            msg = "Docstring contains a Receives section but not Yields."
            raise ValueError(msg)

        for (section, content) in sections:
            if not section.startswith(".."):
                section = (s.capitalize() for s in section.split(" "))
                section = " ".join(section)
                if self.get(section):
                    self._error_location(
                        "The section %s appears twice" % section
                    )

            if section in (
                "Parameters",
                "Other Parameters",
                "Attributes",
                "Methods",
            ):
                self[section] = self._parse_param_list(content)
            elif section in (
                "Returns",
                "Yields",
                "Raises",
                "Warns",
                "Receives",
            ):
                self[section] = self._parse_param_list(
                    content, single_element_is_type=True
                )
            elif section.startswith(".. index::"):
                self["index"] = self._parse_index(section, content)
            elif section == "See Also":
                self["See Also"] = self._parse_see_also(content)
            else:
                self[section] = content

    def _error_location(self, msg):
        if hasattr(self, "_obj"):
            # we know where the docs came from:
            try:
                filename = inspect.getsourcefile(self._obj)
            except TypeError:
                filename = None
            msg = msg + (
                " in the docstring of %s in %s." % (self._obj, filename)
            )

        raise ValueError(msg)

    # string conversion routines

    def _str_header(self, name, symbol="-"):
        return [name, len(name) * symbol]

    def _str_indent(self, doc, indent=4):
        out = []
        for line in doc:
            out += [" " * indent + line]
        return out

    def _str_signature(self):
        if self["Signature"]:
            return [self["Signature"].replace("*", r"\*")] + [""]
        else:
            return [""]

    def _str_summary(self):
        if self["Summary"]:
            return self["Summary"] + [""]
        else:
            return []

    def _str_extended_summary(self):
        if self["Extended Summary"]:
            return self["Extended Summary"] + [""]
        else:
            return []

    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            for param in self[name]:
                parts = []
                if param.name:
                    parts.append(param.name)
                if param.type:
                    parts.append(param.type)
                out += [" : ".join(parts)]
                if param.desc and "".join(param.desc).strip():
                    out += self._str_indent(param.desc)
            out += [""]
        return out

    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += self[name]
            out += [""]
        return out

    def _str_see_also(self, func_role):
        if not self["See Also"]:
            return []
        out = []
        out += self._str_header("See Also")
        last_had_desc = True
        for funcs, desc in self["See Also"]:
            assert isinstance(funcs, list)
            links = []
            for func, role in funcs:
                if role:
                    link = ":%s:`%s`" % (role, func)
                elif func_role:
                    link = ":%s:`%s`" % (func_role, func)
                else:
                    link = "%s" % func
                links.append(link)
            link = ", ".join(links)
            out += [link]
            if desc:
                out += self._str_indent([" ".join(desc)])
                last_had_desc = True
            else:
                last_had_desc = False
                out += self._str_indent([self.empty_description])

        if last_had_desc:
            out += [""]
        return out

    def _str_index(self):
        idx = self["index"]
        out = []
        output_index = False
        default_index = idx.get("default", "")
        if default_index:
            output_index = True
        out += [".. index:: %s" % default_index]
        for section, references in idx.items():
            if section == "default":
                continue
            output_index = True
            out += ["   :%s: %s" % (section, ", ".join(references))]
        if output_index:
            return out
        else:
            return ""

    def __str__(self, func_role=""):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in (
            "Parameters",
            "Returns",
            "Yields",
            "Receives",
            "Other Parameters",
            "Raises",
            "Warns",
        ):
            out += self._str_param_list(param_list)
        out += self._str_section("Warnings")
        out += self._str_see_also(func_role)
        for s in ("Notes", "References", "Examples"):
            out += self._str_section(s)
        for param_list in ("Attributes", "Methods"):
            out += self._str_param_list(param_list)
        out += self._str_index()
        return "\n".join(out)


class Docstring:
    """
    Docstring modification.

    Parameters
    ----------
    docstring : str
        The docstring to modify.
    """

    def __init__(self, docstring):
        self._ds = None
        self._docstring = docstring
        if docstring is None:
            return
        self._ds = NumpyDocString(docstring)

    def remove_parameters(self, parameters):
        """
        Parameters
        ----------
        parameters : str, list[str]
            The names of the parameters to remove.
        """
        if self._docstring is None:
            # Protection against -oo execution
            return
        if isinstance(parameters, str):
            parameters = [parameters]
        repl = [
            param
            for param in self._ds["Parameters"]
            if param.name not in parameters
        ]
        if len(repl) + len(parameters) != len(self._ds["Parameters"]):
            raise ValueError("One or more parameters were not found.")
        self._ds["Parameters"] = repl

    def insert_parameters(self, after, parameters):
        """
        Parameters
        ----------
        after : {None, str}
            If None, inset the parameters before the first parameter in the
            docstring.
        parameters : Parameter, list[Parameter]
            A Parameter of a list of Parameters.
        """
        if self._docstring is None:
            # Protection against -oo execution
            return
        if isinstance(parameters, Parameter):
            parameters = [parameters]
        if after is None:
            self._ds["Parameters"] = parameters + self._ds["Parameters"]
        else:
            loc = -1
            for i, param in enumerate(self._ds["Parameters"]):
                if param.name == after:
                    loc = i + 1
                    break
            if loc < 0:
                raise ValueError()
            params = self._ds["Parameters"][:loc] + parameters
            params += self._ds["Parameters"][loc:]
            self._ds["Parameters"] = params

    def replace_block(self, block_name, block):
        """
        Parameters
        ----------
        block_name : str
            Name of the block to replace, e.g., 'Summary'.
        block : object
            The replacement block. The structure of the replacement block must
            match how the block is stored by NumpyDocString.
        """
        if self._docstring is None:
            # Protection against -oo execution
            return
        block_name = " ".join(map(str.capitalize, block_name.split(" ")))
        if block_name not in self._ds:
            raise ValueError(
                "{0} is not a block in the " "docstring".format(block_name)
            )
        if not isinstance(block, list) and isinstance(
            self._ds[block_name], list
        ):
            block = [block]
        self._ds[block_name] = block

    def extract_parameters(self, parameters, indent=0):
        if self._docstring is None:
            # Protection against -oo execution
            return
        if isinstance(parameters, str):
            parameters = [parameters]
        ds_params = {param.name: param for param in self._ds["Parameters"]}
        missing = set(parameters).difference(ds_params.keys())
        if missing:
            raise ValueError(
                "{0} were not found in the "
                "docstring".format(",".join(missing))
            )
        final = [ds_params[param] for param in parameters]
        ds = copy.deepcopy(self._ds)
        for key in ds:
            if key != "Parameters":
                ds[key] = [] if key != "index" else {}
            else:
                ds[key] = final
        out = str(ds).strip()
        if indent:
            out = textwrap.indent(out, " " * indent)

        out = "\n".join(out.split("\n")[2:])
        return out

    def __str__(self):
        return str(self._ds)


def remove_parameters(docstring, parameters):
    """
    Parameters
    ----------
    docstring : str
        The docstring to modify.
    parameters : str, list[str]
        The names of the parameters to remove.

    Returns
    -------
    str
        The modified docstring.
    """
    if docstring is None:
        return
    ds = Docstring(docstring)
    ds.remove_parameters(parameters)
    return str(ds)


def indent(text, prefix, predicate=None):
    """
    Non-protected indent

    Parameters
    ----------
    text : {None, str}
        If None, function always returns ""
    prefix : str
        Prefix to add to the start of each line
    predicate : callable, optional
        If provided, 'prefix' will only be added to the lines
        where 'predicate(line)' is True. If 'predicate' is not provided,
        it will default to adding 'prefix' to all non-empty lines that do not
        consist solely of whitespace characters.

    Returns
    -------

    """
    if text is None:
        return ""
    return textwrap.indent(text, prefix, predicate=predicate)
