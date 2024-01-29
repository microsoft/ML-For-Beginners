"""Machinery for documenting traitlets config options with Sphinx.

This includes:

- A Sphinx extension defining directives and roles for config options.
- A function to generate an rst file given an Application instance.

To make this documentation, first set this module as an extension in Sphinx's
conf.py::

    extensions = [
        # ...
        'traitlets.config.sphinxdoc',
    ]

Autogenerate the config documentation by running code like this before
Sphinx builds::

    from traitlets.config.sphinxdoc import write_doc
    from myapp import MyApplication

    writedoc('config/options.rst',    # File to write
             'MyApp config options',  # Title
             MyApplication()
            )

The generated rST syntax looks like this::

    .. configtrait:: Application.log_datefmt

        Description goes here.

    Cross reference like this: :configtrait:`Application.log_datefmt`.
"""
from __future__ import annotations

import typing as t
from collections import defaultdict
from textwrap import dedent

from traitlets import HasTraits, Undefined
from traitlets.config.application import Application
from traitlets.utils.text import indent


def setup(app: t.Any) -> dict[str, t.Any]:
    """Registers the Sphinx extension.

    You shouldn't need to call this directly; configure Sphinx to use this
    module instead.
    """
    app.add_object_type("configtrait", "configtrait", objname="Config option")
    return {"parallel_read_safe": True, "parallel_write_safe": True}


def interesting_default_value(dv: t.Any) -> bool:
    if (dv is None) or (dv is Undefined):
        return False
    if isinstance(dv, (str, list, tuple, dict, set)):
        return bool(dv)
    return True


def format_aliases(aliases: list[str]) -> str:
    fmted = []
    for a in aliases:
        dashes = "-" if len(a) == 1 else "--"
        fmted.append(f"``{dashes}{a}``")
    return ", ".join(fmted)


def class_config_rst_doc(cls: type[HasTraits], trait_aliases: dict[str, t.Any]) -> str:
    """Generate rST documentation for this class' config options.

    Excludes traits defined on parent classes.
    """
    lines = []
    classname = cls.__name__
    for _, trait in sorted(cls.class_traits(config=True).items()):
        ttype = trait.__class__.__name__

        fullname = classname + "." + (trait.name or "")
        lines += [".. configtrait:: " + fullname, ""]

        help = trait.help.rstrip() or "No description"
        lines.append(indent(dedent(help)) + "\n")

        # Choices or type
        if "Enum" in ttype:
            # include Enum choices
            lines.append(indent(":options: " + ", ".join("``%r``" % x for x in trait.values)))  # type:ignore[attr-defined]
        else:
            lines.append(indent(":trait type: " + ttype))

        # Default value
        # Ignore boring default values like None, [] or ''
        if interesting_default_value(trait.default_value):
            try:
                dvr = trait.default_value_repr()
            except Exception:
                dvr = None  # ignore defaults we can't construct
            if dvr is not None:
                if len(dvr) > 64:
                    dvr = dvr[:61] + "..."
                # Double up backslashes, so they get to the rendered docs
                dvr = dvr.replace("\\n", "\\\\n")
                lines.append(indent(":default: ``%s``" % dvr))

        # Command line aliases
        if trait_aliases[fullname]:
            fmt_aliases = format_aliases(trait_aliases[fullname])
            lines.append(indent(":CLI option: " + fmt_aliases))

        # Blank line
        lines.append("")

    return "\n".join(lines)


def reverse_aliases(app: Application) -> dict[str, list[str]]:
    """Produce a mapping of trait names to lists of command line aliases."""
    res = defaultdict(list)
    for alias, trait in app.aliases.items():
        res[trait].append(alias)

    # Flags also often act as aliases for a boolean trait.
    # Treat flags which set one trait to True as aliases.
    for flag, (cfg, _) in app.flags.items():
        if len(cfg) == 1:
            classname = next(iter(cfg))
            cls_cfg = cfg[classname]
            if len(cls_cfg) == 1:
                traitname = next(iter(cls_cfg))
                if cls_cfg[traitname] is True:
                    res[classname + "." + traitname].append(flag)

    return res


def write_doc(path: str, title: str, app: Application, preamble: str | None = None) -> None:
    """Write a rst file documenting config options for a traitlets application.

    Parameters
    ----------
    path : str
        The file to be written
    title : str
        The human-readable title of the document
    app : traitlets.config.Application
        An instance of the application class to be documented
    preamble : str
        Extra text to add just after the title (optional)
    """
    trait_aliases = reverse_aliases(app)
    with open(path, "w") as f:
        f.write(title + "\n")
        f.write(("=" * len(title)) + "\n")
        f.write("\n")
        if preamble is not None:
            f.write(preamble + "\n\n")

        for c in app._classes_inc_parents():
            f.write(class_config_rst_doc(c, trait_aliases))
            f.write("\n")
