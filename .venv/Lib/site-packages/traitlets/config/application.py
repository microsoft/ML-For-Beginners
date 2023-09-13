"""A base class for a configurable application."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import functools
import json
import logging
import os
import pprint
import re
import sys
import typing as t
from collections import OrderedDict, defaultdict
from contextlib import suppress
from copy import deepcopy
from logging.config import dictConfig
from textwrap import dedent
from typing import Any, Callable, TypeVar, cast

from traitlets.config.configurable import Configurable, SingletonConfigurable
from traitlets.config.loader import (
    ArgumentError,
    Config,
    ConfigFileNotFound,
    JSONFileConfigLoader,
    KVArgParseConfigLoader,
    PyFileConfigLoader,
)
from traitlets.traitlets import (
    Bool,
    Dict,
    Enum,
    Instance,
    List,
    TraitError,
    Unicode,
    default,
    observe,
    observe_compat,
)
from traitlets.utils.nested_update import nested_update
from traitlets.utils.text import indent, wrap_paragraphs

from ..utils import cast_unicode
from ..utils.importstring import import_item

# -----------------------------------------------------------------------------
# Descriptions for the various sections
# -----------------------------------------------------------------------------
# merge flags&aliases into options
option_description = """
The options below are convenience aliases to configurable class-options,
as listed in the "Equivalent to" description-line of the aliases.
To see all configurable class-options for some <cmd>, use:
    <cmd> --help-all
""".strip()  # trim newlines of front and back

keyvalue_description = """
The command-line option below sets the respective configurable class-parameter:
    --Class.parameter=value
This line is evaluated in Python, so simple expressions are allowed.
For instance, to set `C.a=[0,1,2]`, you may type this:
    --C.a='range(3)'
""".strip()  # trim newlines of front and back

# sys.argv can be missing, for example when python is embedded. See the docs
# for details: http://docs.python.org/2/c-api/intro.html#embedding-python
if not hasattr(sys, "argv"):
    sys.argv = [""]

subcommand_description = """
Subcommands are launched as `{app} cmd [args]`. For information on using
subcommand 'cmd', do: `{app} cmd -h`.
"""
# get running program name

# -----------------------------------------------------------------------------
# Application class
# -----------------------------------------------------------------------------


_envvar = os.environ.get("TRAITLETS_APPLICATION_RAISE_CONFIG_FILE_ERROR", "")
if _envvar.lower() in {"1", "true"}:
    TRAITLETS_APPLICATION_RAISE_CONFIG_FILE_ERROR = True
elif _envvar.lower() in {"0", "false", ""}:
    TRAITLETS_APPLICATION_RAISE_CONFIG_FILE_ERROR = False
else:
    raise ValueError(
        "Unsupported value for environment variable: 'TRAITLETS_APPLICATION_RAISE_CONFIG_FILE_ERROR' is set to '%s' which is none of  {'0', '1', 'false', 'true', ''}."
        % _envvar
    )


IS_PYTHONW = sys.executable and sys.executable.endswith("pythonw.exe")

T = TypeVar("T", bound=Callable[..., Any])


def catch_config_error(method: T) -> T:
    """Method decorator for catching invalid config (Trait/ArgumentErrors) during init.

    On a TraitError (generally caused by bad config), this will print the trait's
    message, and exit the app.

    For use on init methods, to prevent invoking excepthook on invalid input.
    """

    @functools.wraps(method)
    def inner(app, *args, **kwargs):
        try:
            return method(app, *args, **kwargs)
        except (TraitError, ArgumentError) as e:
            app.log.fatal("Bad config encountered during initialization: %s", e)
            app.log.debug("Config at the time: %s", app.config)
            app.exit(1)

    return cast(T, inner)


class ApplicationError(Exception):
    pass


class LevelFormatter(logging.Formatter):
    """Formatter with additional `highlevel` record

    This field is empty if log level is less than highlevel_limit,
    otherwise it is formatted with self.highlevel_format.

    Useful for adding 'WARNING' to warning messages,
    without adding 'INFO' to info, etc.
    """

    highlevel_limit = logging.WARN
    highlevel_format = " %(levelname)s |"

    def format(self, record):
        if record.levelno >= self.highlevel_limit:
            record.highlevel = self.highlevel_format % record.__dict__
        else:
            record.highlevel = ""
        return super().format(record)


class Application(SingletonConfigurable):
    """A singleton application with full configuration support."""

    # The name of the application, will usually match the name of the command
    # line application
    name: t.Union[str, Unicode] = Unicode("application")

    # The description of the application that is printed at the beginning
    # of the help.
    description: t.Union[str, Unicode] = Unicode("This is an application.")
    # default section descriptions
    option_description: t.Union[str, Unicode] = Unicode(option_description)
    keyvalue_description: t.Union[str, Unicode] = Unicode(keyvalue_description)
    subcommand_description: t.Union[str, Unicode] = Unicode(subcommand_description)

    python_config_loader_class = PyFileConfigLoader
    json_config_loader_class = JSONFileConfigLoader

    # The usage and example string that goes at the end of the help string.
    examples: t.Union[str, Unicode] = Unicode()

    # A sequence of Configurable subclasses whose config=True attributes will
    # be exposed at the command line.
    classes: t.List[t.Type[t.Any]] = []

    def _classes_inc_parents(self, classes=None):
        """Iterate through configurable classes, including configurable parents

        :param classes:
            The list of classes to iterate; if not set, uses :attr:`classes`.

        Children should always be after parents, and each class should only be
        yielded once.
        """
        if classes is None:
            classes = self.classes

        seen = set()
        for c in classes:
            # We want to sort parents before children, so we reverse the MRO
            for parent in reversed(c.mro()):
                if issubclass(parent, Configurable) and (parent not in seen):
                    seen.add(parent)
                    yield parent

    # The version string of this application.
    version: t.Union[str, Unicode] = Unicode("0.0")

    # the argv used to initialize the application
    argv: t.Union[t.List[str], List] = List()

    # Whether failing to load config files should prevent startup
    raise_config_file_errors: t.Union[bool, Bool] = Bool(
        TRAITLETS_APPLICATION_RAISE_CONFIG_FILE_ERROR
    )

    # The log level for the application
    log_level: t.Union[str, int, Enum] = Enum(
        (0, 10, 20, 30, 40, 50, "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"),
        default_value=logging.WARN,
        help="Set the log level by value or name.",
    ).tag(config=True)

    _log_formatter_cls = LevelFormatter

    log_datefmt: t.Union[str, Unicode] = Unicode(
        "%Y-%m-%d %H:%M:%S", help="The date format used by logging formatters for %(asctime)s"
    ).tag(config=True)

    log_format: t.Union[str, Unicode] = Unicode(
        "[%(name)s]%(highlevel)s %(message)s",
        help="The Logging format template",
    ).tag(config=True)

    def get_default_logging_config(self):
        """Return the base logging configuration.

        The default is to log to stderr using a StreamHandler, if no default
        handler already exists.

        The log handler level starts at logging.WARN, but this can be adjusted
        by setting the ``log_level`` attribute.

        The ``logging_config`` trait is merged into this allowing for finer
        control of logging.

        """
        config: t.Dict[str, t.Any] = {
            "version": 1,
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "level": logging.getLevelName(self.log_level),
                    "stream": "ext://sys.stderr",
                },
            },
            "formatters": {
                "console": {
                    "class": (
                        f"{self._log_formatter_cls.__module__}"
                        f".{self._log_formatter_cls.__name__}"
                    ),
                    "format": self.log_format,
                    "datefmt": self.log_datefmt,
                },
            },
            "loggers": {
                self.__class__.__name__: {
                    "level": "DEBUG",
                    "handlers": ["console"],
                }
            },
            "disable_existing_loggers": False,
        }

        if IS_PYTHONW:
            # disable logging
            # (this should really go to a file, but file-logging is only
            # hooked up in parallel applications)
            del config["handlers"]
            del config["loggers"]

        return config

    @observe("log_datefmt", "log_format", "log_level", "logging_config")
    def _observe_logging_change(self, change):
        # convert log level strings to ints
        log_level = self.log_level
        if isinstance(log_level, str):
            self.log_level = getattr(logging, log_level)
        self._configure_logging()

    @observe("log", type="default")
    def _observe_logging_default(self, change):
        self._configure_logging()

    def _configure_logging(self):
        config = self.get_default_logging_config()
        nested_update(config, self.logging_config or {})
        dictConfig(config)
        # make a note that we have configured logging
        self._logging_configured = True

    @default("log")
    def _log_default(self):
        """Start logging for this application."""
        log = logging.getLogger(self.__class__.__name__)
        log.propagate = False
        _log = log  # copied from Logger.hasHandlers() (new in Python 3.2)
        while _log:
            if _log.handlers:
                return log
            if not _log.propagate:
                break
            else:
                _log = _log.parent  # type:ignore[assignment]
        return log

    logging_config = Dict(
        help="""
            Configure additional log handlers.

            The default stderr logs handler is configured by the
            log_level, log_datefmt and log_format settings.

            This configuration can be used to configure additional handlers
            (e.g. to output the log to a file) or for finer control over the
            default handlers.

            If provided this should be a logging configuration dictionary, for
            more information see:
            https://docs.python.org/3/library/logging.config.html#logging-config-dictschema

            This dictionary is merged with the base logging configuration which
            defines the following:

            * A logging formatter intended for interactive use called
              ``console``.
            * A logging handler that writes to stderr called
              ``console`` which uses the formatter ``console``.
            * A logger with the name of this application set to ``DEBUG``
              level.

            This example adds a new handler that writes to a file:

            .. code-block:: python

               c.Application.logging_config = {
                   'handlers': {
                       'file': {
                           'class': 'logging.FileHandler',
                           'level': 'DEBUG',
                           'filename': '<path/to/file>',
                       }
                   },
                   'loggers': {
                       '<application-name>': {
                           'level': 'DEBUG',
                           # NOTE: if you don't list the default "console"
                           # handler here then it will be disabled
                           'handlers': ['console', 'file'],
                       },
                   }
               }

        """,
    ).tag(config=True)

    #: the alias map for configurables
    #: Keys might strings or tuples for additional options; single-letter alias accessed like `-v`.
    #: Values might be like "Class.trait" strings of two-tuples: (Class.trait, help-text),
    #  or just the "Class.trait" string, in which case the help text is inferred from the
    #  corresponding trait
    aliases: t.Dict[t.Union[str, t.Tuple[str, ...]], t.Union[str, t.Tuple[str, str]]] = {
        "log-level": "Application.log_level"
    }

    # flags for loading Configurables or store_const style flags
    # flags are loaded from this dict by '--key' flags
    # this must be a dict of two-tuples, the first element being the Config/dict
    # and the second being the help string for the flag
    flags: t.Dict[
        t.Union[str, t.Tuple[str, ...]], t.Tuple[t.Union[t.Dict[str, t.Any], Config], str]
    ] = {
        "debug": (
            {
                "Application": {
                    "log_level": logging.DEBUG,
                },
            },
            "Set log-level to debug, for the most verbose logging.",
        ),
        "show-config": (
            {
                "Application": {
                    "show_config": True,
                },
            },
            "Show the application's configuration (human-readable format)",
        ),
        "show-config-json": (
            {
                "Application": {
                    "show_config_json": True,
                },
            },
            "Show the application's configuration (json format)",
        ),
    }

    # subcommands for launching other applications
    # if this is not empty, this will be a parent Application
    # this must be a dict of two-tuples,
    # the first element being the application class/import string
    # and the second being the help string for the subcommand
    subcommands: t.Union[t.Dict[str, t.Tuple[t.Any, str]], Dict] = Dict()
    # parse_command_line will initialize a subapp, if requested
    subapp = Instance("traitlets.config.application.Application", allow_none=True)

    # extra command-line arguments that don't set config values
    extra_args: t.Union[t.List[str], List] = List(Unicode())

    cli_config = Instance(
        Config,
        (),
        {},
        help="""The subset of our configuration that came from the command-line

        We re-load this configuration after loading config files,
        to ensure that it maintains highest priority.
        """,
    )

    _loaded_config_files = List()

    show_config: t.Union[bool, Bool] = Bool(
        help="Instead of starting the Application, dump configuration to stdout"
    ).tag(config=True)

    show_config_json: t.Union[bool, Bool] = Bool(
        help="Instead of starting the Application, dump configuration to stdout (as JSON)"
    ).tag(config=True)

    @observe("show_config_json")
    def _show_config_json_changed(self, change):
        self.show_config = change.new

    @observe("show_config")
    def _show_config_changed(self, change):
        if change.new:
            self._save_start = self.start
            self.start = self.start_show_config  # type:ignore[assignment]

    def __init__(self, **kwargs):
        SingletonConfigurable.__init__(self, **kwargs)
        # Ensure my class is in self.classes, so my attributes appear in command line
        # options and config files.
        cls = self.__class__
        if cls not in self.classes:
            if self.classes is cls.classes:
                # class attr, assign instead of insert
                self.classes = [cls] + self.classes
            else:
                self.classes.insert(0, self.__class__)

    @observe("config")
    @observe_compat
    def _config_changed(self, change):
        super()._config_changed(change)
        self.log.debug("Config changed: %r", change.new)

    @catch_config_error
    def initialize(self, argv=None):
        """Do the basic steps to configure me.

        Override in subclasses.
        """
        self.parse_command_line(argv)

    def start(self):
        """Start the app mainloop.

        Override in subclasses.
        """
        if self.subapp is not None:
            return self.subapp.start()

    def start_show_config(self):
        """start function used when show_config is True"""
        config = self.config.copy()
        # exclude show_config flags from displayed config
        for cls in self.__class__.mro():
            if cls.__name__ in config:
                cls_config = config[cls.__name__]
                cls_config.pop("show_config", None)
                cls_config.pop("show_config_json", None)

        if self.show_config_json:
            json.dump(config, sys.stdout, indent=1, sort_keys=True, default=repr)
            # add trailing newline
            sys.stdout.write("\n")
            return

        if self._loaded_config_files:
            print("Loaded config files:")
            for f in self._loaded_config_files:
                print("  " + f)
            print()

        for classname in sorted(config):
            class_config = config[classname]
            if not class_config:
                continue
            print(classname)
            pformat_kwargs: t.Dict[str, t.Any] = dict(indent=4, compact=True)

            for traitname in sorted(class_config):
                value = class_config[traitname]
                print(
                    "  .{} = {}".format(
                        traitname,
                        pprint.pformat(value, **pformat_kwargs),
                    )
                )

    def print_alias_help(self):
        """Print the alias parts of the help."""
        print("\n".join(self.emit_alias_help()))

    def emit_alias_help(self):
        """Yield the lines for alias part of the help."""
        if not self.aliases:
            return

        classdict = {}
        for cls in self.classes:
            # include all parents (up to, but excluding Configurable) in available names
            for c in cls.mro()[:-3]:
                classdict[c.__name__] = c

        fhelp: t.Optional[str]
        for alias, longname in self.aliases.items():
            try:
                if isinstance(longname, tuple):
                    longname, fhelp = longname
                else:
                    fhelp = None
                classname, traitname = longname.split(".")[-2:]
                longname = classname + "." + traitname
                cls = classdict[classname]

                trait = cls.class_traits(config=True)[traitname]
                fhelp = cls.class_get_trait_help(trait, helptext=fhelp).splitlines()

                if not isinstance(alias, tuple):
                    alias = (alias,)
                alias = sorted(alias, key=len)  # type:ignore[assignment]
                alias = ", ".join(("--%s" if len(m) > 1 else "-%s") % m for m in alias)

                # reformat first line
                assert fhelp is not None
                fhelp[0] = fhelp[0].replace("--" + longname, alias)  # type:ignore
                yield from fhelp
                yield indent("Equivalent to: [--%s]" % longname)
            except Exception as ex:
                self.log.error("Failed collecting help-message for alias %r, due to: %s", alias, ex)
                raise

    def print_flag_help(self):
        """Print the flag part of the help."""
        print("\n".join(self.emit_flag_help()))

    def emit_flag_help(self):
        """Yield the lines for the flag part of the help."""
        if not self.flags:
            return

        for flags, (cfg, fhelp) in self.flags.items():
            try:
                if not isinstance(flags, tuple):
                    flags = (flags,)
                flags = sorted(flags, key=len)  # type:ignore[assignment]
                flags = ", ".join(("--%s" if len(m) > 1 else "-%s") % m for m in flags)
                yield flags
                yield indent(dedent(fhelp.strip()))
                cfg_list = " ".join(
                    f"--{clname}.{prop}={val}"
                    for clname, props_dict in cfg.items()
                    for prop, val in props_dict.items()
                )
                cfg_txt = "Equivalent to: [%s]" % cfg_list
                yield indent(dedent(cfg_txt))
            except Exception as ex:
                self.log.error("Failed collecting help-message for flag %r, due to: %s", flags, ex)
                raise

    def print_options(self):
        """Print the options part of the help."""
        print("\n".join(self.emit_options_help()))

    def emit_options_help(self):
        """Yield the lines for the options part of the help."""
        if not self.flags and not self.aliases:
            return
        header = "Options"
        yield header
        yield "=" * len(header)
        for p in wrap_paragraphs(self.option_description):
            yield p
            yield ""

        yield from self.emit_flag_help()
        yield from self.emit_alias_help()
        yield ""

    def print_subcommands(self):
        """Print the subcommand part of the help."""
        print("\n".join(self.emit_subcommands_help()))

    def emit_subcommands_help(self):
        """Yield the lines for the subcommand part of the help."""
        if not self.subcommands:
            return

        header = "Subcommands"
        yield header
        yield "=" * len(header)
        for p in wrap_paragraphs(self.subcommand_description.format(app=self.name)):
            yield p
            yield ""
        for subc, (_, help) in self.subcommands.items():
            yield subc
            if help:
                yield indent(dedent(help.strip()))
        yield ""

    def emit_help_epilogue(self, classes):
        """Yield the very bottom lines of the help message.

        If classes=False (the default), print `--help-all` msg.
        """
        if not classes:
            yield "To see all available configurables, use `--help-all`."
            yield ""

    def print_help(self, classes=False):
        """Print the help for each Configurable class in self.classes.

        If classes=False (the default), only flags and aliases are printed.
        """
        print("\n".join(self.emit_help(classes=classes)))

    def emit_help(self, classes=False):
        """Yield the help-lines for each Configurable class in self.classes.

        If classes=False (the default), only flags and aliases are printed.
        """
        yield from self.emit_description()
        yield from self.emit_subcommands_help()
        yield from self.emit_options_help()

        if classes:
            help_classes = self._classes_with_config_traits()
            if help_classes:
                yield "Class options"
                yield "============="
                for p in wrap_paragraphs(self.keyvalue_description):
                    yield p
                    yield ""

            for cls in help_classes:
                yield cls.class_get_help()
                yield ""
        yield from self.emit_examples()

        yield from self.emit_help_epilogue(classes)

    def document_config_options(self):
        """Generate rST format documentation for the config options this application

        Returns a multiline string.
        """
        return "\n".join(c.class_config_rst_doc() for c in self._classes_inc_parents())

    def print_description(self):
        """Print the application description."""
        print("\n".join(self.emit_description()))

    def emit_description(self):
        """Yield lines with the application description."""
        for p in wrap_paragraphs(self.description or self.__doc__ or ""):
            yield p
            yield ""

    def print_examples(self):
        """Print usage and examples (see `emit_examples()`)."""
        print("\n".join(self.emit_examples()))

    def emit_examples(self):
        """Yield lines with the usage and examples.

        This usage string goes at the end of the command line help string
        and should contain examples of the application's usage.
        """
        if self.examples:
            yield "Examples"
            yield "--------"
            yield ""
            yield indent(dedent(self.examples.strip()))
            yield ""

    def print_version(self):
        """Print the version string."""
        print(self.version)

    @catch_config_error
    def initialize_subcommand(self, subc, argv=None):
        """Initialize a subcommand with argv."""
        val = self.subcommands.get(subc)
        assert val is not None
        subapp, _ = val

        if isinstance(subapp, str):
            subapp = import_item(subapp)

        # Cannot issubclass() on a non-type (SOhttp://stackoverflow.com/questions/8692430)
        if isinstance(subapp, type) and issubclass(subapp, Application):
            # Clear existing instances before...
            self.__class__.clear_instance()
            # instantiating subapp...
            self.subapp = subapp.instance(parent=self)
        elif callable(subapp):
            # or ask factory to create it...
            self.subapp = subapp(self)  # type:ignore[call-arg]
        else:
            raise AssertionError("Invalid mappings for subcommand '%s'!" % subc)

        # ... and finally initialize subapp.
        self.subapp.initialize(argv)

    def flatten_flags(self):
        """Flatten flags and aliases for loaders, so cl-args override as expected.

        This prevents issues such as an alias pointing to InteractiveShell,
        but a config file setting the same trait in TerminalInteraciveShell
        getting inappropriate priority over the command-line arg.
        Also, loaders expect ``(key: longname)`` and not ``key: (longname, help)`` items.

        Only aliases with exactly one descendent in the class list
        will be promoted.

        """
        # build a tree of classes in our list that inherit from a particular
        # it will be a dict by parent classname of classes in our list
        # that are descendents
        mro_tree = defaultdict(list)
        for cls in self.classes:
            clsname = cls.__name__
            for parent in cls.mro()[1:-3]:
                # exclude cls itself and Configurable,HasTraits,object
                mro_tree[parent.__name__].append(clsname)
        # flatten aliases, which have the form:
        # { 'alias' : 'Class.trait' }
        aliases: t.Dict[str, str] = {}
        for alias, longname in self.aliases.items():
            if isinstance(longname, tuple):
                longname, _ = longname
            cls, trait = longname.split(".", 1)  # type:ignore
            children = mro_tree[cls]  # type:ignore[index]
            if len(children) == 1:
                # exactly one descendent, promote alias
                cls = children[0]  # type:ignore[assignment]
            if not isinstance(aliases, tuple):
                alias = (alias,)  # type:ignore[assignment]
            for al in alias:
                aliases[al] = ".".join([cls, trait])  # type:ignore[list-item]

        # flatten flags, which are of the form:
        # { 'key' : ({'Cls' : {'trait' : value}}, 'help')}
        flags = {}
        for key, (flagdict, help) in self.flags.items():
            newflag: t.Dict[t.Any, t.Any] = {}
            for cls, subdict in flagdict.items():  # type:ignore
                children = mro_tree[cls]  # type:ignore[index]
                # exactly one descendent, promote flag section
                if len(children) == 1:
                    cls = children[0]  # type:ignore[assignment]

                if cls in newflag:
                    newflag[cls].update(subdict)
                else:
                    newflag[cls] = subdict

            if not isinstance(key, tuple):
                key = (key,)
            for k in key:
                flags[k] = (newflag, help)
        return flags, aliases

    def _create_loader(self, argv, aliases, flags, classes):
        return KVArgParseConfigLoader(
            argv, aliases, flags, classes=classes, log=self.log, subcommands=self.subcommands
        )

    @classmethod
    def _get_sys_argv(cls, check_argcomplete: bool = False) -> t.List[str]:
        """Get `sys.argv` or equivalent from `argcomplete`

        `argcomplete`'s strategy is to call the python script with no arguments,
        so ``len(sys.argv) == 1``, and run until the `ArgumentParser` is constructed
        and determine what completions are available.

        On the other hand, `traitlet`'s subcommand-handling strategy is to check
        ``sys.argv[1]`` and see if it matches a subcommand, and if so then dynamically
        load the subcommand app and initialize it with ``sys.argv[1:]``.

        This helper method helps to take the current tokens for `argcomplete` and pass
        them through as `argv`.
        """
        if check_argcomplete and "_ARGCOMPLETE" in os.environ:
            try:
                from traitlets.config.argcomplete_config import get_argcomplete_cwords

                cwords = get_argcomplete_cwords()
                assert cwords is not None
                return cwords
            except (ImportError, ModuleNotFoundError):
                pass
        return sys.argv

    @classmethod
    def _handle_argcomplete_for_subcommand(cls):
        """Helper for `argcomplete` to recognize `traitlets` subcommands

        `argcomplete` does not know that `traitlets` has already consumed subcommands,
        as it only "sees" the final `argparse.ArgumentParser` that is constructed.
        (Indeed `KVArgParseConfigLoader` does not get passed subcommands at all currently.)
        We explicitly manipulate the environment variables used internally by `argcomplete`
        to get it to skip over the subcommand tokens.
        """
        if "_ARGCOMPLETE" not in os.environ:
            return

        try:
            from traitlets.config.argcomplete_config import increment_argcomplete_index

            increment_argcomplete_index()
        except (ImportError, ModuleNotFoundError):
            pass

    @catch_config_error
    def parse_command_line(self, argv=None):
        """Parse the command line arguments."""
        assert not isinstance(argv, str)
        if argv is None:
            argv = self._get_sys_argv(check_argcomplete=bool(self.subcommands))[1:]
        self.argv = [cast_unicode(arg) for arg in argv]

        if argv and argv[0] == "help":
            # turn `ipython help notebook` into `ipython notebook -h`
            argv = argv[1:] + ["-h"]

        if self.subcommands and len(argv) > 0:
            # we have subcommands, and one may have been specified
            subc, subargv = argv[0], argv[1:]
            if re.match(r"^\w(\-?\w)*$", subc) and subc in self.subcommands:
                # it's a subcommand, and *not* a flag or class parameter
                self._handle_argcomplete_for_subcommand()
                return self.initialize_subcommand(subc, subargv)

        # Arguments after a '--' argument are for the script IPython may be
        # about to run, not IPython iteslf. For arguments parsed here (help and
        # version), we want to only search the arguments up to the first
        # occurrence of '--', which we're calling interpreted_argv.
        try:
            interpreted_argv = argv[: argv.index("--")]
        except ValueError:
            interpreted_argv = argv

        if any(x in interpreted_argv for x in ("-h", "--help-all", "--help")):
            self.print_help("--help-all" in interpreted_argv)
            self.exit(0)

        if "--version" in interpreted_argv or "-V" in interpreted_argv:
            self.print_version()
            self.exit(0)

        # flatten flags&aliases, so cl-args get appropriate priority:
        flags, aliases = self.flatten_flags()
        classes = tuple(self._classes_with_config_traits())
        loader = self._create_loader(argv, aliases, flags, classes=classes)
        try:
            self.cli_config = deepcopy(loader.load_config())
        except SystemExit:
            # traitlets 5: no longer print help output on error
            # help output is huge, and comes after the error
            raise
        self.update_config(self.cli_config)
        # store unparsed args in extra_args
        self.extra_args = loader.extra_args

    @classmethod
    def _load_config_files(cls, basefilename, path=None, log=None, raise_config_file_errors=False):
        """Load config files (py,json) by filename and path.

        yield each config object in turn.
        """

        if not isinstance(path, list):
            path = [path]
        for current in reversed(path):
            # path list is in descending priority order, so load files backwards:
            pyloader = cls.python_config_loader_class(basefilename + ".py", path=current, log=log)
            if log:
                log.debug("Looking for %s in %s", basefilename, current or os.getcwd())
            jsonloader = cls.json_config_loader_class(basefilename + ".json", path=current, log=log)
            loaded: t.List[t.Any] = []
            filenames: t.List[str] = []
            for loader in [pyloader, jsonloader]:
                config = None
                try:
                    config = loader.load_config()
                except ConfigFileNotFound:
                    pass
                except Exception:
                    # try to get the full filename, but it will be empty in the
                    # unlikely event that the error raised before filefind finished
                    filename = loader.full_filename or basefilename
                    # problem while running the file
                    if raise_config_file_errors:
                        raise
                    if log:
                        log.error("Exception while loading config file %s", filename, exc_info=True)
                else:
                    if log:
                        log.debug("Loaded config file: %s", loader.full_filename)
                if config:
                    for filename, earlier_config in zip(filenames, loaded):
                        collisions = earlier_config.collisions(config)
                        if collisions and log:
                            log.warning(
                                "Collisions detected in {0} and {1} config files."
                                " {1} has higher priority: {2}".format(
                                    filename,
                                    loader.full_filename,
                                    json.dumps(collisions, indent=2),
                                )
                            )
                    yield (config, loader.full_filename)
                    loaded.append(config)
                    filenames.append(loader.full_filename)

    @property
    def loaded_config_files(self):
        """Currently loaded configuration files"""
        return self._loaded_config_files[:]

    @catch_config_error
    def load_config_file(self, filename, path=None):
        """Load config files by filename and path."""
        filename, ext = os.path.splitext(filename)
        new_config = Config()
        for (config, fname) in self._load_config_files(
            filename,
            path=path,
            log=self.log,
            raise_config_file_errors=self.raise_config_file_errors,
        ):
            new_config.merge(config)
            if (
                fname not in self._loaded_config_files
            ):  # only add to list of loaded files if not previously loaded
                self._loaded_config_files.append(fname)
        # add self.cli_config to preserve CLI config priority
        new_config.merge(self.cli_config)
        self.update_config(new_config)

    def _classes_with_config_traits(self, classes=None):
        """
        Yields only classes with configurable traits, and their subclasses.

        :param classes:
            The list of classes to iterate; if not set, uses :attr:`classes`.

        Thus, produced sample config-file will contain all classes
        on which a trait-value may be overridden:

        - either on the class owning the trait,
        - or on its subclasses, even if those subclasses do not define
          any traits themselves.
        """
        if classes is None:
            classes = self.classes

        cls_to_config = OrderedDict(
            (cls, bool(cls.class_own_traits(config=True)))
            for cls in self._classes_inc_parents(classes)
        )

        def is_any_parent_included(cls):
            return any(b in cls_to_config and cls_to_config[b] for b in cls.__bases__)

        # Mark "empty" classes for inclusion if their parents own-traits,
        #  and loop until no more classes gets marked.
        #
        while True:
            to_incl_orig = cls_to_config.copy()
            cls_to_config = OrderedDict(
                (cls, inc_yes or is_any_parent_included(cls))
                for cls, inc_yes in cls_to_config.items()
            )
            if cls_to_config == to_incl_orig:
                break
        for cl, inc_yes in cls_to_config.items():
            if inc_yes:
                yield cl

    def generate_config_file(self, classes=None):
        """generate default config file from Configurables"""
        lines = ["# Configuration file for %s." % self.name]
        lines.append("")
        lines.append("c = get_config()  #" + "noqa")
        lines.append("")
        classes = self.classes if classes is None else classes
        config_classes = list(self._classes_with_config_traits(classes))
        for cls in config_classes:
            lines.append(cls.class_config_section(config_classes))
        return "\n".join(lines)

    def close_handlers(self):
        if getattr(self, "_logging_configured", False):
            # don't attempt to close handlers unless they have been opened
            # (note accessing self.log.handlers will create handlers if they
            # have not yet been initialised)
            for handler in self.log.handlers:
                with suppress(Exception):
                    handler.close()
            self._logging_configured = False

    def exit(self, exit_status=0):
        self.log.debug("Exiting application: %s" % self.name)
        self.close_handlers()
        sys.exit(exit_status)

    def __del__(self):
        self.close_handlers()

    @classmethod
    def launch_instance(cls, argv=None, **kwargs):
        """Launch a global instance of this Application

        If a global instance already exists, this reinitializes and starts it
        """
        app = cls.instance(**kwargs)
        app.initialize(argv)
        app.start()


# -----------------------------------------------------------------------------
# utility functions, for convenience
# -----------------------------------------------------------------------------

default_aliases = Application.aliases
default_flags = Application.flags


def boolean_flag(name, configurable, set_help="", unset_help=""):
    """Helper for building basic --trait, --no-trait flags.

    Parameters
    ----------
    name : str
        The name of the flag.
    configurable : str
        The 'Class.trait' string of the trait to be set/unset with the flag
    set_help : unicode
        help string for --name flag
    unset_help : unicode
        help string for --no-name flag

    Returns
    -------
    cfg : dict
        A dict with two keys: 'name', and 'no-name', for setting and unsetting
        the trait, respectively.
    """
    # default helpstrings
    set_help = set_help or "set %s=True" % configurable
    unset_help = unset_help or "set %s=False" % configurable

    cls, trait = configurable.split(".")

    setter = {cls: {trait: True}}
    unsetter = {cls: {trait: False}}
    return {name: (setter, set_help), "no-" + name: (unsetter, unset_help)}


def get_config():
    """Get the config object for the global Application instance, if there is one

    otherwise return an empty config object
    """
    if Application.initialized():
        return Application.instance().config
    else:
        return Config()


if __name__ == "__main__":
    Application.launch_instance()
