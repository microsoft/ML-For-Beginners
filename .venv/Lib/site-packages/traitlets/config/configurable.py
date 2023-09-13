"""A base class for objects that are configurable."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.


import logging
import warnings
from copy import deepcopy
from textwrap import dedent

from traitlets.traitlets import (
    Any,
    Container,
    Dict,
    HasTraits,
    Instance,
    default,
    observe,
    observe_compat,
    validate,
)
from traitlets.utils.text import indent, wrap_paragraphs

from .loader import Config, DeferredConfig, LazyConfigValue, _is_section_key

# -----------------------------------------------------------------------------
# Helper classes for Configurables
# -----------------------------------------------------------------------------


class ConfigurableError(Exception):
    pass


class MultipleInstanceError(ConfigurableError):
    pass


# -----------------------------------------------------------------------------
# Configurable implementation
# -----------------------------------------------------------------------------


class Configurable(HasTraits):

    config = Instance(Config, (), {})
    parent = Instance("traitlets.config.configurable.Configurable", allow_none=True)

    def __init__(self, **kwargs):
        """Create a configurable given a config config.

        Parameters
        ----------
        config : Config
            If this is empty, default values are used. If config is a
            :class:`Config` instance, it will be used to configure the
            instance.
        parent : Configurable instance, optional
            The parent Configurable instance of this object.

        Notes
        -----
        Subclasses of Configurable must call the :meth:`__init__` method of
        :class:`Configurable` *before* doing anything else and using
        :func:`super`::

            class MyConfigurable(Configurable):
                def __init__(self, config=None):
                    super(MyConfigurable, self).__init__(config=config)
                    # Then any other code you need to finish initialization.

        This ensures that instances will be configured properly.
        """
        parent = kwargs.pop("parent", None)
        if parent is not None:
            # config is implied from parent
            if kwargs.get("config", None) is None:
                kwargs["config"] = parent.config
            self.parent = parent

        config = kwargs.pop("config", None)

        # load kwarg traits, other than config
        super().__init__(**kwargs)

        # record traits set by config
        config_override_names = set()

        def notice_config_override(change):
            """Record traits set by both config and kwargs.

            They will need to be overridden again after loading config.
            """
            if change.name in kwargs:
                config_override_names.add(change.name)

        self.observe(notice_config_override)

        # load config
        if config is not None:
            # We used to deepcopy, but for now we are trying to just save
            # by reference.  This *could* have side effects as all components
            # will share config. In fact, I did find such a side effect in
            # _config_changed below. If a config attribute value was a mutable type
            # all instances of a component were getting the same copy, effectively
            # making that a class attribute.
            # self.config = deepcopy(config)
            self.config = config
        else:
            # allow _config_default to return something
            self._load_config(self.config)
        self.unobserve(notice_config_override)

        for name in config_override_names:
            setattr(self, name, kwargs[name])

    # -------------------------------------------------------------------------
    # Static trait notifiations
    # -------------------------------------------------------------------------

    @classmethod
    def section_names(cls):
        """return section names as a list"""
        return [
            c.__name__
            for c in reversed(cls.__mro__)
            if issubclass(c, Configurable) and issubclass(cls, c)
        ]

    def _find_my_config(self, cfg):
        """extract my config from a global Config object

        will construct a Config object of only the config values that apply to me
        based on my mro(), as well as those of my parent(s) if they exist.

        If I am Bar and my parent is Foo, and their parent is Tim,
        this will return merge following config sections, in this order::

            [Bar, Foo.Bar, Tim.Foo.Bar]

        With the last item being the highest priority.
        """
        cfgs = [cfg]
        if self.parent:
            cfgs.append(self.parent._find_my_config(cfg))
        my_config = Config()
        for c in cfgs:
            for sname in self.section_names():
                # Don't do a blind getattr as that would cause the config to
                # dynamically create the section with name Class.__name__.
                if c._has_section(sname):
                    my_config.merge(c[sname])
        return my_config

    def _load_config(self, cfg, section_names=None, traits=None):
        """load traits from a Config object"""

        if traits is None:
            traits = self.traits(config=True)
        if section_names is None:
            section_names = self.section_names()

        my_config = self._find_my_config(cfg)

        # hold trait notifications until after all config has been loaded
        with self.hold_trait_notifications():
            for name, config_value in my_config.items():
                if name in traits:
                    if isinstance(config_value, LazyConfigValue):
                        # ConfigValue is a wrapper for using append / update on containers
                        # without having to copy the initial value
                        initial = getattr(self, name)
                        config_value = config_value.get_value(initial)
                    elif isinstance(config_value, DeferredConfig):
                        # DeferredConfig tends to come from CLI/environment variables
                        config_value = config_value.get_value(traits[name])
                    # We have to do a deepcopy here if we don't deepcopy the entire
                    # config object. If we don't, a mutable config_value will be
                    # shared by all instances, effectively making it a class attribute.
                    setattr(self, name, deepcopy(config_value))
                elif not _is_section_key(name) and not isinstance(config_value, Config):
                    from difflib import get_close_matches

                    if isinstance(self, LoggingConfigurable):
                        warn = self.log.warning
                    else:
                        warn = lambda msg: warnings.warn(msg, stacklevel=9)  # noqa[E371]
                    matches = get_close_matches(name, traits)
                    msg = "Config option `{option}` not recognized by `{klass}`.".format(
                        option=name, klass=self.__class__.__name__
                    )

                    if len(matches) == 1:
                        msg += f"  Did you mean `{matches[0]}`?"
                    elif len(matches) >= 1:
                        msg += "  Did you mean one of: `{matches}`?".format(
                            matches=", ".join(sorted(matches))
                        )
                    warn(msg)

    @observe("config")
    @observe_compat
    def _config_changed(self, change):
        """Update all the class traits having ``config=True`` in metadata.

        For any class trait with a ``config`` metadata attribute that is
        ``True``, we update the trait with the value of the corresponding
        config entry.
        """
        # Get all traits with a config metadata entry that is True
        traits = self.traits(config=True)

        # We auto-load config section for this class as well as any parent
        # classes that are Configurable subclasses.  This starts with Configurable
        # and works down the mro loading the config for each section.
        section_names = self.section_names()
        self._load_config(change.new, traits=traits, section_names=section_names)

    def update_config(self, config):
        """Update config and load the new values"""
        # traitlets prior to 4.2 created a copy of self.config in order to trigger change events.
        # Some projects (IPython < 5) relied upon one side effect of this,
        # that self.config prior to update_config was not modified in-place.
        # For backward-compatibility, we must ensure that self.config
        # is a new object and not modified in-place,
        # but config consumers should not rely on this behavior.
        self.config = deepcopy(self.config)
        # load config
        self._load_config(config)
        # merge it into self.config
        self.config.merge(config)
        # TODO: trigger change event if/when dict-update change events take place
        # DO NOT trigger full trait-change

    @classmethod
    def class_get_help(cls, inst=None):
        """Get the help string for this class in ReST format.

        If `inst` is given, its current trait values will be used in place of
        class defaults.
        """
        assert inst is None or isinstance(inst, cls)
        final_help = []
        base_classes = ", ".join(p.__name__ for p in cls.__bases__)
        final_help.append(f"{cls.__name__}({base_classes}) options")
        final_help.append(len(final_help[0]) * "-")
        for _, v in sorted(cls.class_traits(config=True).items()):
            help = cls.class_get_trait_help(v, inst)
            final_help.append(help)
        return "\n".join(final_help)

    @classmethod
    def class_get_trait_help(cls, trait, inst=None, helptext=None):
        """Get the helptext string for a single trait.

        :param inst:
            If given, its current trait values will be used in place of
            the class default.
        :param helptext:
            If not given, uses the `help` attribute of the current trait.
        """
        assert inst is None or isinstance(inst, cls)
        lines = []
        header = f"--{cls.__name__}.{trait.name}"
        if isinstance(trait, (Container, Dict)):
            multiplicity = trait.metadata.get("multiplicity", "append")
            if isinstance(trait, Dict):
                sample_value = "<key-1>=<value-1>"
            else:
                sample_value = "<%s-item-1>" % trait.__class__.__name__.lower()
            if multiplicity == "append":
                header = f"{header}={sample_value}..."
            else:
                header = f"{header} {sample_value}..."
        else:
            header = f"{header}=<{trait.__class__.__name__}>"
        # header = "--%s.%s=<%s>" % (cls.__name__, trait.name, trait.__class__.__name__)
        lines.append(header)

        if helptext is None:
            helptext = trait.help
        if helptext != "":
            helptext = "\n".join(wrap_paragraphs(helptext, 76))
            lines.append(indent(helptext))

        if "Enum" in trait.__class__.__name__:
            # include Enum choices
            lines.append(indent("Choices: %s" % trait.info()))

        if inst is not None:
            lines.append(indent(f"Current: {getattr(inst, trait.name)!r}"))
        else:
            try:
                dvr = trait.default_value_repr()
            except Exception:
                dvr = None  # ignore defaults we can't construct
            if dvr is not None:
                if len(dvr) > 64:
                    dvr = dvr[:61] + "..."
                lines.append(indent("Default: %s" % dvr))

        return "\n".join(lines)

    @classmethod
    def class_print_help(cls, inst=None):
        """Get the help string for a single trait and print it."""
        print(cls.class_get_help(inst))

    @classmethod
    def _defining_class(cls, trait, classes):
        """Get the class that defines a trait

        For reducing redundant help output in config files.
        Returns the current class if:
        - the trait is defined on this class, or
        - the class where it is defined would not be in the config file

        Parameters
        ----------
        trait : Trait
            The trait to look for
        classes : list
            The list of other classes to consider for redundancy.
            Will return `cls` even if it is not defined on `cls`
            if the defining class is not in `classes`.
        """
        defining_cls = cls
        for parent in cls.mro():
            if (
                issubclass(parent, Configurable)
                and parent in classes
                and parent.class_own_traits(config=True).get(trait.name, None) is trait
            ):
                defining_cls = parent
        return defining_cls

    @classmethod
    def class_config_section(cls, classes=None):
        """Get the config section for this class.

        Parameters
        ----------
        classes : list, optional
            The list of other classes in the config file.
            Used to reduce redundant information.
        """

        def c(s):
            """return a commented, wrapped block."""
            s = "\n\n".join(wrap_paragraphs(s, 78))

            return "## " + s.replace("\n", "\n#  ")

        # section header
        breaker = "#" + "-" * 78
        parent_classes = ", ".join(p.__name__ for p in cls.__bases__ if issubclass(p, Configurable))

        s = f"# {cls.__name__}({parent_classes}) configuration"
        lines = [breaker, s, breaker]
        # get the description trait
        desc = cls.class_traits().get("description")
        if desc:
            desc = desc.default_value
        if not desc:
            # no description from trait, use __doc__
            desc = getattr(cls, "__doc__", "")
        if desc:
            lines.append(c(desc))
            lines.append("")

        for name, trait in sorted(cls.class_traits(config=True).items()):
            default_repr = trait.default_value_repr()

            if classes:
                defining_class = cls._defining_class(trait, classes)
            else:
                defining_class = cls
            if defining_class is cls:
                # cls owns the trait, show full help
                if trait.help:
                    lines.append(c(trait.help))
                if "Enum" in type(trait).__name__:
                    # include Enum choices
                    lines.append("#  Choices: %s" % trait.info())
                lines.append("#  Default: %s" % default_repr)
            else:
                # Trait appears multiple times and isn't defined here.
                # Truncate help to first line + "See also Original.trait"
                if trait.help:
                    lines.append(c(trait.help.split("\n", 1)[0]))
                lines.append(f"#  See also: {defining_class.__name__}.{name}")

            lines.append(f"# c.{cls.__name__}.{name} = {default_repr}")
            lines.append("")
        return "\n".join(lines)

    @classmethod
    def class_config_rst_doc(cls):
        """Generate rST documentation for this class' config options.

        Excludes traits defined on parent classes.
        """
        lines = []
        classname = cls.__name__
        for _, trait in sorted(cls.class_traits(config=True).items()):
            ttype = trait.__class__.__name__

            termline = classname + "." + trait.name

            # Choices or type
            if "Enum" in ttype:
                # include Enum choices
                termline += " : " + trait.info_rst()
            else:
                termline += " : " + ttype
            lines.append(termline)

            # Default value
            try:
                dvr = trait.default_value_repr()
            except Exception:
                dvr = None  # ignore defaults we can't construct
            if dvr is not None:
                if len(dvr) > 64:
                    dvr = dvr[:61] + "..."
                # Double up backslashes, so they get to the rendered docs
                dvr = dvr.replace("\\n", "\\\\n")
                lines.append(indent("Default: ``%s``" % dvr))
                lines.append("")

            help = trait.help or "No description"
            lines.append(indent(dedent(help)))

            # Blank line
            lines.append("")

        return "\n".join(lines)


class LoggingConfigurable(Configurable):
    """A parent class for Configurables that log.

    Subclasses have a log trait, and the default behavior
    is to get the logger from the currently running Application.
    """

    log = Any(help="Logger or LoggerAdapter instance")

    @validate("log")
    def _validate_log(self, proposal):
        if not isinstance(proposal.value, (logging.Logger, logging.LoggerAdapter)):
            # warn about unsupported type, but be lenient to allow for duck typing
            warnings.warn(
                f"{self.__class__.__name__}.log should be a Logger or LoggerAdapter,"
                f" got {proposal.value}."
            )
        return proposal.value

    @default("log")
    def _log_default(self):
        if isinstance(self.parent, LoggingConfigurable):
            return self.parent.log
        from traitlets import log

        return log.get_logger()

    def _get_log_handler(self):
        """Return the default Handler

        Returns None if none can be found

        Deprecated, this now returns the first log handler which may or may
        not be the default one.
        """
        logger = self.log
        if isinstance(logger, logging.LoggerAdapter):
            logger = logger.logger
        if not getattr(logger, "handlers", None):
            # no handlers attribute or empty handlers list
            return None
        return logger.handlers[0]


class SingletonConfigurable(LoggingConfigurable):
    """A configurable that only allows one instance.

    This class is for classes that should only have one instance of itself
    or *any* subclass. To create and retrieve such a class use the
    :meth:`SingletonConfigurable.instance` method.
    """

    _instance = None

    @classmethod
    def _walk_mro(cls):
        """Walk the cls.mro() for parent classes that are also singletons

        For use in instance()
        """

        for subclass in cls.mro():
            if (
                issubclass(cls, subclass)
                and issubclass(subclass, SingletonConfigurable)
                and subclass != SingletonConfigurable
            ):
                yield subclass

    @classmethod
    def clear_instance(cls):
        """unset _instance for this class and singleton parents."""
        if not cls.initialized():
            return
        for subclass in cls._walk_mro():
            if isinstance(subclass._instance, cls):
                # only clear instances that are instances
                # of the calling class
                subclass._instance = None

    @classmethod
    def instance(cls, *args, **kwargs):
        """Returns a global instance of this class.

        This method create a new instance if none have previously been created
        and returns a previously created instance is one already exists.

        The arguments and keyword arguments passed to this method are passed
        on to the :meth:`__init__` method of the class upon instantiation.

        Examples
        --------
        Create a singleton class using instance, and retrieve it::

            >>> from traitlets.config.configurable import SingletonConfigurable
            >>> class Foo(SingletonConfigurable): pass
            >>> foo = Foo.instance()
            >>> foo == Foo.instance()
            True

        Create a subclass that is retrived using the base class instance::

            >>> class Bar(SingletonConfigurable): pass
            >>> class Bam(Bar): pass
            >>> bam = Bam.instance()
            >>> bam == Bar.instance()
            True
        """
        # Create and save the instance
        if cls._instance is None:
            inst = cls(*args, **kwargs)
            # Now make sure that the instance will also be returned by
            # parent classes' _instance attribute.
            for subclass in cls._walk_mro():
                subclass._instance = inst

        if isinstance(cls._instance, cls):
            return cls._instance
        else:
            raise MultipleInstanceError(
                "An incompatible sibling of '%s' is already instantiated"
                " as singleton: %s" % (cls.__name__, type(cls._instance).__name__)
            )

    @classmethod
    def initialized(cls):
        """Has an instance been created?"""
        return hasattr(cls, "_instance") and cls._instance is not None
