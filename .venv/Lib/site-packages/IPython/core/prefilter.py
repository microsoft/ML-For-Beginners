# encoding: utf-8
"""
Prefiltering components.

Prefilters transform user input before it is exec'd by Python.  These
transforms are used to implement additional syntax such as !ls and %magic.
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

from keyword import iskeyword
import re

from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
    ESC_MAGIC,
    ESC_QUOTE,
    ESC_QUOTE2,
    ESC_PAREN,
)
from .macro import Macro
from .splitinput import LineInfo

from traitlets import (
    List, Integer, Unicode, Bool, Instance, CRegExp
)

#-----------------------------------------------------------------------------
# Global utilities, errors and constants
#-----------------------------------------------------------------------------


class PrefilterError(Exception):
    pass


# RegExp to identify potential function names
re_fun_name = re.compile(r'[^\W\d]([\w.]*) *$')

# RegExp to exclude strings with this start from autocalling.  In
# particular, all binary operators should be excluded, so that if foo is
# callable, foo OP bar doesn't become foo(OP bar), which is invalid.  The
# characters '!=()' don't need to be checked for, as the checkPythonChars
# routine explicitly does so, to catch direct calls and rebindings of
# existing names.

# Warning: the '-' HAS TO BE AT THE END of the first group, otherwise
# it affects the rest of the group in square brackets.
re_exclude_auto = re.compile(r'^[,&^\|\*/\+-]'
                             r'|^is |^not |^in |^and |^or ')

# try to catch also methods for stuff in lists/tuples/dicts: off
# (experimental). For this to work, the line_split regexp would need
# to be modified so it wouldn't break things at '['. That line is
# nasty enough that I shouldn't change it until I can test it _well_.
#self.re_fun_name = re.compile (r'[a-zA-Z_]([a-zA-Z0-9_.\[\]]*) ?$')


# Handler Check Utilities
def is_shadowed(identifier, ip):
    """Is the given identifier defined in one of the namespaces which shadow
    the alias and magic namespaces?  Note that an identifier is different
    than ifun, because it can not contain a '.' character."""
    # This is much safer than calling ofind, which can change state
    return (identifier in ip.user_ns \
            or identifier in ip.user_global_ns \
            or identifier in ip.ns_table['builtin']\
            or iskeyword(identifier))


#-----------------------------------------------------------------------------
# Main Prefilter manager
#-----------------------------------------------------------------------------


class PrefilterManager(Configurable):
    """Main prefilter component.

    The IPython prefilter is run on all user input before it is run.  The
    prefilter consumes lines of input and produces transformed lines of
    input.

    The implementation consists of two phases:

    1. Transformers
    2. Checkers and handlers

    Over time, we plan on deprecating the checkers and handlers and doing
    everything in the transformers.

    The transformers are instances of :class:`PrefilterTransformer` and have
    a single method :meth:`transform` that takes a line and returns a
    transformed line.  The transformation can be accomplished using any
    tool, but our current ones use regular expressions for speed.

    After all the transformers have been run, the line is fed to the checkers,
    which are instances of :class:`PrefilterChecker`.  The line is passed to
    the :meth:`check` method, which either returns `None` or a
    :class:`PrefilterHandler` instance.  If `None` is returned, the other
    checkers are tried.  If an :class:`PrefilterHandler` instance is returned,
    the line is passed to the :meth:`handle` method of the returned
    handler and no further checkers are tried.

    Both transformers and checkers have a `priority` attribute, that determines
    the order in which they are called.  Smaller priorities are tried first.

    Both transformers and checkers also have `enabled` attribute, which is
    a boolean that determines if the instance is used.

    Users or developers can change the priority or enabled attribute of
    transformers or checkers, but they must call the :meth:`sort_checkers`
    or :meth:`sort_transformers` method after changing the priority.
    """

    multi_line_specials = Bool(True).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)

    def __init__(self, shell=None, **kwargs):
        super(PrefilterManager, self).__init__(shell=shell, **kwargs)
        self.shell = shell
        self._transformers = []
        self.init_handlers()
        self.init_checkers()

    #-------------------------------------------------------------------------
    # API for managing transformers
    #-------------------------------------------------------------------------

    def sort_transformers(self):
        """Sort the transformers by priority.

        This must be called after the priority of a transformer is changed.
        The :meth:`register_transformer` method calls this automatically.
        """
        self._transformers.sort(key=lambda x: x.priority)

    @property
    def transformers(self):
        """Return a list of checkers, sorted by priority."""
        return self._transformers

    def register_transformer(self, transformer):
        """Register a transformer instance."""
        if transformer not in self._transformers:
            self._transformers.append(transformer)
            self.sort_transformers()

    def unregister_transformer(self, transformer):
        """Unregister a transformer instance."""
        if transformer in self._transformers:
            self._transformers.remove(transformer)

    #-------------------------------------------------------------------------
    # API for managing checkers
    #-------------------------------------------------------------------------

    def init_checkers(self):
        """Create the default checkers."""
        self._checkers = []
        for checker in _default_checkers:
            checker(
                shell=self.shell, prefilter_manager=self, parent=self
            )

    def sort_checkers(self):
        """Sort the checkers by priority.

        This must be called after the priority of a checker is changed.
        The :meth:`register_checker` method calls this automatically.
        """
        self._checkers.sort(key=lambda x: x.priority)

    @property
    def checkers(self):
        """Return a list of checkers, sorted by priority."""
        return self._checkers

    def register_checker(self, checker):
        """Register a checker instance."""
        if checker not in self._checkers:
            self._checkers.append(checker)
            self.sort_checkers()

    def unregister_checker(self, checker):
        """Unregister a checker instance."""
        if checker in self._checkers:
            self._checkers.remove(checker)

    #-------------------------------------------------------------------------
    # API for managing handlers
    #-------------------------------------------------------------------------

    def init_handlers(self):
        """Create the default handlers."""
        self._handlers = {}
        self._esc_handlers = {}
        for handler in _default_handlers:
            handler(
                shell=self.shell, prefilter_manager=self, parent=self
            )

    @property
    def handlers(self):
        """Return a dict of all the handlers."""
        return self._handlers

    def register_handler(self, name, handler, esc_strings):
        """Register a handler instance by name with esc_strings."""
        self._handlers[name] = handler
        for esc_str in esc_strings:
            self._esc_handlers[esc_str] = handler

    def unregister_handler(self, name, handler, esc_strings):
        """Unregister a handler instance by name with esc_strings."""
        try:
            del self._handlers[name]
        except KeyError:
            pass
        for esc_str in esc_strings:
            h = self._esc_handlers.get(esc_str)
            if h is handler:
                del self._esc_handlers[esc_str]

    def get_handler_by_name(self, name):
        """Get a handler by its name."""
        return self._handlers.get(name)

    def get_handler_by_esc(self, esc_str):
        """Get a handler by its escape string."""
        return self._esc_handlers.get(esc_str)

    #-------------------------------------------------------------------------
    # Main prefiltering API
    #-------------------------------------------------------------------------

    def prefilter_line_info(self, line_info):
        """Prefilter a line that has been converted to a LineInfo object.

        This implements the checker/handler part of the prefilter pipe.
        """
        # print "prefilter_line_info: ", line_info
        handler = self.find_handler(line_info)
        return handler.handle(line_info)

    def find_handler(self, line_info):
        """Find a handler for the line_info by trying checkers."""
        for checker in self.checkers:
            if checker.enabled:
                handler = checker.check(line_info)
                if handler:
                    return handler
        return self.get_handler_by_name('normal')

    def transform_line(self, line, continue_prompt):
        """Calls the enabled transformers in order of increasing priority."""
        for transformer in self.transformers:
            if transformer.enabled:
                line = transformer.transform(line, continue_prompt)
        return line

    def prefilter_line(self, line, continue_prompt=False):
        """Prefilter a single input line as text.

        This method prefilters a single line of text by calling the
        transformers and then the checkers/handlers.
        """

        # print "prefilter_line: ", line, continue_prompt
        # All handlers *must* return a value, even if it's blank ('').

        # save the line away in case we crash, so the post-mortem handler can
        # record it
        self.shell._last_input_line = line

        if not line:
            # Return immediately on purely empty lines, so that if the user
            # previously typed some whitespace that started a continuation
            # prompt, he can break out of that loop with just an empty line.
            # This is how the default python prompt works.
            return ''

        # At this point, we invoke our transformers.
        if not continue_prompt or (continue_prompt and self.multi_line_specials):
            line = self.transform_line(line, continue_prompt)

        # Now we compute line_info for the checkers and handlers
        line_info = LineInfo(line, continue_prompt)

        # the input history needs to track even empty lines
        stripped = line.strip()

        normal_handler = self.get_handler_by_name('normal')
        if not stripped:
            return normal_handler.handle(line_info)

        # special handlers are only allowed for single line statements
        if continue_prompt and not self.multi_line_specials:
            return normal_handler.handle(line_info)

        prefiltered = self.prefilter_line_info(line_info)
        # print "prefiltered line: %r" % prefiltered
        return prefiltered

    def prefilter_lines(self, lines, continue_prompt=False):
        """Prefilter multiple input lines of text.

        This is the main entry point for prefiltering multiple lines of
        input.  This simply calls :meth:`prefilter_line` for each line of
        input.

        This covers cases where there are multiple lines in the user entry,
        which is the case when the user goes back to a multiline history
        entry and presses enter.
        """
        llines = lines.rstrip('\n').split('\n')
        # We can get multiple lines in one shot, where multiline input 'blends'
        # into one line, in cases like recalling from the readline history
        # buffer.  We need to make sure that in such cases, we correctly
        # communicate downstream which line is first and which are continuation
        # ones.
        if len(llines) > 1:
            out = '\n'.join([self.prefilter_line(line, lnum>0)
                             for lnum, line in enumerate(llines) ])
        else:
            out = self.prefilter_line(llines[0], continue_prompt)

        return out

#-----------------------------------------------------------------------------
# Prefilter transformers
#-----------------------------------------------------------------------------


class PrefilterTransformer(Configurable):
    """Transform a line of user input."""

    priority = Integer(100).tag(config=True)
    # Transformers don't currently use shell or prefilter_manager, but as we
    # move away from checkers and handlers, they will need them.
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)
    enabled = Bool(True).tag(config=True)

    def __init__(self, shell=None, prefilter_manager=None, **kwargs):
        super(PrefilterTransformer, self).__init__(
            shell=shell, prefilter_manager=prefilter_manager, **kwargs
        )
        self.prefilter_manager.register_transformer(self)

    def transform(self, line, continue_prompt):
        """Transform a line, returning the new one."""
        return None

    def __repr__(self):
        return "<%s(priority=%r, enabled=%r)>" % (
            self.__class__.__name__, self.priority, self.enabled)


#-----------------------------------------------------------------------------
# Prefilter checkers
#-----------------------------------------------------------------------------


class PrefilterChecker(Configurable):
    """Inspect an input line and return a handler for that line."""

    priority = Integer(100).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)
    enabled = Bool(True).tag(config=True)

    def __init__(self, shell=None, prefilter_manager=None, **kwargs):
        super(PrefilterChecker, self).__init__(
            shell=shell, prefilter_manager=prefilter_manager, **kwargs
        )
        self.prefilter_manager.register_checker(self)

    def check(self, line_info):
        """Inspect line_info and return a handler instance or None."""
        return None

    def __repr__(self):
        return "<%s(priority=%r, enabled=%r)>" % (
            self.__class__.__name__, self.priority, self.enabled)


class EmacsChecker(PrefilterChecker):

    priority = Integer(100).tag(config=True)
    enabled = Bool(False).tag(config=True)

    def check(self, line_info):
        "Emacs ipython-mode tags certain input lines."
        if line_info.line.endswith('# PYTHON-MODE'):
            return self.prefilter_manager.get_handler_by_name('emacs')
        else:
            return None


class MacroChecker(PrefilterChecker):

    priority = Integer(250).tag(config=True)

    def check(self, line_info):
        obj = self.shell.user_ns.get(line_info.ifun)
        if isinstance(obj, Macro):
            return self.prefilter_manager.get_handler_by_name('macro')
        else:
            return None


class IPyAutocallChecker(PrefilterChecker):

    priority = Integer(300).tag(config=True)

    def check(self, line_info):
        "Instances of IPyAutocall in user_ns get autocalled immediately"
        obj = self.shell.user_ns.get(line_info.ifun, None)
        if isinstance(obj, IPyAutocall):
            obj.set_ip(self.shell)
            return self.prefilter_manager.get_handler_by_name('auto')
        else:
            return None


class AssignmentChecker(PrefilterChecker):

    priority = Integer(600).tag(config=True)

    def check(self, line_info):
        """Check to see if user is assigning to a var for the first time, in
        which case we want to avoid any sort of automagic / autocall games.

        This allows users to assign to either alias or magic names true python
        variables (the magic/alias systems always take second seat to true
        python code).  E.g. ls='hi', or ls,that=1,2"""
        if line_info.the_rest:
            if line_info.the_rest[0] in '=,':
                return self.prefilter_manager.get_handler_by_name('normal')
        else:
            return None


class AutoMagicChecker(PrefilterChecker):

    priority = Integer(700).tag(config=True)

    def check(self, line_info):
        """If the ifun is magic, and automagic is on, run it.  Note: normal,
        non-auto magic would already have been triggered via '%' in
        check_esc_chars. This just checks for automagic.  Also, before
        triggering the magic handler, make sure that there is nothing in the
        user namespace which could shadow it."""
        if not self.shell.automagic or not self.shell.find_magic(line_info.ifun):
            return None

        # We have a likely magic method.  Make sure we should actually call it.
        if line_info.continue_prompt and not self.prefilter_manager.multi_line_specials:
            return None

        head = line_info.ifun.split('.',1)[0]
        if is_shadowed(head, self.shell):
            return None

        return self.prefilter_manager.get_handler_by_name('magic')


class PythonOpsChecker(PrefilterChecker):

    priority = Integer(900).tag(config=True)

    def check(self, line_info):
        """If the 'rest' of the line begins with a function call or pretty much
        any python operator, we should simply execute the line (regardless of
        whether or not there's a possible autocall expansion).  This avoids
        spurious (and very confusing) geattr() accesses."""
        if line_info.the_rest and line_info.the_rest[0] in '!=()<>,+*/%^&|':
            return self.prefilter_manager.get_handler_by_name('normal')
        else:
            return None


class AutocallChecker(PrefilterChecker):

    priority = Integer(1000).tag(config=True)

    function_name_regexp = CRegExp(re_fun_name,
        help="RegExp to identify potential function names."
        ).tag(config=True)
    exclude_regexp = CRegExp(re_exclude_auto,
        help="RegExp to exclude strings with this start from autocalling."
        ).tag(config=True)

    def check(self, line_info):
        "Check if the initial word/function is callable and autocall is on."
        if not self.shell.autocall:
            return None

        oinfo = line_info.ofind(self.shell) # This can mutate state via getattr
        if not oinfo.found:
            return None

        ignored_funs = ['b', 'f', 'r', 'u', 'br', 'rb', 'fr', 'rf']
        ifun = line_info.ifun
        line = line_info.line
        if ifun.lower() in ignored_funs and (line.startswith(ifun + "'") or line.startswith(ifun + '"')):
            return None

        if (
            callable(oinfo.obj)
            and (not self.exclude_regexp.match(line_info.the_rest))
            and self.function_name_regexp.match(line_info.ifun)
        ):
            return self.prefilter_manager.get_handler_by_name("auto")
        else:
            return None


#-----------------------------------------------------------------------------
# Prefilter handlers
#-----------------------------------------------------------------------------


class PrefilterHandler(Configurable):

    handler_name = Unicode('normal')
    esc_strings = List([])
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)

    def __init__(self, shell=None, prefilter_manager=None, **kwargs):
        super(PrefilterHandler, self).__init__(
            shell=shell, prefilter_manager=prefilter_manager, **kwargs
        )
        self.prefilter_manager.register_handler(
            self.handler_name,
            self,
            self.esc_strings
        )

    def handle(self, line_info):
        # print "normal: ", line_info
        """Handle normal input lines. Use as a template for handlers."""

        # With autoindent on, we need some way to exit the input loop, and I
        # don't want to force the user to have to backspace all the way to
        # clear the line.  The rule will be in this case, that either two
        # lines of pure whitespace in a row, or a line of pure whitespace but
        # of a size different to the indent level, will exit the input loop.
        line = line_info.line
        continue_prompt = line_info.continue_prompt

        if (continue_prompt and
            self.shell.autoindent and
            line.isspace() and
            0 < abs(len(line) - self.shell.indent_current_nsp) <= 2):
            line = ''

        return line

    def __str__(self):
        return "<%s(name=%s)>" % (self.__class__.__name__, self.handler_name)


class MacroHandler(PrefilterHandler):
    handler_name = Unicode("macro")

    def handle(self, line_info):
        obj = self.shell.user_ns.get(line_info.ifun)
        pre_space = line_info.pre_whitespace
        line_sep = "\n" + pre_space
        return pre_space + line_sep.join(obj.value.splitlines())


class MagicHandler(PrefilterHandler):

    handler_name = Unicode('magic')
    esc_strings = List([ESC_MAGIC])

    def handle(self, line_info):
        """Execute magic functions."""
        ifun    = line_info.ifun
        the_rest = line_info.the_rest
        #Prepare arguments for get_ipython().run_line_magic(magic_name, magic_args)
        t_arg_s = ifun + " " + the_rest
        t_magic_name, _, t_magic_arg_s = t_arg_s.partition(' ')
        t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
        cmd = '%sget_ipython().run_line_magic(%r, %r)' % (line_info.pre_whitespace, t_magic_name, t_magic_arg_s)
        return cmd


class AutoHandler(PrefilterHandler):

    handler_name = Unicode('auto')
    esc_strings = List([ESC_PAREN, ESC_QUOTE, ESC_QUOTE2])

    def handle(self, line_info):
        """Handle lines which can be auto-executed, quoting if requested."""
        line    = line_info.line
        ifun    = line_info.ifun
        the_rest = line_info.the_rest
        esc     = line_info.esc
        continue_prompt = line_info.continue_prompt
        obj = line_info.ofind(self.shell).obj

        # This should only be active for single-line input!
        if continue_prompt:
            return line

        force_auto = isinstance(obj, IPyAutocall)

        # User objects sometimes raise exceptions on attribute access other
        # than AttributeError (we've seen it in the past), so it's safest to be
        # ultra-conservative here and catch all.
        try:
            auto_rewrite = obj.rewrite
        except Exception:
            auto_rewrite = True

        if esc == ESC_QUOTE:
            # Auto-quote splitting on whitespace
            newcmd = '%s("%s")' % (ifun,'", "'.join(the_rest.split()) )
        elif esc == ESC_QUOTE2:
            # Auto-quote whole string
            newcmd = '%s("%s")' % (ifun,the_rest)
        elif esc == ESC_PAREN:
            newcmd = '%s(%s)' % (ifun,",".join(the_rest.split()))
        else:
            # Auto-paren.
            if force_auto:
                # Don't rewrite if it is already a call.
                do_rewrite = not the_rest.startswith('(')
            else:
                if not the_rest:
                    # We only apply it to argument-less calls if the autocall
                    # parameter is set to 2.
                    do_rewrite = (self.shell.autocall >= 2)
                elif the_rest.startswith('[') and hasattr(obj, '__getitem__'):
                    # Don't autocall in this case: item access for an object
                    # which is BOTH callable and implements __getitem__.
                    do_rewrite = False
                else:
                    do_rewrite = True

            # Figure out the rewritten command
            if do_rewrite:
                if the_rest.endswith(';'):
                    newcmd = '%s(%s);' % (ifun.rstrip(),the_rest[:-1])
                else:
                    newcmd = '%s(%s)' % (ifun.rstrip(), the_rest)
            else:
                normal_handler = self.prefilter_manager.get_handler_by_name('normal')
                return normal_handler.handle(line_info)

        # Display the rewritten call
        if auto_rewrite:
            self.shell.auto_rewrite_input(newcmd)

        return newcmd


class EmacsHandler(PrefilterHandler):

    handler_name = Unicode('emacs')
    esc_strings = List([])

    def handle(self, line_info):
        """Handle input lines marked by python-mode."""

        # Currently, nothing is done.  Later more functionality can be added
        # here if needed.

        # The input cache shouldn't be updated
        return line_info.line


#-----------------------------------------------------------------------------
# Defaults
#-----------------------------------------------------------------------------


_default_checkers = [
    EmacsChecker,
    MacroChecker,
    IPyAutocallChecker,
    AssignmentChecker,
    AutoMagicChecker,
    PythonOpsChecker,
    AutocallChecker
]

_default_handlers = [
    PrefilterHandler,
    MacroHandler,
    MagicHandler,
    AutoHandler,
    EmacsHandler
]
