# Configuration file for ipython-console.

c = get_config()

# ------------------------------------------------------------------------------
# ZMQTerminalIPythonApp configuration
# ------------------------------------------------------------------------------

# ZMQTerminalIPythonApp will inherit config from: TerminalIPythonApp,
# BaseIPythonApplication, Application, InteractiveShellApp, IPythonConsoleApp,
# ConnectionFileMixin

# Should variables loaded at startup (by startup files, exec_lines, etc.) be
# hidden from tools like %who?
# c.ZMQTerminalIPythonApp.hide_initial_ns = True

# set the heartbeat port [default: random]
# c.ZMQTerminalIPythonApp.hb_port = 0

# A list of dotted module names of IPython extensions to load.
# c.ZMQTerminalIPythonApp.extensions = []

# Execute the given command string.
# c.ZMQTerminalIPythonApp.code_to_run = ''

# Path to the ssh key to use for logging in to the ssh server.
# c.ZMQTerminalIPythonApp.sshkey = ''

# The date format used by logging formatters for %(asctime)s
# c.ZMQTerminalIPythonApp.log_datefmt = '%Y-%m-%d %H:%M:%S'

# set the control (ROUTER) port [default: random]
# c.ZMQTerminalIPythonApp.control_port = 0

# Reraise exceptions encountered loading IPython extensions?
# c.ZMQTerminalIPythonApp.reraise_ipython_extension_failures = False

# Set the log level by value or name.
# c.ZMQTerminalIPythonApp.log_level = 30

# Run the file referenced by the PYTHONSTARTUP environment variable at IPython
# startup.
# c.ZMQTerminalIPythonApp.exec_PYTHONSTARTUP = True

# Pre-load matplotlib and numpy for interactive use, selecting a particular
# matplotlib backend and loop integration.
# c.ZMQTerminalIPythonApp.pylab = None

# Run the module as a script.
# c.ZMQTerminalIPythonApp.module_to_run = ''

# Whether to display a banner upon starting IPython.
# c.ZMQTerminalIPythonApp.display_banner = True

# dotted module name of an IPython extension to load.
# c.ZMQTerminalIPythonApp.extra_extension = ''

# Create a massive crash report when IPython encounters what may be an internal
# error.  The default is to append a short message to the usual traceback
# c.ZMQTerminalIPythonApp.verbose_crash = False

# Whether to overwrite existing config files when copying
# c.ZMQTerminalIPythonApp.overwrite = False

# The IPython profile to use.
# c.ZMQTerminalIPythonApp.profile = 'default'

# If a command or file is given via the command-line, e.g. 'ipython foo.py',
# start an interactive shell after executing the file or command.
# c.ZMQTerminalIPythonApp.force_interact = False

# List of files to run at IPython startup.
# c.ZMQTerminalIPythonApp.exec_files = []

# Start IPython quickly by skipping the loading of config files.
# c.ZMQTerminalIPythonApp.quick = False

# The Logging format template
# c.ZMQTerminalIPythonApp.log_format = '[%(name)s]%(highlevel)s %(message)s'

# Whether to install the default config files into the profile dir. If a new
# profile is being created, and IPython contains config files for that profile,
# then they will be staged into the new directory.  Otherwise, default config
# files will be automatically generated.
# c.ZMQTerminalIPythonApp.copy_config_files = False

# set the stdin (ROUTER) port [default: random]
# c.ZMQTerminalIPythonApp.stdin_port = 0

# Path to an extra config file to load.
#
# If specified, load this config file in addition to any other IPython config.
# c.ZMQTerminalIPythonApp.extra_config_file = ''

# lines of code to run at IPython startup.
# c.ZMQTerminalIPythonApp.exec_lines = []

# Enable GUI event loop integration with any of ('glut', 'gtk', 'gtk3', 'osx',
# 'pyglet', 'qt', 'qt5', 'tk', 'wx').
# c.ZMQTerminalIPythonApp.gui = None

# A file to be run
# c.ZMQTerminalIPythonApp.file_to_run = ''

# Configure matplotlib for interactive use with the default matplotlib backend.
# c.ZMQTerminalIPythonApp.matplotlib = None

# Suppress warning messages about legacy config files
# c.ZMQTerminalIPythonApp.ignore_old_config = False

# set the iopub (PUB) port [default: random]
# c.ZMQTerminalIPythonApp.iopub_port = 0

#
# c.ZMQTerminalIPythonApp.transport = 'tcp'

# JSON file in which to store connection info [default: kernel-<pid>.json]
#
# This file will contain the IP, ports, and authentication key needed to connect
# clients to this kernel. By default, this file will be created in the security
# dir of the current profile, but can be specified by absolute path.
# c.ZMQTerminalIPythonApp.connection_file = ''

# The name of the IPython directory. This directory is used for logging
# configuration (through profiles), history storage, etc. The default is usually
# $HOME/.ipython. This option can also be specified through the environment
# variable IPYTHONDIR.
# c.ZMQTerminalIPythonApp.ipython_dir = ''

# The SSH server to use to connect to the kernel.
# c.ZMQTerminalIPythonApp.sshserver = ''

# Set to display confirmation dialog on exit. You can always use 'exit' or
# 'quit', to force a direct exit without any confirmation.
# c.ZMQTerminalIPythonApp.confirm_exit = True

# set the shell (ROUTER) port [default: random]
# c.ZMQTerminalIPythonApp.shell_port = 0

# The name of the default kernel to start.
# c.ZMQTerminalIPythonApp.kernel_name = 'python'

# If true, IPython will populate the user namespace with numpy, pylab, etc. and
# an ``import *`` is done from numpy and pylab, when using pylab mode.
#
# When False, pylab mode should not import any names into the user namespace.
# c.ZMQTerminalIPythonApp.pylab_import_all = True

# Connect to an already running kernel
# c.ZMQTerminalIPythonApp.existing = ''

# Set the kernel's IP address [default localhost]. If the IP address is
# something other than localhost, then Consoles on other machines will be able
# to connect to the Kernel, so be careful!
# c.ZMQTerminalIPythonApp.ip = ''

# ------------------------------------------------------------------------------
# ZMQTerminalInteractiveShell configuration
# ------------------------------------------------------------------------------

# A subclass of TerminalInteractiveShell that uses the 0MQ kernel

# ZMQTerminalInteractiveShell will inherit config from:
# TerminalInteractiveShell, InteractiveShell

#
# c.ZMQTerminalInteractiveShell.history_length = 10000

# auto editing of files with syntax errors.
# c.ZMQTerminalInteractiveShell.autoedit_syntax = False

# If True, anything that would be passed to the pager will be displayed as
# regular output instead.
# c.ZMQTerminalInteractiveShell.display_page = False

#
# c.ZMQTerminalInteractiveShell.debug = False

# 'all', 'last', 'last_expr' or 'none', specifying which nodes should be run
# interactively (displaying output from expressions).
# c.ZMQTerminalInteractiveShell.ast_node_interactivity = 'last_expr'

# Start logging to the default log file in overwrite mode. Use `logappend` to
# specify a log file to **append** logs to.
# c.ZMQTerminalInteractiveShell.logstart = False

# Set the size of the output cache.  The default is 1000, you can change it
# permanently in your config file.  Setting it to 0 completely disables the
# caching system, and the minimum value accepted is 20 (if you provide a value
# less than 20, it is reset to 0 and a warning is issued).  This limit is
# defined because otherwise you'll spend more time re-flushing a too small cache
# than working
# c.ZMQTerminalInteractiveShell.cache_size = 1000

# The shell program to be used for paging.
# c.ZMQTerminalInteractiveShell.pager = 'less'

# The name of the logfile to use.
# c.ZMQTerminalInteractiveShell.logfile = ''

# Save multi-line entries as one entry in readline history
# c.ZMQTerminalInteractiveShell.multiline_history = True

#
# c.ZMQTerminalInteractiveShell.readline_remove_delims = '-/~'

# Enable magic commands to be called without the leading %.
# c.ZMQTerminalInteractiveShell.automagic = True

# Prefix to add to outputs coming from clients other than this one.
#
# Only relevant if include_other_output is True.
# c.ZMQTerminalInteractiveShell.other_output_prefix = '[remote] '

#
# c.ZMQTerminalInteractiveShell.readline_parse_and_bind = ['tab: complete', '"\\C-l": clear-screen', 'set show-all-if-ambiguous on', '"\\C-o": tab-insert', '"\\C-r": reverse-search-history', '"\\C-s": forward-search-history', '"\\C-p": history-search-backward', '"\\C-n": history-search-forward', '"\\e[A": history-search-backward', '"\\e[B": history-search-forward', '"\\C-k": kill-line', '"\\C-u": unix-line-discard']

# Use colors for displaying information about objects. Because this information
# is passed through a pager (like 'less'), and some pagers get confused with
# color codes, this capability can be turned off.
# c.ZMQTerminalInteractiveShell.color_info = True

# Callable object called via 'callable' image handler with one argument, `data`,
# which is `msg["content"]["data"]` where `msg` is the message from iopub
# channel.  For exmaple, you can find base64 encoded PNG data as
# `data['image/png']`.
# c.ZMQTerminalInteractiveShell.callable_image_handler = None

# Command to invoke an image viewer program when you are using 'stream' image
# handler.  This option is a list of string where the first element is the
# command itself and reminders are the options for the command.  Raw image data
# is given as STDIN to the program.
# c.ZMQTerminalInteractiveShell.stream_image_handler = []

#
# c.ZMQTerminalInteractiveShell.separate_out2 = ''

# Autoindent IPython code entered interactively.
# c.ZMQTerminalInteractiveShell.autoindent = True

# The part of the banner to be printed after the profile
# c.ZMQTerminalInteractiveShell.banner2 = ''

# Don't call post-execute functions that have failed in the past.
# c.ZMQTerminalInteractiveShell.disable_failing_post_execute = False

# Deprecated, use PromptManager.out_template
# c.ZMQTerminalInteractiveShell.prompt_out = 'Out[\\#]: '

#
# c.ZMQTerminalInteractiveShell.object_info_string_level = 0

#
# c.ZMQTerminalInteractiveShell.separate_out = ''

# Automatically call the pdb debugger after every exception.
# c.ZMQTerminalInteractiveShell.pdb = False

# Deprecated, use PromptManager.in_template
# c.ZMQTerminalInteractiveShell.prompt_in1 = 'In [\\#]: '

#
# c.ZMQTerminalInteractiveShell.separate_in = '\n'

#
# c.ZMQTerminalInteractiveShell.wildcards_case_sensitive = True

# Enable auto setting the terminal title.
# c.ZMQTerminalInteractiveShell.term_title = False

# Enable deep (recursive) reloading by default. IPython can use the deep_reload
# module which reloads changes in modules recursively (it replaces the reload()
# function, so you don't need to change anything to use it). deep_reload()
# forces a full reload of modules whose code may have changed, which the default
# reload() function does not.  When deep_reload is off, IPython will use the
# normal reload(), but deep_reload will still be available as dreload().
# c.ZMQTerminalInteractiveShell.deep_reload = False

# Deprecated, use PromptManager.in2_template
# c.ZMQTerminalInteractiveShell.prompt_in2 = '   .\\D.: '

# Whether to include output from clients other than this one sharing the same
# kernel.
#
# Outputs are not displayed until enter is pressed.
# c.ZMQTerminalInteractiveShell.include_other_output = False

# Preferred object representation MIME type in order.  First matched MIME type
# will be used.
# c.ZMQTerminalInteractiveShell.mime_preference = ['image/png', 'image/jpeg', 'image/svg+xml']

#
# c.ZMQTerminalInteractiveShell.readline_use = True

# Make IPython automatically call any callable object even if you didn't type
# explicit parentheses. For example, 'str 43' becomes 'str(43)' automatically.
# The value can be '0' to disable the feature, '1' for 'smart' autocall, where
# it is not applied if there are no more arguments on the line, and '2' for
# 'full' autocall, where all callable objects are automatically called (even if
# no arguments are present).
# c.ZMQTerminalInteractiveShell.autocall = 0

# The part of the banner to be printed before the profile
# c.ZMQTerminalInteractiveShell.banner1 = 'Python 3.4.3 |Continuum Analytics, Inc.| (default, Mar  6 2015, 12:07:41) \nType "copyright", "credits" or "license" for more information.\n\nIPython 3.1.0 -- An enhanced Interactive Python.\nAnaconda is brought to you by Continuum Analytics.\nPlease check out: http://continuum.io/thanks and https://binstar.org\n?         -> Introduction and overview of IPython\'s features.\n%quickref -> Quick reference.\nhelp      -> Python\'s own help system.\nobject?   -> Details about \'object\', use \'object??\' for extra details.\n'

# Handler for image type output.  This is useful, for example, when connecting
# to the kernel in which pylab inline backend is activated.  There are four
# handlers defined.  'PIL': Use Python Imaging Library to popup image; 'stream':
# Use an external program to show the image.  Image will be fed into the STDIN
# of the program.  You will need to configure `stream_image_handler`;
# 'tempfile': Use an external program to show the image.  Image will be saved in
# a temporally file and the program is called with the temporally file.  You
# will need to configure `tempfile_image_handler`; 'callable': You can set any
# Python callable which is called with the image data.  You will need to
# configure `callable_image_handler`.
# c.ZMQTerminalInteractiveShell.image_handler = None

# Set the color scheme (NoColor, Linux, or LightBG).
# c.ZMQTerminalInteractiveShell.colors = 'LightBG'

# Set the editor used by IPython (default to $EDITOR/vi/notepad).
# c.ZMQTerminalInteractiveShell.editor = 'mate -w'

# Show rewritten input, e.g. for autocall.
# c.ZMQTerminalInteractiveShell.show_rewritten_input = True

#
# c.ZMQTerminalInteractiveShell.xmode = 'Context'

#
# c.ZMQTerminalInteractiveShell.quiet = False

# A list of ast.NodeTransformer subclass instances, which will be applied to
# user input before code is run.
# c.ZMQTerminalInteractiveShell.ast_transformers = []

#
# c.ZMQTerminalInteractiveShell.ipython_dir = ''

# Set to confirm when you try to exit IPython with an EOF (Control-D in Unix,
# Control-Z/Enter in Windows). By typing 'exit' or 'quit', you can force a
# direct exit without any confirmation.
# c.ZMQTerminalInteractiveShell.confirm_exit = True

# Deprecated, use PromptManager.justify
# c.ZMQTerminalInteractiveShell.prompts_pad_left = True

# Timeout for giving up on a kernel (in seconds).
#
# On first connect and restart, the console tests whether the kernel is running
# and responsive by sending kernel_info_requests. This sets the timeout in
# seconds for how long the kernel can take before being presumed dead.
# c.ZMQTerminalInteractiveShell.kernel_timeout = 60

# Number of lines of your screen, used to control printing of very long strings.
# Strings longer than this number of lines will be sent through a pager instead
# of directly printed.  The default value for this is 0, which means IPython
# will auto-detect your screen size every time it needs to print certain
# potentially long strings (this doesn't change the behavior of the 'print'
# keyword, it's only triggered internally). If for some reason this isn't
# working well (it needs curses support), specify it yourself. Otherwise don't
# change the default.
# c.ZMQTerminalInteractiveShell.screen_length = 0

# Start logging to the given file in append mode. Use `logfile` to specify a log
# file to **overwrite** logs to.
# c.ZMQTerminalInteractiveShell.logappend = ''

# Command to invoke an image viewer program when you are using 'tempfile' image
# handler.  This option is a list of string where the first element is the
# command itself and reminders are the options for the command.  You can use
# {file} and {format} in the string to represent the location of the generated
# image file and image format.
# c.ZMQTerminalInteractiveShell.tempfile_image_handler = []

# ------------------------------------------------------------------------------
# KernelManager configuration
# ------------------------------------------------------------------------------

# Manages a single kernel in a subprocess on this host.
#
# This version starts kernels with Popen.

# KernelManager will inherit config from: ConnectionFileMixin

# set the heartbeat port [default: random]
# c.KernelManager.hb_port = 0

# set the stdin (ROUTER) port [default: random]
# c.KernelManager.stdin_port = 0

#
# c.KernelManager.transport = 'tcp'

# JSON file in which to store connection info [default: kernel-<pid>.json]
#
# This file will contain the IP, ports, and authentication key needed to connect
# clients to this kernel. By default, this file will be created in the security
# dir of the current profile, but can be specified by absolute path.
# c.KernelManager.connection_file = ''

# set the control (ROUTER) port [default: random]
# c.KernelManager.control_port = 0

# set the shell (ROUTER) port [default: random]
# c.KernelManager.shell_port = 0

# Should we autorestart the kernel if it dies.
# c.KernelManager.autorestart = False

# DEPRECATED: Use kernel_name instead.
#
# The Popen Command to launch the kernel. Override this if you have a custom
# kernel. If kernel_cmd is specified in a configuration file, IPython does not
# pass any arguments to the kernel, because it cannot make any assumptions about
# the  arguments that the kernel understands. In particular, this means that the
# kernel does not receive the option --debug if it given on the IPython command
# line.
# c.KernelManager.kernel_cmd = []

# Set the kernel's IP address [default localhost]. If the IP address is
# something other than localhost, then Consoles on other machines will be able
# to connect to the Kernel, so be careful!
# c.KernelManager.ip = ''

# set the iopub (PUB) port [default: random]
# c.KernelManager.iopub_port = 0

# ------------------------------------------------------------------------------
# ProfileDir configuration
# ------------------------------------------------------------------------------

# An object to manage the profile directory and its resources.
#
# The profile directory is used by all IPython applications, to manage
# configuration, logging and security.
#
# This object knows how to find, create and manage these directories. This
# should be used by any code that wants to handle profiles.

# Set the profile location directly. This overrides the logic used by the
# `profile` option.
# c.ProfileDir.location = ''

# ------------------------------------------------------------------------------
# Session configuration
# ------------------------------------------------------------------------------

# Object for handling serialization and sending of messages.
#
# The Session object handles building messages and sending them with ZMQ sockets
# or ZMQStream objects.  Objects can communicate with each other over the
# network via Session objects, and only need to work with the dict-based IPython
# message spec. The Session will handle serialization/deserialization, security,
# and metadata.
#
# Sessions support configurable serialization via packer/unpacker traits, and
# signing with HMAC digests via the key/keyfile traits.
#
# Parameters ----------
#
# debug : bool
#     whether to trigger extra debugging statements
# packer/unpacker : str : 'json', 'pickle' or import_string
#     importstrings for methods to serialize message parts.  If just
#     'json' or 'pickle', predefined JSON and pickle packers will be used.
#     Otherwise, the entire importstring must be used.
#
#     The functions must accept at least valid JSON input, and output *bytes*.
#
#     For example, to use msgpack:
#     packer = 'msgpack.packb', unpacker='msgpack.unpackb'
# pack/unpack : callables
#     You can also set the pack/unpack callables for serialization directly.
# session : bytes
#     the ID of this Session object.  The default is to generate a new UUID.
# username : unicode
#     username added to message headers.  The default is to ask the OS.
# key : bytes
#     The key used to initialize an HMAC signature.  If unset, messages
#     will not be signed or checked.
# keyfile : filepath
#     The file containing a key.  If this is set, `key` will be initialized
#     to the contents of the file.

# The digest scheme used to construct the message signatures. Must have the form
# 'hmac-HASH'.
# c.Session.signature_scheme = 'hmac-sha256'

# The maximum number of digests to remember.
#
# The digest history will be culled when it exceeds this value.
# c.Session.digest_history_size = 65536

# The name of the unpacker for unserializing messages. Only used with custom
# functions for `packer`.
# c.Session.unpacker = 'json'

# The name of the packer for serializing messages. Should be one of 'json',
# 'pickle', or an import name for a custom callable serializer.
# c.Session.packer = 'json'

# Username for the Session. Default is your system username.
# c.Session.username = 'minrk'

# Debug output in the Session
# c.Session.debug = False

# path to file containing execution key.
# c.Session.keyfile = ''

# The maximum number of items for a container to be introspected for custom
# serialization. Containers larger than this are pickled outright.
# c.Session.item_threshold = 64

# Threshold (in bytes) beyond which an object's buffer should be extracted to
# avoid pickling.
# c.Session.buffer_threshold = 1024

# The UUID identifying this session.
# c.Session.session = ''

# Threshold (in bytes) beyond which a buffer should be sent without copying.
# c.Session.copy_threshold = 65536

# execution key, for signing messages.
# c.Session.key = b''

# Metadata dictionary, which serves as the default top-level metadata dict for
# each message.
# c.Session.metadata = {}
