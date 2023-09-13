"""Implementation of magic functions for IPython's own logging.
"""
#-----------------------------------------------------------------------------
#  Copyright (c) 2012 The IPython Development Team.
#
#  Distributed under the terms of the Modified BSD License.
#
#  The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib
import os
import sys

# Our own packages
from IPython.core.magic import Magics, magics_class, line_magic
from warnings import warn
from traitlets import Bool

#-----------------------------------------------------------------------------
# Magic implementation classes
#-----------------------------------------------------------------------------

@magics_class
class LoggingMagics(Magics):
    """Magics related to all logging machinery."""

    quiet = Bool(False, help=
        """
        Suppress output of log state when logging is enabled
        """
    ).tag(config=True)

    @line_magic
    def logstart(self, parameter_s=''):
        """Start logging anywhere in a session.

        %logstart [-o|-r|-t|-q] [log_name [log_mode]]

        If no name is given, it defaults to a file named 'ipython_log.py' in your
        current directory, in 'rotate' mode (see below).

        '%logstart name' saves to file 'name' in 'backup' mode.  It saves your
        history up to that point and then continues logging.

        %logstart takes a second optional parameter: logging mode. This can be one
        of (note that the modes are given unquoted):

        append
            Keep logging at the end of any existing file.

        backup
            Rename any existing file to name~ and start name.

        global
            Append to  a single logfile in your home directory.

        over
            Overwrite any existing log.

        rotate
            Create rotating logs: name.1~, name.2~, etc.

        Options:

          -o
            log also IPython's output. In this mode, all commands which
            generate an Out[NN] prompt are recorded to the logfile, right after
            their corresponding input line. The output lines are always
            prepended with a '#[Out]# ' marker, so that the log remains valid
            Python code.

          Since this marker is always the same, filtering only the output from
          a log is very easy, using for example a simple awk call::

            awk -F'#\\[Out\\]# ' '{if($2) {print $2}}' ipython_log.py

          -r
            log 'raw' input.  Normally, IPython's logs contain the processed
            input, so that user lines are logged in their final form, converted
            into valid Python.  For example, %Exit is logged as
            _ip.magic("Exit").  If the -r flag is given, all input is logged
            exactly as typed, with no transformations applied.

          -t
            put timestamps before each input line logged (these are put in
            comments).

          -q 
            suppress output of logstate message when logging is invoked
        """

        opts,par = self.parse_options(parameter_s,'ortq')
        log_output = 'o' in opts
        log_raw_input = 'r' in opts
        timestamp = 't' in opts
        quiet = 'q' in opts

        logger = self.shell.logger

        # if no args are given, the defaults set in the logger constructor by
        # ipython remain valid
        if par:
            try:
                logfname,logmode = par.split()
            except:
                logfname = par
                logmode = 'backup'
        else:
            logfname = logger.logfname
            logmode = logger.logmode
        # put logfname into rc struct as if it had been called on the command
        # line, so it ends up saved in the log header Save it in case we need
        # to restore it...
        old_logfile = self.shell.logfile
        if logfname:
            logfname = os.path.expanduser(logfname)
        self.shell.logfile = logfname

        loghead = u'# IPython log file\n\n'
        try:
            logger.logstart(logfname, loghead, logmode, log_output, timestamp,
                            log_raw_input)
        except:
            self.shell.logfile = old_logfile
            warn("Couldn't start log: %s" % sys.exc_info()[1])
        else:
            # log input history up to this point, optionally interleaving
            # output if requested

            if timestamp:
                # disable timestamping for the previous history, since we've
                # lost those already (no time machine here).
                logger.timestamp = False

            if log_raw_input:
                input_hist = self.shell.history_manager.input_hist_raw
            else:
                input_hist = self.shell.history_manager.input_hist_parsed

            if log_output:
                log_write = logger.log_write
                output_hist = self.shell.history_manager.output_hist
                for n in range(1,len(input_hist)-1):
                    log_write(input_hist[n].rstrip() + u'\n')
                    if n in output_hist:
                        log_write(repr(output_hist[n]),'output')
            else:
                logger.log_write(u'\n'.join(input_hist[1:]))
                logger.log_write(u'\n')
            if timestamp:
                # re-enable timestamping
                logger.timestamp = True

            if not (self.quiet or quiet):
                print ('Activating auto-logging. '
                       'Current session state plus future input saved.')
                logger.logstate()

    @line_magic
    def logstop(self, parameter_s=''):
        """Fully stop logging and close log file.

        In order to start logging again, a new %logstart call needs to be made,
        possibly (though not necessarily) with a new filename, mode and other
        options."""
        self.shell.logger.logstop()

    @line_magic
    def logoff(self, parameter_s=''):
        """Temporarily stop logging.

        You must have previously started logging."""
        self.shell.logger.switch_log(0)

    @line_magic
    def logon(self, parameter_s=''):
        """Restart logging.

        This function is for restarting logging which you've temporarily
        stopped with %logoff. For starting logging for the first time, you
        must use the %logstart function, which allows you to specify an
        optional log filename."""

        self.shell.logger.switch_log(1)

    @line_magic
    def logstate(self, parameter_s=''):
        """Print the status of the logging system."""

        self.shell.logger.logstate()
