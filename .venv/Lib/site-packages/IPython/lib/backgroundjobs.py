# -*- coding: utf-8 -*-
"""Manage background (threaded) jobs conveniently from an interactive shell.

This module provides a BackgroundJobManager class.  This is the main class
meant for public usage, it implements an object which can create and manage
new background jobs.

It also provides the actual job classes managed by these BackgroundJobManager
objects, see their docstrings below.


This system was inspired by discussions with B. Granger and the
BackgroundCommand class described in the book Python Scripting for
Computational Science, by H. P. Langtangen:

http://folk.uio.no/hpl/scripting

(although ultimately no code from this text was used, as IPython's system is a
separate implementation).

An example notebook is provided in our documentation illustrating interactive
use of the system.
"""

#*****************************************************************************
#       Copyright (C) 2005-2006 Fernando Perez <fperez@colorado.edu>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#*****************************************************************************

# Code begins
import sys
import threading

from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB
from logging import error, debug


class BackgroundJobManager(object):
    """Class to manage a pool of backgrounded threaded jobs.

    Below, we assume that 'jobs' is a BackgroundJobManager instance.
    
    Usage summary (see the method docstrings for details):

      jobs.new(...) -> start a new job
      
      jobs() or jobs.status() -> print status summary of all jobs

      jobs[N] -> returns job number N.

      foo = jobs[N].result -> assign to variable foo the result of job N

      jobs[N].traceback() -> print the traceback of dead job N

      jobs.remove(N) -> remove (finished) job N

      jobs.flush() -> remove all finished jobs
      
    As a convenience feature, BackgroundJobManager instances provide the
    utility result and traceback methods which retrieve the corresponding
    information from the jobs list:

      jobs.result(N) <--> jobs[N].result
      jobs.traceback(N) <--> jobs[N].traceback()

    While this appears minor, it allows you to use tab completion
    interactively on the job manager instance.
    """

    def __init__(self):
        # Lists for job management, accessed via a property to ensure they're
        # up to date.x
        self._running  = []
        self._completed = []
        self._dead = []
        # A dict of all jobs, so users can easily access any of them
        self.all = {}
        # For reporting
        self._comp_report = []
        self._dead_report = []
        # Store status codes locally for fast lookups
        self._s_created   = BackgroundJobBase.stat_created_c
        self._s_running   = BackgroundJobBase.stat_running_c
        self._s_completed = BackgroundJobBase.stat_completed_c
        self._s_dead      = BackgroundJobBase.stat_dead_c
        self._current_job_id = 0

    @property
    def running(self):
        self._update_status()
        return self._running

    @property
    def dead(self):
        self._update_status()
        return self._dead

    @property
    def completed(self):
        self._update_status()
        return self._completed

    def new(self, func_or_exp, *args, **kwargs):
        """Add a new background job and start it in a separate thread.

        There are two types of jobs which can be created:

        1. Jobs based on expressions which can be passed to an eval() call.
        The expression must be given as a string.  For example:

          job_manager.new('myfunc(x,y,z=1)'[,glob[,loc]])

        The given expression is passed to eval(), along with the optional
        global/local dicts provided.  If no dicts are given, they are
        extracted automatically from the caller's frame.

        A Python statement is NOT a valid eval() expression.  Basically, you
        can only use as an eval() argument something which can go on the right
        of an '=' sign and be assigned to a variable.

        For example,"print 'hello'" is not valid, but '2+3' is.

        2. Jobs given a function object, optionally passing additional
        positional arguments:

          job_manager.new(myfunc, x, y)

        The function is called with the given arguments.

        If you need to pass keyword arguments to your function, you must
        supply them as a dict named kw:

          job_manager.new(myfunc, x, y, kw=dict(z=1))

        The reason for this asymmetry is that the new() method needs to
        maintain access to its own keywords, and this prevents name collisions
        between arguments to new() and arguments to your own functions.

        In both cases, the result is stored in the job.result field of the
        background job object.

        You can set `daemon` attribute of the thread by giving the keyword
        argument `daemon`.

        Notes and caveats:

        1. All threads running share the same standard output.  Thus, if your
        background jobs generate output, it will come out on top of whatever
        you are currently writing.  For this reason, background jobs are best
        used with silent functions which simply return their output.

        2. Threads also all work within the same global namespace, and this
        system does not lock interactive variables.  So if you send job to the
        background which operates on a mutable object for a long time, and
        start modifying that same mutable object interactively (or in another
        backgrounded job), all sorts of bizarre behaviour will occur.

        3. If a background job is spending a lot of time inside a C extension
        module which does not release the Python Global Interpreter Lock
        (GIL), this will block the IPython prompt.  This is simply because the
        Python interpreter can only switch between threads at Python
        bytecodes.  While the execution is inside C code, the interpreter must
        simply wait unless the extension module releases the GIL.

        4. There is no way, due to limitations in the Python threads library,
        to kill a thread once it has started."""
        
        if callable(func_or_exp):
            kw  = kwargs.get('kw',{})
            job = BackgroundJobFunc(func_or_exp,*args,**kw)
        elif isinstance(func_or_exp, str):
            if not args:
                frame = sys._getframe(1)
                glob, loc = frame.f_globals, frame.f_locals
            elif len(args)==1:
                glob = loc = args[0]
            elif len(args)==2:
                glob,loc = args
            else:
                raise ValueError(
                      'Expression jobs take at most 2 args (globals,locals)')
            job = BackgroundJobExpr(func_or_exp, glob, loc)
        else:
            raise TypeError('invalid args for new job')

        if kwargs.get('daemon', False):
            job.daemon = True
        job.num = self._current_job_id
        self._current_job_id += 1
        self.running.append(job)
        self.all[job.num] = job
        debug('Starting job # %s in a separate thread.' % job.num)
        job.start()
        return job

    def __getitem__(self, job_key):
        num = job_key if isinstance(job_key, int) else job_key.num
        return self.all[num]

    def __call__(self):
        """An alias to self.status(),

        This allows you to simply call a job manager instance much like the
        Unix `jobs` shell command."""

        return self.status()

    def _update_status(self):
        """Update the status of the job lists.

        This method moves finished jobs to one of two lists:
          - self.completed: jobs which completed successfully
          - self.dead: jobs which finished but died.

        It also copies those jobs to corresponding _report lists.  These lists
        are used to report jobs completed/dead since the last update, and are
        then cleared by the reporting function after each call."""

        # Status codes
        srun, scomp, sdead = self._s_running, self._s_completed, self._s_dead
        # State lists, use the actual lists b/c the public names are properties
        # that call this very function on access
        running, completed, dead = self._running, self._completed, self._dead

        # Now, update all state lists
        for num, job in enumerate(running):
            stat = job.stat_code
            if stat == srun:
                continue
            elif stat == scomp:
                completed.append(job)
                self._comp_report.append(job)
                running[num] = False
            elif stat == sdead:
                dead.append(job)
                self._dead_report.append(job)
                running[num] = False
        # Remove dead/completed jobs from running list
        running[:] = filter(None, running)

    def _group_report(self,group,name):
        """Report summary for a given job group.

        Return True if the group had any elements."""

        if group:
            print('%s jobs:' % name)
            for job in group:
                print('%s : %s' % (job.num,job))
            print()
            return True

    def _group_flush(self,group,name):
        """Flush a given job group

        Return True if the group had any elements."""

        njobs = len(group)
        if njobs:
            plural = {1:''}.setdefault(njobs,'s')
            print('Flushing %s %s job%s.' % (njobs,name,plural))
            group[:] = []
            return True
        
    def _status_new(self):
        """Print the status of newly finished jobs.

        Return True if any new jobs are reported.

        This call resets its own state every time, so it only reports jobs
        which have finished since the last time it was called."""

        self._update_status()
        new_comp = self._group_report(self._comp_report, 'Completed')
        new_dead = self._group_report(self._dead_report,
                                      'Dead, call jobs.traceback() for details')
        self._comp_report[:] = []
        self._dead_report[:] = []
        return new_comp or new_dead
                
    def status(self,verbose=0):
        """Print a status of all jobs currently being managed."""

        self._update_status()
        self._group_report(self.running,'Running')
        self._group_report(self.completed,'Completed')
        self._group_report(self.dead,'Dead')
        # Also flush the report queues
        self._comp_report[:] = []
        self._dead_report[:] = []

    def remove(self,num):
        """Remove a finished (completed or dead) job."""

        try:
            job = self.all[num]
        except KeyError:
            error('Job #%s not found' % num)
        else:
            stat_code = job.stat_code
            if stat_code == self._s_running:
                error('Job #%s is still running, it can not be removed.' % num)
                return
            elif stat_code == self._s_completed:
                self.completed.remove(job)
            elif stat_code == self._s_dead:
                self.dead.remove(job)

    def flush(self):
        """Flush all finished jobs (completed and dead) from lists.

        Running jobs are never flushed.

        It first calls _status_new(), to update info. If any jobs have
        completed since the last _status_new() call, the flush operation
        aborts."""

        # Remove the finished jobs from the master dict
        alljobs = self.all
        for job in self.completed+self.dead:
            del(alljobs[job.num])

        # Now flush these lists completely
        fl_comp = self._group_flush(self.completed, 'Completed')
        fl_dead = self._group_flush(self.dead, 'Dead')
        if not (fl_comp or fl_dead):
            print('No jobs to flush.')

    def result(self,num):
        """result(N) -> return the result of job N."""
        try:
            return self.all[num].result
        except KeyError:
            error('Job #%s not found' % num)

    def _traceback(self, job):
        num = job if isinstance(job, int) else job.num
        try:
            self.all[num].traceback()
        except KeyError:
            error('Job #%s not found' % num)

    def traceback(self, job=None):
        if job is None:
            self._update_status()
            for deadjob in self.dead:
                print("Traceback for: %r" % deadjob)
                self._traceback(deadjob)
                print()
        else:
            self._traceback(job)


class BackgroundJobBase(threading.Thread):
    """Base class to build BackgroundJob classes.

    The derived classes must implement:

    - Their own __init__, since the one here raises NotImplementedError.  The
      derived constructor must call self._init() at the end, to provide common
      initialization.

    - A strform attribute used in calls to __str__.

    - A call() method, which will make the actual execution call and must
      return a value to be held in the 'result' field of the job object.
    """

    # Class constants for status, in string and as numerical codes (when
    # updating jobs lists, we don't want to do string comparisons).  This will
    # be done at every user prompt, so it has to be as fast as possible
    stat_created   = 'Created'; stat_created_c = 0
    stat_running   = 'Running'; stat_running_c = 1
    stat_completed = 'Completed'; stat_completed_c = 2
    stat_dead      = 'Dead (Exception), call jobs.traceback() for details'
    stat_dead_c = -1

    def __init__(self):
        """Must be implemented in subclasses.

        Subclasses must call :meth:`_init` for standard initialisation.
        """
        raise NotImplementedError("This class can not be instantiated directly.")

    def _init(self):
        """Common initialization for all BackgroundJob objects"""
        
        for attr in ['call','strform']:
            assert hasattr(self,attr), "Missing attribute <%s>" % attr
        
        # The num tag can be set by an external job manager
        self.num = None
      
        self.status    = BackgroundJobBase.stat_created
        self.stat_code = BackgroundJobBase.stat_created_c
        self.finished  = False
        self.result    = '<BackgroundJob has not completed>'
        
        # reuse the ipython traceback handler if we can get to it, otherwise
        # make a new one
        try:
            make_tb = get_ipython().InteractiveTB.text
        except:
            make_tb = AutoFormattedTB(mode = 'Context',
                                      color_scheme='NoColor',
                                      tb_offset = 1).text
        # Note that the actual API for text() requires the three args to be
        # passed in, so we wrap it in a simple lambda.
        self._make_tb = lambda : make_tb(None, None, None)

        # Hold a formatted traceback if one is generated.
        self._tb = None
        
        threading.Thread.__init__(self)

    def __str__(self):
        return self.strform

    def __repr__(self):
        return '<BackgroundJob #%d: %s>' % (self.num, self.strform)

    def traceback(self):
        print(self._tb)
        
    def run(self):
        try:
            self.status    = BackgroundJobBase.stat_running
            self.stat_code = BackgroundJobBase.stat_running_c
            self.result    = self.call()
        except:
            self.status    = BackgroundJobBase.stat_dead
            self.stat_code = BackgroundJobBase.stat_dead_c
            self.finished  = None
            self.result    = ('<BackgroundJob died, call jobs.traceback() for details>')
            self._tb       = self._make_tb()
        else:
            self.status    = BackgroundJobBase.stat_completed
            self.stat_code = BackgroundJobBase.stat_completed_c
            self.finished  = True


class BackgroundJobExpr(BackgroundJobBase):
    """Evaluate an expression as a background job (uses a separate thread)."""

    def __init__(self, expression, glob=None, loc=None):
        """Create a new job from a string which can be fed to eval().

        global/locals dicts can be provided, which will be passed to the eval
        call."""

        # fail immediately if the given expression can't be compiled
        self.code = compile(expression,'<BackgroundJob compilation>','eval')
                
        glob = {} if glob is None else glob
        loc = {} if loc is None else loc
        self.expression = self.strform = expression
        self.glob = glob
        self.loc = loc
        self._init()
        
    def call(self):
        return eval(self.code,self.glob,self.loc)


class BackgroundJobFunc(BackgroundJobBase):
    """Run a function call as a background job (uses a separate thread)."""

    def __init__(self, func, *args, **kwargs):
        """Create a new job from a callable object.

        Any positional arguments and keyword args given to this constructor
        after the initial callable are passed directly to it."""

        if not callable(func):
            raise TypeError(
                'first argument to BackgroundJobFunc must be callable')
        
        self.func = func
        self.args = args
        self.kwargs = kwargs
        # The string form will only include the function passed, because
        # generating string representations of the arguments is a potentially
        # _very_ expensive operation (e.g. with large arrays).
        self.strform = str(func)
        self._init()

    def call(self):
        return self.func(*self.args, **self.kwargs)
