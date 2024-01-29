# encoding: utf-8
"""
Utilities for timing code execution.
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import time

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

# If possible (Unix), use the resource module instead of time.clock()
try:
    import resource
except ModuleNotFoundError:
    resource = None  # type: ignore [assignment]

# Some implementations (like jyputerlite) don't have getrusage
if resource is not None and hasattr(resource, "getrusage"):
    def clocku():
        """clocku() -> floating point number

        Return the *USER* CPU time in seconds since the start of the process.
        This is done via a call to resource.getrusage, so it avoids the
        wraparound problems in time.clock()."""

        return resource.getrusage(resource.RUSAGE_SELF)[0]

    def clocks():
        """clocks() -> floating point number

        Return the *SYSTEM* CPU time in seconds since the start of the process.
        This is done via a call to resource.getrusage, so it avoids the
        wraparound problems in time.clock()."""

        return resource.getrusage(resource.RUSAGE_SELF)[1]

    def clock():
        """clock() -> floating point number

        Return the *TOTAL USER+SYSTEM* CPU time in seconds since the start of
        the process.  This is done via a call to resource.getrusage, so it
        avoids the wraparound problems in time.clock()."""

        u,s = resource.getrusage(resource.RUSAGE_SELF)[:2]
        return u+s

    def clock2():
        """clock2() -> (t_user,t_system)

        Similar to clock(), but return a tuple of user/system times."""
        return resource.getrusage(resource.RUSAGE_SELF)[:2]

else:
    # There is no distinction of user/system time under windows, so we just use
    # time.process_time() for everything...
    clocku = clocks = clock = time.process_time

    def clock2():
        """Under windows, system CPU time can't be measured.

        This just returns process_time() and zero."""
        return time.process_time(), 0.0

    
def timings_out(reps,func,*args,**kw):
    """timings_out(reps,func,*args,**kw) -> (t_total,t_per_call,output)

    Execute a function reps times, return a tuple with the elapsed total
    CPU time in seconds, the time per call and the function's output.

    Under Unix, the return value is the sum of user+system time consumed by
    the process, computed via the resource module.  This prevents problems
    related to the wraparound effect which the time.clock() function has.

    Under Windows the return value is in wall clock seconds. See the
    documentation for the time module for more details."""

    reps = int(reps)
    assert reps >=1, 'reps must be >= 1'
    if reps==1:
        start = clock()
        out = func(*args,**kw)
        tot_time = clock()-start
    else:
        rng = range(reps-1) # the last time is executed separately to store output
        start = clock()
        for dummy in rng: func(*args,**kw)
        out = func(*args,**kw)  # one last time
        tot_time = clock()-start
    av_time = tot_time / reps
    return tot_time,av_time,out


def timings(reps,func,*args,**kw):
    """timings(reps,func,*args,**kw) -> (t_total,t_per_call)

    Execute a function reps times, return a tuple with the elapsed total CPU
    time in seconds and the time per call. These are just the first two values
    in timings_out()."""

    return timings_out(reps,func,*args,**kw)[0:2]


def timing(func,*args,**kw):
    """timing(func,*args,**kw) -> t_total

    Execute a function once, return the elapsed total CPU time in
    seconds. This is just the first value in timings_out()."""

    return timings_out(1,func,*args,**kw)[0]

