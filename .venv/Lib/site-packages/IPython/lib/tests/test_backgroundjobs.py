"""Tests for pylab tools module.
"""
#-----------------------------------------------------------------------------
# Copyright (c) 2011, the IPython Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

# Stdlib imports
import time

# Our own imports
from IPython.lib import backgroundjobs as bg

#-----------------------------------------------------------------------------
# Globals and constants
#-----------------------------------------------------------------------------
t_short = 0.0001  # very short interval to wait on jobs

#-----------------------------------------------------------------------------
# Local utilities
#-----------------------------------------------------------------------------
def sleeper(interval=t_short, *a, **kw):
    args = dict(interval=interval,
                other_args=a,
                kw_args=kw)
    time.sleep(interval)
    return args

def crasher(interval=t_short, *a, **kw):
    time.sleep(interval)
    raise Exception("Dead job with interval %s" % interval)

#-----------------------------------------------------------------------------
# Classes and functions
#-----------------------------------------------------------------------------

def test_result():
    """Test job submission and result retrieval"""
    jobs = bg.BackgroundJobManager()
    j = jobs.new(sleeper)
    j.join()
    assert j.result["interval"] == t_short


def test_flush():
    """Test job control"""
    jobs = bg.BackgroundJobManager()
    j = jobs.new(sleeper)
    j.join()
    assert len(jobs.completed) == 1
    assert len(jobs.dead) == 0
    jobs.flush()
    assert len(jobs.completed) == 0
    

def test_dead():
    """Test control of dead jobs"""
    jobs = bg.BackgroundJobManager()
    j = jobs.new(crasher)
    j.join()
    assert len(jobs.completed) == 0
    assert len(jobs.dead) == 1
    jobs.flush()    
    assert len(jobs.dead) == 0


def test_longer():
    """Test control of longer-running jobs"""
    jobs = bg.BackgroundJobManager()
    # Sleep for long enough for the following two checks to still report the
    # job as running, but not so long that it makes the test suite noticeably
    # slower. 
    j = jobs.new(sleeper, 0.1)
    assert len(jobs.running) == 1
    assert len(jobs.completed) == 0
    j.join()
    assert len(jobs.running) == 0
    assert len(jobs.completed) == 1
