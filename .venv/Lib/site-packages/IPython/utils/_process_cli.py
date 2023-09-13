"""cli-specific implementation of process utilities.

cli - Common Language Infrastructure for IronPython. Code
      can run on any operating system. Check os.name for os-
      specific settings.

This file is only meant to be imported by process.py, not by end-users.

This file is largely untested. To become a full drop-in process
interface for IronPython will probably require you to help fill
in the details. 
"""

# Import cli libraries:
import clr
import System

# Import Python libraries:
import os

# Import IPython libraries:
from ._process_common import arg_split


def system(cmd):
    """
    system(cmd) should work in a cli environment on Mac OSX, Linux,
    and Windows
    """
    psi = System.Diagnostics.ProcessStartInfo(cmd)
    psi.RedirectStandardOutput = True
    psi.RedirectStandardError = True
    psi.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal
    psi.UseShellExecute = False
    # Start up process:
    reg = System.Diagnostics.Process.Start(psi)

def getoutput(cmd):
    """
    getoutput(cmd) should work in a cli environment on Mac OSX, Linux,
    and Windows
    """
    psi = System.Diagnostics.ProcessStartInfo(cmd)
    psi.RedirectStandardOutput = True
    psi.RedirectStandardError = True
    psi.WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal
    psi.UseShellExecute = False
    # Start up process:
    reg = System.Diagnostics.Process.Start(psi)
    myOutput = reg.StandardOutput
    output = myOutput.ReadToEnd()
    myError = reg.StandardError
    error = myError.ReadToEnd()
    return output

def check_pid(pid):
    """
    Check if a process with the given PID (pid) exists
    """
    try:
        System.Diagnostics.Process.GetProcessById(pid)
        # process with given pid is running
        return True
    except System.InvalidOperationException:
        # process wasn't started by this object (but is running)
        return True
    except System.ArgumentException:
        # process with given pid isn't running
        return False 
