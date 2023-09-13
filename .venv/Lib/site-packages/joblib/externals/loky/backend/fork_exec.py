###############################################################################
# Launch a subprocess using forkexec and make sure only the needed fd are
# shared in the two process.
#
# author: Thomas Moreau and Olivier Grisel
#
import os
import sys


def close_fds(keep_fds):  # pragma: no cover
    """Close all the file descriptors except those in keep_fds."""

    # Make sure to keep stdout and stderr open for logging purpose
    keep_fds = {*keep_fds, 1, 2}

    # We try to retrieve all the open fds
    try:
        open_fds = {int(fd) for fd in os.listdir("/proc/self/fd")}
    except FileNotFoundError:
        import resource

        max_nfds = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        open_fds = {*range(max_nfds)}

    for i in open_fds - keep_fds:
        try:
            os.close(i)
        except OSError:
            pass


def fork_exec(cmd, keep_fds, env=None):
    # copy the environment variables to set in the child process
    env = env or {}
    child_env = {**os.environ, **env}

    pid = os.fork()
    if pid == 0:  # pragma: no cover
        close_fds(keep_fds)
        os.execve(sys.executable, cmd, child_env)
    else:
        return pid
