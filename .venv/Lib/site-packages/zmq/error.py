"""0MQ Error classes and functions."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

from errno import EINTR
from typing import Optional, Tuple, Union


class ZMQBaseError(Exception):
    """Base exception class for 0MQ errors in Python."""


class ZMQError(ZMQBaseError):
    """Wrap an errno style error.

    Parameters
    ----------
    errno : int
        The ZMQ errno or None.  If None, then ``zmq_errno()`` is called and
        used.
    msg : string
        Description of the error or None.
    """

    errno: Optional[int] = None

    def __init__(self, errno: Optional[int] = None, msg: Optional[str] = None):
        """Wrap an errno style error.

        Parameters
        ----------
        errno : int
            The ZMQ errno or None.  If None, then ``zmq_errno()`` is called and
            used.
        msg : string
            Description of the error or None.
        """
        from zmq.backend import strerror, zmq_errno

        if errno is None:
            errno = zmq_errno()
        if isinstance(errno, int):
            self.errno = errno
            if msg is None:
                self.strerror = strerror(errno)
            else:
                self.strerror = msg
        else:
            if msg is None:
                self.strerror = str(errno)
            else:
                self.strerror = msg
        # flush signals, because there could be a SIGINT
        # waiting to pounce, resulting in uncaught exceptions.
        # Doing this here means getting SIGINT during a blocking
        # libzmq call will raise a *catchable* KeyboardInterrupt
        # PyErr_CheckSignals()

    def __str__(self) -> str:
        return self.strerror

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{str(self)}')"


class ZMQBindError(ZMQBaseError):
    """An error for ``Socket.bind_to_random_port()``.

    See Also
    --------
    .Socket.bind_to_random_port
    """


class NotDone(ZMQBaseError):
    """Raised when timeout is reached while waiting for 0MQ to finish with a Message

    See Also
    --------
    .MessageTracker.wait : object for tracking when ZeroMQ is done
    """


class ContextTerminated(ZMQError):
    """Wrapper for zmq.ETERM

    .. versionadded:: 13.0
    """

    def __init__(self, errno="ignored", msg="ignored"):
        from zmq import ETERM

        super().__init__(ETERM)


class Again(ZMQError):
    """Wrapper for zmq.EAGAIN

    .. versionadded:: 13.0
    """

    def __init__(self, errno="ignored", msg="ignored"):
        from zmq import EAGAIN

        super().__init__(EAGAIN)


class InterruptedSystemCall(ZMQError, InterruptedError):
    """Wrapper for EINTR

    This exception should be caught internally in pyzmq
    to retry system calls, and not propagate to the user.

    .. versionadded:: 14.7
    """

    errno = EINTR

    def __init__(self, errno="ignored", msg="ignored"):
        super().__init__(EINTR)

    def __str__(self):
        s = super().__str__()
        return s + ": This call should have been retried. Please report this to pyzmq."


def _check_rc(rc, errno=None, error_without_errno=True):
    """internal utility for checking zmq return condition

    and raising the appropriate Exception class
    """
    if rc == -1:
        if errno is None:
            from zmq.backend import zmq_errno

            errno = zmq_errno()
        if errno == 0 and not error_without_errno:
            return
        from zmq import EAGAIN, ETERM

        if errno == EINTR:
            raise InterruptedSystemCall(errno)
        elif errno == EAGAIN:
            raise Again(errno)
        elif errno == ETERM:
            raise ContextTerminated(errno)
        else:
            raise ZMQError(errno)


_zmq_version_info = None
_zmq_version = None


class ZMQVersionError(NotImplementedError):
    """Raised when a feature is not provided by the linked version of libzmq.

    .. versionadded:: 14.2
    """

    min_version = None

    def __init__(self, min_version: str, msg: str = "Feature"):
        global _zmq_version
        if _zmq_version is None:
            from zmq import zmq_version

            _zmq_version = zmq_version()
        self.msg = msg
        self.min_version = min_version
        self.version = _zmq_version

    def __repr__(self):
        return "ZMQVersionError('%s')" % str(self)

    def __str__(self):
        return "{} requires libzmq >= {}, have {}".format(
            self.msg,
            self.min_version,
            self.version,
        )


def _check_version(
    min_version_info: Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]],
    msg: str = "Feature",
):
    """Check for libzmq

    raises ZMQVersionError if current zmq version is not at least min_version

    min_version_info is a tuple of integers, and will be compared against zmq.zmq_version_info().
    """
    global _zmq_version_info
    if _zmq_version_info is None:
        from zmq import zmq_version_info

        _zmq_version_info = zmq_version_info()
    if _zmq_version_info < min_version_info:
        min_version = ".".join(str(v) for v in min_version_info)
        raise ZMQVersionError(min_version, msg)


__all__ = [
    "ZMQBaseError",
    "ZMQBindError",
    "ZMQError",
    "NotDone",
    "ContextTerminated",
    "InterruptedSystemCall",
    "Again",
    "ZMQVersionError",
]
