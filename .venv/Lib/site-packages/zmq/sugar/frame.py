"""0MQ Frame pure Python methods."""

# Copyright (C) PyZMQ Developers
# Distributed under the terms of the Modified BSD License.

import zmq
from zmq.backend import Frame as FrameBase

from .attrsettr import AttributeSetter


def _draft(v, feature):
    zmq.error._check_version(v, feature)
    if not zmq.DRAFT_API:
        raise RuntimeError(
            "libzmq and pyzmq must be built with draft support for %s" % feature
        )


class Frame(FrameBase, AttributeSetter):
    """Frame(data=None, track=False, copy=None, copy_threshold=zmq.COPY_THRESHOLD)

    A zmq message Frame class for non-copying send/recvs and access to message properties.

    A ``zmq.Frame`` wraps an underlying ``zmq_msg_t``.

    Message *properties* can be accessed by treating a Frame like a dictionary (``frame["User-Id"]``).

    .. versionadded:: 14.4, libzmq 4

        Frames created by ``recv(copy=False)`` can be used to access message properties and attributes,
        such as the CURVE User-Id.

        For example::

            frames = socket.recv_multipart(copy=False)
            user_id = frames[0]["User-Id"]

    This class is used if you want to do non-copying send and recvs.
    When you pass a chunk of bytes to this class, e.g. ``Frame(buf)``, the
    ref-count of `buf` is increased by two: once because the Frame saves `buf` as
    an instance attribute and another because a ZMQ message is created that
    points to the buffer of `buf`. This second ref-count increase makes sure
    that `buf` lives until all messages that use it have been sent.
    Once 0MQ sends all the messages and it doesn't need the buffer of ``buf``,
    0MQ will call ``Py_DECREF(s)``.

    Parameters
    ----------

    data : object, optional
        any object that provides the buffer interface will be used to
        construct the 0MQ message data.
    track : bool [default: False]
        whether a MessageTracker_ should be created to track this object.
        Tracking a message has a cost at creation, because it creates a threadsafe
        Event object.
    copy : bool [default: use copy_threshold]
        Whether to create a copy of the data to pass to libzmq
        or share the memory with libzmq.
        If unspecified, copy_threshold is used.
    copy_threshold: int [default: zmq.COPY_THRESHOLD]
        If copy is unspecified, messages smaller than this many bytes
        will be copied and messages larger than this will be shared with libzmq.
    """

    def __getitem__(self, key):
        # map Frame['User-Id'] to Frame.get('User-Id')
        return self.get(key)

    @property
    def group(self):
        """The RADIO-DISH group of the message.

        Requires libzmq >= 4.2 and pyzmq built with draft APIs enabled.

        .. versionadded:: 17
        """
        _draft((4, 2), "RADIO-DISH")
        return self.get('group')

    @group.setter
    def group(self, group):
        _draft((4, 2), "RADIO-DISH")
        self.set('group', group)

    @property
    def routing_id(self):
        """The CLIENT-SERVER routing id of the message.

        Requires libzmq >= 4.2 and pyzmq built with draft APIs enabled.

        .. versionadded:: 17
        """
        _draft((4, 2), "CLIENT-SERVER")
        return self.get('routing_id')

    @routing_id.setter
    def routing_id(self, routing_id):
        _draft((4, 2), "CLIENT-SERVER")
        self.set('routing_id', routing_id)


# keep deprecated alias
Message = Frame
__all__ = ['Frame', 'Message']
