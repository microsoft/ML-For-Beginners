# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root
# for license information.

import functools
import threading


class Singleton(object):
    """A base class for a class of a singleton object.

    For any derived class T, the first invocation of T() will create the instance,
    and any future invocations of T() will return that instance.

    Concurrent invocations of T() from different threads are safe.
    """

    # A dual-lock scheme is necessary to be thread safe while avoiding deadlocks.
    # _lock_lock is shared by all singleton types, and is used to construct their
    # respective _lock instances when invoked for a new type. Then _lock is used
    # to synchronize all further access for that type, including __init__. This way,
    # __init__ for any given singleton can access another singleton, and not get
    # deadlocked if that other singleton is trying to access it.
    _lock_lock = threading.RLock()
    _lock = None

    # Specific subclasses will get their own _instance set in __new__.
    _instance = None

    _is_shared = None  # True if shared, False if exclusive

    def __new__(cls, *args, **kwargs):
        # Allow arbitrary args and kwargs if shared=False, because that is guaranteed
        # to construct a new singleton if it succeeds. Otherwise, this call might end
        # up returning an existing instance, which might have been constructed with
        # different arguments, so allowing them is misleading.
        assert not kwargs.get("shared", False) or (len(args) + len(kwargs)) == 0, (
            "Cannot use constructor arguments when accessing a Singleton without "
            "specifying shared=False."
        )

        # Avoid locking as much as possible with repeated double-checks - the most
        # common path is when everything is already allocated.
        if not cls._instance:
            # If there's no per-type lock, allocate it.
            if cls._lock is None:
                with cls._lock_lock:
                    if cls._lock is None:
                        cls._lock = threading.RLock()

            # Now that we have a per-type lock, we can synchronize construction.
            if not cls._instance:
                with cls._lock:
                    if not cls._instance:
                        cls._instance = object.__new__(cls)
                        # To prevent having __init__ invoked multiple times, call
                        # it here directly, and then replace it with a stub that
                        # does nothing - that stub will get auto-invoked on return,
                        # and on all future singleton accesses.
                        cls._instance.__init__()
                        cls.__init__ = lambda *args, **kwargs: None

        return cls._instance

    def __init__(self, *args, **kwargs):
        """Initializes the singleton instance. Guaranteed to only be invoked once for
        any given type derived from Singleton.

        If shared=False, the caller is requesting a singleton instance for their own
        exclusive use. This is only allowed if the singleton has not been created yet;
        if so, it is created and marked as being in exclusive use. While it is marked
        as such, all attempts to obtain an existing instance of it immediately raise
        an exception. The singleton can eventually be promoted to shared use by calling
        share() on it.
        """

        shared = kwargs.pop("shared", True)
        with self:
            if shared:
                assert (
                    type(self)._is_shared is not False
                ), "Cannot access a non-shared Singleton."
                type(self)._is_shared = True
            else:
                assert type(self)._is_shared is None, "Singleton is already created."

    def __enter__(self):
        """Lock this singleton to prevent concurrent access."""
        type(self)._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Unlock this singleton to allow concurrent access."""
        type(self)._lock.release()

    def share(self):
        """Share this singleton, if it was originally created with shared=False."""
        type(self)._is_shared = True


class ThreadSafeSingleton(Singleton):
    """A singleton that incorporates a lock for thread-safe access to its members.

    The lock can be acquired using the context manager protocol, and thus idiomatic
    use is in conjunction with a with-statement. For example, given derived class T::

        with T() as t:
            t.x = t.frob(t.y)

    All access to the singleton from the outside should follow this pattern for both
    attributes and method calls. Singleton members can assume that self is locked by
    the caller while they're executing, but recursive locking of the same singleton
    on the same thread is also permitted.
    """

    threadsafe_attrs = frozenset()
    """Names of attributes that are guaranteed to be used in a thread-safe manner.

    This is typically used in conjunction with share() to simplify synchronization.
    """

    readonly_attrs = frozenset()
    """Names of attributes that are readonly. These can be read without locking, but
    cannot be written at all.

    Every derived class gets its own separate set. Thus, for any given singleton type
    T, an attribute can be made readonly after setting it, with T.readonly_attrs.add().
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make sure each derived class gets a separate copy.
        type(self).readonly_attrs = set(type(self).readonly_attrs)

    # Prevent callers from reading or writing attributes without locking, except for
    # reading attributes listed in threadsafe_attrs, and methods specifically marked
    # with @threadsafe_method. Such methods should perform the necessary locking to
    # ensure thread safety for the callers.

    @staticmethod
    def assert_locked(self):
        lock = type(self)._lock
        assert lock.acquire(blocking=False), (
            "ThreadSafeSingleton accessed without locking. Either use with-statement, "
            "or if it is a method or property, mark it as @threadsafe_method or with "
            "@autolocked_method, as appropriate."
        )
        lock.release()

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name not in (type(self).threadsafe_attrs | type(self).readonly_attrs):
            if not getattr(value, "is_threadsafe_method", False):
                ThreadSafeSingleton.assert_locked(self)
        return value

    def __setattr__(self, name, value):
        assert name not in type(self).readonly_attrs, "This attribute is read-only."
        if name not in type(self).threadsafe_attrs:
            ThreadSafeSingleton.assert_locked(self)
        return object.__setattr__(self, name, value)


def threadsafe_method(func):
    """Marks a method of a ThreadSafeSingleton-derived class as inherently thread-safe.

    A method so marked must either not use any singleton state, or lock it appropriately.
    """

    func.is_threadsafe_method = True
    return func


def autolocked_method(func):
    """Automatically synchronizes all calls of a method of a ThreadSafeSingleton-derived
    class by locking the singleton for the duration of each call.
    """

    @functools.wraps(func)
    @threadsafe_method
    def lock_and_call(self, *args, **kwargs):
        with self:
            return func(self, *args, **kwargs)

    return lock_and_call
