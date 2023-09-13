"""Decorators for running functions with context/sockets.

.. versionadded:: 15.3

Like using Contexts and Sockets as context managers, but with decorator syntax.
Context and sockets are closed at the end of the function.

For example::

    from zmq.decorators import context, socket
    
    @context()
    @socket(zmq.PUSH)
    def work(ctx, push):
        ...
"""

# Copyright (c) PyZMQ Developers.
# Distributed under the terms of the Modified BSD License.

__all__ = (
    'context',
    'socket',
)

from functools import wraps

import zmq


class _Decorator:
    '''The mini decorator factory'''

    def __init__(self, target=None):
        self._target = target

    def __call__(self, *dec_args, **dec_kwargs):
        """
        The main logic of decorator

        Here is how those arguments works::

            @out_decorator(*dec_args, *dec_kwargs)
            def func(*wrap_args, **wrap_kwargs):
                ...

        And in the ``wrapper``, we simply create ``self.target`` instance via
        ``with``::

            target = self.get_target(*args, **kwargs)
            with target(*dec_args, **dec_kwargs) as obj:
                ...

        """
        kw_name, dec_args, dec_kwargs = self.process_decorator_args(
            *dec_args, **dec_kwargs
        )

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                target = self.get_target(*args, **kwargs)

                with target(*dec_args, **dec_kwargs) as obj:
                    # insert our object into args
                    if kw_name and kw_name not in kwargs:
                        kwargs[kw_name] = obj
                    elif kw_name and kw_name in kwargs:
                        raise TypeError(
                            "{}() got multiple values for"
                            " argument '{}'".format(func.__name__, kw_name)
                        )
                    else:
                        args = args + (obj,)

                    return func(*args, **kwargs)

            return wrapper

        return decorator

    def get_target(self, *args, **kwargs):
        """Return the target function

        Allows modifying args/kwargs to be passed.
        """
        return self._target

    def process_decorator_args(self, *args, **kwargs):
        """Process args passed to the decorator.

        args not consumed by the decorator will be passed to the target factory
        (Context/Socket constructor).
        """
        kw_name = None

        if isinstance(kwargs.get('name'), str):
            kw_name = kwargs.pop('name')
        elif len(args) >= 1 and isinstance(args[0], str):
            kw_name = args[0]
            args = args[1:]

        return kw_name, args, kwargs


class _ContextDecorator(_Decorator):
    """Decorator subclass for Contexts"""

    def __init__(self):
        super().__init__(zmq.Context)


class _SocketDecorator(_Decorator):
    """Decorator subclass for sockets

    Gets the context from other args.
    """

    def process_decorator_args(self, *args, **kwargs):
        """Also grab context_name out of kwargs"""
        kw_name, args, kwargs = super().process_decorator_args(*args, **kwargs)
        self.context_name = kwargs.pop('context_name', 'context')
        return kw_name, args, kwargs

    def get_target(self, *args, **kwargs):
        """Get context, based on call-time args"""
        context = self._get_context(*args, **kwargs)
        return context.socket

    def _get_context(self, *args, **kwargs):
        """
        Find the ``zmq.Context`` from ``args`` and ``kwargs`` at call time.

        First, if there is an keyword argument named ``context`` and it is a
        ``zmq.Context`` instance , we will take it.

        Second, we check all the ``args``, take the first ``zmq.Context``
        instance.

        Finally, we will provide default Context -- ``zmq.Context.instance``

        :return: a ``zmq.Context`` instance
        """
        if self.context_name in kwargs:
            ctx = kwargs[self.context_name]

            if isinstance(ctx, zmq.Context):
                return ctx

        for arg in args:
            if isinstance(arg, zmq.Context):
                return arg
        # not specified by any decorator
        return zmq.Context.instance()


def context(*args, **kwargs):
    """Decorator for adding a Context to a function.

    Usage::

        @context()
        def foo(ctx):
            ...

    .. versionadded:: 15.3

    :param str name: the keyword argument passed to decorated function
    """
    return _ContextDecorator()(*args, **kwargs)


def socket(*args, **kwargs):
    """Decorator for adding a socket to a function.

    Usage::

        @socket(zmq.PUSH)
        def foo(push):
            ...

    .. versionadded:: 15.3

    :param str name: the keyword argument passed to decorated function
    :param str context_name: the keyword only argument to identify context
                             object
    """
    return _SocketDecorator()(*args, **kwargs)
