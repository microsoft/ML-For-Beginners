# -*- coding: utf-8 -*-
"""Being removed
"""

class LazyEvaluate(object):
    """This is used for formatting strings with values that need to be updated
    at that time, such as the current time or working directory."""
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        self.kwargs.update(kwargs)
        return self.func(*self.args, **self.kwargs)

    def __str__(self):
        return str(self())
    
    def __format__(self, format_spec):
        return format(self(), format_spec)
