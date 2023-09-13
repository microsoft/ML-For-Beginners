"""Grab the global logger instance."""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import logging

_logger = None


def get_logger():
    """Grab the global logger instance.

    If a global Application is instantiated, grab its logger.
    Otherwise, grab the root logger.
    """
    global _logger

    if _logger is None:
        from .config import Application

        if Application.initialized():
            _logger = Application.instance().log
        else:
            _logger = logging.getLogger("traitlets")
            # Add a NullHandler to silence warnings about not being
            # initialized, per best practice for libraries.
            _logger.addHandler(logging.NullHandler())
    return _logger
