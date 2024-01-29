"""A PUB log handler."""
import warnings

from zmq.log.handlers import PUBHandler

warnings.warn(
    "ipykernel.log is deprecated. It has moved to ipyparallel.engine.log",
    DeprecationWarning,
    stacklevel=2,
)


class EnginePUBHandler(PUBHandler):
    """A simple PUBHandler subclass that sets root_topic"""

    engine = None

    def __init__(self, engine, *args, **kwargs):
        """Initialize the handler."""
        PUBHandler.__init__(self, *args, **kwargs)
        self.engine = engine

    @property  # type:ignore[misc]
    def root_topic(self):
        """this is a property, in case the handler is created
        before the engine gets registered with an id"""
        if isinstance(getattr(self.engine, "id", None), int):
            return "engine.%i" % self.engine.id  # type:ignore[union-attr]
        return "engine"
