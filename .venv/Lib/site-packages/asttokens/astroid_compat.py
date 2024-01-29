try:
  from astroid import nodes as astroid_node_classes

  # astroid_node_classes should be whichever module has the NodeNG class
  from astroid.nodes import NodeNG
  from astroid.nodes import BaseContainer
except Exception:
  try:
    from astroid import node_classes as astroid_node_classes
    from astroid.node_classes import NodeNG
    from astroid.node_classes import _BaseContainer as BaseContainer
  except Exception:  # pragma: no cover
    astroid_node_classes = None
    NodeNG = None
    BaseContainer = None


__all__ = ["astroid_node_classes", "NodeNG", "BaseContainer"]
