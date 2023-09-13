try:
  from astroid import nodes as astroid_node_classes

  # astroid_node_classes should be whichever module has the NodeNG class
  from astroid.nodes import NodeNG
except Exception:
  try:
    from astroid import node_classes as astroid_node_classes
    from astroid.node_classes import NodeNG
  except Exception:  # pragma: no cover
    astroid_node_classes = None
    NodeNG = None

__all__ = ["astroid_node_classes", "NodeNG"]
