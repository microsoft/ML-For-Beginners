# Natural Language Toolkit: Graphical Representations for Trees
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Graphically display a Tree.
"""

from tkinter import IntVar, Menu, Tk

from nltk.draw.util import (
    BoxWidget,
    CanvasFrame,
    CanvasWidget,
    OvalWidget,
    ParenWidget,
    TextWidget,
)
from nltk.tree import Tree
from nltk.util import in_idle

##//////////////////////////////////////////////////////
##  Tree Segment
##//////////////////////////////////////////////////////


class TreeSegmentWidget(CanvasWidget):
    """
    A canvas widget that displays a single segment of a hierarchical
    tree.  Each ``TreeSegmentWidget`` connects a single "node widget"
    to a sequence of zero or more "subtree widgets".  By default, the
    bottom of the node is connected to the top of each subtree by a
    single line.  However, if the ``roof`` attribute is set, then a
    single triangular "roof" will connect the node to all of its
    children.

    Attributes:
      - ``roof``: What sort of connection to draw between the node and
        its subtrees.  If ``roof`` is true, draw a single triangular
        "roof" over the subtrees.  If ``roof`` is false, draw a line
        between each subtree and the node.  Default value is false.
      - ``xspace``: The amount of horizontal space to leave between
        subtrees when managing this widget.  Default value is 10.
      - ``yspace``: The amount of space to place between the node and
        its children when managing this widget.  Default value is 15.
      - ``color``: The color of the lines connecting the node to its
        subtrees; and of the outline of the triangular roof.  Default
        value is ``'#006060'``.
      - ``fill``: The fill color for the triangular roof.  Default
        value is ``''`` (no fill).
      - ``width``: The width of the lines connecting the node to its
        subtrees; and of the outline of the triangular roof.  Default
        value is 1.
      - ``orientation``: Determines whether the tree branches downwards
        or rightwards.  Possible values are ``'horizontal'`` and
        ``'vertical'``.  The default value is ``'vertical'`` (i.e.,
        branch downwards).
      - ``draggable``: whether the widget can be dragged by the user.
    """

    def __init__(self, canvas, label, subtrees, **attribs):
        """
        :type node:
        :type subtrees: list(CanvasWidgetI)
        """
        self._label = label
        self._subtrees = subtrees

        # Attributes
        self._horizontal = 0
        self._roof = 0
        self._xspace = 10
        self._yspace = 15
        self._ordered = False

        # Create canvas objects.
        self._lines = [canvas.create_line(0, 0, 0, 0, fill="#006060") for c in subtrees]
        self._polygon = canvas.create_polygon(
            0, 0, fill="", state="hidden", outline="#006060"
        )

        # Register child widgets (label + subtrees)
        self._add_child_widget(label)
        for subtree in subtrees:
            self._add_child_widget(subtree)

        # Are we currently managing?
        self._managing = False

        CanvasWidget.__init__(self, canvas, **attribs)

    def __setitem__(self, attr, value):
        canvas = self.canvas()
        if attr == "roof":
            self._roof = value
            if self._roof:
                for l in self._lines:
                    canvas.itemconfig(l, state="hidden")
                canvas.itemconfig(self._polygon, state="normal")
            else:
                for l in self._lines:
                    canvas.itemconfig(l, state="normal")
                canvas.itemconfig(self._polygon, state="hidden")
        elif attr == "orientation":
            if value == "horizontal":
                self._horizontal = 1
            elif value == "vertical":
                self._horizontal = 0
            else:
                raise ValueError("orientation must be horizontal or vertical")
        elif attr == "color":
            for l in self._lines:
                canvas.itemconfig(l, fill=value)
            canvas.itemconfig(self._polygon, outline=value)
        elif isinstance(attr, tuple) and attr[0] == "color":
            # Set the color of an individual line.
            l = self._lines[int(attr[1])]
            canvas.itemconfig(l, fill=value)
        elif attr == "fill":
            canvas.itemconfig(self._polygon, fill=value)
        elif attr == "width":
            canvas.itemconfig(self._polygon, {attr: value})
            for l in self._lines:
                canvas.itemconfig(l, {attr: value})
        elif attr in ("xspace", "yspace"):
            if attr == "xspace":
                self._xspace = value
            elif attr == "yspace":
                self._yspace = value
            self.update(self._label)
        elif attr == "ordered":
            self._ordered = value
        else:
            CanvasWidget.__setitem__(self, attr, value)

    def __getitem__(self, attr):
        if attr == "roof":
            return self._roof
        elif attr == "width":
            return self.canvas().itemcget(self._polygon, attr)
        elif attr == "color":
            return self.canvas().itemcget(self._polygon, "outline")
        elif isinstance(attr, tuple) and attr[0] == "color":
            l = self._lines[int(attr[1])]
            return self.canvas().itemcget(l, "fill")
        elif attr == "xspace":
            return self._xspace
        elif attr == "yspace":
            return self._yspace
        elif attr == "orientation":
            if self._horizontal:
                return "horizontal"
            else:
                return "vertical"
        elif attr == "ordered":
            return self._ordered
        else:
            return CanvasWidget.__getitem__(self, attr)

    def label(self):
        return self._label

    def subtrees(self):
        return self._subtrees[:]

    def set_label(self, label):
        """
        Set the node label to ``label``.
        """
        self._remove_child_widget(self._label)
        self._add_child_widget(label)
        self._label = label
        self.update(self._label)

    def replace_child(self, oldchild, newchild):
        """
        Replace the child ``oldchild`` with ``newchild``.
        """
        index = self._subtrees.index(oldchild)
        self._subtrees[index] = newchild
        self._remove_child_widget(oldchild)
        self._add_child_widget(newchild)
        self.update(newchild)

    def remove_child(self, child):
        index = self._subtrees.index(child)
        del self._subtrees[index]
        self._remove_child_widget(child)
        self.canvas().delete(self._lines.pop())
        self.update(self._label)

    def insert_child(self, index, child):
        canvas = self.canvas()
        self._subtrees.insert(index, child)
        self._add_child_widget(child)
        self._lines.append(canvas.create_line(0, 0, 0, 0, fill="#006060"))
        self.update(self._label)

    # but.. lines???

    def _tags(self):
        if self._roof:
            return [self._polygon]
        else:
            return self._lines

    def _subtree_top(self, child):
        if isinstance(child, TreeSegmentWidget):
            bbox = child.label().bbox()
        else:
            bbox = child.bbox()
        if self._horizontal:
            return (bbox[0], (bbox[1] + bbox[3]) / 2.0)
        else:
            return ((bbox[0] + bbox[2]) / 2.0, bbox[1])

    def _node_bottom(self):
        bbox = self._label.bbox()
        if self._horizontal:
            return (bbox[2], (bbox[1] + bbox[3]) / 2.0)
        else:
            return ((bbox[0] + bbox[2]) / 2.0, bbox[3])

    def _update(self, child):
        if len(self._subtrees) == 0:
            return
        if self._label.bbox() is None:
            return  # [XX] ???

        # Which lines need to be redrawn?
        if child is self._label:
            need_update = self._subtrees
        else:
            need_update = [child]

        if self._ordered and not self._managing:
            need_update = self._maintain_order(child)

        # Update the polygon.
        (nodex, nodey) = self._node_bottom()
        (xmin, ymin, xmax, ymax) = self._subtrees[0].bbox()
        for subtree in self._subtrees[1:]:
            bbox = subtree.bbox()
            xmin = min(xmin, bbox[0])
            ymin = min(ymin, bbox[1])
            xmax = max(xmax, bbox[2])
            ymax = max(ymax, bbox[3])

        if self._horizontal:
            self.canvas().coords(
                self._polygon, nodex, nodey, xmin, ymin, xmin, ymax, nodex, nodey
            )
        else:
            self.canvas().coords(
                self._polygon, nodex, nodey, xmin, ymin, xmax, ymin, nodex, nodey
            )

        # Redraw all lines that need it.
        for subtree in need_update:
            (nodex, nodey) = self._node_bottom()
            line = self._lines[self._subtrees.index(subtree)]
            (subtreex, subtreey) = self._subtree_top(subtree)
            self.canvas().coords(line, nodex, nodey, subtreex, subtreey)

    def _maintain_order(self, child):
        if self._horizontal:
            return self._maintain_order_horizontal(child)
        else:
            return self._maintain_order_vertical(child)

    def _maintain_order_vertical(self, child):
        (left, top, right, bot) = child.bbox()

        if child is self._label:
            # Check all the leaves
            for subtree in self._subtrees:
                (x1, y1, x2, y2) = subtree.bbox()
                if bot + self._yspace > y1:
                    subtree.move(0, bot + self._yspace - y1)

            return self._subtrees
        else:
            moved = [child]
            index = self._subtrees.index(child)

            # Check leaves to our right.
            x = right + self._xspace
            for i in range(index + 1, len(self._subtrees)):
                (x1, y1, x2, y2) = self._subtrees[i].bbox()
                if x > x1:
                    self._subtrees[i].move(x - x1, 0)
                    x += x2 - x1 + self._xspace
                    moved.append(self._subtrees[i])

            # Check leaves to our left.
            x = left - self._xspace
            for i in range(index - 1, -1, -1):
                (x1, y1, x2, y2) = self._subtrees[i].bbox()
                if x < x2:
                    self._subtrees[i].move(x - x2, 0)
                    x -= x2 - x1 + self._xspace
                    moved.append(self._subtrees[i])

            # Check the node
            (x1, y1, x2, y2) = self._label.bbox()
            if y2 > top - self._yspace:
                self._label.move(0, top - self._yspace - y2)
                moved = self._subtrees

        # Return a list of the nodes we moved
        return moved

    def _maintain_order_horizontal(self, child):
        (left, top, right, bot) = child.bbox()

        if child is self._label:
            # Check all the leaves
            for subtree in self._subtrees:
                (x1, y1, x2, y2) = subtree.bbox()
                if right + self._xspace > x1:
                    subtree.move(right + self._xspace - x1)

            return self._subtrees
        else:
            moved = [child]
            index = self._subtrees.index(child)

            # Check leaves below us.
            y = bot + self._yspace
            for i in range(index + 1, len(self._subtrees)):
                (x1, y1, x2, y2) = self._subtrees[i].bbox()
                if y > y1:
                    self._subtrees[i].move(0, y - y1)
                    y += y2 - y1 + self._yspace
                    moved.append(self._subtrees[i])

            # Check leaves above us
            y = top - self._yspace
            for i in range(index - 1, -1, -1):
                (x1, y1, x2, y2) = self._subtrees[i].bbox()
                if y < y2:
                    self._subtrees[i].move(0, y - y2)
                    y -= y2 - y1 + self._yspace
                    moved.append(self._subtrees[i])

            # Check the node
            (x1, y1, x2, y2) = self._label.bbox()
            if x2 > left - self._xspace:
                self._label.move(left - self._xspace - x2, 0)
                moved = self._subtrees

        # Return a list of the nodes we moved
        return moved

    def _manage_horizontal(self):
        (nodex, nodey) = self._node_bottom()

        # Put the subtrees in a line.
        y = 20
        for subtree in self._subtrees:
            subtree_bbox = subtree.bbox()
            dx = nodex - subtree_bbox[0] + self._xspace
            dy = y - subtree_bbox[1]
            subtree.move(dx, dy)
            y += subtree_bbox[3] - subtree_bbox[1] + self._yspace

        # Find the center of their tops.
        center = 0.0
        for subtree in self._subtrees:
            center += self._subtree_top(subtree)[1]
        center /= len(self._subtrees)

        # Center the subtrees with the node.
        for subtree in self._subtrees:
            subtree.move(0, nodey - center)

    def _manage_vertical(self):
        (nodex, nodey) = self._node_bottom()

        # Put the subtrees in a line.
        x = 0
        for subtree in self._subtrees:
            subtree_bbox = subtree.bbox()
            dy = nodey - subtree_bbox[1] + self._yspace
            dx = x - subtree_bbox[0]
            subtree.move(dx, dy)
            x += subtree_bbox[2] - subtree_bbox[0] + self._xspace

        # Find the center of their tops.
        center = 0.0
        for subtree in self._subtrees:
            center += self._subtree_top(subtree)[0] / len(self._subtrees)

        # Center the subtrees with the node.
        for subtree in self._subtrees:
            subtree.move(nodex - center, 0)

    def _manage(self):
        self._managing = True
        (nodex, nodey) = self._node_bottom()
        if len(self._subtrees) == 0:
            return

        if self._horizontal:
            self._manage_horizontal()
        else:
            self._manage_vertical()

        # Update lines to subtrees.
        for subtree in self._subtrees:
            self._update(subtree)

        self._managing = False

    def __repr__(self):
        return f"[TreeSeg {self._label}: {self._subtrees}]"


def _tree_to_treeseg(
    canvas,
    t,
    make_node,
    make_leaf,
    tree_attribs,
    node_attribs,
    leaf_attribs,
    loc_attribs,
):
    if isinstance(t, Tree):
        label = make_node(canvas, t.label(), **node_attribs)
        subtrees = [
            _tree_to_treeseg(
                canvas,
                child,
                make_node,
                make_leaf,
                tree_attribs,
                node_attribs,
                leaf_attribs,
                loc_attribs,
            )
            for child in t
        ]
        return TreeSegmentWidget(canvas, label, subtrees, **tree_attribs)
    else:
        return make_leaf(canvas, t, **leaf_attribs)


def tree_to_treesegment(
    canvas, t, make_node=TextWidget, make_leaf=TextWidget, **attribs
):
    """
    Convert a Tree into a ``TreeSegmentWidget``.

    :param make_node: A ``CanvasWidget`` constructor or a function that
        creates ``CanvasWidgets``.  ``make_node`` is used to convert
        the Tree's nodes into ``CanvasWidgets``.  If no constructor
        is specified, then ``TextWidget`` will be used.
    :param make_leaf: A ``CanvasWidget`` constructor or a function that
        creates ``CanvasWidgets``.  ``make_leaf`` is used to convert
        the Tree's leafs into ``CanvasWidgets``.  If no constructor
        is specified, then ``TextWidget`` will be used.
    :param attribs: Attributes for the canvas widgets that make up the
        returned ``TreeSegmentWidget``.  Any attribute beginning with
        ``'tree_'`` will be passed to all ``TreeSegmentWidgets`` (with
        the ``'tree_'`` prefix removed.  Any attribute beginning with
        ``'node_'`` will be passed to all nodes.  Any attribute
        beginning with ``'leaf_'`` will be passed to all leaves.  And
        any attribute beginning with ``'loc_'`` will be passed to all
        text locations (for Trees).
    """
    # Process attribs.
    tree_attribs = {}
    node_attribs = {}
    leaf_attribs = {}
    loc_attribs = {}

    for (key, value) in list(attribs.items()):
        if key[:5] == "tree_":
            tree_attribs[key[5:]] = value
        elif key[:5] == "node_":
            node_attribs[key[5:]] = value
        elif key[:5] == "leaf_":
            leaf_attribs[key[5:]] = value
        elif key[:4] == "loc_":
            loc_attribs[key[4:]] = value
        else:
            raise ValueError("Bad attribute: %s" % key)
    return _tree_to_treeseg(
        canvas,
        t,
        make_node,
        make_leaf,
        tree_attribs,
        node_attribs,
        leaf_attribs,
        loc_attribs,
    )


##//////////////////////////////////////////////////////
##  Tree Widget
##//////////////////////////////////////////////////////


class TreeWidget(CanvasWidget):
    """
    A canvas widget that displays a single Tree.
    ``TreeWidget`` manages a group of ``TreeSegmentWidgets`` that are
    used to display a Tree.

    Attributes:

      - ``node_attr``: Sets the attribute ``attr`` on all of the
        node widgets for this ``TreeWidget``.
      - ``node_attr``: Sets the attribute ``attr`` on all of the
        leaf widgets for this ``TreeWidget``.
      - ``loc_attr``: Sets the attribute ``attr`` on all of the
        location widgets for this ``TreeWidget`` (if it was built from
        a Tree).  Note that a location widget is a ``TextWidget``.

      - ``xspace``: The amount of horizontal space to leave between
        subtrees when managing this widget.  Default value is 10.
      - ``yspace``: The amount of space to place between the node and
        its children when managing this widget.  Default value is 15.

      - ``line_color``: The color of the lines connecting each expanded
        node to its subtrees.
      - ``roof_color``: The color of the outline of the triangular roof
        for collapsed trees.
      - ``roof_fill``: The fill color for the triangular roof for
        collapsed trees.
      - ``width``

      - ``orientation``: Determines whether the tree branches downwards
        or rightwards.  Possible values are ``'horizontal'`` and
        ``'vertical'``.  The default value is ``'vertical'`` (i.e.,
        branch downwards).

      - ``shapeable``: whether the subtrees can be independently
        dragged by the user.  THIS property simply sets the
        ``DRAGGABLE`` property on all of the ``TreeWidget``'s tree
        segments.
      - ``draggable``: whether the widget can be dragged by the user.
    """

    def __init__(
        self, canvas, t, make_node=TextWidget, make_leaf=TextWidget, **attribs
    ):
        # Node & leaf canvas widget constructors
        self._make_node = make_node
        self._make_leaf = make_leaf
        self._tree = t

        # Attributes.
        self._nodeattribs = {}
        self._leafattribs = {}
        self._locattribs = {"color": "#008000"}
        self._line_color = "#008080"
        self._line_width = 1
        self._roof_color = "#008080"
        self._roof_fill = "#c0c0c0"
        self._shapeable = False
        self._xspace = 10
        self._yspace = 10
        self._orientation = "vertical"
        self._ordered = False

        # Build trees.
        self._keys = {}  # treeseg -> key
        self._expanded_trees = {}
        self._collapsed_trees = {}
        self._nodes = []
        self._leaves = []
        # self._locs = []
        self._make_collapsed_trees(canvas, t, ())
        self._treeseg = self._make_expanded_tree(canvas, t, ())
        self._add_child_widget(self._treeseg)

        CanvasWidget.__init__(self, canvas, **attribs)

    def expanded_tree(self, *path_to_tree):
        """
        Return the ``TreeSegmentWidget`` for the specified subtree.

        :param path_to_tree: A list of indices i1, i2, ..., in, where
            the desired widget is the widget corresponding to
            ``tree.children()[i1].children()[i2]....children()[in]``.
            For the root, the path is ``()``.
        """
        return self._expanded_trees[path_to_tree]

    def collapsed_tree(self, *path_to_tree):
        """
        Return the ``TreeSegmentWidget`` for the specified subtree.

        :param path_to_tree: A list of indices i1, i2, ..., in, where
            the desired widget is the widget corresponding to
            ``tree.children()[i1].children()[i2]....children()[in]``.
            For the root, the path is ``()``.
        """
        return self._collapsed_trees[path_to_tree]

    def bind_click_trees(self, callback, button=1):
        """
        Add a binding to all tree segments.
        """
        for tseg in list(self._expanded_trees.values()):
            tseg.bind_click(callback, button)
        for tseg in list(self._collapsed_trees.values()):
            tseg.bind_click(callback, button)

    def bind_drag_trees(self, callback, button=1):
        """
        Add a binding to all tree segments.
        """
        for tseg in list(self._expanded_trees.values()):
            tseg.bind_drag(callback, button)
        for tseg in list(self._collapsed_trees.values()):
            tseg.bind_drag(callback, button)

    def bind_click_leaves(self, callback, button=1):
        """
        Add a binding to all leaves.
        """
        for leaf in self._leaves:
            leaf.bind_click(callback, button)
        for leaf in self._leaves:
            leaf.bind_click(callback, button)

    def bind_drag_leaves(self, callback, button=1):
        """
        Add a binding to all leaves.
        """
        for leaf in self._leaves:
            leaf.bind_drag(callback, button)
        for leaf in self._leaves:
            leaf.bind_drag(callback, button)

    def bind_click_nodes(self, callback, button=1):
        """
        Add a binding to all nodes.
        """
        for node in self._nodes:
            node.bind_click(callback, button)
        for node in self._nodes:
            node.bind_click(callback, button)

    def bind_drag_nodes(self, callback, button=1):
        """
        Add a binding to all nodes.
        """
        for node in self._nodes:
            node.bind_drag(callback, button)
        for node in self._nodes:
            node.bind_drag(callback, button)

    def _make_collapsed_trees(self, canvas, t, key):
        if not isinstance(t, Tree):
            return
        make_node = self._make_node
        make_leaf = self._make_leaf

        node = make_node(canvas, t.label(), **self._nodeattribs)
        self._nodes.append(node)
        leaves = [make_leaf(canvas, l, **self._leafattribs) for l in t.leaves()]
        self._leaves += leaves
        treeseg = TreeSegmentWidget(
            canvas,
            node,
            leaves,
            roof=1,
            color=self._roof_color,
            fill=self._roof_fill,
            width=self._line_width,
        )

        self._collapsed_trees[key] = treeseg
        self._keys[treeseg] = key
        # self._add_child_widget(treeseg)
        treeseg.hide()

        # Build trees for children.
        for i in range(len(t)):
            child = t[i]
            self._make_collapsed_trees(canvas, child, key + (i,))

    def _make_expanded_tree(self, canvas, t, key):
        make_node = self._make_node
        make_leaf = self._make_leaf

        if isinstance(t, Tree):
            node = make_node(canvas, t.label(), **self._nodeattribs)
            self._nodes.append(node)
            children = t
            subtrees = [
                self._make_expanded_tree(canvas, children[i], key + (i,))
                for i in range(len(children))
            ]
            treeseg = TreeSegmentWidget(
                canvas, node, subtrees, color=self._line_color, width=self._line_width
            )
            self._expanded_trees[key] = treeseg
            self._keys[treeseg] = key
            return treeseg
        else:
            leaf = make_leaf(canvas, t, **self._leafattribs)
            self._leaves.append(leaf)
            return leaf

    def __setitem__(self, attr, value):
        if attr[:5] == "node_":
            for node in self._nodes:
                node[attr[5:]] = value
        elif attr[:5] == "leaf_":
            for leaf in self._leaves:
                leaf[attr[5:]] = value
        elif attr == "line_color":
            self._line_color = value
            for tseg in list(self._expanded_trees.values()):
                tseg["color"] = value
        elif attr == "line_width":
            self._line_width = value
            for tseg in list(self._expanded_trees.values()):
                tseg["width"] = value
            for tseg in list(self._collapsed_trees.values()):
                tseg["width"] = value
        elif attr == "roof_color":
            self._roof_color = value
            for tseg in list(self._collapsed_trees.values()):
                tseg["color"] = value
        elif attr == "roof_fill":
            self._roof_fill = value
            for tseg in list(self._collapsed_trees.values()):
                tseg["fill"] = value
        elif attr == "shapeable":
            self._shapeable = value
            for tseg in list(self._expanded_trees.values()):
                tseg["draggable"] = value
            for tseg in list(self._collapsed_trees.values()):
                tseg["draggable"] = value
            for leaf in self._leaves:
                leaf["draggable"] = value
        elif attr == "xspace":
            self._xspace = value
            for tseg in list(self._expanded_trees.values()):
                tseg["xspace"] = value
            for tseg in list(self._collapsed_trees.values()):
                tseg["xspace"] = value
            self.manage()
        elif attr == "yspace":
            self._yspace = value
            for tseg in list(self._expanded_trees.values()):
                tseg["yspace"] = value
            for tseg in list(self._collapsed_trees.values()):
                tseg["yspace"] = value
            self.manage()
        elif attr == "orientation":
            self._orientation = value
            for tseg in list(self._expanded_trees.values()):
                tseg["orientation"] = value
            for tseg in list(self._collapsed_trees.values()):
                tseg["orientation"] = value
            self.manage()
        elif attr == "ordered":
            self._ordered = value
            for tseg in list(self._expanded_trees.values()):
                tseg["ordered"] = value
            for tseg in list(self._collapsed_trees.values()):
                tseg["ordered"] = value
        else:
            CanvasWidget.__setitem__(self, attr, value)

    def __getitem__(self, attr):
        if attr[:5] == "node_":
            return self._nodeattribs.get(attr[5:], None)
        elif attr[:5] == "leaf_":
            return self._leafattribs.get(attr[5:], None)
        elif attr[:4] == "loc_":
            return self._locattribs.get(attr[4:], None)
        elif attr == "line_color":
            return self._line_color
        elif attr == "line_width":
            return self._line_width
        elif attr == "roof_color":
            return self._roof_color
        elif attr == "roof_fill":
            return self._roof_fill
        elif attr == "shapeable":
            return self._shapeable
        elif attr == "xspace":
            return self._xspace
        elif attr == "yspace":
            return self._yspace
        elif attr == "orientation":
            return self._orientation
        else:
            return CanvasWidget.__getitem__(self, attr)

    def _tags(self):
        return []

    def _manage(self):
        segs = list(self._expanded_trees.values()) + list(
            self._collapsed_trees.values()
        )
        for tseg in segs:
            if tseg.hidden():
                tseg.show()
                tseg.manage()
                tseg.hide()

    def toggle_collapsed(self, treeseg):
        """
        Collapse/expand a tree.
        """
        old_treeseg = treeseg
        if old_treeseg["roof"]:
            new_treeseg = self._expanded_trees[self._keys[old_treeseg]]
        else:
            new_treeseg = self._collapsed_trees[self._keys[old_treeseg]]

        # Replace the old tree with the new tree.
        if old_treeseg.parent() is self:
            self._remove_child_widget(old_treeseg)
            self._add_child_widget(new_treeseg)
            self._treeseg = new_treeseg
        else:
            old_treeseg.parent().replace_child(old_treeseg, new_treeseg)

        # Move the new tree to where the old tree was.  Show it first,
        # so we can find its bounding box.
        new_treeseg.show()
        (newx, newy) = new_treeseg.label().bbox()[:2]
        (oldx, oldy) = old_treeseg.label().bbox()[:2]
        new_treeseg.move(oldx - newx, oldy - newy)

        # Hide the old tree
        old_treeseg.hide()

        # We could do parent.manage() here instead, if we wanted.
        new_treeseg.parent().update(new_treeseg)


##//////////////////////////////////////////////////////
##  draw_trees
##//////////////////////////////////////////////////////


class TreeView:
    def __init__(self, *trees):
        from math import ceil, sqrt

        self._trees = trees

        self._top = Tk()
        self._top.title("NLTK")
        self._top.bind("<Control-x>", self.destroy)
        self._top.bind("<Control-q>", self.destroy)

        cf = self._cframe = CanvasFrame(self._top)
        self._top.bind("<Control-p>", self._cframe.print_to_file)

        # Size is variable.
        self._size = IntVar(self._top)
        self._size.set(12)
        bold = ("helvetica", -self._size.get(), "bold")
        helv = ("helvetica", -self._size.get())

        # Lay the trees out in a square.
        self._width = int(ceil(sqrt(len(trees))))
        self._widgets = []
        for i in range(len(trees)):
            widget = TreeWidget(
                cf.canvas(),
                trees[i],
                node_font=bold,
                leaf_color="#008040",
                node_color="#004080",
                roof_color="#004040",
                roof_fill="white",
                line_color="#004040",
                draggable=1,
                leaf_font=helv,
            )
            widget.bind_click_trees(widget.toggle_collapsed)
            self._widgets.append(widget)
            cf.add_widget(widget, 0, 0)

        self._layout()
        self._cframe.pack(expand=1, fill="both")
        self._init_menubar()

    def _layout(self):
        i = x = y = ymax = 0
        width = self._width
        for i in range(len(self._widgets)):
            widget = self._widgets[i]
            (oldx, oldy) = widget.bbox()[:2]
            if i % width == 0:
                y = ymax
                x = 0
            widget.move(x - oldx, y - oldy)
            x = widget.bbox()[2] + 10
            ymax = max(ymax, widget.bbox()[3] + 10)

    def _init_menubar(self):
        menubar = Menu(self._top)

        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(
            label="Print to Postscript",
            underline=0,
            command=self._cframe.print_to_file,
            accelerator="Ctrl-p",
        )
        filemenu.add_command(
            label="Exit", underline=1, command=self.destroy, accelerator="Ctrl-x"
        )
        menubar.add_cascade(label="File", underline=0, menu=filemenu)

        zoommenu = Menu(menubar, tearoff=0)
        zoommenu.add_radiobutton(
            label="Tiny",
            variable=self._size,
            underline=0,
            value=10,
            command=self.resize,
        )
        zoommenu.add_radiobutton(
            label="Small",
            variable=self._size,
            underline=0,
            value=12,
            command=self.resize,
        )
        zoommenu.add_radiobutton(
            label="Medium",
            variable=self._size,
            underline=0,
            value=14,
            command=self.resize,
        )
        zoommenu.add_radiobutton(
            label="Large",
            variable=self._size,
            underline=0,
            value=28,
            command=self.resize,
        )
        zoommenu.add_radiobutton(
            label="Huge",
            variable=self._size,
            underline=0,
            value=50,
            command=self.resize,
        )
        menubar.add_cascade(label="Zoom", underline=0, menu=zoommenu)

        self._top.config(menu=menubar)

    def resize(self, *e):
        bold = ("helvetica", -self._size.get(), "bold")
        helv = ("helvetica", -self._size.get())
        xspace = self._size.get()
        yspace = self._size.get()
        for widget in self._widgets:
            widget["node_font"] = bold
            widget["leaf_font"] = helv
            widget["xspace"] = xspace
            widget["yspace"] = yspace
            if self._size.get() < 20:
                widget["line_width"] = 1
            elif self._size.get() < 30:
                widget["line_width"] = 2
            else:
                widget["line_width"] = 3
        self._layout()

    def destroy(self, *e):
        if self._top is None:
            return
        self._top.destroy()
        self._top = None

    def mainloop(self, *args, **kwargs):
        """
        Enter the Tkinter mainloop.  This function must be called if
        this demo is created from a non-interactive program (e.g.
        from a secript); otherwise, the demo will close as soon as
        the script completes.
        """
        if in_idle():
            return
        self._top.mainloop(*args, **kwargs)


def draw_trees(*trees):
    """
    Open a new window containing a graphical diagram of the given
    trees.

    :rtype: None
    """
    TreeView(*trees).mainloop()
    return


##//////////////////////////////////////////////////////
##  Demo Code
##//////////////////////////////////////////////////////


def demo():
    import random

    def fill(cw):
        cw["fill"] = "#%06d" % random.randint(0, 999999)

    cf = CanvasFrame(width=550, height=450, closeenough=2)

    t = Tree.fromstring(
        """
    (S (NP the very big cat)
       (VP (Adv sorta) (V saw) (NP (Det the) (N dog))))"""
    )

    tc = TreeWidget(
        cf.canvas(),
        t,
        draggable=1,
        node_font=("helvetica", -14, "bold"),
        leaf_font=("helvetica", -12, "italic"),
        roof_fill="white",
        roof_color="black",
        leaf_color="green4",
        node_color="blue2",
    )
    cf.add_widget(tc, 10, 10)

    def boxit(canvas, text):
        big = ("helvetica", -16, "bold")
        return BoxWidget(canvas, TextWidget(canvas, text, font=big), fill="green")

    def ovalit(canvas, text):
        return OvalWidget(canvas, TextWidget(canvas, text), fill="cyan")

    treetok = Tree.fromstring("(S (NP this tree) (VP (V is) (AdjP shapeable)))")
    tc2 = TreeWidget(cf.canvas(), treetok, boxit, ovalit, shapeable=1)

    def color(node):
        node["color"] = "#%04d00" % random.randint(0, 9999)

    def color2(treeseg):
        treeseg.label()["fill"] = "#%06d" % random.randint(0, 9999)
        treeseg.label().child()["color"] = "white"

    tc.bind_click_trees(tc.toggle_collapsed)
    tc2.bind_click_trees(tc2.toggle_collapsed)
    tc.bind_click_nodes(color, 3)
    tc2.expanded_tree(1).bind_click(color2, 3)
    tc2.expanded_tree().bind_click(color2, 3)

    paren = ParenWidget(cf.canvas(), tc2)
    cf.add_widget(paren, tc.bbox()[2] + 10, 10)

    tree3 = Tree.fromstring(
        """
    (S (NP this tree) (AUX was)
       (VP (V built) (PP (P with) (NP (N tree_to_treesegment)))))"""
    )
    tc3 = tree_to_treesegment(
        cf.canvas(), tree3, tree_color="green4", tree_xspace=2, tree_width=2
    )
    tc3["draggable"] = 1
    cf.add_widget(tc3, 10, tc.bbox()[3] + 10)

    def orientswitch(treewidget):
        if treewidget["orientation"] == "horizontal":
            treewidget.expanded_tree(1, 1).subtrees()[0].set_text("vertical")
            treewidget.collapsed_tree(1, 1).subtrees()[0].set_text("vertical")
            treewidget.collapsed_tree(1).subtrees()[1].set_text("vertical")
            treewidget.collapsed_tree().subtrees()[3].set_text("vertical")
            treewidget["orientation"] = "vertical"
        else:
            treewidget.expanded_tree(1, 1).subtrees()[0].set_text("horizontal")
            treewidget.collapsed_tree(1, 1).subtrees()[0].set_text("horizontal")
            treewidget.collapsed_tree(1).subtrees()[1].set_text("horizontal")
            treewidget.collapsed_tree().subtrees()[3].set_text("horizontal")
            treewidget["orientation"] = "horizontal"

    text = """
Try clicking, right clicking, and dragging
different elements of each of the trees.
The top-left tree is a TreeWidget built from
a Tree.  The top-right is a TreeWidget built
from a Tree, using non-default widget
constructors for the nodes & leaves (BoxWidget
and OvalWidget).  The bottom-left tree is
built from tree_to_treesegment."""
    twidget = TextWidget(cf.canvas(), text.strip())
    textbox = BoxWidget(cf.canvas(), twidget, fill="white", draggable=1)
    cf.add_widget(textbox, tc3.bbox()[2] + 10, tc2.bbox()[3] + 10)

    tree4 = Tree.fromstring("(S (NP this tree) (VP (V is) (Adj horizontal)))")
    tc4 = TreeWidget(
        cf.canvas(),
        tree4,
        draggable=1,
        line_color="brown2",
        roof_color="brown2",
        node_font=("helvetica", -12, "bold"),
        node_color="brown4",
        orientation="horizontal",
    )
    tc4.manage()
    cf.add_widget(tc4, tc3.bbox()[2] + 10, textbox.bbox()[3] + 10)
    tc4.bind_click(orientswitch)
    tc4.bind_click_trees(tc4.toggle_collapsed, 3)

    # Run mainloop
    cf.mainloop()


if __name__ == "__main__":
    demo()
