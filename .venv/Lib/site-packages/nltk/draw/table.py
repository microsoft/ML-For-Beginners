# Natural Language Toolkit: Table widget
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT

"""
Tkinter widgets for displaying multi-column listboxes and tables.
"""

import operator
from tkinter import Frame, Label, Listbox, Scrollbar, Tk

######################################################################
# Multi-Column Listbox
######################################################################


class MultiListbox(Frame):
    """
    A multi-column listbox, where the current selection applies to an
    entire row.  Based on the MultiListbox Tkinter widget
    recipe from the Python Cookbook (https://code.activestate.com/recipes/52266/)

    For the most part, ``MultiListbox`` methods delegate to its
    contained listboxes.  For any methods that do not have docstrings,
    see ``Tkinter.Listbox`` for a description of what that method does.
    """

    # /////////////////////////////////////////////////////////////////
    # Configuration
    # /////////////////////////////////////////////////////////////////

    #: Default configuration values for the frame.
    FRAME_CONFIG = dict(background="#888", takefocus=True, highlightthickness=1)

    #: Default configurations for the column labels.
    LABEL_CONFIG = dict(
        borderwidth=1,
        relief="raised",
        font="helvetica -16 bold",
        background="#444",
        foreground="white",
    )

    #: Default configuration for the column listboxes.
    LISTBOX_CONFIG = dict(
        borderwidth=1,
        selectborderwidth=0,
        highlightthickness=0,
        exportselection=False,
        selectbackground="#888",
        activestyle="none",
        takefocus=False,
    )

    # /////////////////////////////////////////////////////////////////
    # Constructor
    # /////////////////////////////////////////////////////////////////

    def __init__(self, master, columns, column_weights=None, cnf={}, **kw):
        """
        Construct a new multi-column listbox widget.

        :param master: The widget that should contain the new
            multi-column listbox.

        :param columns: Specifies what columns should be included in
            the new multi-column listbox.  If ``columns`` is an integer,
            then it is the number of columns to include.  If it is
            a list, then its length indicates the number of columns
            to include; and each element of the list will be used as
            a label for the corresponding column.

        :param cnf, kw: Configuration parameters for this widget.
            Use ``label_*`` to configure all labels; and ``listbox_*``
            to configure all listboxes.  E.g.:
                >>> root = Tk()  # doctest: +SKIP
                >>> MultiListbox(root, ["Subject", "Sender", "Date"], label_foreground='red').pack()  # doctest: +SKIP
        """
        # If columns was specified as an int, convert it to a list.
        if isinstance(columns, int):
            columns = list(range(columns))
            include_labels = False
        else:
            include_labels = True

        if len(columns) == 0:
            raise ValueError("Expected at least one column")

        # Instance variables
        self._column_names = tuple(columns)
        self._listboxes = []
        self._labels = []

        # Pick a default value for column_weights, if none was specified.
        if column_weights is None:
            column_weights = [1] * len(columns)
        elif len(column_weights) != len(columns):
            raise ValueError("Expected one column_weight for each column")
        self._column_weights = column_weights

        # Configure our widgets.
        Frame.__init__(self, master, **self.FRAME_CONFIG)
        self.grid_rowconfigure(1, weight=1)
        for i, label in enumerate(self._column_names):
            self.grid_columnconfigure(i, weight=column_weights[i])

            # Create a label for the column
            if include_labels:
                l = Label(self, text=label, **self.LABEL_CONFIG)
                self._labels.append(l)
                l.grid(column=i, row=0, sticky="news", padx=0, pady=0)
                l.column_index = i

            # Create a listbox for the column
            lb = Listbox(self, **self.LISTBOX_CONFIG)
            self._listboxes.append(lb)
            lb.grid(column=i, row=1, sticky="news", padx=0, pady=0)
            lb.column_index = i

            # Clicking or dragging selects:
            lb.bind("<Button-1>", self._select)
            lb.bind("<B1-Motion>", self._select)
            # Scroll wheel scrolls:
            lb.bind("<Button-4>", lambda e: self._scroll(-1))
            lb.bind("<Button-5>", lambda e: self._scroll(+1))
            lb.bind("<MouseWheel>", lambda e: self._scroll(e.delta))
            # Button 2 can be used to scan:
            lb.bind("<Button-2>", lambda e: self.scan_mark(e.x, e.y))
            lb.bind("<B2-Motion>", lambda e: self.scan_dragto(e.x, e.y))
            # Dragging outside the window has no effect (disable
            # the default listbox behavior, which scrolls):
            lb.bind("<B1-Leave>", lambda e: "break")
            # Columns can be resized by dragging them:
            lb.bind("<Button-1>", self._resize_column)

        # Columns can be resized by dragging them.  (This binding is
        # used if they click on the grid between columns:)
        self.bind("<Button-1>", self._resize_column)

        # Set up key bindings for the widget:
        self.bind("<Up>", lambda e: self.select(delta=-1))
        self.bind("<Down>", lambda e: self.select(delta=1))
        self.bind("<Prior>", lambda e: self.select(delta=-self._pagesize()))
        self.bind("<Next>", lambda e: self.select(delta=self._pagesize()))

        # Configuration customizations
        self.configure(cnf, **kw)

    # /////////////////////////////////////////////////////////////////
    # Column Resizing
    # /////////////////////////////////////////////////////////////////

    def _resize_column(self, event):
        """
        Callback used to resize a column of the table.  Return ``True``
        if the column is actually getting resized (if the user clicked
        on the far left or far right 5 pixels of a label); and
        ``False`` otherwies.
        """
        # If we're already waiting for a button release, then ignore
        # the new button press.
        if event.widget.bind("<ButtonRelease>"):
            return False

        # Decide which column (if any) to resize.
        self._resize_column_index = None
        if event.widget is self:
            for i, lb in enumerate(self._listboxes):
                if abs(event.x - (lb.winfo_x() + lb.winfo_width())) < 10:
                    self._resize_column_index = i
        elif event.x > (event.widget.winfo_width() - 5):
            self._resize_column_index = event.widget.column_index
        elif event.x < 5 and event.widget.column_index != 0:
            self._resize_column_index = event.widget.column_index - 1

        # Bind callbacks that are used to resize it.
        if self._resize_column_index is not None:
            event.widget.bind("<Motion>", self._resize_column_motion_cb)
            event.widget.bind(
                "<ButtonRelease-%d>" % event.num, self._resize_column_buttonrelease_cb
            )
            return True
        else:
            return False

    def _resize_column_motion_cb(self, event):
        lb = self._listboxes[self._resize_column_index]
        charwidth = lb.winfo_width() / lb["width"]

        x1 = event.x + event.widget.winfo_x()
        x2 = lb.winfo_x() + lb.winfo_width()

        lb["width"] = max(3, lb["width"] + (x1 - x2) // charwidth)

    def _resize_column_buttonrelease_cb(self, event):
        event.widget.unbind("<ButtonRelease-%d>" % event.num)
        event.widget.unbind("<Motion>")

    # /////////////////////////////////////////////////////////////////
    # Properties
    # /////////////////////////////////////////////////////////////////

    @property
    def column_names(self):
        """
        A tuple containing the names of the columns used by this
        multi-column listbox.
        """
        return self._column_names

    @property
    def column_labels(self):
        """
        A tuple containing the ``Tkinter.Label`` widgets used to
        display the label of each column.  If this multi-column
        listbox was created without labels, then this will be an empty
        tuple.  These widgets will all be augmented with a
        ``column_index`` attribute, which can be used to determine
        which column they correspond to.  This can be convenient,
        e.g., when defining callbacks for bound events.
        """
        return tuple(self._labels)

    @property
    def listboxes(self):
        """
        A tuple containing the ``Tkinter.Listbox`` widgets used to
        display individual columns.  These widgets will all be
        augmented with a ``column_index`` attribute, which can be used
        to determine which column they correspond to.  This can be
        convenient, e.g., when defining callbacks for bound events.
        """
        return tuple(self._listboxes)

    # /////////////////////////////////////////////////////////////////
    # Mouse & Keyboard Callback Functions
    # /////////////////////////////////////////////////////////////////

    def _select(self, e):
        i = e.widget.nearest(e.y)
        self.selection_clear(0, "end")
        self.selection_set(i)
        self.activate(i)
        self.focus()

    def _scroll(self, delta):
        for lb in self._listboxes:
            lb.yview_scroll(delta, "unit")
        return "break"

    def _pagesize(self):
        """:return: The number of rows that makes up one page"""
        return int(self.index("@0,1000000")) - int(self.index("@0,0"))

    # /////////////////////////////////////////////////////////////////
    # Row selection
    # /////////////////////////////////////////////////////////////////

    def select(self, index=None, delta=None, see=True):
        """
        Set the selected row.  If ``index`` is specified, then select
        row ``index``.  Otherwise, if ``delta`` is specified, then move
        the current selection by ``delta`` (negative numbers for up,
        positive numbers for down).  This will not move the selection
        past the top or the bottom of the list.

        :param see: If true, then call ``self.see()`` with the newly
            selected index, to ensure that it is visible.
        """
        if (index is not None) and (delta is not None):
            raise ValueError("specify index or delta, but not both")

        # If delta was given, then calculate index.
        if delta is not None:
            if len(self.curselection()) == 0:
                index = -1 + delta
            else:
                index = int(self.curselection()[0]) + delta

        # Clear all selected rows.
        self.selection_clear(0, "end")

        # Select the specified index
        if index is not None:
            index = min(max(index, 0), self.size() - 1)
            # self.activate(index)
            self.selection_set(index)
            if see:
                self.see(index)

    # /////////////////////////////////////////////////////////////////
    # Configuration
    # /////////////////////////////////////////////////////////////////

    def configure(self, cnf={}, **kw):
        """
        Configure this widget.  Use ``label_*`` to configure all
        labels; and ``listbox_*`` to configure all listboxes.  E.g.:

                >>> master = Tk()  # doctest: +SKIP
                >>> mlb = MultiListbox(master, 5)  # doctest: +SKIP
                >>> mlb.configure(label_foreground='red')  # doctest: +SKIP
                >>> mlb.configure(listbox_foreground='red')  # doctest: +SKIP
        """
        cnf = dict(list(cnf.items()) + list(kw.items()))
        for (key, val) in list(cnf.items()):
            if key.startswith("label_") or key.startswith("label-"):
                for label in self._labels:
                    label.configure({key[6:]: val})
            elif key.startswith("listbox_") or key.startswith("listbox-"):
                for listbox in self._listboxes:
                    listbox.configure({key[8:]: val})
            else:
                Frame.configure(self, {key: val})

    def __setitem__(self, key, val):
        """
        Configure this widget.  This is equivalent to
        ``self.configure({key,val``)}.  See ``configure()``.
        """
        self.configure({key: val})

    def rowconfigure(self, row_index, cnf={}, **kw):
        """
        Configure all table cells in the given row.  Valid keyword
        arguments are: ``background``, ``bg``, ``foreground``, ``fg``,
        ``selectbackground``, ``selectforeground``.
        """
        for lb in self._listboxes:
            lb.itemconfigure(row_index, cnf, **kw)

    def columnconfigure(self, col_index, cnf={}, **kw):
        """
        Configure all table cells in the given column.  Valid keyword
        arguments are: ``background``, ``bg``, ``foreground``, ``fg``,
        ``selectbackground``, ``selectforeground``.
        """
        lb = self._listboxes[col_index]

        cnf = dict(list(cnf.items()) + list(kw.items()))
        for (key, val) in list(cnf.items()):
            if key in (
                "background",
                "bg",
                "foreground",
                "fg",
                "selectbackground",
                "selectforeground",
            ):
                for i in range(lb.size()):
                    lb.itemconfigure(i, {key: val})
            else:
                lb.configure({key: val})

    def itemconfigure(self, row_index, col_index, cnf=None, **kw):
        """
        Configure the table cell at the given row and column.  Valid
        keyword arguments are: ``background``, ``bg``, ``foreground``,
        ``fg``, ``selectbackground``, ``selectforeground``.
        """
        lb = self._listboxes[col_index]
        return lb.itemconfigure(row_index, cnf, **kw)

    # /////////////////////////////////////////////////////////////////
    # Value Access
    # /////////////////////////////////////////////////////////////////

    def insert(self, index, *rows):
        """
        Insert the given row or rows into the table, at the given
        index.  Each row value should be a tuple of cell values, one
        for each column in the row.  Index may be an integer or any of
        the special strings (such as ``'end'``) accepted by
        ``Tkinter.Listbox``.
        """
        for elt in rows:
            if len(elt) != len(self._column_names):
                raise ValueError(
                    "rows should be tuples whose length "
                    "is equal to the number of columns"
                )
        for (lb, elts) in zip(self._listboxes, list(zip(*rows))):
            lb.insert(index, *elts)

    def get(self, first, last=None):
        """
        Return the value(s) of the specified row(s).  If ``last`` is
        not specified, then return a single row value; otherwise,
        return a list of row values.  Each row value is a tuple of
        cell values, one for each column in the row.
        """
        values = [lb.get(first, last) for lb in self._listboxes]
        if last:
            return [tuple(row) for row in zip(*values)]
        else:
            return tuple(values)

    def bbox(self, row, col):
        """
        Return the bounding box for the given table cell, relative to
        this widget's top-left corner.  The bounding box is a tuple
        of integers ``(left, top, width, height)``.
        """
        dx, dy, _, _ = self.grid_bbox(row=0, column=col)
        x, y, w, h = self._listboxes[col].bbox(row)
        return int(x) + int(dx), int(y) + int(dy), int(w), int(h)

    # /////////////////////////////////////////////////////////////////
    # Hide/Show Columns
    # /////////////////////////////////////////////////////////////////

    def hide_column(self, col_index):
        """
        Hide the given column.  The column's state is still
        maintained: its values will still be returned by ``get()``, and
        you must supply its values when calling ``insert()``.  It is
        safe to call this on a column that is already hidden.

        :see: ``show_column()``
        """
        if self._labels:
            self._labels[col_index].grid_forget()
        self.listboxes[col_index].grid_forget()
        self.grid_columnconfigure(col_index, weight=0)

    def show_column(self, col_index):
        """
        Display a column that has been hidden using ``hide_column()``.
        It is safe to call this on a column that is not hidden.
        """
        weight = self._column_weights[col_index]
        if self._labels:
            self._labels[col_index].grid(
                column=col_index, row=0, sticky="news", padx=0, pady=0
            )
        self._listboxes[col_index].grid(
            column=col_index, row=1, sticky="news", padx=0, pady=0
        )
        self.grid_columnconfigure(col_index, weight=weight)

    # /////////////////////////////////////////////////////////////////
    # Binding Methods
    # /////////////////////////////////////////////////////////////////

    def bind_to_labels(self, sequence=None, func=None, add=None):
        """
        Add a binding to each ``Tkinter.Label`` widget in this
        mult-column listbox that will call ``func`` in response to the
        event sequence.

        :return: A list of the identifiers of replaced binding
            functions (if any), allowing for their deletion (to
            prevent a memory leak).
        """
        return [label.bind(sequence, func, add) for label in self.column_labels]

    def bind_to_listboxes(self, sequence=None, func=None, add=None):
        """
        Add a binding to each ``Tkinter.Listbox`` widget in this
        mult-column listbox that will call ``func`` in response to the
        event sequence.

        :return: A list of the identifiers of replaced binding
            functions (if any), allowing for their deletion (to
            prevent a memory leak).
        """
        for listbox in self.listboxes:
            listbox.bind(sequence, func, add)

    def bind_to_columns(self, sequence=None, func=None, add=None):
        """
        Add a binding to each ``Tkinter.Label`` and ``Tkinter.Listbox``
        widget in this mult-column listbox that will call ``func`` in
        response to the event sequence.

        :return: A list of the identifiers of replaced binding
            functions (if any), allowing for their deletion (to
            prevent a memory leak).
        """
        return self.bind_to_labels(sequence, func, add) + self.bind_to_listboxes(
            sequence, func, add
        )

    # /////////////////////////////////////////////////////////////////
    # Simple Delegation
    # /////////////////////////////////////////////////////////////////

    # These methods delegate to the first listbox:
    def curselection(self, *args, **kwargs):
        return self._listboxes[0].curselection(*args, **kwargs)

    def selection_includes(self, *args, **kwargs):
        return self._listboxes[0].selection_includes(*args, **kwargs)

    def itemcget(self, *args, **kwargs):
        return self._listboxes[0].itemcget(*args, **kwargs)

    def size(self, *args, **kwargs):
        return self._listboxes[0].size(*args, **kwargs)

    def index(self, *args, **kwargs):
        return self._listboxes[0].index(*args, **kwargs)

    def nearest(self, *args, **kwargs):
        return self._listboxes[0].nearest(*args, **kwargs)

    # These methods delegate to each listbox (and return None):
    def activate(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.activate(*args, **kwargs)

    def delete(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.delete(*args, **kwargs)

    def scan_mark(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.scan_mark(*args, **kwargs)

    def scan_dragto(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.scan_dragto(*args, **kwargs)

    def see(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.see(*args, **kwargs)

    def selection_anchor(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.selection_anchor(*args, **kwargs)

    def selection_clear(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.selection_clear(*args, **kwargs)

    def selection_set(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.selection_set(*args, **kwargs)

    def yview(self, *args, **kwargs):
        for lb in self._listboxes:
            v = lb.yview(*args, **kwargs)
        return v  # if called with no arguments

    def yview_moveto(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.yview_moveto(*args, **kwargs)

    def yview_scroll(self, *args, **kwargs):
        for lb in self._listboxes:
            lb.yview_scroll(*args, **kwargs)

    # /////////////////////////////////////////////////////////////////
    # Aliases
    # /////////////////////////////////////////////////////////////////

    itemconfig = itemconfigure
    rowconfig = rowconfigure
    columnconfig = columnconfigure
    select_anchor = selection_anchor
    select_clear = selection_clear
    select_includes = selection_includes
    select_set = selection_set

    # /////////////////////////////////////////////////////////////////
    # These listbox methods are not defined for multi-listbox
    # /////////////////////////////////////////////////////////////////
    # def xview(self, *what): pass
    # def xview_moveto(self, fraction): pass
    # def xview_scroll(self, number, what): pass


######################################################################
# Table
######################################################################


class Table:
    """
    A display widget for a table of values, based on a ``MultiListbox``
    widget.  For many purposes, ``Table`` can be treated as a
    list-of-lists.  E.g., table[i] is a list of the values for row i;
    and table.append(row) adds a new row with the given list of
    values.  Individual cells can be accessed using table[i,j], which
    refers to the j-th column of the i-th row.  This can be used to
    both read and write values from the table.  E.g.:

        >>> table[i,j] = 'hello'  # doctest: +SKIP

    The column (j) can be given either as an index number, or as a
    column name.  E.g., the following prints the value in the 3rd row
    for the 'First Name' column:

        >>> print(table[3, 'First Name'])  # doctest: +SKIP
        John

    You can configure the colors for individual rows, columns, or
    cells using ``rowconfig()``, ``columnconfig()``, and ``itemconfig()``.
    The color configuration for each row will be preserved if the
    table is modified; however, when new rows are added, any color
    configurations that have been made for *columns* will not be
    applied to the new row.

    Note: Although ``Table`` acts like a widget in some ways (e.g., it
    defines ``grid()``, ``pack()``, and ``bind()``), it is not itself a
    widget; it just contains one.  This is because widgets need to
    define ``__getitem__()``, ``__setitem__()``, and ``__nonzero__()`` in
    a way that's incompatible with the fact that ``Table`` behaves as a
    list-of-lists.

    :ivar _mlb: The multi-column listbox used to display this table's data.
    :ivar _rows: A list-of-lists used to hold the cell values of this
        table.  Each element of _rows is a row value, i.e., a list of
        cell values, one for each column in the row.
    """

    def __init__(
        self,
        master,
        column_names,
        rows=None,
        column_weights=None,
        scrollbar=True,
        click_to_sort=True,
        reprfunc=None,
        cnf={},
        **kw
    ):
        """
        Construct a new Table widget.

        :type master: Tkinter.Widget
        :param master: The widget that should contain the new table.
        :type column_names: list(str)
        :param column_names: A list of names for the columns; these
            names will be used to create labels for each column;
            and can be used as an index when reading or writing
            cell values from the table.
        :type rows: list(list)
        :param rows: A list of row values used to initialize the table.
            Each row value should be a tuple of cell values, one for
            each column in the row.
        :type scrollbar: bool
        :param scrollbar: If true, then create a scrollbar for the
            new table widget.
        :type click_to_sort: bool
        :param click_to_sort: If true, then create bindings that will
            sort the table's rows by a given column's values if the
            user clicks on that colum's label.
        :type reprfunc: function
        :param reprfunc: If specified, then use this function to
            convert each table cell value to a string suitable for
            display.  ``reprfunc`` has the following signature:
            reprfunc(row_index, col_index, cell_value) -> str
            (Note that the column is specified by index, not by name.)
        :param cnf, kw: Configuration parameters for this widget's
            contained ``MultiListbox``.  See ``MultiListbox.__init__()``
            for details.
        """
        self._num_columns = len(column_names)
        self._reprfunc = reprfunc
        self._frame = Frame(master)

        self._column_name_to_index = {c: i for (i, c) in enumerate(column_names)}

        # Make a copy of the rows & check that it's valid.
        if rows is None:
            self._rows = []
        else:
            self._rows = [[v for v in row] for row in rows]
        for row in self._rows:
            self._checkrow(row)

        # Create our multi-list box.
        self._mlb = MultiListbox(self._frame, column_names, column_weights, cnf, **kw)
        self._mlb.pack(side="left", expand=True, fill="both")

        # Optional scrollbar
        if scrollbar:
            sb = Scrollbar(self._frame, orient="vertical", command=self._mlb.yview)
            self._mlb.listboxes[0]["yscrollcommand"] = sb.set
            # for listbox in self._mlb.listboxes:
            #    listbox['yscrollcommand'] = sb.set
            sb.pack(side="right", fill="y")
            self._scrollbar = sb

        # Set up sorting
        self._sortkey = None
        if click_to_sort:
            for i, l in enumerate(self._mlb.column_labels):
                l.bind("<Button-1>", self._sort)

        # Fill in our multi-list box.
        self._fill_table()

    # /////////////////////////////////////////////////////////////////
    # { Widget-like Methods
    # /////////////////////////////////////////////////////////////////
    # These all just delegate to either our frame or our MLB.

    def pack(self, *args, **kwargs):
        """Position this table's main frame widget in its parent
        widget.  See ``Tkinter.Frame.pack()`` for more info."""
        self._frame.pack(*args, **kwargs)

    def grid(self, *args, **kwargs):
        """Position this table's main frame widget in its parent
        widget.  See ``Tkinter.Frame.grid()`` for more info."""
        self._frame.grid(*args, **kwargs)

    def focus(self):
        """Direct (keyboard) input foxus to this widget."""
        self._mlb.focus()

    def bind(self, sequence=None, func=None, add=None):
        """Add a binding to this table's main frame that will call
        ``func`` in response to the event sequence."""
        self._mlb.bind(sequence, func, add)

    def rowconfigure(self, row_index, cnf={}, **kw):
        """:see: ``MultiListbox.rowconfigure()``"""
        self._mlb.rowconfigure(row_index, cnf, **kw)

    def columnconfigure(self, col_index, cnf={}, **kw):
        """:see: ``MultiListbox.columnconfigure()``"""
        col_index = self.column_index(col_index)
        self._mlb.columnconfigure(col_index, cnf, **kw)

    def itemconfigure(self, row_index, col_index, cnf=None, **kw):
        """:see: ``MultiListbox.itemconfigure()``"""
        col_index = self.column_index(col_index)
        return self._mlb.itemconfigure(row_index, col_index, cnf, **kw)

    def bind_to_labels(self, sequence=None, func=None, add=None):
        """:see: ``MultiListbox.bind_to_labels()``"""
        return self._mlb.bind_to_labels(sequence, func, add)

    def bind_to_listboxes(self, sequence=None, func=None, add=None):
        """:see: ``MultiListbox.bind_to_listboxes()``"""
        return self._mlb.bind_to_listboxes(sequence, func, add)

    def bind_to_columns(self, sequence=None, func=None, add=None):
        """:see: ``MultiListbox.bind_to_columns()``"""
        return self._mlb.bind_to_columns(sequence, func, add)

    rowconfig = rowconfigure
    columnconfig = columnconfigure
    itemconfig = itemconfigure

    # /////////////////////////////////////////////////////////////////
    # { Table as list-of-lists
    # /////////////////////////////////////////////////////////////////

    def insert(self, row_index, rowvalue):
        """
        Insert a new row into the table, so that its row index will be
        ``row_index``.  If the table contains any rows whose row index
        is greater than or equal to ``row_index``, then they will be
        shifted down.

        :param rowvalue: A tuple of cell values, one for each column
            in the new row.
        """
        self._checkrow(rowvalue)
        self._rows.insert(row_index, rowvalue)
        if self._reprfunc is not None:
            rowvalue = [
                self._reprfunc(row_index, j, v) for (j, v) in enumerate(rowvalue)
            ]
        self._mlb.insert(row_index, rowvalue)
        if self._DEBUG:
            self._check_table_vs_mlb()

    def extend(self, rowvalues):
        """
        Add new rows at the end of the table.

        :param rowvalues: A list of row values used to initialize the
            table.  Each row value should be a tuple of cell values,
            one for each column in the row.
        """
        for rowvalue in rowvalues:
            self.append(rowvalue)
        if self._DEBUG:
            self._check_table_vs_mlb()

    def append(self, rowvalue):
        """
        Add a new row to the end of the table.

        :param rowvalue: A tuple of cell values, one for each column
            in the new row.
        """
        self.insert(len(self._rows), rowvalue)
        if self._DEBUG:
            self._check_table_vs_mlb()

    def clear(self):
        """
        Delete all rows in this table.
        """
        self._rows = []
        self._mlb.delete(0, "end")
        if self._DEBUG:
            self._check_table_vs_mlb()

    def __getitem__(self, index):
        """
        Return the value of a row or a cell in this table.  If
        ``index`` is an integer, then the row value for the ``index``th
        row.  This row value consists of a tuple of cell values, one
        for each column in the row.  If ``index`` is a tuple of two
        integers, ``(i,j)``, then return the value of the cell in the
        ``i``th row and the ``j``th column.
        """
        if isinstance(index, slice):
            raise ValueError("Slicing not supported")
        elif isinstance(index, tuple) and len(index) == 2:
            return self._rows[index[0]][self.column_index(index[1])]
        else:
            return tuple(self._rows[index])

    def __setitem__(self, index, val):
        """
        Replace the value of a row or a cell in this table with
        ``val``.

        If ``index`` is an integer, then ``val`` should be a row value
        (i.e., a tuple of cell values, one for each column).  In this
        case, the values of the ``index``th row of the table will be
        replaced with the values in ``val``.

        If ``index`` is a tuple of integers, ``(i,j)``, then replace the
        value of the cell in the ``i``th row and ``j``th column with
        ``val``.
        """
        if isinstance(index, slice):
            raise ValueError("Slicing not supported")

        # table[i,j] = val
        elif isinstance(index, tuple) and len(index) == 2:
            i, j = index[0], self.column_index(index[1])
            config_cookie = self._save_config_info([i])
            self._rows[i][j] = val
            if self._reprfunc is not None:
                val = self._reprfunc(i, j, val)
            self._mlb.listboxes[j].insert(i, val)
            self._mlb.listboxes[j].delete(i + 1)
            self._restore_config_info(config_cookie)

        # table[i] = val
        else:
            config_cookie = self._save_config_info([index])
            self._checkrow(val)
            self._rows[index] = list(val)
            if self._reprfunc is not None:
                val = [self._reprfunc(index, j, v) for (j, v) in enumerate(val)]
            self._mlb.insert(index, val)
            self._mlb.delete(index + 1)
            self._restore_config_info(config_cookie)

    def __delitem__(self, row_index):
        """
        Delete the ``row_index``th row from this table.
        """
        if isinstance(row_index, slice):
            raise ValueError("Slicing not supported")
        if isinstance(row_index, tuple) and len(row_index) == 2:
            raise ValueError("Cannot delete a single cell!")
        del self._rows[row_index]
        self._mlb.delete(row_index)
        if self._DEBUG:
            self._check_table_vs_mlb()

    def __len__(self):
        """
        :return: the number of rows in this table.
        """
        return len(self._rows)

    def _checkrow(self, rowvalue):
        """
        Helper function: check that a given row value has the correct
        number of elements; and if not, raise an exception.
        """
        if len(rowvalue) != self._num_columns:
            raise ValueError(
                "Row %r has %d columns; expected %d"
                % (rowvalue, len(rowvalue), self._num_columns)
            )

    # /////////////////////////////////////////////////////////////////
    # Columns
    # /////////////////////////////////////////////////////////////////

    @property
    def column_names(self):
        """A list of the names of the columns in this table."""
        return self._mlb.column_names

    def column_index(self, i):
        """
        If ``i`` is a valid column index integer, then return it as is.
        Otherwise, check if ``i`` is used as the name for any column;
        if so, return that column's index.  Otherwise, raise a
        ``KeyError`` exception.
        """
        if isinstance(i, int) and 0 <= i < self._num_columns:
            return i
        else:
            # This raises a key error if the column is not found.
            return self._column_name_to_index[i]

    def hide_column(self, column_index):
        """:see: ``MultiListbox.hide_column()``"""
        self._mlb.hide_column(self.column_index(column_index))

    def show_column(self, column_index):
        """:see: ``MultiListbox.show_column()``"""
        self._mlb.show_column(self.column_index(column_index))

    # /////////////////////////////////////////////////////////////////
    # Selection
    # /////////////////////////////////////////////////////////////////

    def selected_row(self):
        """
        Return the index of the currently selected row, or None if
        no row is selected.  To get the row value itself, use
        ``table[table.selected_row()]``.
        """
        sel = self._mlb.curselection()
        if sel:
            return int(sel[0])
        else:
            return None

    def select(self, index=None, delta=None, see=True):
        """:see: ``MultiListbox.select()``"""
        self._mlb.select(index, delta, see)

    # /////////////////////////////////////////////////////////////////
    # Sorting
    # /////////////////////////////////////////////////////////////////

    def sort_by(self, column_index, order="toggle"):
        """
        Sort the rows in this table, using the specified column's
        values as a sort key.

        :param column_index: Specifies which column to sort, using
            either a column index (int) or a column's label name
            (str).

        :param order: Specifies whether to sort the values in
            ascending or descending order:

              - ``'ascending'``: Sort from least to greatest.
              - ``'descending'``: Sort from greatest to least.
              - ``'toggle'``: If the most recent call to ``sort_by()``
                sorted the table by the same column (``column_index``),
                then reverse the rows; otherwise sort in ascending
                order.
        """
        if order not in ("ascending", "descending", "toggle"):
            raise ValueError(
                'sort_by(): order should be "ascending", ' '"descending", or "toggle".'
            )
        column_index = self.column_index(column_index)
        config_cookie = self._save_config_info(index_by_id=True)

        # Sort the rows.
        if order == "toggle" and column_index == self._sortkey:
            self._rows.reverse()
        else:
            self._rows.sort(
                key=operator.itemgetter(column_index), reverse=(order == "descending")
            )
            self._sortkey = column_index

        # Redraw the table.
        self._fill_table()
        self._restore_config_info(config_cookie, index_by_id=True, see=True)
        if self._DEBUG:
            self._check_table_vs_mlb()

    def _sort(self, event):
        """Event handler for clicking on a column label -- sort by
        that column."""
        column_index = event.widget.column_index

        # If they click on the far-left of far-right of a column's
        # label, then resize rather than sorting.
        if self._mlb._resize_column(event):
            return "continue"

        # Otherwise, sort.
        else:
            self.sort_by(column_index)
            return "continue"

    # /////////////////////////////////////////////////////////////////
    # { Table Drawing Helpers
    # /////////////////////////////////////////////////////////////////

    def _fill_table(self, save_config=True):
        """
        Re-draw the table from scratch, by clearing out the table's
        multi-column listbox; and then filling it in with values from
        ``self._rows``.  Note that any cell-, row-, or column-specific
        color configuration that has been done will be lost.  The
        selection will also be lost -- i.e., no row will be selected
        after this call completes.
        """
        self._mlb.delete(0, "end")
        for i, row in enumerate(self._rows):
            if self._reprfunc is not None:
                row = [self._reprfunc(i, j, v) for (j, v) in enumerate(row)]
            self._mlb.insert("end", row)

    def _get_itemconfig(self, r, c):
        return {
            k: self._mlb.itemconfig(r, c, k)[-1]
            for k in (
                "foreground",
                "selectforeground",
                "background",
                "selectbackground",
            )
        }

    def _save_config_info(self, row_indices=None, index_by_id=False):
        """
        Return a 'cookie' containing information about which row is
        selected, and what color configurations have been applied.
        this information can the be re-applied to the table (after
        making modifications) using ``_restore_config_info()``.  Color
        configuration information will be saved for any rows in
        ``row_indices``, or in the entire table, if
        ``row_indices=None``.  If ``index_by_id=True``, the the cookie
        will associate rows with their configuration information based
        on the rows' python id.  This is useful when performing
        operations that re-arrange the rows (e.g. ``sort``).  If
        ``index_by_id=False``, then it is assumed that all rows will be
        in the same order when ``_restore_config_info()`` is called.
        """
        # Default value for row_indices is all rows.
        if row_indices is None:
            row_indices = list(range(len(self._rows)))

        # Look up our current selection.
        selection = self.selected_row()
        if index_by_id and selection is not None:
            selection = id(self._rows[selection])

        # Look up the color configuration info for each row.
        if index_by_id:
            config = {
                id(self._rows[r]): [
                    self._get_itemconfig(r, c) for c in range(self._num_columns)
                ]
                for r in row_indices
            }
        else:
            config = {
                r: [self._get_itemconfig(r, c) for c in range(self._num_columns)]
                for r in row_indices
            }

        return selection, config

    def _restore_config_info(self, cookie, index_by_id=False, see=False):
        """
        Restore selection & color configuration information that was
        saved using ``_save_config_info``.
        """
        selection, config = cookie

        # Clear the selection.
        if selection is None:
            self._mlb.selection_clear(0, "end")

        # Restore selection & color config
        if index_by_id:
            for r, row in enumerate(self._rows):
                if id(row) in config:
                    for c in range(self._num_columns):
                        self._mlb.itemconfigure(r, c, config[id(row)][c])
                if id(row) == selection:
                    self._mlb.select(r, see=see)
        else:
            if selection is not None:
                self._mlb.select(selection, see=see)
            for r in config:
                for c in range(self._num_columns):
                    self._mlb.itemconfigure(r, c, config[r][c])

    # /////////////////////////////////////////////////////////////////
    # Debugging (Invariant Checker)
    # /////////////////////////////////////////////////////////////////

    _DEBUG = False
    """If true, then run ``_check_table_vs_mlb()`` after any operation
       that modifies the table."""

    def _check_table_vs_mlb(self):
        """
        Verify that the contents of the table's ``_rows`` variable match
        the contents of its multi-listbox (``_mlb``).  This is just
        included for debugging purposes, to make sure that the
        list-modifying operations are working correctly.
        """
        for col in self._mlb.listboxes:
            assert len(self) == col.size()
        for row in self:
            assert len(row) == self._num_columns
        assert self._num_columns == len(self._mlb.column_names)
        # assert self._column_names == self._mlb.column_names
        for i, row in enumerate(self):
            for j, cell in enumerate(row):
                if self._reprfunc is not None:
                    cell = self._reprfunc(i, j, cell)
                assert self._mlb.get(i)[j] == cell


######################################################################
# Demo/Test Function
######################################################################

# update this to use new WordNet API
def demo():
    root = Tk()
    root.bind("<Control-q>", lambda e: root.destroy())

    table = Table(
        root,
        "Word Synset Hypernym Hyponym".split(),
        column_weights=[0, 1, 1, 1],
        reprfunc=(lambda i, j, s: "  %s" % s),
    )
    table.pack(expand=True, fill="both")

    from nltk.corpus import brown, wordnet

    for word, pos in sorted(set(brown.tagged_words()[:500])):
        if pos[0] != "N":
            continue
        word = word.lower()
        for synset in wordnet.synsets(word):
            try:
                hyper_def = synset.hypernyms()[0].definition()
            except:
                hyper_def = "*none*"
            try:
                hypo_def = synset.hypernyms()[0].definition()
            except:
                hypo_def = "*none*"
            table.append([word, synset.definition(), hyper_def, hypo_def])

    table.columnconfig("Word", background="#afa")
    table.columnconfig("Synset", background="#efe")
    table.columnconfig("Hypernym", background="#fee")
    table.columnconfig("Hyponym", background="#ffe")
    for row in range(len(table)):
        for column in ("Hypernym", "Hyponym"):
            if table[row, column] == "*none*":
                table.itemconfig(
                    row, column, foreground="#666", selectforeground="#666"
                )
    root.mainloop()


if __name__ == "__main__":
    demo()
