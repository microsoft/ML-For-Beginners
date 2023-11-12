"""
Provides a simple table class.  A SimpleTable is essentially
a list of lists plus some formatting functionality.

Dependencies: the Python 2.5+ standard library.

Installation: just copy this module into your working directory (or
   anywhere in your pythonpath).

Basic use::

   mydata = [[11,12],[21,22]]  # data MUST be 2-dimensional
   myheaders = [ "Column 1", "Column 2" ]
   mystubs = [ "Row 1", "Row 2" ]
   tbl = SimpleTable(mydata, myheaders, mystubs, title="Title")
   print( tbl )
   print( tbl.as_csv() )

A SimpleTable is inherently (but not rigidly) rectangular.
You should create it from a *rectangular* (2d!) iterable of data.
Each item in your rectangular iterable will become the data
of a single Cell.  In principle, items can be any object,
not just numbers and strings.  However, default conversion
during table production is by simple string interpolation.
(So you cannot have a tuple as a data item *and* rely on
the default conversion.)

A SimpleTable allows only one column (the first) of stubs at
initilization, concatenation of tables allows you to produce tables
with interior stubs.  (You can also assign the datatype 'stub' to the
cells in any column, or use ``insert_stubs``.) A SimpleTable can be
concatenated with another SimpleTable or extended by another
SimpleTable. ::

    table1.extend_right(table2)
    table1.extend(table2)


A SimpleTable can be initialized with `datatypes`: a list of ints that
provide indexes into `data_fmts` and `data_aligns`.  Each data cell is
assigned a datatype, which will control formatting.  If you do not
specify the `datatypes` list, it will be set to ``range(ncols)`` where
`ncols` is the number of columns in the data.  (I.e., cells in a
column have their own datatype.) This means that you can just specify
`data_fmts` without bothering to provide a `datatypes` list.  If
``len(datatypes)<ncols`` then datatype assignment will cycle across a
row.  E.g., if you provide 10 columns of data with ``datatypes=[0,1]``
then you will have 5 columns of datatype 0 and 5 columns of datatype
1, alternating.  Corresponding to this specification, you should provide
a list of two ``data_fmts`` and a list of two ``data_aligns``.

Cells can be assigned labels as their `datatype` attribute.
You can then provide a format for that lable.
Us the SimpleTable's `label_cells` method to do this.  ::

    def mylabeller(cell):
        if cell.data is np.nan:
            return 'missing'

    mytable.label_cells(mylabeller)
    print(mytable.as_text(missing='-'))


Potential problems for Python 3
-------------------------------

- Calls ``next`` instead of ``__next__``.
  The 2to3 tool should handle that no problem.
  (We will switch to the `next` function if 2.5 support is ever dropped.)
- Let me know if you find other problems.

:contact: alan dot isaac at gmail dot com
:requires: Python 2.5.1+
:note: current version
:note: HTML data format currently specifies tags
:todo: support a bit more of http://www.oasis-open.org/specs/tr9503.html
:todo: add labels2formatters method, that associates a cell formatter with a
       datatype
:todo: add colspan support to Cell
:since: 2008-12-21
:change: 2010-05-02 eliminate newlines that came before and after table
:change: 2010-05-06 add `label_cells` to `SimpleTable`
"""

from statsmodels.compat.python import lmap, lrange

from itertools import cycle, zip_longest
import csv


def csv2st(csvfile, headers=False, stubs=False, title=None):
    """Return SimpleTable instance,
    created from the data in `csvfile`,
    which is in comma separated values format.
    The first row may contain headers: set headers=True.
    The first column may contain stubs: set stubs=True.
    Can also supply headers and stubs as tuples of strings.
    """
    rows = list()
    with open(csvfile, 'r', encoding="utf-8") as fh:
        reader = csv.reader(fh)
        if headers is True:
            headers = next(reader)
        elif headers is False:
            headers = ()
        if stubs is True:
            stubs = list()
            for row in reader:
                if row:
                    stubs.append(row[0])
                    rows.append(row[1:])
        else:  # no stubs, or stubs provided
            for row in reader:
                if row:
                    rows.append(row)
        if stubs is False:
            stubs = ()
    ncols = len(rows[0])
    if any(len(row) != ncols for row in rows):
        raise IOError('All rows of CSV file must have same length.')
    return SimpleTable(data=rows, headers=headers, stubs=stubs)


class SimpleTable(list):
    """Produce a simple ASCII, CSV, HTML, or LaTeX table from a
    *rectangular* (2d!) array of data, not necessarily numerical.
    Directly supports at most one header row,
    which should be the length of data[0].
    Directly supports at most one stubs column,
    which must be the length of data.
    (But see `insert_stubs` method.)
    See globals `default_txt_fmt`, `default_csv_fmt`, `default_html_fmt`,
    and `default_latex_fmt` for formatting options.

    Sample uses::

        mydata = [[11,12],[21,22]]  # data MUST be 2-dimensional
        myheaders = [ "Column 1", "Column 2" ]
        mystubs = [ "Row 1", "Row 2" ]
        tbl = text.SimpleTable(mydata, myheaders, mystubs, title="Title")
        print( tbl )
        print( tbl.as_html() )
        # set column specific data formatting
        tbl = text.SimpleTable(mydata, myheaders, mystubs,
            data_fmts=["%3.2f","%d"])
        print( tbl.as_csv() )
        with open('c:/temp/temp.tex','w') as fh:
            fh.write( tbl.as_latex_tabular() )
    """
    def __init__(self, data, headers=None, stubs=None, title='',
                 datatypes=None, csv_fmt=None, txt_fmt=None, ltx_fmt=None,
                 html_fmt=None, celltype=None, rowtype=None, **fmt_dict):
        """
        Parameters
        ----------
        data : list of lists or 2d array (not matrix!)
            R rows by K columns of table elements
        headers : list (or tuple) of str
            sequence of K strings, one per header
        stubs : list (or tuple) of str
            sequence of R strings, one per stub
        title : str
            title of the table
        datatypes : list of int
            indexes to `data_fmts`
        txt_fmt : dict
            text formatting options
        ltx_fmt : dict
            latex formatting options
        csv_fmt : dict
            csv formatting options
        hmtl_fmt : dict
            hmtl formatting options
        celltype : class
            the cell class for the table (default: Cell)
        rowtype : class
            the row class for the table (default: Row)
        fmt_dict : dict
            general formatting options
        """
        self.title = title
        self._datatypes = datatypes
        if self._datatypes is None:
            self._datatypes = [] if len(data) == 0 else lrange(len(data[0]))
        # start with default formatting
        self._txt_fmt = default_txt_fmt.copy()
        self._latex_fmt = default_latex_fmt.copy()
        self._csv_fmt = default_csv_fmt.copy()
        self._html_fmt = default_html_fmt.copy()
        # substitute any general user specified formatting
        # :note: these will be overridden by output specific arguments
        self._csv_fmt.update(fmt_dict)
        self._txt_fmt.update(fmt_dict)
        self._latex_fmt.update(fmt_dict)
        self._html_fmt.update(fmt_dict)
        # substitute any output-type specific formatting
        self._csv_fmt.update(csv_fmt or dict())
        self._txt_fmt.update(txt_fmt or dict())
        self._latex_fmt.update(ltx_fmt or dict())
        self._html_fmt.update(html_fmt or dict())
        self.output_formats = dict(
            txt=self._txt_fmt,
            csv=self._csv_fmt,
            html=self._html_fmt,
            latex=self._latex_fmt
        )
        self._Cell = celltype or Cell
        self._Row = rowtype or Row
        rows = self._data2rows(data)  # a list of Row instances
        list.__init__(self, rows)
        self._add_headers_stubs(headers, stubs)
        self._colwidths = dict()

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self))

    def _repr_html_(self, **fmt_dict):
        return self.as_html(**fmt_dict)

    def _repr_latex_(self, center=True, **fmt_dict):
        return self.as_latex_tabular(center, **fmt_dict)

    def _add_headers_stubs(self, headers, stubs):
        """Return None.  Adds headers and stubs to table,
        if these were provided at initialization.
        Parameters
        ----------
        headers : list[str]
            K strings, where K is number of columns
        stubs : list[str]
            R strings, where R is number of non-header rows

        :note: a header row does not receive a stub!
        """
        if headers:
            self.insert_header_row(0, headers, dec_below='header_dec_below')
        if stubs:
            self.insert_stubs(0, stubs)

    def insert(self, idx, row, datatype=None):
        """Return None.  Insert a row into a table.
        """
        if datatype is None:
            try:
                datatype = row.datatype
            except AttributeError:
                pass
        row = self._Row(row, datatype=datatype, table=self)
        list.insert(self, idx, row)

    def insert_header_row(self, rownum, headers, dec_below='header_dec_below'):
        """Return None.  Insert a row of headers,
        where ``headers`` is a sequence of strings.
        (The strings may contain newlines, to indicated multiline headers.)
        """
        header_rows = [header.split('\n') for header in headers]
        # rows in reverse order
        rows = list(zip_longest(*header_rows, **dict(fillvalue='')))
        rows.reverse()
        for i, row in enumerate(rows):
            self.insert(rownum, row, datatype='header')
            if i == 0:
                self[rownum].dec_below = dec_below
            else:
                self[rownum].dec_below = None

    def insert_stubs(self, loc, stubs):
        """Return None.  Insert column of stubs at column `loc`.
        If there is a header row, it gets an empty cell.
        So ``len(stubs)`` should equal the number of non-header rows.
        """
        _Cell = self._Cell
        stubs = iter(stubs)
        for row in self:
            if row.datatype == 'header':
                empty_cell = _Cell('', datatype='empty')
                row.insert(loc, empty_cell)
            else:
                try:
                    row.insert_stub(loc, next(stubs))
                except StopIteration:
                    raise ValueError('length of stubs must match table length')

    def _data2rows(self, raw_data):
        """Return list of Row,
        the raw data as rows of cells.
        """

        _Cell = self._Cell
        _Row = self._Row
        rows = []
        for datarow in raw_data:
            dtypes = cycle(self._datatypes)
            newrow = _Row(datarow, datatype='data', table=self, celltype=_Cell)
            for cell in newrow:
                cell.datatype = next(dtypes)
                cell.row = newrow  # a cell knows its row
            rows.append(newrow)

        return rows

    def pad(self, s, width, align):
        """DEPRECATED: just use the pad function"""
        return pad(s, width, align)

    def _get_colwidths(self, output_format, **fmt_dict):
        """Return list, the calculated widths of each column."""
        output_format = get_output_format(output_format)
        fmt = self.output_formats[output_format].copy()
        fmt.update(fmt_dict)
        ncols = max(len(row) for row in self)
        request = fmt.get('colwidths')
        if request == 0:  # assume no extra space desired (e.g, CSV)
            return [0] * ncols
        elif request is None:  # assume no extra space desired (e.g, CSV)
            request = [0] * ncols
        elif isinstance(request, int):
            request = [request] * ncols
        elif len(request) < ncols:
            request = [request[i % len(request)] for i in range(ncols)]
        min_widths = []
        for col in zip(*self):
            maxwidth = max(len(c.format(0, output_format, **fmt)) for c in col)
            min_widths.append(maxwidth)
        result = lmap(max, min_widths, request)
        return result

    def get_colwidths(self, output_format, **fmt_dict):
        """Return list, the widths of each column."""
        call_args = [output_format]
        for k, v in sorted(fmt_dict.items()):
            if isinstance(v, list):
                call_args.append((k, tuple(v)))
            elif isinstance(v, dict):
                call_args.append((k, tuple(sorted(v.items()))))
            else:
                call_args.append((k, v))
        key = tuple(call_args)
        try:
            return self._colwidths[key]
        except KeyError:
            self._colwidths[key] = self._get_colwidths(output_format,
                                                       **fmt_dict)
            return self._colwidths[key]

    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        output_format = get_output_format(output_format)
        # first get the default formatting
        try:
            fmt = self.output_formats[output_format].copy()
        except KeyError:
            raise ValueError('Unknown format: %s' % output_format)
        # then, add formatting specific to this call
        fmt.update(fmt_dict)
        return fmt

    def as_csv(self, **fmt_dict):
        """Return string, the table in CSV format.
        Currently only supports comma separator."""
        # fetch the format, which may just be default_csv_format
        fmt = self._get_fmt('csv', **fmt_dict)
        return self.as_text(**fmt)

    def as_text(self, **fmt_dict):
        """Return string, the table as text."""
        # fetch the text format, override with fmt_dict
        fmt = self._get_fmt('txt', **fmt_dict)
        # get rows formatted as strings
        formatted_rows = [row.as_string('text', **fmt) for row in self]
        rowlen = len(formatted_rows[-1])  # do not use header row

        # place decoration above the table body, if desired
        table_dec_above = fmt.get('table_dec_above', '=')
        if table_dec_above:
            formatted_rows.insert(0, table_dec_above * rowlen)
        # next place a title at the very top, if desired
        # :note: user can include a newlines at end of title if desired
        title = self.title
        if title:
            title = pad(self.title, rowlen, fmt.get('title_align', 'c'))
            formatted_rows.insert(0, title)
        # add decoration below the table, if desired
        table_dec_below = fmt.get('table_dec_below', '-')
        if table_dec_below:
            formatted_rows.append(table_dec_below * rowlen)
        return '\n'.join(formatted_rows)

    def as_html(self, **fmt_dict):
        """Return string.
        This is the default formatter for HTML tables.
        An HTML table formatter must accept as arguments
        a table and a format dictionary.
        """
        # fetch the text format, override with fmt_dict
        fmt = self._get_fmt('html', **fmt_dict)
        formatted_rows = ['<table class="simpletable">']
        if self.title:
            title = '<caption>%s</caption>' % self.title
            formatted_rows.append(title)
        formatted_rows.extend(row.as_string('html', **fmt) for row in self)
        formatted_rows.append('</table>')
        return '\n'.join(formatted_rows)

    def as_latex_tabular(self, center=True, **fmt_dict):
        '''Return string, the table as a LaTeX tabular environment.
        Note: will require the booktabs package.'''
        # fetch the text format, override with fmt_dict
        fmt = self._get_fmt('latex', **fmt_dict)

        formatted_rows = []
        if center:
            formatted_rows.append(r'\begin{center}')

        table_dec_above = fmt['table_dec_above'] or ''
        table_dec_below = fmt['table_dec_below'] or ''

        prev_aligns = None
        last = None
        for row in self + [last]:
            if row == last:
                aligns = None
            else:
                aligns = row.get_aligns('latex', **fmt)

            if aligns != prev_aligns:
                # When the number/type of columns changes...
                if prev_aligns:
                    # ... if there is a tabular to close, close it...
                    formatted_rows.append(table_dec_below)
                    formatted_rows.append(r'\end{tabular}')
                if aligns:
                    # ... and if there are more lines, open a new one:
                    formatted_rows.append(r'\begin{tabular}{%s}' % aligns)
                    if not prev_aligns:
                        # (with a nice line if it's the top of the whole table)
                        formatted_rows.append(table_dec_above)
            if row != last:
                formatted_rows.append(
                    row.as_string(output_format='latex', **fmt))
            prev_aligns = aligns
        # tabular does not support caption, but make it available for
        # figure environment
        if self.title:
            title = r'%%\caption{%s}' % self.title
            formatted_rows.append(title)
        if center:
            formatted_rows.append(r'\end{center}')

        # Replace $$ due to bug in GH 5444
        return '\n'.join(formatted_rows).replace('$$', ' ')

    def extend_right(self, table):
        """Return None.
        Extend each row of `self` with corresponding row of `table`.
        Does **not** import formatting from ``table``.
        This generally makes sense only if the two tables have
        the same number of rows, but that is not enforced.
        :note: To extend append a table below, just use `extend`,
        which is the ordinary list method.  This generally makes sense
        only if the two tables have the same number of columns,
        but that is not enforced.
        """
        for row1, row2 in zip(self, table):
            row1.extend(row2)

    def label_cells(self, func):
        """Return None.  Labels cells based on `func`.
        If ``func(cell) is None`` then its datatype is
        not changed; otherwise it is set to ``func(cell)``.
        """
        for row in self:
            for cell in row:
                label = func(cell)
                if label is not None:
                    cell.datatype = label

    @property
    def data(self):
        return [row.data for row in self]


def pad(s, width, align):
    """Return string padded with spaces,
    based on alignment parameter."""
    if align == 'l':
        s = s.ljust(width)
    elif align == 'r':
        s = s.rjust(width)
    else:
        s = s.center(width)
    return s


class Row(list):
    """Provides a table row as a list of cells.
    A row can belong to a SimpleTable, but does not have to.
    """
    def __init__(self, seq, datatype='data', table=None, celltype=None,
                 dec_below='row_dec_below', **fmt_dict):
        """
        Parameters
        ----------
        seq : sequence of data or cells
        table : SimpleTable
        datatype : str ('data' or 'header')
        dec_below : str
          (e.g., 'header_dec_below' or 'row_dec_below')
          decoration tag, identifies the decoration to go below the row.
          (Decoration is repeated as needed for text formats.)
        """
        self.datatype = datatype
        self.table = table
        if celltype is None:
            if table is None:
                celltype = Cell
            else:
                celltype = table._Cell
        self._Cell = celltype
        self._fmt = fmt_dict
        self.special_fmts = dict()  # special formatting for any output format
        self.dec_below = dec_below
        list.__init__(self, (celltype(cell, row=self) for cell in seq))

    def add_format(self, output_format, **fmt_dict):
        """
        Return None. Adds row-instance specific formatting
        for the specified output format.
        Example: myrow.add_format('txt', row_dec_below='+-')
        """
        output_format = get_output_format(output_format)
        if output_format not in self.special_fmts:
            self.special_fmts[output_format] = dict()
        self.special_fmts[output_format].update(fmt_dict)

    def insert_stub(self, loc, stub):
        """Return None.  Inserts a stub cell
        in the row at `loc`.
        """
        _Cell = self._Cell
        if not isinstance(stub, _Cell):
            stub = stub
            stub = _Cell(stub, datatype='stub', row=self)
        self.insert(loc, stub)

    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        output_format = get_output_format(output_format)
        # first get the default formatting
        try:
            fmt = default_fmts[output_format].copy()
        except KeyError:
            raise ValueError('Unknown format: %s' % output_format)
        # second get table specific formatting (if possible)
        try:
            fmt.update(self.table.output_formats[output_format])
        except AttributeError:
            pass
        # finally, add formatting for this row and this call
        fmt.update(self._fmt)
        fmt.update(fmt_dict)
        special_fmt = self.special_fmts.get(output_format, None)
        if special_fmt is not None:
            fmt.update(special_fmt)
        return fmt

    def get_aligns(self, output_format, **fmt_dict):
        """Return string, sequence of column alignments.
        Ensure comformable data_aligns in `fmt_dict`."""
        fmt = self._get_fmt(output_format, **fmt_dict)
        return ''.join(cell.alignment(output_format, **fmt) for cell in self)

    def as_string(self, output_format='txt', **fmt_dict):
        """Return string: the formatted row.
        This is the default formatter for rows.
        Override this to get different formatting.
        A row formatter must accept as arguments
        a row (self) and an output format,
        one of ('html', 'txt', 'csv', 'latex').
        """
        fmt = self._get_fmt(output_format, **fmt_dict)

        # get column widths
        try:
            colwidths = self.table.get_colwidths(output_format, **fmt)
        except AttributeError:
            colwidths = fmt.get('colwidths')
        if colwidths is None:
            colwidths = (0,) * len(self)

        colsep = fmt['colsep']
        row_pre = fmt.get('row_pre', '')
        row_post = fmt.get('row_post', '')
        formatted_cells = []
        for cell, width in zip(self, colwidths):
            content = cell.format(width, output_format=output_format, **fmt)
            formatted_cells.append(content)
        formatted_row = row_pre + colsep.join(formatted_cells) + row_post
        formatted_row = self._decorate_below(formatted_row, output_format,
                                             **fmt)
        return formatted_row

    def _decorate_below(self, row_as_string, output_format, **fmt_dict):
        """This really only makes sense for the text and latex output formats.
        """
        dec_below = fmt_dict.get(self.dec_below, None)
        if dec_below is None:
            result = row_as_string
        else:
            output_format = get_output_format(output_format)
            if output_format == 'txt':
                row0len = len(row_as_string)
                dec_len = len(dec_below)
                repeat, addon = divmod(row0len, dec_len)
                result = row_as_string + "\n" + (dec_below * repeat +
                                                 dec_below[:addon])
            elif output_format == 'latex':
                result = row_as_string + "\n" + dec_below
            else:
                raise ValueError("I cannot decorate a %s header." %
                                 output_format)
        return result

    @property
    def data(self):
        return [cell.data for cell in self]


class Cell:
    """Provides a table cell.
    A cell can belong to a Row, but does not have to.
    """
    def __init__(self, data='', datatype=None, row=None, **fmt_dict):
        if isinstance(data, Cell):
            # might have passed a Cell instance
            self.data = data.data
            self._datatype = data.datatype
            self._fmt = data._fmt
        else:
            self.data = data
            self._datatype = datatype
            self._fmt = dict()
        self._fmt.update(fmt_dict)
        self.row = row

    def __str__(self):
        return '%s' % self.data

    def _get_fmt(self, output_format, **fmt_dict):
        """Return dict, the formatting options.
        """
        output_format = get_output_format(output_format)
        # first get the default formatting
        try:
            fmt = default_fmts[output_format].copy()
        except KeyError:
            raise ValueError('Unknown format: %s' % output_format)
        # then get any table specific formtting
        try:
            fmt.update(self.row.table.output_formats[output_format])
        except AttributeError:
            pass
        # then get any row specific formtting
        try:
            fmt.update(self.row._fmt)
        except AttributeError:
            pass
        # finally add formatting for this instance and call
        fmt.update(self._fmt)
        fmt.update(fmt_dict)
        return fmt

    def alignment(self, output_format, **fmt_dict):
        fmt = self._get_fmt(output_format, **fmt_dict)
        datatype = self.datatype
        data_aligns = fmt.get('data_aligns', 'c')
        if isinstance(datatype, int):
            align = data_aligns[datatype % len(data_aligns)]
        elif datatype == 'stub':
            # still support deprecated `stubs_align`
            align = fmt.get('stubs_align') or fmt.get('stub_align', 'l')
        elif datatype in fmt:
            label_align = '%s_align' % datatype
            align = fmt.get(label_align, 'c')
        else:
            raise ValueError('Unknown cell datatype: %s' % datatype)
        return align

    @staticmethod
    def _latex_escape(data, fmt, output_format):
        if output_format != 'latex':
            return data
        if "replacements" in fmt:
            if isinstance(data, str):
                for repl in sorted(fmt["replacements"]):
                    data = data.replace(repl, fmt["replacements"][repl])
        return data

    def format(self, width, output_format='txt', **fmt_dict):
        """Return string.
        This is the default formatter for cells.
        Override this to get different formating.
        A cell formatter must accept as arguments
        a cell (self) and an output format,
        one of ('html', 'txt', 'csv', 'latex').
        It will generally respond to the datatype,
        one of (int, 'header', 'stub').
        """
        fmt = self._get_fmt(output_format, **fmt_dict)

        data = self.data
        datatype = self.datatype
        data_fmts = fmt.get('data_fmts')
        if data_fmts is None:
            # chk allow for deprecated use of data_fmt
            data_fmt = fmt.get('data_fmt')
            if data_fmt is None:
                data_fmt = '%s'
            data_fmts = [data_fmt]
        if isinstance(datatype, int):
            datatype = datatype % len(data_fmts)  # constrain to indexes
            data_fmt = data_fmts[datatype]
            if isinstance(data_fmt, str):
                content = data_fmt % (data,)
            elif callable(data_fmt):
                content = data_fmt(data)
            else:
                raise TypeError("Must be a string or a callable")
            if datatype == 0:
                content = self._latex_escape(content, fmt, output_format)
        elif datatype in fmt:
            data = self._latex_escape(data, fmt, output_format)

            dfmt = fmt.get(datatype)
            try:
                content = dfmt % (data,)
            except TypeError:  # dfmt is not a substitution string
                content = dfmt
        else:
            raise ValueError('Unknown cell datatype: %s' % datatype)
        align = self.alignment(output_format, **fmt)
        return pad(content, width, align)

    def get_datatype(self):
        if self._datatype is None:
            dtype = self.row.datatype
        else:
            dtype = self._datatype
        return dtype

    def set_datatype(self, val):
        # TODO: add checking
        self._datatype = val
    datatype = property(get_datatype, set_datatype)


# begin: default formats for SimpleTable
""" Some formatting suggestions:

- if you want rows to have no extra spacing,
  set colwidths=0 and colsep=''.
  (Naturally the columns will not align.)
- if you want rows to have minimal extra spacing,
  set colwidths=1.  The columns will align.
- to get consistent formatting, you should leave
  all field width handling to SimpleTable:
  use 0 as the field width in data_fmts.  E.g., ::

        data_fmts = ["%#0.6g","%#0.6g","%#0.4g","%#0.4g"],
        colwidths = 14,
        data_aligns = "r",
"""
default_txt_fmt = dict(
    fmt='txt',
    # basic table formatting
    table_dec_above='=',
    table_dec_below='-',
    title_align='c',
    # basic row formatting
    row_pre='',
    row_post='',
    header_dec_below='-',
    row_dec_below=None,
    colwidths=None,
    colsep=' ',
    data_aligns="r",  # GH 1477
    # data formats
    # data_fmt="%s",  #deprecated; use data_fmts
    data_fmts=["%s"],
    # labeled alignments
    # stubs_align='l',   #deprecated; use data_fmts
    stub_align='l',
    header_align='c',
    # labeled formats
    header_fmt='%s',  # deprecated; just use 'header'
    stub_fmt='%s',  # deprecated; just use 'stub'
    header='%s',
    stub='%s',
    empty_cell='',  # deprecated; just use 'empty'
    empty='',
    missing='--',
)

default_csv_fmt = dict(
    fmt='csv',
    table_dec_above=None,  # '',
    table_dec_below=None,  # '',
    # basic row formatting
    row_pre='',
    row_post='',
    header_dec_below=None,  # '',
    row_dec_below=None,
    title_align='',
    data_aligns="l",
    colwidths=None,
    colsep=',',
    # data formats
    data_fmt='%s',  # deprecated; use data_fmts
    data_fmts=['%s'],
    # labeled alignments
    # stubs_align='l',   # deprecated; use data_fmts
    stub_align="l",
    header_align='c',
    # labeled formats
    header_fmt='"%s"',  # deprecated; just use 'header'
    stub_fmt='"%s"',  # deprecated; just use 'stub'
    empty_cell='',  # deprecated; just use 'empty'
    header='%s',
    stub='%s',
    empty='',
    missing='--',
)

default_html_fmt = dict(
    # basic table formatting
    table_dec_above=None,
    table_dec_below=None,
    header_dec_below=None,
    row_dec_below=None,
    title_align='c',
    # basic row formatting
    colwidths=None,
    colsep=' ',
    row_pre='<tr>\n  ',
    row_post='\n</tr>',
    data_aligns="c",
    # data formats
    data_fmts=['<td>%s</td>'],
    data_fmt="<td>%s</td>",  # deprecated; use data_fmts
    # labeled alignments
    # stubs_align='l',   #deprecated; use data_fmts
    stub_align='l',
    header_align='c',
    # labeled formats
    header_fmt='<th>%s</th>',  # deprecated; just use `header`
    stub_fmt='<th>%s</th>',  # deprecated; just use `stub`
    empty_cell='<td></td>',  # deprecated; just use `empty`
    header='<th>%s</th>',
    stub='<th>%s</th>',
    empty='<td></td>',
    missing='<td>--</td>',
)

default_latex_fmt = dict(
    fmt='ltx',
    # basic table formatting
    table_dec_above=r'\toprule',
    table_dec_below=r'\bottomrule',
    header_dec_below=r'\midrule',
    row_dec_below=None,
    strip_backslash=True,  # NotImplemented
    # row formatting
    row_post=r'  \\',
    data_aligns='c',
    colwidths=None,
    colsep=' & ',
    # data formats
    data_fmts=['%s'],
    data_fmt='%s',  # deprecated; use data_fmts
    # labeled alignments
    # stubs_align='l',   # deprecated; use data_fmts
    stub_align='l',
    header_align='c',
    empty_align='l',
    # labeled formats
    header_fmt=r'\textbf{%s}',  # deprecated; just use 'header'
    stub_fmt=r'\textbf{%s}',  # deprecated; just use 'stub'
    empty_cell='',  # deprecated; just use 'empty'
    header=r'\textbf{%s}',
    stub=r'\textbf{%s}',
    empty='',
    missing='--',
    # replacements will be processed in lexicographical order
    replacements={"#": r"\#",
                  "$": r"\$",
                  "%": r"\%",
                  "&": r"\&",
                  ">": r"$>$",
                  "_": r"\_",
                  "|": r"$|$"}
)

default_fmts = dict(
    html=default_html_fmt,
    txt=default_txt_fmt,
    latex=default_latex_fmt,
    csv=default_csv_fmt
)
output_format_translations = dict(
    htm='html',
    text='txt',
    ltx='latex'
)


def get_output_format(output_format):
    if output_format not in ('html', 'txt', 'latex', 'csv'):
        try:
            output_format = output_format_translations[output_format]
        except KeyError:
            raise ValueError('unknown output format %s' % output_format)
    return output_format
