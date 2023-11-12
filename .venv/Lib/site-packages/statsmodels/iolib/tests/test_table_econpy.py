'''
Unit tests table.py.

:see: http://docs.python.org/lib/minimal-example.html for an intro to unittest
:see: http://agiletesting.blogspot.com/2005/01/python-unit-testing-part-1-unittest.html
:see: http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/305292
'''
import numpy as np
from numpy.testing import assert_equal

__docformat__ = "restructuredtext en"

from statsmodels.iolib.table import Cell, SimpleTable
from statsmodels.iolib.table import default_latex_fmt
from statsmodels.iolib.table import default_html_fmt

ltx_fmt1 = default_latex_fmt.copy()
html_fmt1 = default_html_fmt.copy()

txt_fmt1 = dict(
    data_fmts = ['%0.2f', '%d'],
    empty_cell = ' ',
    colwidths = 1,
    colsep=' * ',
    row_pre = '* ',
    row_post = ' *',
    table_dec_above='*',
    table_dec_below='*',
    header_dec_below='*',
    header_fmt = '%s',
    stub_fmt = '%s',
    title_align='r',
    header_align = 'r',
    data_aligns = "r",
    stubs_align = "l",
    fmt = 'txt'
)
cell0data = 0.0000
cell1data = 1
row0data = [cell0data, cell1data]
row1data = [2, 3.333]
table1data = [ row0data, row1data ]
test1stubs = ('stub1', 'stub2')
test1header = ('header1', 'header2')
#test1header = ('header1\nheader1a', 'header2\nheader2a')
tbl = SimpleTable(table1data, test1header, test1stubs,
    txt_fmt=txt_fmt1, ltx_fmt=ltx_fmt1, html_fmt=html_fmt1)


def custom_labeller(cell):
    if cell.data is np.nan:
        return 'missing'


class TestCell:
    def test_celldata(self):
        celldata = cell0data, cell1data, row1data[0], row1data[1]
        cells = [Cell(datum, datatype=i % 2)
                 for i, datum in enumerate(celldata)]
        for cell, datum in zip(cells, celldata):
            assert_equal(cell.data, datum)


class TestSimpleTable:
    def test_txt_fmt1(self):
        # Limited test of custom txt_fmt
        desired = """
*****************************
*       * header1 * header2 *
*****************************
* stub1 *    0.00 *       1 *
* stub2 *    2.00 *       3 *
*****************************
"""
        actual = '\n%s\n' % tbl.as_text()
        #print('actual')
        #print(actual)
        #print('desired')
        #print(desired)
        assert_equal(actual, desired)
    def test_ltx_fmt1(self):
        # Limited test of custom ltx_fmt
        desired = r"""
\begin{center}
\begin{tabular}{lcc}
\toprule
               & \textbf{header1} & \textbf{header2}  \\
\midrule
\textbf{stub1} &       0.0        &        1          \\
\textbf{stub2} &        2         &      3.333        \\
\bottomrule
\end{tabular}
\end{center}
"""
        actual = '\n%s\n' % tbl.as_latex_tabular()
        #print(actual)
        #print(desired)
        assert_equal(actual, desired)

    def test_html_fmt1(self):
        # Limited test of custom html_fmt
        desired = """
<table class="simpletable">
<tr>
    <td></td>    <th>header1</th> <th>header2</th>
</tr>
<tr>
  <th>stub1</th>   <td>0.0</td>      <td>1</td>
</tr>
<tr>
  <th>stub2</th>    <td>2</td>     <td>3.333</td>
</tr>
</table>
"""
        #the previous has significant trailing whitespace that got removed
        #desired = '''\n<table class="simpletable">\n<tr>\n    <td></td>    <th>header1</th> <th>header2</th>\n</tr>\n<tr>\n  <th>stub1</th>   <td>0.0</td>      <td>1</td>   \n</tr>\n<tr>\n  <th>stub2</th>    <td>2</td>     <td>3.333</td> \n</tr>\n</table>\n'''
        actual = '\n%s\n' % tbl.as_html()
        actual = '\n'.join((line.rstrip() for line in actual.split('\n')))
        #print(actual)
        #print(desired)
        #print len(actual), len(desired)
        assert_equal(actual, desired)

    def test_customlabel(self):
        # Limited test of custom custom labeling
        tbl = SimpleTable(table1data, test1header, test1stubs, txt_fmt=txt_fmt1)
        tbl[1][1].data = np.nan
        tbl.label_cells(custom_labeller)
        #print([[c.datatype for c in row] for row in tbl])
        desired = """
*****************************
*       * header1 * header2 *
*****************************
* stub1 *    --   *       1 *
* stub2 *    2.00 *       3 *
*****************************
"""
        actual = '\n%s\n' % tbl.as_text(missing='--')
        assert_equal(actual, desired)
