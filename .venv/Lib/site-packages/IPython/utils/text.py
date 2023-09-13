# encoding: utf-8
"""
Utilities for working with strings and text.

Inheritance diagram:

.. inheritance-diagram:: IPython.utils.text
   :parts: 3
"""

import os
import re
import string
import sys
import textwrap
from string import Formatter
from pathlib import Path


# datetime.strftime date format for ipython
if sys.platform == 'win32':
    date_format = "%B %d, %Y"
else:
    date_format = "%B %-d, %Y"

class LSString(str):
    """String derivative with a special access attributes.

    These are normal strings, but with the special attributes:

        .l (or .list) : value as list (split on newlines).
        .n (or .nlstr): original value (the string itself).
        .s (or .spstr): value as whitespace-separated string.
        .p (or .paths): list of path objects (requires path.py package)

    Any values which require transformations are computed only once and
    cached.

    Such strings are very useful to efficiently interact with the shell, which
    typically only understands whitespace-separated options for commands."""

    def get_list(self):
        try:
            return self.__list
        except AttributeError:
            self.__list = self.split('\n')
            return self.__list

    l = list = property(get_list)

    def get_spstr(self):
        try:
            return self.__spstr
        except AttributeError:
            self.__spstr = self.replace('\n',' ')
            return self.__spstr

    s = spstr = property(get_spstr)

    def get_nlstr(self):
        return self

    n = nlstr = property(get_nlstr)

    def get_paths(self):
        try:
            return self.__paths
        except AttributeError:
            self.__paths = [Path(p) for p in self.split('\n') if os.path.exists(p)]
            return self.__paths

    p = paths = property(get_paths)

# FIXME: We need to reimplement type specific displayhook and then add this
# back as a custom printer. This should also be moved outside utils into the
# core.

# def print_lsstring(arg):
#     """ Prettier (non-repr-like) and more informative printer for LSString """
#     print "LSString (.p, .n, .l, .s available). Value:"
#     print arg
#
#
# print_lsstring = result_display.register(LSString)(print_lsstring)


class SList(list):
    """List derivative with a special access attributes.

    These are normal lists, but with the special attributes:

    * .l (or .list) : value as list (the list itself).
    * .n (or .nlstr): value as a string, joined on newlines.
    * .s (or .spstr): value as a string, joined on spaces.
    * .p (or .paths): list of path objects (requires path.py package)

    Any values which require transformations are computed only once and
    cached."""

    def get_list(self):
        return self

    l = list = property(get_list)

    def get_spstr(self):
        try:
            return self.__spstr
        except AttributeError:
            self.__spstr = ' '.join(self)
            return self.__spstr

    s = spstr = property(get_spstr)

    def get_nlstr(self):
        try:
            return self.__nlstr
        except AttributeError:
            self.__nlstr = '\n'.join(self)
            return self.__nlstr

    n = nlstr = property(get_nlstr)

    def get_paths(self):
        try:
            return self.__paths
        except AttributeError:
            self.__paths = [Path(p) for p in self if os.path.exists(p)]
            return self.__paths

    p = paths = property(get_paths)

    def grep(self, pattern, prune = False, field = None):
        """ Return all strings matching 'pattern' (a regex or callable)

        This is case-insensitive. If prune is true, return all items
        NOT matching the pattern.

        If field is specified, the match must occur in the specified
        whitespace-separated field.

        Examples::

            a.grep( lambda x: x.startswith('C') )
            a.grep('Cha.*log', prune=1)
            a.grep('chm', field=-1)
        """

        def match_target(s):
            if field is None:
                return s
            parts = s.split()
            try:
                tgt = parts[field]
                return tgt
            except IndexError:
                return ""

        if isinstance(pattern, str):
            pred = lambda x : re.search(pattern, x, re.IGNORECASE)
        else:
            pred = pattern
        if not prune:
            return SList([el for el in self if pred(match_target(el))])
        else:
            return SList([el for el in self if not pred(match_target(el))])

    def fields(self, *fields):
        """ Collect whitespace-separated fields from string list

        Allows quick awk-like usage of string lists.

        Example data (in var a, created by 'a = !ls -l')::

            -rwxrwxrwx  1 ville None      18 Dec 14  2006 ChangeLog
            drwxrwxrwx+ 6 ville None       0 Oct 24 18:05 IPython

        * ``a.fields(0)`` is ``['-rwxrwxrwx', 'drwxrwxrwx+']``
        * ``a.fields(1,0)`` is ``['1 -rwxrwxrwx', '6 drwxrwxrwx+']``
          (note the joining by space).
        * ``a.fields(-1)`` is ``['ChangeLog', 'IPython']``

        IndexErrors are ignored.

        Without args, fields() just split()'s the strings.
        """
        if len(fields) == 0:
            return [el.split() for el in self]

        res = SList()
        for el in [f.split() for f in self]:
            lineparts = []

            for fd in fields:
                try:
                    lineparts.append(el[fd])
                except IndexError:
                    pass
            if lineparts:
                res.append(" ".join(lineparts))

        return res

    def sort(self,field= None,  nums = False):
        """ sort by specified fields (see fields())

        Example::

            a.sort(1, nums = True)

        Sorts a by second field, in numerical order (so that 21 > 3)

        """

        #decorate, sort, undecorate
        if field is not None:
            dsu = [[SList([line]).fields(field),  line] for line in self]
        else:
            dsu = [[line,  line] for line in self]
        if nums:
            for i in range(len(dsu)):
                numstr = "".join([ch for ch in dsu[i][0] if ch.isdigit()])
                try:
                    n = int(numstr)
                except ValueError:
                    n = 0
                dsu[i][0] = n


        dsu.sort()
        return SList([t[1] for t in dsu])


# FIXME: We need to reimplement type specific displayhook and then add this
# back as a custom printer. This should also be moved outside utils into the
# core.

# def print_slist(arg):
#     """ Prettier (non-repr-like) and more informative printer for SList """
#     print "SList (.p, .n, .l, .s, .grep(), .fields(), sort() available):"
#     if hasattr(arg,  'hideonce') and arg.hideonce:
#         arg.hideonce = False
#         return
#
#     nlprint(arg)   # This was a nested list printer, now removed.
#
# print_slist = result_display.register(SList)(print_slist)


def indent(instr,nspaces=4, ntabs=0, flatten=False):
    """Indent a string a given number of spaces or tabstops.

    indent(str,nspaces=4,ntabs=0) -> indent str by ntabs+nspaces.

    Parameters
    ----------
    instr : basestring
        The string to be indented.
    nspaces : int (default: 4)
        The number of spaces to be indented.
    ntabs : int (default: 0)
        The number of tabs to be indented.
    flatten : bool (default: False)
        Whether to scrub existing indentation.  If True, all lines will be
        aligned to the same indentation.  If False, existing indentation will
        be strictly increased.

    Returns
    -------
    str|unicode : string indented by ntabs and nspaces.

    """
    if instr is None:
        return
    ind = '\t'*ntabs+' '*nspaces
    if flatten:
        pat = re.compile(r'^\s*', re.MULTILINE)
    else:
        pat = re.compile(r'^', re.MULTILINE)
    outstr = re.sub(pat, ind, instr)
    if outstr.endswith(os.linesep+ind):
        return outstr[:-len(ind)]
    else:
        return outstr


def list_strings(arg):
    """Always return a list of strings, given a string or list of strings
    as input.

    Examples
    --------
    ::

        In [7]: list_strings('A single string')
        Out[7]: ['A single string']

        In [8]: list_strings(['A single string in a list'])
        Out[8]: ['A single string in a list']

        In [9]: list_strings(['A','list','of','strings'])
        Out[9]: ['A', 'list', 'of', 'strings']
    """

    if isinstance(arg, str):
        return [arg]
    else:
        return arg


def marquee(txt='',width=78,mark='*'):
    """Return the input string centered in a 'marquee'.

    Examples
    --------
    ::

        In [16]: marquee('A test',40)
        Out[16]: '**************** A test ****************'

        In [17]: marquee('A test',40,'-')
        Out[17]: '---------------- A test ----------------'

        In [18]: marquee('A test',40,' ')
        Out[18]: '                 A test                 '

    """
    if not txt:
        return (mark*width)[:width]
    nmark = (width-len(txt)-2)//len(mark)//2
    if nmark < 0: nmark =0
    marks = mark*nmark
    return '%s %s %s' % (marks,txt,marks)


ini_spaces_re = re.compile(r'^(\s+)')

def num_ini_spaces(strng):
    """Return the number of initial spaces in a string"""

    ini_spaces = ini_spaces_re.match(strng)
    if ini_spaces:
        return ini_spaces.end()
    else:
        return 0


def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    # Paragraph continue
    par_re = re.compile(r'\\$',re.MULTILINE)
    strng = par_re.sub('',strng)
    return strng


def dedent(text):
    """Equivalent of textwrap.dedent that ignores unindented first line.

    This means it will still dedent strings like:
    '''foo
    is a bar
    '''

    For use in wrap_paragraphs.
    """

    if text.startswith('\n'):
        # text starts with blank line, don't ignore the first line
        return textwrap.dedent(text)

    # split first line
    splits = text.split('\n',1)
    if len(splits) == 1:
        # only one line
        return textwrap.dedent(text)

    first, rest = splits
    # dedent everything but the first line
    rest = textwrap.dedent(rest)
    return '\n'.join([first, rest])


def wrap_paragraphs(text, ncols=80):
    """Wrap multiple paragraphs to fit a specified width.

    This is equivalent to textwrap.wrap, but with support for multiple
    paragraphs, as separated by empty lines.

    Returns
    -------
    list of complete paragraphs, wrapped to fill `ncols` columns.
    """
    paragraph_re = re.compile(r'\n(\s*\n)+', re.MULTILINE)
    text = dedent(text).strip()
    paragraphs = paragraph_re.split(text)[::2] # every other entry is space
    out_ps = []
    indent_re = re.compile(r'\n\s+', re.MULTILINE)
    for p in paragraphs:
        # presume indentation that survives dedent is meaningful formatting,
        # so don't fill unless text is flush.
        if indent_re.search(p) is None:
            # wrap paragraph
            p = textwrap.fill(p, ncols)
        out_ps.append(p)
    return out_ps


def strip_email_quotes(text):
    """Strip leading email quotation characters ('>').

    Removes any combination of leading '>' interspersed with whitespace that
    appears *identically* in all lines of the input text.

    Parameters
    ----------
    text : str

    Examples
    --------

    Simple uses::

        In [2]: strip_email_quotes('> > text')
        Out[2]: 'text'

        In [3]: strip_email_quotes('> > text\\n> > more')
        Out[3]: 'text\\nmore'

    Note how only the common prefix that appears in all lines is stripped::

        In [4]: strip_email_quotes('> > text\\n> > more\\n> more...')
        Out[4]: '> text\\n> more\\nmore...'

    So if any line has no quote marks ('>'), then none are stripped from any
    of them ::

        In [5]: strip_email_quotes('> > text\\n> > more\\nlast different')
        Out[5]: '> > text\\n> > more\\nlast different'
    """
    lines = text.splitlines()
    strip_len = 0

    for characters in zip(*lines):
        # Check if all characters in this position are the same
        if len(set(characters)) > 1:
            break
        prefix_char = characters[0]

        if prefix_char in string.whitespace or prefix_char == ">":
            strip_len += 1
        else:
            break

    text = "\n".join([ln[strip_len:] for ln in lines])
    return text


def strip_ansi(source):
    """
    Remove ansi escape codes from text.

    Parameters
    ----------
    source : str
        Source to remove the ansi from
    """
    return re.sub(r'\033\[(\d|;)+?m', '', source)


class EvalFormatter(Formatter):
    """A String Formatter that allows evaluation of simple expressions.

    Note that this version interprets a `:`  as specifying a format string (as per
    standard string formatting), so if slicing is required, you must explicitly
    create a slice.

    This is to be used in templating cases, such as the parallel batch
    script templates, where simple arithmetic on arguments is useful.

    Examples
    --------
    ::

        In [1]: f = EvalFormatter()
        In [2]: f.format('{n//4}', n=8)
        Out[2]: '2'

        In [3]: f.format("{greeting[slice(2,4)]}", greeting="Hello")
        Out[3]: 'll'
    """
    def get_field(self, name, args, kwargs):
        v = eval(name, kwargs)
        return v, name

#XXX: As of Python 3.4, the format string parsing no longer splits on a colon
# inside [], so EvalFormatter can handle slicing. Once we only support 3.4 and
# above, it should be possible to remove FullEvalFormatter.

class FullEvalFormatter(Formatter):
    """A String Formatter that allows evaluation of simple expressions.
    
    Any time a format key is not found in the kwargs,
    it will be tried as an expression in the kwargs namespace.
    
    Note that this version allows slicing using [1:2], so you cannot specify
    a format string. Use :class:`EvalFormatter` to permit format strings.
    
    Examples
    --------
    ::

        In [1]: f = FullEvalFormatter()
        In [2]: f.format('{n//4}', n=8)
        Out[2]: '2'

        In [3]: f.format('{list(range(5))[2:4]}')
        Out[3]: '[2, 3]'

        In [4]: f.format('{3*2}')
        Out[4]: '6'
    """
    # copied from Formatter._vformat with minor changes to allow eval
    # and replace the format_spec code with slicing
    def vformat(self, format_string:str, args, kwargs)->str:
        result = []
        for literal_text, field_name, format_spec, conversion in \
                self.parse(format_string):

            # output the literal text
            if literal_text:
                result.append(literal_text)

            # if there's a field, output it
            if field_name is not None:
                # this is some markup, find the object and do
                # the formatting

                if format_spec:
                    # override format spec, to allow slicing:
                    field_name = ':'.join([field_name, format_spec])

                # eval the contents of the field for the object
                # to be formatted
                obj = eval(field_name, kwargs)

                # do any conversion on the resulting object
                obj = self.convert_field(obj, conversion)

                # format the object and append to the result
                result.append(self.format_field(obj, ''))

        return ''.join(result)


class DollarFormatter(FullEvalFormatter):
    """Formatter allowing Itpl style $foo replacement, for names and attribute
    access only. Standard {foo} replacement also works, and allows full
    evaluation of its arguments.

    Examples
    --------
    ::

        In [1]: f = DollarFormatter()
        In [2]: f.format('{n//4}', n=8)
        Out[2]: '2'

        In [3]: f.format('23 * 76 is $result', result=23*76)
        Out[3]: '23 * 76 is 1748'

        In [4]: f.format('$a or {b}', a=1, b=2)
        Out[4]: '1 or 2'
    """
    _dollar_pattern_ignore_single_quote = re.compile(r"(.*?)\$(\$?[\w\.]+)(?=([^']*'[^']*')*[^']*$)")
    def parse(self, fmt_string):
        for literal_txt, field_name, format_spec, conversion \
                    in Formatter.parse(self, fmt_string):
            
            # Find $foo patterns in the literal text.
            continue_from = 0
            txt = ""
            for m in self._dollar_pattern_ignore_single_quote.finditer(literal_txt):
                new_txt, new_field = m.group(1,2)
                # $$foo --> $foo
                if new_field.startswith("$"):
                    txt += new_txt + new_field
                else:
                    yield (txt + new_txt, new_field, "", None)
                    txt = ""
                continue_from = m.end()
            
            # Re-yield the {foo} style pattern
            yield (txt + literal_txt[continue_from:], field_name, format_spec, conversion)

    def __repr__(self):
        return "<DollarFormatter>"

#-----------------------------------------------------------------------------
# Utils to columnize a list of string
#-----------------------------------------------------------------------------

def _col_chunks(l, max_rows, row_first=False):
    """Yield successive max_rows-sized column chunks from l."""
    if row_first:
        ncols = (len(l) // max_rows) + (len(l) % max_rows > 0)
        for i in range(ncols):
            yield [l[j] for j in range(i, len(l), ncols)]
    else:
        for i in range(0, len(l), max_rows):
            yield l[i:(i + max_rows)]


def _find_optimal(rlist, row_first=False, separator_size=2, displaywidth=80):
    """Calculate optimal info to columnize a list of string"""
    for max_rows in range(1, len(rlist) + 1):
        col_widths = list(map(max, _col_chunks(rlist, max_rows, row_first)))
        sumlength = sum(col_widths)
        ncols = len(col_widths)
        if sumlength + separator_size * (ncols - 1) <= displaywidth:
            break
    return {'num_columns': ncols,
            'optimal_separator_width': (displaywidth - sumlength) // (ncols - 1) if (ncols - 1) else 0,
            'max_rows': max_rows,
            'column_widths': col_widths
            }


def _get_or_default(mylist, i, default=None):
    """return list item number, or default if don't exist"""
    if i >= len(mylist):
        return default
    else :
        return mylist[i]


def compute_item_matrix(items, row_first=False, empty=None, *args, **kwargs) :
    """Returns a nested list, and info to columnize items

    Parameters
    ----------
    items
        list of strings to columize
    row_first : (default False)
        Whether to compute columns for a row-first matrix instead of
        column-first (default).
    empty : (default None)
        default value to fill list if needed
    separator_size : int (default=2)
        How much characters will be used as a separation between each columns.
    displaywidth : int (default=80)
        The width of the area onto which the columns should enter

    Returns
    -------
    strings_matrix
        nested list of string, the outer most list contains as many list as
        rows, the innermost lists have each as many element as columns. If the
        total number of elements in `items` does not equal the product of
        rows*columns, the last element of some lists are filled with `None`.
    dict_info
        some info to make columnize easier:

        num_columns
          number of columns
        max_rows
          maximum number of rows (final number may be less)
        column_widths
          list of with of each columns
        optimal_separator_width
          best separator width between columns

    Examples
    --------
    ::

        In [1]: l = ['aaa','b','cc','d','eeeee','f','g','h','i','j','k','l']
        In [2]: list, info = compute_item_matrix(l, displaywidth=12)
        In [3]: list
        Out[3]: [['aaa', 'f', 'k'], ['b', 'g', 'l'], ['cc', 'h', None], ['d', 'i', None], ['eeeee', 'j', None]]
        In [4]: ideal = {'num_columns': 3, 'column_widths': [5, 1, 1], 'optimal_separator_width': 2, 'max_rows': 5}
        In [5]: all((info[k] == ideal[k] for k in ideal.keys()))
        Out[5]: True
    """
    info = _find_optimal(list(map(len, items)), row_first, *args, **kwargs)
    nrow, ncol = info['max_rows'], info['num_columns']
    if row_first:
        return ([[_get_or_default(items, r * ncol + c, default=empty) for c in range(ncol)] for r in range(nrow)], info)
    else:
        return ([[_get_or_default(items, c * nrow + r, default=empty) for c in range(ncol)] for r in range(nrow)], info)


def columnize(items, row_first=False, separator="  ", displaywidth=80, spread=False):
    """Transform a list of strings into a single string with columns.

    Parameters
    ----------
    items : sequence of strings
        The strings to process.
    row_first : (default False)
        Whether to compute columns for a row-first matrix instead of
        column-first (default).
    separator : str, optional [default is two spaces]
        The string that separates columns.
    displaywidth : int, optional [default is 80]
        Width of the display in number of characters.

    Returns
    -------
    The formatted string.
    """
    if not items:
        return '\n'
    matrix, info = compute_item_matrix(items, row_first=row_first, separator_size=len(separator), displaywidth=displaywidth)
    if spread:
        separator = separator.ljust(int(info['optimal_separator_width']))
    fmatrix = [filter(None, x) for x in matrix]
    sjoin = lambda x : separator.join([ y.ljust(w, ' ') for y, w in zip(x, info['column_widths'])])
    return '\n'.join(map(sjoin, fmatrix))+'\n'


def get_text_list(list_, last_sep=' and ', sep=", ", wrap_item_with=""):
    """
    Return a string with a natural enumeration of items

    >>> get_text_list(['a', 'b', 'c', 'd'])
    'a, b, c and d'
    >>> get_text_list(['a', 'b', 'c'], ' or ')
    'a, b or c'
    >>> get_text_list(['a', 'b', 'c'], ', ')
    'a, b, c'
    >>> get_text_list(['a', 'b'], ' or ')
    'a or b'
    >>> get_text_list(['a'])
    'a'
    >>> get_text_list([])
    ''
    >>> get_text_list(['a', 'b'], wrap_item_with="`")
    '`a` and `b`'
    >>> get_text_list(['a', 'b', 'c', 'd'], " = ", sep=" + ")
    'a + b + c = d'
    """
    if len(list_) == 0:
        return ''
    if wrap_item_with:
        list_ = ['%s%s%s' % (wrap_item_with, item, wrap_item_with) for
                 item in list_]
    if len(list_) == 1:
        return list_[0]
    return '%s%s%s' % (
        sep.join(i for i in list_[:-1]),
        last_sep, list_[-1])
