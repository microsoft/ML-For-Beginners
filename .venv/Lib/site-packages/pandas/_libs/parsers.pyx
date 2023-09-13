# Copyright (c) 2012, Lambda Foundry, Inc.
# See LICENSE for the license
from collections import defaultdict
from csv import (
    QUOTE_MINIMAL,
    QUOTE_NONE,
    QUOTE_NONNUMERIC,
)
import sys
import time
import warnings

from pandas.util._exceptions import find_stack_level

from pandas import StringDtype
from pandas.core.arrays import (
    ArrowExtensionArray,
    BooleanArray,
    FloatingArray,
    IntegerArray,
)

cimport cython
from cpython.bytes cimport PyBytes_AsString
from cpython.exc cimport (
    PyErr_Fetch,
    PyErr_Occurred,
)
from cpython.object cimport PyObject
from cpython.ref cimport (
    Py_INCREF,
    Py_XDECREF,
)
from cpython.unicode cimport (
    PyUnicode_AsUTF8String,
    PyUnicode_Decode,
    PyUnicode_DecodeUTF8,
)
from cython cimport Py_ssize_t
from libc.stdlib cimport free
from libc.string cimport (
    strcasecmp,
    strlen,
    strncpy,
)


cdef extern from "Python.h":
    # TODO(cython3): get this from cpython.unicode
    object PyUnicode_FromString(char *v)


import numpy as np

cimport numpy as cnp
from numpy cimport (
    float64_t,
    int64_t,
    ndarray,
    uint8_t,
    uint64_t,
)

cnp.import_array()

from pandas._libs cimport util
from pandas._libs.util cimport (
    INT64_MAX,
    INT64_MIN,
    UINT64_MAX,
)

from pandas._libs import lib

from pandas._libs.khash cimport (
    kh_destroy_float64,
    kh_destroy_str,
    kh_destroy_str_starts,
    kh_destroy_strbox,
    kh_exist_str,
    kh_float64_t,
    kh_get_float64,
    kh_get_str,
    kh_get_str_starts_item,
    kh_get_strbox,
    kh_init_float64,
    kh_init_str,
    kh_init_str_starts,
    kh_init_strbox,
    kh_put_float64,
    kh_put_str,
    kh_put_str_starts_item,
    kh_put_strbox,
    kh_resize_float64,
    kh_resize_str_starts,
    kh_str_starts_t,
    kh_str_t,
    kh_strbox_t,
    khiter_t,
)

from pandas.errors import (
    EmptyDataError,
    ParserError,
    ParserWarning,
)

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.inference import is_dict_like

from pandas.core.arrays.boolean import BooleanDtype

cdef:
    float64_t INF = <float64_t>np.inf
    float64_t NEGINF = -INF
    int64_t DEFAULT_CHUNKSIZE = 256 * 1024

DEFAULT_BUFFER_HEURISTIC = 2 ** 20


cdef extern from "pandas/portable.h":
    # I *think* this is here so that strcasecmp is defined on Windows
    # so we don't get
    # `parsers.obj : error LNK2001: unresolved external symbol strcasecmp`
    # in Appveyor.
    # In a sane world, the `from libc.string cimport` above would fail
    # loudly.
    pass


cdef extern from "pandas/parser/tokenizer.h":

    ctypedef enum ParserState:
        START_RECORD
        START_FIELD
        ESCAPED_CHAR
        IN_FIELD
        IN_QUOTED_FIELD
        ESCAPE_IN_QUOTED_FIELD
        QUOTE_IN_QUOTED_FIELD
        EAT_CRNL
        EAT_CRNL_NOP
        EAT_WHITESPACE
        EAT_COMMENT
        EAT_LINE_COMMENT
        WHITESPACE_LINE
        SKIP_LINE
        FINISHED

    enum: ERROR_OVERFLOW

    ctypedef enum BadLineHandleMethod:
        ERROR,
        WARN,
        SKIP

    ctypedef void* (*io_callback)(void *src, size_t nbytes, size_t *bytes_read,
                                  int *status, const char *encoding_errors)
    ctypedef int (*io_cleanup)(void *src)

    ctypedef struct parser_t:
        void *source
        io_callback cb_io
        io_cleanup cb_cleanup

        int64_t chunksize  # Number of bytes to prepare for each chunk
        char *data         # pointer to data to be processed
        int64_t datalen    # amount of data available
        int64_t datapos

        # where to write out tokenized data
        char *stream
        uint64_t stream_len
        uint64_t stream_cap

        # Store words in (potentially ragged) matrix for now, hmm
        char **words
        int64_t *word_starts  # where we are in the stream
        uint64_t words_len
        uint64_t words_cap
        uint64_t max_words_cap   # maximum word cap encountered

        char *pword_start        # pointer to stream start of current field
        int64_t word_start       # position start of current field

        int64_t *line_start      # position in words for start of line
        int64_t *line_fields     # Number of fields in each line
        uint64_t lines           # Number of lines observed
        uint64_t file_lines      # Number of lines observed (with bad/skipped)
        uint64_t lines_cap       # Vector capacity

        # Tokenizing stuff
        ParserState state
        int doublequote            # is " represented by ""? */
        char delimiter             # field separator */
        int delim_whitespace       # consume tabs / spaces instead
        char quotechar             # quote character */
        char escapechar            # escape character */
        char lineterminator
        int skipinitialspace       # ignore spaces following delimiter? */
        int quoting                # style of quoting to write */

        char commentchar
        int allow_embedded_newline

        int usecols

        Py_ssize_t expected_fields
        BadLineHandleMethod on_bad_lines

        # floating point options
        char decimal
        char sci

        # thousands separator (comma, period)
        char thousands

        int header                  # Boolean: 1: has header, 0: no header
        int64_t header_start        # header row start
        uint64_t header_end         # header row end

        void *skipset
        PyObject *skipfunc
        int64_t skip_first_N_rows
        int64_t skipfooter
        # pick one, depending on whether the converter requires GIL
        double (*double_converter)(const char *, char **,
                                   char, char, char,
                                   int, int *, int *) nogil

        #  error handling
        char *warn_msg
        char *error_msg

        int64_t skip_empty_lines

    ctypedef struct coliter_t:
        char **words
        int64_t *line_start
        int64_t col

    ctypedef struct uint_state:
        int seen_sint
        int seen_uint
        int seen_null

    void COLITER_NEXT(coliter_t, const char *) nogil

cdef extern from "pandas/parser/pd_parser.h":
    void *new_rd_source(object obj) except NULL

    int del_rd_source(void *src)

    void* buffer_rd_bytes(void *source, size_t nbytes,
                          size_t *bytes_read, int *status, const char *encoding_errors)

    void uint_state_init(uint_state *self)
    int uint64_conflict(uint_state *self)

    void coliter_setup(coliter_t *it, parser_t *parser,
                       int64_t i, int64_t start) nogil
    void COLITER_NEXT(coliter_t, const char *) nogil

    parser_t* parser_new()

    int parser_init(parser_t *self) nogil
    void parser_free(parser_t *self) nogil
    void parser_del(parser_t *self) nogil
    int parser_add_skiprow(parser_t *self, int64_t row)

    int parser_set_skipfirstnrows(parser_t *self, int64_t nrows)

    void parser_set_default_options(parser_t *self)

    int parser_consume_rows(parser_t *self, size_t nrows)

    int parser_trim_buffers(parser_t *self)

    int tokenize_all_rows(parser_t *self, const char *encoding_errors) nogil
    int tokenize_nrows(parser_t *self, size_t nrows, const char *encoding_errors) nogil

    int64_t str_to_int64(char *p_item, int64_t int_min,
                         int64_t int_max, int *error, char tsep) nogil
    uint64_t str_to_uint64(uint_state *state, char *p_item, int64_t int_max,
                           uint64_t uint_max, int *error, char tsep) nogil

    double xstrtod(const char *p, char **q, char decimal,
                   char sci, char tsep, int skip_trailing,
                   int *error, int *maybe_int) nogil
    double precise_xstrtod(const char *p, char **q, char decimal,
                           char sci, char tsep, int skip_trailing,
                           int *error, int *maybe_int) nogil
    double round_trip(const char *p, char **q, char decimal,
                      char sci, char tsep, int skip_trailing,
                      int *error, int *maybe_int) nogil

    int to_boolean(const char *item, uint8_t *val) nogil

    void PandasParser_IMPORT()

PandasParser_IMPORT

# When not invoked directly but rather assigned as a function,
# cdef extern'ed declarations seem to leave behind an undefined symbol
cdef double xstrtod_wrapper(const char *p, char **q, char decimal,
                            char sci, char tsep, int skip_trailing,
                            int *error, int *maybe_int) noexcept nogil:
    return xstrtod(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)


cdef double precise_xstrtod_wrapper(const char *p, char **q, char decimal,
                                    char sci, char tsep, int skip_trailing,
                                    int *error, int *maybe_int) noexcept nogil:
    return precise_xstrtod(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)


cdef double round_trip_wrapper(const char *p, char **q, char decimal,
                               char sci, char tsep, int skip_trailing,
                               int *error, int *maybe_int) noexcept nogil:
    return round_trip(p, q, decimal, sci, tsep, skip_trailing, error, maybe_int)


cdef void* buffer_rd_bytes_wrapper(void *source, size_t nbytes,
                                   size_t *bytes_read, int *status,
                                   const char *encoding_errors) noexcept:
    return buffer_rd_bytes(source, nbytes, bytes_read, status, encoding_errors)

cdef int del_rd_source_wrapper(void *src) noexcept:
    return del_rd_source(src)


cdef class TextReader:
    """

    # source: StringIO or file object

    ..versionchange:: 1.2.0
        removed 'compression', 'memory_map', and 'encoding' argument.
        These arguments are outsourced to CParserWrapper.
        'source' has to be a file handle.
    """

    cdef:
        parser_t *parser
        object na_fvalues
        object true_values, false_values
        object handle
        object orig_header
        bint na_filter, keep_default_na, verbose, has_usecols, has_mi_columns
        bint allow_leading_cols
        uint64_t parser_start  # this is modified after __init__
        list clocks
        const char *encoding_errors
        kh_str_starts_t *false_set
        kh_str_starts_t *true_set
        int64_t buffer_lines, skipfooter
        list dtype_cast_order  # list[np.dtype]
        list names   # can be None
        set noconvert  # set[int]

    cdef public:
        int64_t leading_cols, table_width
        object delimiter  # bytes or str
        object converters
        object na_values
        list header  # list[list[non-negative integers]]
        object index_col
        object skiprows
        object dtype
        object usecols
        set unnamed_cols  # set[str]
        str dtype_backend

    def __cinit__(self, source,
                  delimiter=b",",  # bytes | str
                  header=0,
                  int64_t header_start=0,
                  uint64_t header_end=0,
                  index_col=None,
                  names=None,
                  tokenize_chunksize=DEFAULT_CHUNKSIZE,
                  bint delim_whitespace=False,
                  converters=None,
                  bint skipinitialspace=False,
                  escapechar=None,      # bytes | str
                  bint doublequote=True,
                  quotechar=b'"',
                  quoting=0,            # int
                  lineterminator=None,  # bytes | str
                  comment=None,
                  decimal=b".",         # bytes | str
                  thousands=None,       # bytes | str
                  dtype=None,
                  usecols=None,
                  on_bad_lines=ERROR,
                  bint na_filter=True,
                  na_values=None,
                  na_fvalues=None,
                  bint keep_default_na=True,
                  true_values=None,
                  false_values=None,
                  bint allow_leading_cols=True,
                  skiprows=None,
                  skipfooter=0,         # int64_t
                  bint verbose=False,
                  float_precision=None,
                  bint skip_blank_lines=True,
                  encoding_errors=b"strict",
                  dtype_backend="numpy"):

        # set encoding for native Python and C library
        if isinstance(encoding_errors, str):
            encoding_errors = encoding_errors.encode("utf-8")
        elif encoding_errors is None:
            encoding_errors = b"strict"
        Py_INCREF(encoding_errors)
        self.encoding_errors = PyBytes_AsString(encoding_errors)

        self.parser = parser_new()
        self.parser.chunksize = tokenize_chunksize

        # For timekeeping
        self.clocks = []

        self.parser.usecols = (usecols is not None)

        self._setup_parser_source(source)
        parser_set_default_options(self.parser)

        parser_init(self.parser)

        if delim_whitespace:
            self.parser.delim_whitespace = delim_whitespace
        else:
            if len(delimiter) > 1:
                raise ValueError("only length-1 separators excluded right now")
            self.parser.delimiter = <char>ord(delimiter)

        # ----------------------------------------
        # parser options

        self.parser.doublequote = doublequote
        self.parser.skipinitialspace = skipinitialspace
        self.parser.skip_empty_lines = skip_blank_lines

        if lineterminator is not None:
            if len(lineterminator) != 1:
                raise ValueError("Only length-1 line terminators supported")
            self.parser.lineterminator = <char>ord(lineterminator)

        if len(decimal) != 1:
            raise ValueError("Only length-1 decimal markers supported")
        self.parser.decimal = <char>ord(decimal)

        if thousands is not None:
            if len(thousands) != 1:
                raise ValueError("Only length-1 thousands markers supported")
            self.parser.thousands = <char>ord(thousands)

        if escapechar is not None:
            if len(escapechar) != 1:
                raise ValueError("Only length-1 escapes supported")
            self.parser.escapechar = <char>ord(escapechar)

        self._set_quoting(quotechar, quoting)

        dtype_order = ["int64", "float64", "bool", "object"]
        if quoting == QUOTE_NONNUMERIC:
            # consistent with csv module semantics, cast all to float
            dtype_order = dtype_order[1:]
        self.dtype_cast_order = [np.dtype(x) for x in dtype_order]

        if comment is not None:
            if len(comment) > 1:
                raise ValueError("Only length-1 comment characters supported")
            self.parser.commentchar = <char>ord(comment)

        self.parser.on_bad_lines = on_bad_lines

        self.skiprows = skiprows
        if skiprows is not None:
            self._make_skiprow_set()

        self.skipfooter = skipfooter

        if usecols is not None:
            self.has_usecols = 1
            # GH-20558, validate usecols at higher level and only pass clean
            # usecols into TextReader.
            self.usecols = usecols

        if skipfooter > 0:
            self.parser.on_bad_lines = SKIP

        self.delimiter = delimiter

        self.na_values = na_values
        if na_fvalues is None:
            na_fvalues = set()
        self.na_fvalues = na_fvalues

        self.true_values = _maybe_encode(true_values) + _true_values
        self.false_values = _maybe_encode(false_values) + _false_values

        self.true_set = kset_from_list(self.true_values)
        self.false_set = kset_from_list(self.false_values)

        self.keep_default_na = keep_default_na
        self.converters = converters
        self.na_filter = na_filter

        self.verbose = verbose

        if float_precision == "round_trip":
            # see gh-15140
            self.parser.double_converter = round_trip_wrapper
        elif float_precision == "legacy":
            self.parser.double_converter = xstrtod_wrapper
        elif float_precision == "high" or float_precision is None:
            self.parser.double_converter = precise_xstrtod_wrapper
        else:
            raise ValueError(f"Unrecognized float_precision option: "
                             f"{float_precision}")

        # Caller is responsible for ensuring we have one of
        # - None
        # - DtypeObj
        # - dict[Any, DtypeObj]
        self.dtype = dtype
        self.dtype_backend = dtype_backend

        self.noconvert = set()

        self.index_col = index_col

        # ----------------------------------------
        # header stuff

        self.allow_leading_cols = allow_leading_cols
        self.leading_cols = 0  # updated in _get_header

        # TODO: no header vs. header is not the first row
        self.has_mi_columns = 0
        self.orig_header = header
        if header is None:
            # sentinel value
            self.parser.header_start = -1
            self.parser.header_end = -1
            self.parser.header = -1
            self.parser_start = 0
            prelim_header = []
        else:
            if isinstance(header, list):
                if len(header) > 1:
                    # need to artificially skip the final line
                    # which is still a header line
                    header = list(header)
                    header.append(header[-1] + 1)
                    self.parser.header_end = header[-1]
                    self.has_mi_columns = 1
                else:
                    self.parser.header_end = header[0]

                self.parser_start = header[-1] + 1
                self.parser.header_start = header[0]
                self.parser.header = header[0]
                prelim_header = header
            else:
                self.parser.header_start = header
                self.parser.header_end = header
                self.parser_start = header + 1
                self.parser.header = header
                prelim_header = [header]

        self.names = names
        header, table_width, unnamed_cols = self._get_header(prelim_header)
        # header, table_width, and unnamed_cols are set here, never changed
        self.header = header
        self.table_width = table_width
        self.unnamed_cols = unnamed_cols

        if not self.table_width:
            raise EmptyDataError("No columns to parse from file")

        # Compute buffer_lines as function of table width.
        heuristic = DEFAULT_BUFFER_HEURISTIC // self.table_width
        self.buffer_lines = 1
        while self.buffer_lines * 2 < heuristic:
            self.buffer_lines *= 2

    def __init__(self, *args, **kwargs):
        pass

    def __dealloc__(self):
        _close(self)
        parser_del(self.parser)

    def close(self):
        _close(self)

    def _set_quoting(self, quote_char: str | bytes | None, quoting: int):
        if not isinstance(quoting, int):
            raise TypeError('"quoting" must be an integer')

        if not QUOTE_MINIMAL <= quoting <= QUOTE_NONE:
            raise TypeError('bad "quoting" value')

        if not isinstance(quote_char, (str, bytes)) and quote_char is not None:
            dtype = type(quote_char).__name__
            raise TypeError(f'"quotechar" must be string, not {dtype}')

        if quote_char is None or quote_char == "":
            if quoting != QUOTE_NONE:
                raise TypeError("quotechar must be set if quoting enabled")
            self.parser.quoting = quoting
            self.parser.quotechar = -1
        elif len(quote_char) > 1:  # 0-len case handled earlier
            raise TypeError('"quotechar" must be a 1-character string')
        else:
            self.parser.quoting = quoting
            self.parser.quotechar = <char>ord(quote_char)

    cdef _make_skiprow_set(self):
        if util.is_integer_object(self.skiprows):
            parser_set_skipfirstnrows(self.parser, self.skiprows)
        elif not callable(self.skiprows):
            for i in self.skiprows:
                parser_add_skiprow(self.parser, i)
        else:
            self.parser.skipfunc = <PyObject *>self.skiprows

    cdef _setup_parser_source(self, source):
        cdef:
            void *ptr

        ptr = new_rd_source(source)
        self.parser.source = ptr
        self.parser.cb_io = buffer_rd_bytes_wrapper
        self.parser.cb_cleanup = del_rd_source_wrapper

    cdef _get_header(self, list prelim_header):
        # header is now a list of lists, so field_count should use header[0]
        #
        # modifies:
        #   self.parser attributes
        #   self.parser_start
        #   self.leading_cols

        cdef:
            Py_ssize_t i, start, field_count, passed_count, unnamed_count, level
            char *word
            str name
            uint64_t hr, data_line = 0
            list header = []
            set unnamed_cols = set()

        if self.parser.header_start >= 0:

            # Header is in the file
            for level, hr in enumerate(prelim_header):

                this_header = []

                if self.parser.lines < hr + 1:
                    self._tokenize_rows(hr + 2)

                if self.parser.lines == 0:
                    field_count = 0
                    start = self.parser.line_start[0]

                # e.g., if header=3 and file only has 2 lines
                elif (self.parser.lines < hr + 1
                      and not isinstance(self.orig_header, list)) or (
                          self.parser.lines < hr):
                    msg = self.orig_header
                    if isinstance(msg, list):
                        joined = ",".join(str(m) for m in msg)
                        msg = f"[{joined}], len of {len(msg)},"
                    raise ParserError(
                        f"Passed header={msg} but only "
                        f"{self.parser.lines} lines in file")

                else:
                    field_count = self.parser.line_fields[hr]
                    start = self.parser.line_start[hr]

                unnamed_count = 0
                unnamed_col_indices = []

                for i in range(field_count):
                    word = self.parser.words[start + i]

                    name = PyUnicode_DecodeUTF8(word, strlen(word),
                                                self.encoding_errors)

                    if name == "":
                        if self.has_mi_columns:
                            name = f"Unnamed: {i}_level_{level}"
                        else:
                            name = f"Unnamed: {i}"

                        unnamed_count += 1
                        unnamed_col_indices.append(i)

                    this_header.append(name)

                if not self.has_mi_columns:
                    # Ensure that regular columns are used before unnamed ones
                    # to keep given names and mangle unnamed columns
                    col_loop_order = [i for i in range(len(this_header))
                                      if i not in unnamed_col_indices
                                      ] + unnamed_col_indices
                    counts = {}

                    for i in col_loop_order:
                        col = this_header[i]
                        old_col = col
                        cur_count = counts.get(col, 0)

                        if cur_count > 0:
                            while cur_count > 0:
                                counts[old_col] = cur_count + 1
                                col = f"{old_col}.{cur_count}"
                                if col in this_header:
                                    cur_count += 1
                                else:
                                    cur_count = counts.get(col, 0)

                            if (
                                self.dtype is not None
                                and is_dict_like(self.dtype)
                                and self.dtype.get(old_col) is not None
                                and self.dtype.get(col) is None
                            ):
                                self.dtype.update({col: self.dtype.get(old_col)})

                        this_header[i] = col
                        counts[col] = cur_count + 1

                if self.has_mi_columns:

                    # If we have grabbed an extra line, but it's not in our
                    # format, save in the buffer, and create an blank extra
                    # line for the rest of the parsing code.
                    if hr == prelim_header[-1]:
                        lc = len(this_header)
                        ic = (len(self.index_col) if self.index_col
                              is not None else 0)

                        # if wrong number of blanks or no index, not our format
                        if (lc != unnamed_count and lc - ic > unnamed_count) or ic == 0:
                            hr -= 1
                            self.parser_start -= 1
                            this_header = [None] * lc

                data_line = hr + 1
                header.append(this_header)
                unnamed_cols.update({this_header[i] for i in unnamed_col_indices})

            if self.names is not None:
                header = [self.names]

        elif self.names is not None:
            # Names passed
            if self.parser.lines < 1:
                if not self.has_usecols:
                    self.parser.expected_fields = len(self.names)
                self._tokenize_rows(1)

            header = [self.names]

            if self.parser.lines < 1:
                field_count = len(header[0])
            else:
                field_count = self.parser.line_fields[data_line]

            # Enforce this unless usecols
            if not self.has_usecols:
                self.parser.expected_fields = max(field_count, len(self.names))

        else:
            # No header passed nor to be found in the file
            if self.parser.lines < 1:
                self._tokenize_rows(1)

            return None, self.parser.line_fields[0], unnamed_cols

        # Corner case, not enough lines in the file
        if self.parser.lines < data_line + 1:
            field_count = len(header[0])
        else:

            field_count = self.parser.line_fields[data_line]

            # #2981
            if self.names is not None:
                field_count = max(field_count, len(self.names))

            passed_count = len(header[0])

            if (self.has_usecols and self.allow_leading_cols and
                    not callable(self.usecols)):
                nuse = len(self.usecols)
                if nuse == passed_count:
                    self.leading_cols = 0
                elif self.names is None and nuse < passed_count:
                    self.leading_cols = field_count - passed_count
                elif passed_count != field_count:
                    raise ValueError("Number of passed names did not match number of "
                                     "header fields in the file")
            # oh boy, #2442, #2981
            elif self.allow_leading_cols and passed_count < field_count:
                self.leading_cols = field_count - passed_count

        return header, field_count, unnamed_cols

    def read(self, rows: int | None = None) -> dict[int, "ArrayLike"]:
        """
        rows=None --> read all rows
        """
        # Don't care about memory usage
        columns = self._read_rows(rows, 1)

        return columns

    def read_low_memory(self, rows: int | None)-> list[dict[int, "ArrayLike"]]:
        """
        rows=None --> read all rows
        """
        # Conserve intermediate space
        # Caller is responsible for concatenating chunks,
        #  see c_parser_wrapper._concatenate_chunks
        cdef:
            size_t rows_read = 0
            list chunks = []

        if rows is None:
            while True:
                try:
                    chunk = self._read_rows(self.buffer_lines, 0)
                    if len(chunk) == 0:
                        break
                except StopIteration:
                    break
                else:
                    chunks.append(chunk)
        else:
            while rows_read < rows:
                try:
                    crows = min(self.buffer_lines, rows - rows_read)

                    chunk = self._read_rows(crows, 0)
                    if len(chunk) == 0:
                        break

                    rows_read += len(list(chunk.values())[0])
                except StopIteration:
                    break
                else:
                    chunks.append(chunk)

        parser_trim_buffers(self.parser)

        if len(chunks) == 0:
            raise StopIteration

        return chunks

    cdef _tokenize_rows(self, size_t nrows):
        cdef:
            int status

        with nogil:
            status = tokenize_nrows(self.parser, nrows, self.encoding_errors)

        self._check_tokenize_status(status)

    cdef _check_tokenize_status(self, int status):
        if self.parser.warn_msg != NULL:
            print(PyUnicode_DecodeUTF8(
                self.parser.warn_msg, strlen(self.parser.warn_msg),
                self.encoding_errors), file=sys.stderr)
            free(self.parser.warn_msg)
            self.parser.warn_msg = NULL

        if status < 0:
            raise_parser_error("Error tokenizing data", self.parser)

    #  -> dict[int, "ArrayLike"]
    cdef _read_rows(self, rows, bint trim):
        cdef:
            int64_t buffered_lines
            int64_t irows

        self._start_clock()

        if rows is not None:
            irows = rows
            buffered_lines = self.parser.lines - self.parser_start
            if buffered_lines < irows:
                self._tokenize_rows(irows - buffered_lines)

            if self.skipfooter > 0:
                raise ValueError("skipfooter can only be used to read "
                                 "the whole file")
        else:
            with nogil:
                status = tokenize_all_rows(self.parser, self.encoding_errors)

            self._check_tokenize_status(status)

        if self.parser_start >= self.parser.lines:
            raise StopIteration
        self._end_clock("Tokenization")

        self._start_clock()
        columns = self._convert_column_data(rows)
        self._end_clock("Type conversion")
        self._start_clock()
        if len(columns) > 0:
            rows_read = len(list(columns.values())[0])
            # trim
            parser_consume_rows(self.parser, rows_read)
            if trim:
                parser_trim_buffers(self.parser)
            self.parser_start -= rows_read

        self._end_clock("Parser memory cleanup")

        return columns

    cdef _start_clock(self):
        self.clocks.append(time.time())

    cdef _end_clock(self, str what):
        if self.verbose:
            elapsed = time.time() - self.clocks.pop(-1)
            print(f"{what} took: {elapsed * 1000:.2f} ms")

    def set_noconvert(self, i: int) -> None:
        self.noconvert.add(i)

    def remove_noconvert(self, i: int) -> None:
        self.noconvert.remove(i)

    def _convert_column_data(self, rows: int | None) -> dict[int, "ArrayLike"]:
        cdef:
            int64_t i
            int nused
            kh_str_starts_t *na_hashset = NULL
            int64_t start, end
            object name, na_flist, col_dtype = None
            bint na_filter = 0
            int64_t num_cols
            dict results

        start = self.parser_start

        if rows is None:
            end = self.parser.lines
        else:
            end = min(start + rows, self.parser.lines)

        num_cols = -1
        # Py_ssize_t cast prevents build warning
        for i in range(<Py_ssize_t>self.parser.lines):
            num_cols = (num_cols < self.parser.line_fields[i]) * \
                self.parser.line_fields[i] + \
                (num_cols >= self.parser.line_fields[i]) * num_cols

        usecols_not_callable_and_exists = not callable(self.usecols) and self.usecols
        names_larger_num_cols = (self.names and
                                 len(self.names) - self.leading_cols > num_cols)

        if self.table_width - self.leading_cols > num_cols:
            if (usecols_not_callable_and_exists
                    and self.table_width - self.leading_cols < len(self.usecols)
                    or names_larger_num_cols):
                raise ParserError(f"Too many columns specified: expected "
                                  f"{self.table_width - self.leading_cols} "
                                  f"and found {num_cols}")

        if (usecols_not_callable_and_exists and
                all(isinstance(u, int) for u in self.usecols)):
            missing_usecols = [col for col in self.usecols if col >= num_cols]
            if missing_usecols:
                raise ParserError(
                    "Defining usecols without of bounds indices is not allowed. "
                    f"{missing_usecols} are out of bounds.",
                )

        results = {}
        nused = 0
        is_default_dict_dtype = isinstance(self.dtype, defaultdict)

        for i in range(self.table_width):
            if i < self.leading_cols:
                # Pass through leading columns always
                name = i
            elif (self.usecols and not callable(self.usecols) and
                    nused == len(self.usecols)):
                # Once we've gathered all requested columns, stop. GH5766
                break
            else:
                name = self._get_column_name(i, nused)
                usecols = set()
                if callable(self.usecols):
                    if self.usecols(name):
                        usecols = {i}
                else:
                    usecols = self.usecols
                if self.has_usecols and not (i in usecols or
                                             name in usecols):
                    continue
                nused += 1

            conv = self._get_converter(i, name)

            col_dtype = None
            if self.dtype is not None:
                if isinstance(self.dtype, dict):
                    if name in self.dtype:
                        col_dtype = self.dtype[name]
                    elif i in self.dtype:
                        col_dtype = self.dtype[i]
                    elif is_default_dict_dtype:
                        col_dtype = self.dtype[name]
                else:
                    if self.dtype.names:
                        # structured array
                        col_dtype = np.dtype(self.dtype.descr[i][1])
                    else:
                        col_dtype = self.dtype

            if conv:
                if col_dtype is not None:
                    warnings.warn((f"Both a converter and dtype were specified "
                                   f"for column {name} - only the converter will "
                                   f"be used."), ParserWarning,
                                  stacklevel=find_stack_level())
                results[i] = _apply_converter(conv, self.parser, i, start, end)
                continue

            # Collect the list of NaN values associated with the column.
            # If we aren't supposed to do that, or none are collected,
            # we set `na_filter` to `0` (`1` otherwise).
            na_flist = set()

            if self.na_filter:
                na_list, na_flist = self._get_na_list(i, name)
                if na_list is None:
                    na_filter = 0
                else:
                    na_filter = 1
                    na_hashset = kset_from_list(na_list)
            else:
                na_filter = 0

            # Attempt to parse tokens and infer dtype of the column.
            # Should return as the desired dtype (inferred or specified).
            try:
                col_res, na_count = self._convert_tokens(
                    i, start, end, name, na_filter, na_hashset,
                    na_flist, col_dtype)
            finally:
                # gh-21353
                #
                # Cleanup the NaN hash that we generated
                # to avoid memory leaks.
                if na_filter:
                    self._free_na_set(na_hashset)

            # don't try to upcast EAs
            if (
                na_count > 0 and not isinstance(col_dtype, ExtensionDtype)
                or self.dtype_backend != "numpy"
            ):
                use_dtype_backend = self.dtype_backend != "numpy" and col_dtype is None
                col_res = _maybe_upcast(
                    col_res,
                    use_dtype_backend=use_dtype_backend,
                    dtype_backend=self.dtype_backend,
                )

            if col_res is None:
                raise ParserError(f"Unable to parse column {i}")

            results[i] = col_res

        self.parser_start += end - start

        return results

    # -> tuple["ArrayLike", int]:
    cdef _convert_tokens(self, Py_ssize_t i, int64_t start,
                         int64_t end, object name, bint na_filter,
                         kh_str_starts_t *na_hashset,
                         object na_flist, object col_dtype):

        if col_dtype is not None:
            col_res, na_count = self._convert_with_dtype(
                col_dtype, i, start, end, na_filter,
                1, na_hashset, na_flist)

            # Fallback on the parse (e.g. we requested int dtype,
            # but its actually a float).
            if col_res is not None:
                return col_res, na_count

        if i in self.noconvert:
            return self._string_convert(i, start, end, na_filter, na_hashset)
        else:
            col_res = None
            for dt in self.dtype_cast_order:
                try:
                    col_res, na_count = self._convert_with_dtype(
                        dt, i, start, end, na_filter, 0, na_hashset, na_flist)
                except ValueError:
                    # This error is raised from trying to convert to uint64,
                    # and we discover that we cannot convert to any numerical
                    # dtype successfully. As a result, we leave the data
                    # column AS IS with object dtype.
                    col_res, na_count = self._convert_with_dtype(
                        np.dtype("object"), i, start, end, 0,
                        0, na_hashset, na_flist)
                except OverflowError:
                    col_res, na_count = self._convert_with_dtype(
                        np.dtype("object"), i, start, end, na_filter,
                        0, na_hashset, na_flist)

                if col_res is not None:
                    break

        # we had a fallback parse on the dtype, so now try to cast
        if col_res is not None and col_dtype is not None:
            # If col_res is bool, it might actually be a bool array mixed with NaNs
            # (see _try_bool_flex()). Usually this would be taken care of using
            # _maybe_upcast(), but if col_dtype is a floating type we should just
            # take care of that cast here.
            if col_res.dtype == np.bool_ and col_dtype.kind == "f":
                mask = col_res.view(np.uint8) == na_values[np.uint8]
                col_res = col_res.astype(col_dtype)
                np.putmask(col_res, mask, np.nan)
                return col_res, na_count

            # NaNs are already cast to True here, so can not use astype
            if col_res.dtype == np.bool_ and col_dtype.kind in "iu":
                if na_count > 0:
                    raise ValueError(
                        f"cannot safely convert passed user dtype of "
                        f"{col_dtype} for {np.bool_} dtyped data in "
                        f"column {i} due to NA values"
                    )

            # only allow safe casts, eg. with a nan you cannot safely cast to int
            try:
                col_res = col_res.astype(col_dtype, casting="safe")
            except TypeError:

                # float -> int conversions can fail the above
                # even with no nans
                col_res_orig = col_res
                col_res = col_res.astype(col_dtype)
                if (col_res != col_res_orig).any():
                    raise ValueError(
                        f"cannot safely convert passed user dtype of "
                        f"{col_dtype} for {col_res_orig.dtype.name} dtyped data in "
                        f"column {i}")

        return col_res, na_count

    cdef _convert_with_dtype(self, object dtype, Py_ssize_t i,
                             int64_t start, int64_t end,
                             bint na_filter,
                             bint user_dtype,
                             kh_str_starts_t *na_hashset,
                             object na_flist):
        if isinstance(dtype, CategoricalDtype):
            # TODO: I suspect that _categorical_convert could be
            # optimized when dtype is an instance of CategoricalDtype
            codes, cats, na_count = _categorical_convert(
                self.parser, i, start, end, na_filter, na_hashset)

            # Method accepts list of strings, not encoded ones.
            true_values = [x.decode() for x in self.true_values]
            array_type = dtype.construct_array_type()
            cat = array_type._from_inferred_categories(
                cats, codes, dtype, true_values=true_values)
            return cat, na_count

        elif isinstance(dtype, ExtensionDtype):
            result, na_count = self._string_convert(i, start, end, na_filter,
                                                    na_hashset)

            array_type = dtype.construct_array_type()
            try:
                # use _from_sequence_of_strings if the class defines it
                if isinstance(dtype, BooleanDtype):
                    # xref GH 47534: BooleanArray._from_sequence_of_strings has extra
                    # kwargs
                    true_values = [x.decode() for x in self.true_values]
                    false_values = [x.decode() for x in self.false_values]
                    result = array_type._from_sequence_of_strings(
                        result, dtype=dtype, true_values=true_values,
                        false_values=false_values)
                else:
                    result = array_type._from_sequence_of_strings(result, dtype=dtype)
            except NotImplementedError:
                raise NotImplementedError(
                    f"Extension Array: {array_type} must implement "
                    f"_from_sequence_of_strings in order "
                    f"to be used in parser methods")

            return result, na_count

        elif dtype.kind in "iu":
            try:
                result, na_count = _try_int64(self.parser, i, start,
                                              end, na_filter, na_hashset)
                if user_dtype and na_count is not None:
                    if na_count > 0:
                        raise ValueError(f"Integer column has NA values in column {i}")
            except OverflowError:
                result = _try_uint64(self.parser, i, start, end,
                                     na_filter, na_hashset)
                na_count = 0

            if result is not None and dtype != "int64":
                result = result.astype(dtype)

            return result, na_count

        elif dtype.kind == "f":
            result, na_count = _try_double(self.parser, i, start, end,
                                           na_filter, na_hashset, na_flist)

            if result is not None and dtype != "float64":
                result = result.astype(dtype)
            return result, na_count
        elif dtype.kind == "b":
            result, na_count = _try_bool_flex(self.parser, i, start, end,
                                              na_filter, na_hashset,
                                              self.true_set, self.false_set)
            if user_dtype and na_count is not None:
                if na_count > 0:
                    raise ValueError(f"Bool column has NA values in column {i}")
            return result, na_count

        elif dtype.kind == "S":
            # TODO: na handling
            width = dtype.itemsize
            if width > 0:
                result = _to_fw_string(self.parser, i, start, end, width)
                return result, 0

            # treat as a regular string parsing
            return self._string_convert(i, start, end, na_filter,
                                        na_hashset)
        elif dtype.kind == "U":
            width = dtype.itemsize
            if width > 0:
                raise TypeError(f"the dtype {dtype} is not supported for parsing")

            # unicode variable width
            return self._string_convert(i, start, end, na_filter,
                                        na_hashset)
        elif dtype == object:
            return self._string_convert(i, start, end, na_filter,
                                        na_hashset)
        elif dtype.kind == "M":
            raise TypeError(f"the dtype {dtype} is not supported "
                            f"for parsing, pass this column "
                            f"using parse_dates instead")
        else:
            raise TypeError(f"the dtype {dtype} is not supported for parsing")

    # -> tuple[ndarray[object], int]
    cdef _string_convert(self, Py_ssize_t i, int64_t start, int64_t end,
                         bint na_filter, kh_str_starts_t *na_hashset):

        return _string_box_utf8(self.parser, i, start, end, na_filter,
                                na_hashset, self.encoding_errors)

    def _get_converter(self, i: int, name):
        if self.converters is None:
            return None

        if name is not None and name in self.converters:
            return self.converters[name]

        # Converter for position, if any
        return self.converters.get(i)

    cdef _get_na_list(self, Py_ssize_t i, name):
        # Note: updates self.na_values, self.na_fvalues
        if self.na_values is None:
            return None, set()

        if isinstance(self.na_values, dict):
            key = None
            values = None

            if name is not None and name in self.na_values:
                key = name
            elif i in self.na_values:
                key = i
            else:  # No na_values provided for this column.
                if self.keep_default_na:
                    return _NA_VALUES, set()

                return list(), set()

            values = self.na_values[key]
            if values is not None and not isinstance(values, list):
                values = list(values)

            fvalues = self.na_fvalues[key]
            if fvalues is not None and not isinstance(fvalues, set):
                fvalues = set(fvalues)

            return _ensure_encoded(values), fvalues
        else:
            if not isinstance(self.na_values, list):
                self.na_values = list(self.na_values)
            if not isinstance(self.na_fvalues, set):
                self.na_fvalues = set(self.na_fvalues)

            return _ensure_encoded(self.na_values), self.na_fvalues

    cdef _free_na_set(self, kh_str_starts_t *table):
        kh_destroy_str_starts(table)

    cdef _get_column_name(self, Py_ssize_t i, Py_ssize_t nused):
        cdef int64_t j
        if self.has_usecols and self.names is not None:
            if (not callable(self.usecols) and
                    len(self.names) == len(self.usecols)):
                return self.names[nused]
            else:
                return self.names[i - self.leading_cols]
        else:
            if self.header is not None:
                j = i - self.leading_cols
                # generate extra (bogus) headers if there are more columns than headers
                # These should be strings, not integers, because otherwise we might get
                # issues with callables as usecols GH#46997
                if j >= len(self.header[0]):
                    return str(j)
                elif self.has_mi_columns:
                    return tuple(header_row[j] for header_row in self.header)
                else:
                    return self.header[0][j]
            else:
                return None


# Factor out code common to TextReader.__dealloc__ and TextReader.close
# It cannot be a class method, since calling self.close() in __dealloc__
# which causes a class attribute lookup and violates best practices
# https://cython.readthedocs.io/en/latest/src/userguide/special_methods.html#finalization-method-dealloc
cdef _close(TextReader reader):
    # also preemptively free all allocated memory
    parser_free(reader.parser)
    if reader.true_set:
        kh_destroy_str_starts(reader.true_set)
        reader.true_set = NULL
    if reader.false_set:
        kh_destroy_str_starts(reader.false_set)
        reader.false_set = NULL


cdef:
    object _true_values = [b"True", b"TRUE", b"true"]
    object _false_values = [b"False", b"FALSE", b"false"]


def _ensure_encoded(list lst):
    cdef:
        list result = []
    for x in lst:
        if isinstance(x, str):
            x = PyUnicode_AsUTF8String(x)
        elif not isinstance(x, bytes):
            x = str(x).encode("utf-8")

        result.append(x)
    return result


# common NA values
# no longer excluding inf representations
# '1.#INF','-1.#INF', '1.#INF000000',
STR_NA_VALUES = {
    "-1.#IND",
    "1.#QNAN",
    "1.#IND",
    "-1.#QNAN",
    "#N/A N/A",
    "#N/A",
    "N/A",
    "n/a",
    "NA",
    "<NA>",
    "#NA",
    "NULL",
    "null",
    "NaN",
    "-NaN",
    "nan",
    "-nan",
    "",
    "None",
}
_NA_VALUES = _ensure_encoded(list(STR_NA_VALUES))


def _maybe_upcast(
    arr, use_dtype_backend: bool = False, dtype_backend: str = "numpy"
):
    """Sets nullable dtypes or upcasts if nans are present.

    Upcast, if use_dtype_backend is false and nans are present so that the
    current dtype can not hold the na value. We use nullable dtypes if the
    flag is true for every array.

    Parameters
    ----------
    arr: ndarray
        Numpy array that is potentially being upcast.

    use_dtype_backend: bool, default False
        If true, we cast to the associated nullable dtypes.

    Returns
    -------
    The casted array.
    """
    if isinstance(arr.dtype, ExtensionDtype):
        # TODO: the docstring says arr is an ndarray, in which case this cannot
        #  be reached. Is that incorrect?
        return arr

    na_value = na_values[arr.dtype]

    if issubclass(arr.dtype.type, np.integer):
        mask = arr == na_value

        if use_dtype_backend:
            arr = IntegerArray(arr, mask)
        else:
            arr = arr.astype(float)
            np.putmask(arr, mask, np.nan)

    elif arr.dtype == np.bool_:
        mask = arr.view(np.uint8) == na_value

        if use_dtype_backend:
            arr = BooleanArray(arr, mask)
        else:
            arr = arr.astype(object)
            np.putmask(arr, mask, np.nan)

    elif issubclass(arr.dtype.type, float) or arr.dtype.type == np.float32:
        if use_dtype_backend:
            mask = np.isnan(arr)
            arr = FloatingArray(arr, mask)

    elif arr.dtype == np.object_:
        if use_dtype_backend:
            arr = StringDtype().construct_array_type()._from_sequence(arr)

    if use_dtype_backend and dtype_backend == "pyarrow":
        import pyarrow as pa
        if isinstance(arr, IntegerArray) and arr.isna().all():
            # use null instead of int64 in pyarrow
            arr = arr.to_numpy()
        arr = ArrowExtensionArray(pa.array(arr, from_pandas=True))

    return arr


# ----------------------------------------------------------------------
# Type conversions / inference support code


# -> tuple[ndarray[object], int]
cdef _string_box_utf8(parser_t *parser, int64_t col,
                      int64_t line_start, int64_t line_end,
                      bint na_filter, kh_str_starts_t *na_hashset,
                      const char *encoding_errors):
    cdef:
        int na_count = 0
        Py_ssize_t i, lines
        coliter_t it
        const char *word = NULL
        ndarray[object] result

        int ret = 0
        kh_strbox_t *table

        object pyval

        object NA = na_values[np.object_]
        khiter_t k

    table = kh_init_strbox()
    lines = line_end - line_start
    result = np.empty(lines, dtype=np.object_)
    coliter_setup(&it, parser, col, line_start)

    for i in range(lines):
        COLITER_NEXT(it, word)

        if na_filter:
            if kh_get_str_starts_item(na_hashset, word):
                # in the hash table
                na_count += 1
                result[i] = NA
                continue

        k = kh_get_strbox(table, word)

        # in the hash table
        if k != table.n_buckets:
            # this increments the refcount, but need to test
            pyval = <object>table.vals[k]
        else:
            # box it. new ref?
            pyval = PyUnicode_Decode(word, strlen(word), "utf-8", encoding_errors)

            k = kh_put_strbox(table, word, &ret)
            table.vals[k] = <PyObject *>pyval

        result[i] = pyval

    kh_destroy_strbox(table)

    return result, na_count


@cython.boundscheck(False)
cdef _categorical_convert(parser_t *parser, int64_t col,
                          int64_t line_start, int64_t line_end,
                          bint na_filter, kh_str_starts_t *na_hashset):
    "Convert column data into codes, categories"
    cdef:
        int na_count = 0
        Py_ssize_t i, lines
        coliter_t it
        const char *word = NULL

        int64_t NA = -1
        int64_t[::1] codes
        int64_t current_category = 0

        int ret = 0
        kh_str_t *table
        khiter_t k

    lines = line_end - line_start
    codes = np.empty(lines, dtype=np.int64)

    # factorize parsed values, creating a hash table
    # bytes -> category code
    with nogil:
        table = kh_init_str()
        coliter_setup(&it, parser, col, line_start)

        for i in range(lines):
            COLITER_NEXT(it, word)

            if na_filter:
                if kh_get_str_starts_item(na_hashset, word):
                    # is in NA values
                    na_count += 1
                    codes[i] = NA
                    continue

            k = kh_get_str(table, word)
            # not in the hash table
            if k == table.n_buckets:
                k = kh_put_str(table, word, &ret)
                table.vals[k] = current_category
                current_category += 1

            codes[i] = table.vals[k]

    # parse and box categories to python strings
    result = np.empty(table.n_occupied, dtype=np.object_)
    for k in range(table.n_buckets):
        if kh_exist_str(table, k):
            result[table.vals[k]] = PyUnicode_FromString(table.keys[k])

    kh_destroy_str(table)
    return np.asarray(codes), result, na_count


# -> ndarray[f'|S{width}']
cdef _to_fw_string(parser_t *parser, int64_t col, int64_t line_start,
                   int64_t line_end, int64_t width):
    cdef:
        char *data
        ndarray result

    result = np.empty(line_end - line_start, dtype=f"|S{width}")
    data = <char*>result.data

    with nogil:
        _to_fw_string_nogil(parser, col, line_start, line_end, width, data)

    return result


cdef void _to_fw_string_nogil(parser_t *parser, int64_t col,
                              int64_t line_start, int64_t line_end,
                              size_t width, char *data) nogil:
    cdef:
        int64_t i
        coliter_t it
        const char *word = NULL

    coliter_setup(&it, parser, col, line_start)

    for i in range(line_end - line_start):
        COLITER_NEXT(it, word)
        strncpy(data, word, width)
        data += width


cdef:
    char* cinf = b"inf"
    char* cposinf = b"+inf"
    char* cneginf = b"-inf"

    char* cinfty = b"Infinity"
    char* cposinfty = b"+Infinity"
    char* cneginfty = b"-Infinity"


# -> tuple[ndarray[float64_t], int]  | tuple[None, None]
cdef _try_double(parser_t *parser, int64_t col,
                 int64_t line_start, int64_t line_end,
                 bint na_filter, kh_str_starts_t *na_hashset, object na_flist):
    cdef:
        int error, na_count = 0
        Py_ssize_t lines
        float64_t *data
        float64_t NA = na_values[np.float64]
        kh_float64_t *na_fset
        ndarray[float64_t] result
        bint use_na_flist = len(na_flist) > 0

    lines = line_end - line_start
    result = np.empty(lines, dtype=np.float64)
    data = <float64_t *>result.data
    na_fset = kset_float64_from_list(na_flist)
    with nogil:
        error = _try_double_nogil(parser, parser.double_converter,
                                  col, line_start, line_end,
                                  na_filter, na_hashset, use_na_flist,
                                  na_fset, NA, data, &na_count)

    kh_destroy_float64(na_fset)
    if error != 0:
        return None, None
    return result, na_count


cdef int _try_double_nogil(parser_t *parser,
                           float64_t (*double_converter)(
                               const char *, char **, char,
                               char, char, int, int *, int *) nogil,
                           int64_t col, int64_t line_start, int64_t line_end,
                           bint na_filter, kh_str_starts_t *na_hashset,
                           bint use_na_flist,
                           const kh_float64_t *na_flist,
                           float64_t NA, float64_t *data,
                           int *na_count) nogil:
    cdef:
        int error = 0,
        Py_ssize_t i, lines = line_end - line_start
        coliter_t it
        const char *word = NULL
        char *p_end
        khiter_t k64

    na_count[0] = 0
    coliter_setup(&it, parser, col, line_start)

    if na_filter:
        for i in range(lines):
            COLITER_NEXT(it, word)

            if kh_get_str_starts_item(na_hashset, word):
                # in the hash table
                na_count[0] += 1
                data[0] = NA
            else:
                data[0] = double_converter(word, &p_end, parser.decimal,
                                           parser.sci, parser.thousands,
                                           1, &error, NULL)
                if error != 0 or p_end == word or p_end[0]:
                    error = 0
                    if (strcasecmp(word, cinf) == 0 or
                            strcasecmp(word, cposinf) == 0 or
                            strcasecmp(word, cinfty) == 0 or
                            strcasecmp(word, cposinfty) == 0):
                        data[0] = INF
                    elif (strcasecmp(word, cneginf) == 0 or
                            strcasecmp(word, cneginfty) == 0):
                        data[0] = NEGINF
                    else:
                        return 1
                if use_na_flist:
                    k64 = kh_get_float64(na_flist, data[0])
                    if k64 != na_flist.n_buckets:
                        na_count[0] += 1
                        data[0] = NA
            data += 1
    else:
        for i in range(lines):
            COLITER_NEXT(it, word)
            data[0] = double_converter(word, &p_end, parser.decimal,
                                       parser.sci, parser.thousands,
                                       1, &error, NULL)
            if error != 0 or p_end == word or p_end[0]:
                error = 0
                if (strcasecmp(word, cinf) == 0 or
                        strcasecmp(word, cposinf) == 0 or
                        strcasecmp(word, cinfty) == 0 or
                        strcasecmp(word, cposinfty) == 0):
                    data[0] = INF
                elif (strcasecmp(word, cneginf) == 0 or
                        strcasecmp(word, cneginfty) == 0):
                    data[0] = NEGINF
                else:
                    return 1
            data += 1

    return 0


cdef _try_uint64(parser_t *parser, int64_t col,
                 int64_t line_start, int64_t line_end,
                 bint na_filter, kh_str_starts_t *na_hashset):
    cdef:
        int error
        Py_ssize_t lines
        coliter_t it
        uint64_t *data
        ndarray result
        uint_state state

    lines = line_end - line_start
    result = np.empty(lines, dtype=np.uint64)
    data = <uint64_t *>result.data

    uint_state_init(&state)
    coliter_setup(&it, parser, col, line_start)
    with nogil:
        error = _try_uint64_nogil(parser, col, line_start, line_end,
                                  na_filter, na_hashset, data, &state)
    if error != 0:
        if error == ERROR_OVERFLOW:
            # Can't get the word variable
            raise OverflowError("Overflow")
        return None

    if uint64_conflict(&state):
        raise ValueError("Cannot convert to numerical dtype")

    if state.seen_sint:
        raise OverflowError("Overflow")

    return result


cdef int _try_uint64_nogil(parser_t *parser, int64_t col,
                           int64_t line_start,
                           int64_t line_end, bint na_filter,
                           const kh_str_starts_t *na_hashset,
                           uint64_t *data, uint_state *state) nogil:
    cdef:
        int error
        Py_ssize_t i, lines = line_end - line_start
        coliter_t it
        const char *word = NULL

    coliter_setup(&it, parser, col, line_start)

    if na_filter:
        for i in range(lines):
            COLITER_NEXT(it, word)
            if kh_get_str_starts_item(na_hashset, word):
                # in the hash table
                state.seen_null = 1
                data[i] = 0
                continue

            data[i] = str_to_uint64(state, word, INT64_MAX, UINT64_MAX,
                                    &error, parser.thousands)
            if error != 0:
                return error
    else:
        for i in range(lines):
            COLITER_NEXT(it, word)
            data[i] = str_to_uint64(state, word, INT64_MAX, UINT64_MAX,
                                    &error, parser.thousands)
            if error != 0:
                return error

    return 0


cdef _try_int64(parser_t *parser, int64_t col,
                int64_t line_start, int64_t line_end,
                bint na_filter, kh_str_starts_t *na_hashset):
    cdef:
        int error, na_count = 0
        Py_ssize_t lines
        coliter_t it
        int64_t *data
        ndarray result
        int64_t NA = na_values[np.int64]

    lines = line_end - line_start
    result = np.empty(lines, dtype=np.int64)
    data = <int64_t *>result.data
    coliter_setup(&it, parser, col, line_start)
    with nogil:
        error = _try_int64_nogil(parser, col, line_start, line_end,
                                 na_filter, na_hashset, NA, data, &na_count)
    if error != 0:
        if error == ERROR_OVERFLOW:
            # Can't get the word variable
            raise OverflowError("Overflow")
        return None, None

    return result, na_count


cdef int _try_int64_nogil(parser_t *parser, int64_t col,
                          int64_t line_start,
                          int64_t line_end, bint na_filter,
                          const kh_str_starts_t *na_hashset, int64_t NA,
                          int64_t *data, int *na_count) nogil:
    cdef:
        int error
        Py_ssize_t i, lines = line_end - line_start
        coliter_t it
        const char *word = NULL

    na_count[0] = 0
    coliter_setup(&it, parser, col, line_start)

    if na_filter:
        for i in range(lines):
            COLITER_NEXT(it, word)
            if kh_get_str_starts_item(na_hashset, word):
                # in the hash table
                na_count[0] += 1
                data[i] = NA
                continue

            data[i] = str_to_int64(word, INT64_MIN, INT64_MAX,
                                   &error, parser.thousands)
            if error != 0:
                return error
    else:
        for i in range(lines):
            COLITER_NEXT(it, word)
            data[i] = str_to_int64(word, INT64_MIN, INT64_MAX,
                                   &error, parser.thousands)
            if error != 0:
                return error

    return 0


# -> tuple[ndarray[bool], int]
cdef _try_bool_flex(parser_t *parser, int64_t col,
                    int64_t line_start, int64_t line_end,
                    bint na_filter, const kh_str_starts_t *na_hashset,
                    const kh_str_starts_t *true_hashset,
                    const kh_str_starts_t *false_hashset):
    cdef:
        int error, na_count = 0
        Py_ssize_t lines
        uint8_t *data
        ndarray result
        uint8_t NA = na_values[np.bool_]

    lines = line_end - line_start
    result = np.empty(lines, dtype=np.uint8)
    data = <uint8_t *>result.data
    with nogil:
        error = _try_bool_flex_nogil(parser, col, line_start, line_end,
                                     na_filter, na_hashset, true_hashset,
                                     false_hashset, NA, data, &na_count)
    if error != 0:
        return None, None
    return result.view(np.bool_), na_count


cdef int _try_bool_flex_nogil(parser_t *parser, int64_t col,
                              int64_t line_start,
                              int64_t line_end, bint na_filter,
                              const kh_str_starts_t *na_hashset,
                              const kh_str_starts_t *true_hashset,
                              const kh_str_starts_t *false_hashset,
                              uint8_t NA, uint8_t *data,
                              int *na_count) nogil:
    cdef:
        int error = 0
        Py_ssize_t i, lines = line_end - line_start
        coliter_t it
        const char *word = NULL

    na_count[0] = 0
    coliter_setup(&it, parser, col, line_start)

    if na_filter:
        for i in range(lines):
            COLITER_NEXT(it, word)

            if kh_get_str_starts_item(na_hashset, word):
                # in the hash table
                na_count[0] += 1
                data[0] = NA
                data += 1
                continue

            if kh_get_str_starts_item(true_hashset, word):
                data[0] = 1
                data += 1
                continue
            if kh_get_str_starts_item(false_hashset, word):
                data[0] = 0
                data += 1
                continue

            error = to_boolean(word, data)
            if error != 0:
                return error
            data += 1
    else:
        for i in range(lines):
            COLITER_NEXT(it, word)

            if kh_get_str_starts_item(true_hashset, word):
                data[0] = 1
                data += 1
                continue

            if kh_get_str_starts_item(false_hashset, word):
                data[0] = 0
                data += 1
                continue

            error = to_boolean(word, data)
            if error != 0:
                return error
            data += 1

    return 0


cdef kh_str_starts_t* kset_from_list(list values) except NULL:
    # caller takes responsibility for freeing the hash table
    cdef:
        Py_ssize_t i
        kh_str_starts_t *table
        int ret = 0
        object val

    table = kh_init_str_starts()

    for i in range(len(values)):
        val = values[i]

        # None creeps in sometimes, which isn't possible here
        if not isinstance(val, bytes):
            kh_destroy_str_starts(table)
            raise ValueError("Must be all encoded bytes")

        kh_put_str_starts_item(table, PyBytes_AsString(val), &ret)

    if table.table.n_buckets <= 128:
        # Resize the hash table to make it almost empty, this
        # reduces amount of hash collisions on lookup thus
        # "key not in table" case is faster.
        # Note that this trades table memory footprint for lookup speed.
        kh_resize_str_starts(table, table.table.n_buckets * 8)

    return table


cdef kh_float64_t* kset_float64_from_list(values) except NULL:
    # caller takes responsibility for freeing the hash table
    cdef:
        kh_float64_t *table
        int ret = 0
        float64_t val
        object value

    table = kh_init_float64()

    for value in values:
        val = float(value)

        kh_put_float64(table, val, &ret)

    if table.n_buckets <= 128:
        # See reasoning in kset_from_list
        kh_resize_float64(table, table.n_buckets * 8)
    return table


cdef raise_parser_error(object base, parser_t *parser):
    cdef:
        object old_exc
        object exc_type
        PyObject *type
        PyObject *value
        PyObject *traceback

    if PyErr_Occurred():
        PyErr_Fetch(&type, &value, &traceback)
        Py_XDECREF(traceback)

        if value != NULL:
            old_exc = <object>value
            Py_XDECREF(value)

            # PyErr_Fetch only returned the error message in *value,
            # so the Exception class must be extracted from *type.
            if isinstance(old_exc, str):
                if type != NULL:
                    exc_type = <object>type
                else:
                    exc_type = ParserError

                Py_XDECREF(type)
                raise exc_type(old_exc)
            else:
                Py_XDECREF(type)
                raise old_exc

    message = f"{base}. C error: "
    if parser.error_msg != NULL:
        message += parser.error_msg.decode("utf-8")
    else:
        message += "no error message set"

    raise ParserError(message)


# ----------------------------------------------------------------------
# NA values
def _compute_na_values():
    int64info = np.iinfo(np.int64)
    int32info = np.iinfo(np.int32)
    int16info = np.iinfo(np.int16)
    int8info = np.iinfo(np.int8)
    uint64info = np.iinfo(np.uint64)
    uint32info = np.iinfo(np.uint32)
    uint16info = np.iinfo(np.uint16)
    uint8info = np.iinfo(np.uint8)
    na_values = {
        np.float32: np.nan,
        np.float64: np.nan,
        np.int64: int64info.min,
        np.int32: int32info.min,
        np.int16: int16info.min,
        np.int8: int8info.min,
        np.uint64: uint64info.max,
        np.uint32: uint32info.max,
        np.uint16: uint16info.max,
        np.uint8: uint8info.max,
        np.bool_: uint8info.max,
        np.object_: np.nan,
    }
    return na_values


na_values = _compute_na_values()

for k in list(na_values):
    na_values[np.dtype(k)] = na_values[k]


# -> ArrayLike
cdef _apply_converter(object f, parser_t *parser, int64_t col,
                      int64_t line_start, int64_t line_end):
    cdef:
        Py_ssize_t i, lines
        coliter_t it
        const char *word = NULL
        ndarray[object] result
        object val

    lines = line_end - line_start
    result = np.empty(lines, dtype=np.object_)

    coliter_setup(&it, parser, col, line_start)

    for i in range(lines):
        COLITER_NEXT(it, word)
        val = PyUnicode_FromString(word)
        result[i] = f(val)

    return lib.maybe_convert_objects(result)


cdef list _maybe_encode(list values):
    if values is None:
        return []
    return [x.encode("utf-8") if isinstance(x, str) else x for x in values]


def sanitize_objects(ndarray[object] values, set na_values) -> int:
    """
    Convert specified values, including the given set na_values to np.nan.

    Parameters
    ----------
    values : ndarray[object]
    na_values : set

    Returns
    -------
    na_count : int
    """
    cdef:
        Py_ssize_t i, n
        object val, onan
        Py_ssize_t na_count = 0
        dict memo = {}

    n = len(values)
    onan = np.nan

    for i in range(n):
        val = values[i]
        if val in na_values:
            values[i] = onan
            na_count += 1
        elif val in memo:
            values[i] = memo[val]
        else:
            memo[val] = val

    return na_count
