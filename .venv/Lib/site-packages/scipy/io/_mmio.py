"""
  Matrix Market I/O in Python.
  See http://math.nist.gov/MatrixMarket/formats.html
  for information about the Matrix Market format.
"""
#
# Author: Pearu Peterson <pearu@cens.ioc.ee>
# Created: October, 2004
#
# References:
#  http://math.nist.gov/MatrixMarket/
#
import os

import numpy as np
from numpy import (asarray, real, imag, conj, zeros, ndarray, concatenate,
                   ones, can_cast)

from scipy.sparse import coo_matrix, issparse

__all__ = ['mminfo', 'mmread', 'mmwrite', 'MMFile']


# -----------------------------------------------------------------------------
def asstr(s):
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)


def mminfo(source):
    """
    Return size and storage parameters from Matrix Market file-like 'source'.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extension .mtx) or open file-like object

    Returns
    -------
    rows : int
        Number of matrix rows.
    cols : int
        Number of matrix columns.
    entries : int
        Number of non-zero entries of a sparse matrix
        or rows*cols for a dense matrix.
    format : str
        Either 'coordinate' or 'array'.
    field : str
        Either 'real', 'complex', 'pattern', or 'integer'.
    symmetry : str
        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mminfo

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''


    ``mminfo(source)`` returns the number of rows, number of columns,
    format, field type and symmetry attribute of the source file.

    >>> mminfo(StringIO(text))
    (5, 5, 7, 'coordinate', 'real', 'general')
    """
    return MMFile.info(source)

# -----------------------------------------------------------------------------


def mmread(source):
    """
    Reads the contents of a Matrix Market file-like 'source' into a matrix.

    Parameters
    ----------
    source : str or file-like
        Matrix Market filename (extensions .mtx, .mtz.gz)
        or open file-like object.

    Returns
    -------
    a : ndarray or coo_matrix
        Dense or sparse matrix depending on the matrix format in the
        Matrix Market file.

    Examples
    --------
    >>> from io import StringIO
    >>> from scipy.io import mmread

    >>> text = '''%%MatrixMarket matrix coordinate real general
    ...  5 5 7
    ...  2 3 1.0
    ...  3 4 2.0
    ...  3 5 3.0
    ...  4 1 4.0
    ...  4 2 5.0
    ...  4 3 6.0
    ...  4 4 7.0
    ... '''

    ``mmread(source)`` returns the data as sparse matrix in COO format.

    >>> m = mmread(StringIO(text))
    >>> m
    <5x5 sparse matrix of type '<class 'numpy.float64'>'
    with 7 stored elements in COOrdinate format>
    >>> m.A
    array([[0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 2., 3.],
           [4., 5., 6., 7., 0.],
           [0., 0., 0., 0., 0.]])
    """
    return MMFile().read(source)

# -----------------------------------------------------------------------------


def mmwrite(target, a, comment='', field=None, precision=None, symmetry=None):
    r"""
    Writes the sparse or dense array `a` to Matrix Market file-like `target`.

    Parameters
    ----------
    target : str or file-like
        Matrix Market filename (extension .mtx) or open file-like object.
    a : array like
        Sparse or dense 2-D array.
    comment : str, optional
        Comments to be prepended to the Matrix Market file.
    field : None or str, optional
        Either 'real', 'complex', 'pattern', or 'integer'.
    precision : None or int, optional
        Number of digits to display for real or complex values.
    symmetry : None or str, optional
        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
        If symmetry is None the symmetry type of 'a' is determined by its
        values.

    Returns
    -------
    None

    Examples
    --------
    >>> from io import BytesIO
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> from scipy.io import mmwrite

    Write a small NumPy array to a matrix market file.  The file will be
    written in the ``'array'`` format.

    >>> a = np.array([[1.0, 0, 0, 0], [0, 2.5, 0, 6.25]])
    >>> target = BytesIO()
    >>> mmwrite(target, a)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    2 4
    1.0000000000000000e+00
    0.0000000000000000e+00
    0.0000000000000000e+00
    2.5000000000000000e+00
    0.0000000000000000e+00
    0.0000000000000000e+00
    0.0000000000000000e+00
    6.2500000000000000e+00

    Add a comment to the output file, and set the precision to 3.

    >>> target = BytesIO()
    >>> mmwrite(target, a, comment='\n Some test data.\n', precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array real general
    %
    % Some test data.
    %
    2 4
    1.000e+00
    0.000e+00
    0.000e+00
    2.500e+00
    0.000e+00
    0.000e+00
    0.000e+00
    6.250e+00

    Convert to a sparse matrix before calling ``mmwrite``.  This will
    result in the output format being ``'coordinate'`` rather than
    ``'array'``.

    >>> target = BytesIO()
    >>> mmwrite(target, coo_matrix(a), precision=3)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix coordinate real general
    %
    2 4 3
    1 1 1.00e+00
    2 2 2.50e+00
    2 4 6.25e+00

    Write a complex Hermitian array to a matrix market file.  Note that
    only six values are actually written to the file; the other values
    are implied by the symmetry.

    >>> z = np.array([[3, 1+2j, 4-3j], [1-2j, 1, -5j], [4+3j, 5j, 2.5]])
    >>> z
    array([[ 3. +0.j,  1. +2.j,  4. -3.j],
           [ 1. -2.j,  1. +0.j, -0. -5.j],
           [ 4. +3.j,  0. +5.j,  2.5+0.j]])

    >>> target = BytesIO()
    >>> mmwrite(target, z, precision=2)
    >>> print(target.getvalue().decode('latin1'))
    %%MatrixMarket matrix array complex hermitian
    %
    3 3
    3.00e+00 0.00e+00
    1.00e+00 -2.00e+00
    4.00e+00 3.00e+00
    1.00e+00 0.00e+00
    0.00e+00 5.00e+00
    2.50e+00 0.00e+00

    """
    MMFile().write(target, a, comment, field, precision, symmetry)


###############################################################################
class MMFile:
    __slots__ = ('_rows',
                 '_cols',
                 '_entries',
                 '_format',
                 '_field',
                 '_symmetry')

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def entries(self):
        return self._entries

    @property
    def format(self):
        return self._format

    @property
    def field(self):
        return self._field

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def has_symmetry(self):
        return self._symmetry in (self.SYMMETRY_SYMMETRIC,
                                  self.SYMMETRY_SKEW_SYMMETRIC,
                                  self.SYMMETRY_HERMITIAN)

    # format values
    FORMAT_COORDINATE = 'coordinate'
    FORMAT_ARRAY = 'array'
    FORMAT_VALUES = (FORMAT_COORDINATE, FORMAT_ARRAY)

    @classmethod
    def _validate_format(self, format):
        if format not in self.FORMAT_VALUES:
            msg = f'unknown format type {format}, must be one of {self.FORMAT_VALUES}'
            raise ValueError(msg)

    # field values
    FIELD_INTEGER = 'integer'
    FIELD_UNSIGNED = 'unsigned-integer'
    FIELD_REAL = 'real'
    FIELD_COMPLEX = 'complex'
    FIELD_PATTERN = 'pattern'
    FIELD_VALUES = (FIELD_INTEGER, FIELD_UNSIGNED, FIELD_REAL, FIELD_COMPLEX,
                    FIELD_PATTERN)

    @classmethod
    def _validate_field(self, field):
        if field not in self.FIELD_VALUES:
            msg = f'unknown field type {field}, must be one of {self.FIELD_VALUES}'
            raise ValueError(msg)

    # symmetry values
    SYMMETRY_GENERAL = 'general'
    SYMMETRY_SYMMETRIC = 'symmetric'
    SYMMETRY_SKEW_SYMMETRIC = 'skew-symmetric'
    SYMMETRY_HERMITIAN = 'hermitian'
    SYMMETRY_VALUES = (SYMMETRY_GENERAL, SYMMETRY_SYMMETRIC,
                       SYMMETRY_SKEW_SYMMETRIC, SYMMETRY_HERMITIAN)

    @classmethod
    def _validate_symmetry(self, symmetry):
        if symmetry not in self.SYMMETRY_VALUES:
            raise ValueError(f'unknown symmetry type {symmetry}, '
                             f'must be one of {self.SYMMETRY_VALUES}')

    DTYPES_BY_FIELD = {FIELD_INTEGER: 'intp',
                       FIELD_UNSIGNED: 'uint64',
                       FIELD_REAL: 'd',
                       FIELD_COMPLEX: 'D',
                       FIELD_PATTERN: 'd'}

    # -------------------------------------------------------------------------
    @staticmethod
    def reader():
        pass

    # -------------------------------------------------------------------------
    @staticmethod
    def writer():
        pass

    # -------------------------------------------------------------------------
    @classmethod
    def info(self, source):
        """
        Return size, storage parameters from Matrix Market file-like 'source'.

        Parameters
        ----------
        source : str or file-like
            Matrix Market filename (extension .mtx) or open file-like object

        Returns
        -------
        rows : int
            Number of matrix rows.
        cols : int
            Number of matrix columns.
        entries : int
            Number of non-zero entries of a sparse matrix
            or rows*cols for a dense matrix.
        format : str
            Either 'coordinate' or 'array'.
        field : str
            Either 'real', 'complex', 'pattern', or 'integer'.
        symmetry : str
            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
        """

        stream, close_it = self._open(source)

        try:

            # read and validate header line
            line = stream.readline()
            mmid, matrix, format, field, symmetry = \
                (asstr(part.strip()) for part in line.split())
            if not mmid.startswith('%%MatrixMarket'):
                raise ValueError('source is not in Matrix Market format')
            if not matrix.lower() == 'matrix':
                raise ValueError("Problem reading file header: " + line)

            # http://math.nist.gov/MatrixMarket/formats.html
            if format.lower() == 'array':
                format = self.FORMAT_ARRAY
            elif format.lower() == 'coordinate':
                format = self.FORMAT_COORDINATE

            # skip comments
            # line.startswith('%')
            while line:
                if line.lstrip() and line.lstrip()[0] in ['%', 37]:
                    line = stream.readline()
                else:
                    break

            # skip empty lines
            while not line.strip():
                line = stream.readline()

            split_line = line.split()
            if format == self.FORMAT_ARRAY:
                if not len(split_line) == 2:
                    raise ValueError("Header line not of length 2: " +
                                     line.decode('ascii'))
                rows, cols = map(int, split_line)
                entries = rows * cols
            else:
                if not len(split_line) == 3:
                    raise ValueError("Header line not of length 3: " +
                                     line.decode('ascii'))
                rows, cols, entries = map(int, split_line)

            return (rows, cols, entries, format, field.lower(),
                    symmetry.lower())

        finally:
            if close_it:
                stream.close()

    # -------------------------------------------------------------------------
    @staticmethod
    def _open(filespec, mode='rb'):
        """ Return an open file stream for reading based on source.

        If source is a file name, open it (after trying to find it with mtx and
        gzipped mtx extensions). Otherwise, just return source.

        Parameters
        ----------
        filespec : str or file-like
            String giving file name or file-like object
        mode : str, optional
            Mode with which to open file, if `filespec` is a file name.

        Returns
        -------
        fobj : file-like
            Open file-like object.
        close_it : bool
            True if the calling function should close this file when done,
            false otherwise.
        """
        # If 'filespec' is path-like (str, pathlib.Path, os.DirEntry, other class
        # implementing a '__fspath__' method), try to convert it to str. If this
        # fails by throwing a 'TypeError', assume it's an open file handle and
        # return it as-is.
        try:
            filespec = os.fspath(filespec)
        except TypeError:
            return filespec, False

        # 'filespec' is definitely a str now

        # open for reading
        if mode[0] == 'r':

            # determine filename plus extension
            if not os.path.isfile(filespec):
                if os.path.isfile(filespec+'.mtx'):
                    filespec = filespec + '.mtx'
                elif os.path.isfile(filespec+'.mtx.gz'):
                    filespec = filespec + '.mtx.gz'
                elif os.path.isfile(filespec+'.mtx.bz2'):
                    filespec = filespec + '.mtx.bz2'
            # open filename
            if filespec.endswith('.gz'):
                import gzip
                stream = gzip.open(filespec, mode)
            elif filespec.endswith('.bz2'):
                import bz2
                stream = bz2.BZ2File(filespec, 'rb')
            else:
                stream = open(filespec, mode)

        # open for writing
        else:
            if filespec[-4:] != '.mtx':
                filespec = filespec + '.mtx'
            stream = open(filespec, mode)

        return stream, True

    # -------------------------------------------------------------------------
    @staticmethod
    def _get_symmetry(a):
        m, n = a.shape
        if m != n:
            return MMFile.SYMMETRY_GENERAL
        issymm = True
        isskew = True
        isherm = a.dtype.char in 'FD'

        # sparse input
        if issparse(a):
            # check if number of nonzero entries of lower and upper triangle
            # matrix are equal
            a = a.tocoo()
            (row, col) = a.nonzero()
            if (row < col).sum() != (row > col).sum():
                return MMFile.SYMMETRY_GENERAL

            # define iterator over symmetric pair entries
            a = a.todok()

            def symm_iterator():
                for ((i, j), aij) in a.items():
                    if i > j:
                        aji = a[j, i]
                        yield (aij, aji, False)
                    elif i == j:
                        yield (aij, aij, True)

        # non-sparse input
        else:
            # define iterator over symmetric pair entries
            def symm_iterator():
                for j in range(n):
                    for i in range(j, n):
                        aij, aji = a[i][j], a[j][i]
                        yield (aij, aji, i == j)

        # check for symmetry
        # yields aij, aji, is_diagonal
        for (aij, aji, is_diagonal) in symm_iterator():
            if isskew and is_diagonal and aij != 0:
                isskew = False
            else:
                if issymm and aij != aji:
                    issymm = False
                with np.errstate(over="ignore"):
                    # This can give a warning for uint dtypes, so silence that
                    if isskew and aij != -aji:
                        isskew = False
                if isherm and aij != conj(aji):
                    isherm = False
            if not (issymm or isskew or isherm):
                break

        # return symmetry value
        if issymm:
            return MMFile.SYMMETRY_SYMMETRIC
        if isskew:
            return MMFile.SYMMETRY_SKEW_SYMMETRIC
        if isherm:
            return MMFile.SYMMETRY_HERMITIAN
        return MMFile.SYMMETRY_GENERAL

    # -------------------------------------------------------------------------
    @staticmethod
    def _field_template(field, precision):
        return {MMFile.FIELD_REAL: '%%.%ie\n' % precision,
                MMFile.FIELD_INTEGER: '%i\n',
                MMFile.FIELD_UNSIGNED: '%u\n',
                MMFile.FIELD_COMPLEX: '%%.%ie %%.%ie\n' %
                    (precision, precision)
                }.get(field, None)

    # -------------------------------------------------------------------------
    def __init__(self, **kwargs):
        self._init_attrs(**kwargs)

    # -------------------------------------------------------------------------
    def read(self, source):
        """
        Reads the contents of a Matrix Market file-like 'source' into a matrix.

        Parameters
        ----------
        source : str or file-like
            Matrix Market filename (extensions .mtx, .mtz.gz)
            or open file object.

        Returns
        -------
        a : ndarray or coo_matrix
            Dense or sparse matrix depending on the matrix format in the
            Matrix Market file.
        """
        stream, close_it = self._open(source)

        try:
            self._parse_header(stream)
            return self._parse_body(stream)

        finally:
            if close_it:
                stream.close()

    # -------------------------------------------------------------------------
    def write(self, target, a, comment='', field=None, precision=None,
              symmetry=None):
        """
        Writes sparse or dense array `a` to Matrix Market file-like `target`.

        Parameters
        ----------
        target : str or file-like
            Matrix Market filename (extension .mtx) or open file-like object.
        a : array like
            Sparse or dense 2-D array.
        comment : str, optional
            Comments to be prepended to the Matrix Market file.
        field : None or str, optional
            Either 'real', 'complex', 'pattern', or 'integer'.
        precision : None or int, optional
            Number of digits to display for real or complex values.
        symmetry : None or str, optional
            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.
            If symmetry is None the symmetry type of 'a' is determined by its
            values.
        """

        stream, close_it = self._open(target, 'wb')

        try:
            self._write(stream, a, comment, field, precision, symmetry)

        finally:
            if close_it:
                stream.close()
            else:
                stream.flush()

    # -------------------------------------------------------------------------
    def _init_attrs(self, **kwargs):
        """
        Initialize each attributes with the corresponding keyword arg value
        or a default of None
        """

        attrs = self.__class__.__slots__
        public_attrs = [attr[1:] for attr in attrs]
        invalid_keys = set(kwargs.keys()) - set(public_attrs)

        if invalid_keys:
            raise ValueError('''found {} invalid keyword arguments, please only
                                use {}'''.format(tuple(invalid_keys),
                                             public_attrs))

        for attr in attrs:
            setattr(self, attr, kwargs.get(attr[1:], None))

    # -------------------------------------------------------------------------
    def _parse_header(self, stream):
        rows, cols, entries, format, field, symmetry = \
            self.__class__.info(stream)
        self._init_attrs(rows=rows, cols=cols, entries=entries, format=format,
                         field=field, symmetry=symmetry)

    # -------------------------------------------------------------------------
    def _parse_body(self, stream):
        rows, cols, entries, format, field, symm = (self.rows, self.cols,
                                                    self.entries, self.format,
                                                    self.field, self.symmetry)

        dtype = self.DTYPES_BY_FIELD.get(field, None)

        has_symmetry = self.has_symmetry
        is_integer = field == self.FIELD_INTEGER
        is_unsigned_integer = field == self.FIELD_UNSIGNED
        is_complex = field == self.FIELD_COMPLEX
        is_skew = symm == self.SYMMETRY_SKEW_SYMMETRIC
        is_herm = symm == self.SYMMETRY_HERMITIAN
        is_pattern = field == self.FIELD_PATTERN

        if format == self.FORMAT_ARRAY:
            a = zeros((rows, cols), dtype=dtype)
            line = 1
            i, j = 0, 0
            if is_skew:
                a[i, j] = 0
                if i < rows - 1:
                    i += 1
            while line:
                line = stream.readline()
                # line.startswith('%')
                if not line or line[0] in ['%', 37] or not line.strip():
                    continue
                if is_integer:
                    aij = int(line)
                elif is_unsigned_integer:
                    aij = int(line)
                elif is_complex:
                    aij = complex(*map(float, line.split()))
                else:
                    aij = float(line)
                a[i, j] = aij
                if has_symmetry and i != j:
                    if is_skew:
                        a[j, i] = -aij
                    elif is_herm:
                        a[j, i] = conj(aij)
                    else:
                        a[j, i] = aij
                if i < rows-1:
                    i = i + 1
                else:
                    j = j + 1
                    if not has_symmetry:
                        i = 0
                    else:
                        i = j
                        if is_skew:
                            a[i, j] = 0
                            if i < rows-1:
                                i += 1

            if is_skew:
                if not (i in [0, j] and j == cols - 1):
                    raise ValueError("Parse error, did not read all lines.")
            else:
                if not (i in [0, j] and j == cols):
                    raise ValueError("Parse error, did not read all lines.")

        elif format == self.FORMAT_COORDINATE:
            # Read sparse COOrdinate format

            if entries == 0:
                # empty matrix
                return coo_matrix((rows, cols), dtype=dtype)

            I = zeros(entries, dtype='intc')
            J = zeros(entries, dtype='intc')
            if is_pattern:
                V = ones(entries, dtype='int8')
            elif is_integer:
                V = zeros(entries, dtype='intp')
            elif is_unsigned_integer:
                V = zeros(entries, dtype='uint64')
            elif is_complex:
                V = zeros(entries, dtype='complex')
            else:
                V = zeros(entries, dtype='float')

            entry_number = 0
            for line in stream:
                # line.startswith('%')
                if not line or line[0] in ['%', 37] or not line.strip():
                    continue

                if entry_number+1 > entries:
                    raise ValueError("'entries' in header is smaller than "
                                     "number of entries")
                l = line.split()
                I[entry_number], J[entry_number] = map(int, l[:2])

                if not is_pattern:
                    if is_integer:
                        V[entry_number] = int(l[2])
                    elif is_unsigned_integer:
                        V[entry_number] = int(l[2])
                    elif is_complex:
                        V[entry_number] = complex(*map(float, l[2:]))
                    else:
                        V[entry_number] = float(l[2])
                entry_number += 1
            if entry_number < entries:
                raise ValueError("'entries' in header is larger than "
                                 "number of entries")

            I -= 1  # adjust indices (base 1 -> base 0)
            J -= 1

            if has_symmetry:
                mask = (I != J)       # off diagonal mask
                od_I = I[mask]
                od_J = J[mask]
                od_V = V[mask]

                I = concatenate((I, od_J))
                J = concatenate((J, od_I))

                if is_skew:
                    od_V *= -1
                elif is_herm:
                    od_V = od_V.conjugate()

                V = concatenate((V, od_V))

            a = coo_matrix((V, (I, J)), shape=(rows, cols), dtype=dtype)
        else:
            raise NotImplementedError(format)

        return a

    #  ------------------------------------------------------------------------
    def _write(self, stream, a, comment='', field=None, precision=None,
               symmetry=None):
        if isinstance(a, list) or isinstance(a, ndarray) or \
           isinstance(a, tuple) or hasattr(a, '__array__'):
            rep = self.FORMAT_ARRAY
            a = asarray(a)
            if len(a.shape) != 2:
                raise ValueError('Expected 2 dimensional array')
            rows, cols = a.shape

            if field is not None:

                if field == self.FIELD_INTEGER:
                    if not can_cast(a.dtype, 'intp'):
                        raise OverflowError("mmwrite does not support integer "
                                            "dtypes larger than native 'intp'.")
                    a = a.astype('intp')
                elif field == self.FIELD_REAL:
                    if a.dtype.char not in 'fd':
                        a = a.astype('d')
                elif field == self.FIELD_COMPLEX:
                    if a.dtype.char not in 'FD':
                        a = a.astype('D')

        else:
            if not issparse(a):
                raise ValueError('unknown matrix type: %s' % type(a))

            rep = 'coordinate'
            rows, cols = a.shape

        typecode = a.dtype.char

        if precision is None:
            if typecode in 'fF':
                precision = 8
            else:
                precision = 16
        if field is None:
            kind = a.dtype.kind
            if kind == 'i':
                if not can_cast(a.dtype, 'intp'):
                    raise OverflowError("mmwrite does not support integer "
                                        "dtypes larger than native 'intp'.")
                field = 'integer'
            elif kind == 'f':
                field = 'real'
            elif kind == 'c':
                field = 'complex'
            elif kind == 'u':
                field = 'unsigned-integer'
            else:
                raise TypeError('unexpected dtype kind ' + kind)

        if symmetry is None:
            symmetry = self._get_symmetry(a)

        # validate rep, field, and symmetry
        self.__class__._validate_format(rep)
        self.__class__._validate_field(field)
        self.__class__._validate_symmetry(symmetry)

        # write initial header line
        data = f'%%MatrixMarket matrix {rep} {field} {symmetry}\n'
        stream.write(data.encode('latin1'))

        # write comments
        for line in comment.split('\n'):
            data = '%%%s\n' % (line)
            stream.write(data.encode('latin1'))

        template = self._field_template(field, precision)
        # write dense format
        if rep == self.FORMAT_ARRAY:
            # write shape spec
            data = '%i %i\n' % (rows, cols)
            stream.write(data.encode('latin1'))

            if field in (self.FIELD_INTEGER, self.FIELD_REAL,
                         self.FIELD_UNSIGNED):
                if symmetry == self.SYMMETRY_GENERAL:
                    for j in range(cols):
                        for i in range(rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))

                elif symmetry == self.SYMMETRY_SKEW_SYMMETRIC:
                    for j in range(cols):
                        for i in range(j + 1, rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))

                else:
                    for j in range(cols):
                        for i in range(j, rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))

            elif field == self.FIELD_COMPLEX:

                if symmetry == self.SYMMETRY_GENERAL:
                    for j in range(cols):
                        for i in range(rows):
                            aij = a[i, j]
                            data = template % (real(aij), imag(aij))
                            stream.write(data.encode('latin1'))
                else:
                    for j in range(cols):
                        for i in range(j, rows):
                            aij = a[i, j]
                            data = template % (real(aij), imag(aij))
                            stream.write(data.encode('latin1'))

            elif field == self.FIELD_PATTERN:
                raise ValueError('pattern type inconsisted with dense format')

            else:
                raise TypeError('Unknown field type %s' % field)

        # write sparse format
        else:
            coo = a.tocoo()  # convert to COOrdinate format

            # if symmetry format used, remove values above main diagonal
            if symmetry != self.SYMMETRY_GENERAL:
                lower_triangle_mask = coo.row >= coo.col
                coo = coo_matrix((coo.data[lower_triangle_mask],
                                 (coo.row[lower_triangle_mask],
                                  coo.col[lower_triangle_mask])),
                                 shape=coo.shape)

            # write shape spec
            data = '%i %i %i\n' % (rows, cols, coo.nnz)
            stream.write(data.encode('latin1'))

            template = self._field_template(field, precision-1)

            if field == self.FIELD_PATTERN:
                for r, c in zip(coo.row+1, coo.col+1):
                    data = "%i %i\n" % (r, c)
                    stream.write(data.encode('latin1'))
            elif field in (self.FIELD_INTEGER, self.FIELD_REAL,
                           self.FIELD_UNSIGNED):
                for r, c, d in zip(coo.row+1, coo.col+1, coo.data):
                    data = ("%i %i " % (r, c)) + (template % d)
                    stream.write(data.encode('latin1'))
            elif field == self.FIELD_COMPLEX:
                for r, c, d in zip(coo.row+1, coo.col+1, coo.data):
                    data = ("%i %i " % (r, c)) + (template % (d.real, d.imag))
                    stream.write(data.encode('latin1'))
            else:
                raise TypeError('Unknown field type %s' % field)


def _is_fromfile_compatible(stream):
    """
    Check whether `stream` is compatible with numpy.fromfile.

    Passing a gzipped file object to ``fromfile/fromstring`` doesn't work with
    Python 3.
    """

    bad_cls = []
    try:
        import gzip
        bad_cls.append(gzip.GzipFile)
    except ImportError:
        pass
    try:
        import bz2
        bad_cls.append(bz2.BZ2File)
    except ImportError:
        pass

    bad_cls = tuple(bad_cls)
    return not isinstance(stream, bad_cls)
