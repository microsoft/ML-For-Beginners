import numpy as np
import scipy as sp

__all__ = ['save_npz', 'load_npz']


# Make loading safe vs. malicious input
PICKLE_KWARGS = dict(allow_pickle=False)


def save_npz(file, matrix, compressed=True):
    """ Save a sparse matrix or array to a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already
        there.
    matrix: spmatrix or sparray
        The sparse matrix or array to save.
        Supported formats: ``csc``, ``csr``, ``bsr``, ``dia`` or ``coo``.
    compressed : bool, optional
        Allow compressing the file. Default: True

    See Also
    --------
    scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.
    numpy.savez: Save several arrays into a ``.npz`` archive.
    numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.

    Examples
    --------
    Store sparse matrix to disk, and load it again:

    >>> import numpy as np
    >>> import scipy as sp
    >>> sparse_matrix = sp.sparse.csc_matrix([[0, 0, 3], [4, 0, 0]])
    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    >>> sp.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> sparse_matrix = sp.sparse.load_npz('/tmp/sparse_matrix.npz')

    >>> sparse_matrix
    <2x3 sparse matrix of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_matrix.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)
    """
    arrays_dict = {}
    if matrix.format in ('csc', 'csr', 'bsr'):
        arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
    elif matrix.format == 'dia':
        arrays_dict.update(offsets=matrix.offsets)
    elif matrix.format == 'coo':
        arrays_dict.update(row=matrix.row, col=matrix.col)
    else:
        msg = f'Save is not implemented for sparse matrix of format {matrix.format}.'
        raise NotImplementedError(msg)
    arrays_dict.update(
        format=matrix.format.encode('ascii'),
        shape=matrix.shape,
        data=matrix.data
    )
    if isinstance(matrix, sp.sparse.sparray):
        arrays_dict.update(_is_array=True)
    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)


def load_npz(file):
    """ Load a sparse array/matrix from a file using ``.npz`` format.

    Parameters
    ----------
    file : str or file-like object
        Either the file name (string) or an open file (file-like object)
        where the data will be loaded.

    Returns
    -------
    result : csc_array, csr_array, bsr_array, dia_array or coo_array
        A sparse array/matrix containing the loaded data.

    Raises
    ------
    OSError
        If the input file does not exist or cannot be read.

    See Also
    --------
    scipy.sparse.save_npz: Save a sparse array/matrix to a file using ``.npz`` format.
    numpy.load: Load several arrays from a ``.npz`` archive.

    Examples
    --------
    Store sparse array/matrix to disk, and load it again:

    >>> import numpy as np
    >>> import scipy as sp
    >>> sparse_array = sp.sparse.csc_array([[0, 0, 3], [4, 0, 0]])
    >>> sparse_array
    <2x3 sparse array of type '<class 'numpy.int64'>'
       with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_array.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    >>> sp.sparse.save_npz('/tmp/sparse_array.npz', sparse_array)
    >>> sparse_array = sp.sparse.load_npz('/tmp/sparse_array.npz')

    >>> sparse_array
    <2x3 sparse array of type '<class 'numpy.int64'>'
        with 2 stored elements in Compressed Sparse Column format>
    >>> sparse_array.toarray()
    array([[0, 0, 3],
           [4, 0, 0]], dtype=int64)

    In this example we force the result to be csr_array from csr_matrix
    >>> sparse_matrix = sp.sparse.csc_matrix([[0, 0, 3], [4, 0, 0]])
    >>> sp.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)
    >>> tmp = sp.sparse.load_npz('/tmp/sparse_matrix.npz')
    >>> sparse_array = sp.sparse.csr_array(tmp)
    """
    with np.load(file, **PICKLE_KWARGS) as loaded:
        sparse_format = loaded.get('format')
        if sparse_format is None:
            raise ValueError(f'The file {file} does not contain '
                             f'a sparse array or matrix.')
        sparse_format = sparse_format.item()

        if not isinstance(sparse_format, str):
            # Play safe with Python 2 vs 3 backward compatibility;
            # files saved with SciPy < 1.0.0 may contain unicode or bytes.
            sparse_format = sparse_format.decode('ascii')

        if loaded.get('_is_array'):
            sparse_type = sparse_format + '_array'
        else:
            sparse_type = sparse_format + '_matrix'

        try:
            cls = getattr(sp.sparse, f'{sparse_type}')
        except AttributeError as e:
            raise ValueError(f'Unknown format "{sparse_type}"') from e

        if sparse_format in ('csc', 'csr', 'bsr'):
            return cls((loaded['data'], loaded['indices'], loaded['indptr']),
                       shape=loaded['shape'])
        elif sparse_format == 'dia':
            return cls((loaded['data'], loaded['offsets']),
                       shape=loaded['shape'])
        elif sparse_format == 'coo':
            return cls((loaded['data'], (loaded['row'], loaded['col'])),
                       shape=loaded['shape'])
        else:
            raise NotImplementedError(f'Load is not implemented for '
                                      f'sparse matrix of format {sparse_format}.')
