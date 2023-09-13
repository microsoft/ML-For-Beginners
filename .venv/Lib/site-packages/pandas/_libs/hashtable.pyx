cimport cython
from cpython.mem cimport (
    PyMem_Free,
    PyMem_Malloc,
)
from cpython.ref cimport (
    Py_INCREF,
    PyObject,
)
from libc.stdlib cimport (
    free,
    malloc,
)

import numpy as np

cimport numpy as cnp
from numpy cimport ndarray

cnp.import_array()


from pandas._libs cimport util
from pandas._libs.dtypes cimport numeric_object_t
from pandas._libs.khash cimport (
    KHASH_TRACE_DOMAIN,
    are_equivalent_float32_t,
    are_equivalent_float64_t,
    are_equivalent_khcomplex64_t,
    are_equivalent_khcomplex128_t,
    kh_needed_n_buckets,
    kh_python_hash_equal,
    kh_python_hash_func,
    khiter_t,
)
from pandas._libs.missing cimport checknull


def get_hashtable_trace_domain():
    return KHASH_TRACE_DOMAIN


def object_hash(obj):
    return kh_python_hash_func(obj)


def objects_are_equal(a, b):
    return kh_python_hash_equal(a, b)


cdef int64_t NPY_NAT = util.get_nat()
SIZE_HINT_LIMIT = (1 << 20) + 7


cdef Py_ssize_t _INIT_VEC_CAP = 128

include "hashtable_class_helper.pxi"
include "hashtable_func_helper.pxi"


# map derived hash-map types onto basic hash-map types:
if np.dtype(np.intp) == np.dtype(np.int64):
    IntpHashTable = Int64HashTable
    unique_label_indices = _unique_label_indices_int64
elif np.dtype(np.intp) == np.dtype(np.int32):
    IntpHashTable = Int32HashTable
    unique_label_indices = _unique_label_indices_int32
else:
    raise ValueError(np.dtype(np.intp))


cdef class Factorizer:
    cdef readonly:
        Py_ssize_t count

    def __cinit__(self, size_hint: int):
        self.count = 0

    def get_count(self) -> int:
        return self.count

    def factorize(self, values, na_sentinel=-1, na_value=None, mask=None) -> np.ndarray:
        raise NotImplementedError


cdef class ObjectFactorizer(Factorizer):
    cdef public:
        PyObjectHashTable table
        ObjectVector uniques

    def __cinit__(self, size_hint: int):
        self.table = PyObjectHashTable(size_hint)
        self.uniques = ObjectVector()

    def factorize(
        self, ndarray[object] values, na_sentinel=-1, na_value=None, mask=None
    ) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray[np.intp]

        Examples
        --------
        Factorize values with nans replaced by na_sentinel

        >>> fac = ObjectFactorizer(3)
        >>> fac.factorize(np.array([1,2,np.nan], dtype='O'), na_sentinel=20)
        array([ 0,  1, 20])
        """
        cdef:
            ndarray[intp_t] labels

        if mask is not None:
            raise NotImplementedError("mask not supported for ObjectFactorizer.")

        if self.uniques.external_view_exists:
            uniques = ObjectVector()
            uniques.extend(self.uniques.to_array())
            self.uniques = uniques
        labels = self.table.get_labels(values, self.uniques,
                                       self.count, na_sentinel, na_value)
        self.count = len(self.uniques)
        return labels
