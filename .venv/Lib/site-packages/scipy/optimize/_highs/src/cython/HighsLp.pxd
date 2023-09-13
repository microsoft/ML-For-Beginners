# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from .HConst cimport HighsBasisStatus, ObjSense, HighsVarType
from .HighsSparseMatrix cimport HighsSparseMatrix


cdef extern from "HighsLp.h" nogil:
    # From HiGHS/src/lp_data/HighsLp.h
    cdef cppclass HighsLp:
        int num_col_
        int num_row_

        vector[double] col_cost_
        vector[double] col_lower_
        vector[double] col_upper_
        vector[double] row_lower_
        vector[double] row_upper_

        HighsSparseMatrix a_matrix_

        ObjSense sense_
        double offset_

        string model_name_

        vector[string] row_names_
        vector[string] col_names_

        vector[HighsVarType] integrality_

        bool isMip() const

    cdef cppclass HighsSolution:
        vector[double] col_value
        vector[double] col_dual
        vector[double] row_value
        vector[double] row_dual

    cdef cppclass HighsBasis:
        bool valid_
        vector[HighsBasisStatus] col_status
        vector[HighsBasisStatus] row_status
