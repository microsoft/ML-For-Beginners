# cython: language_level=3

from libcpp.string cimport string

cdef extern from "HighsStatus.h" nogil:
    ctypedef enum HighsStatus:
        HighsStatusError "HighsStatus::kError" = -1
        HighsStatusOK "HighsStatus::kOk" = 0
        HighsStatusWarning "HighsStatus::kWarning" = 1


    string highsStatusToString(HighsStatus status)
