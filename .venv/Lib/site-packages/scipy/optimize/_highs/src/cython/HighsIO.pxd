# distutils: language=c++
# cython: language_level=3


cdef extern from "HighsIO.h" nogil:
    # workaround for lack of enum class support in Cython < 3.x
    # cdef enum class HighsLogType(int):
    #     kInfo "HighsLogType::kInfo" = 1
    #     kDetailed "HighsLogType::kDetailed"
    #     kVerbose "HighsLogType::kVerbose"
    #     kWarning "HighsLogType::kWarning"
    #     kError "HighsLogType::kError"

    cdef cppclass HighsLogType:
        pass

    cdef HighsLogType kInfo "HighsLogType::kInfo"
    cdef HighsLogType kDetailed "HighsLogType::kDetailed"
    cdef HighsLogType kVerbose "HighsLogType::kVerbose"
    cdef HighsLogType kWarning "HighsLogType::kWarning"
    cdef HighsLogType kError "HighsLogType::kError"
