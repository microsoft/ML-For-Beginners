from cpython.datetime cimport datetime

from pandas._libs.tslibs.np_datetime cimport NPY_DATETIMEUNIT


cpdef str get_rule_month(str source)
cpdef quarter_to_myear(int year, int quarter, str freq)

cdef datetime parse_datetime_string(
    str date_string,
    bint dayfirst,
    bint yearfirst,
    NPY_DATETIMEUNIT* out_bestunit
)
