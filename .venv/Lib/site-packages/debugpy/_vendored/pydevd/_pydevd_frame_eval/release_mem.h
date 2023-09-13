#include "Python.h"

void release_co_extra(void *obj) {
    Py_XDECREF(obj);
}
