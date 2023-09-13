#include "Python.h"

// default to Python's own target Windows version(s)
// override by setting WINVER, _WIN32_WINNT, (maybe also NTDDI_VERSION?) macros
#ifdef Py_WINVER
#ifndef WINVER
#define WINVER Py_WINVER
#endif
#ifndef _WIN32_WINNT
#define _WIN32_WINNT Py_WINVER
#endif
#endif
