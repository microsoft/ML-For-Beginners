from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef extern from *:
    ctypedef void PyObject
    ctypedef struct PyCodeObject:
        int co_argcount;		# arguments, except *args */
        int co_kwonlyargcount;	# keyword only arguments */
        int co_nlocals;		    # local variables */
        int co_stacksize;		# entries needed for evaluation stack */
        int co_flags;		    # CO_..., see below */
        int co_firstlineno;     # first source line number */
        PyObject *co_code;		# instruction opcodes */
        PyObject *co_consts;	# list (constants used) */
        PyObject *co_names;		# list of strings (names used) */
        PyObject *co_varnames;	# tuple of strings (local variable names) */
        PyObject *co_freevars;	# tuple of strings (free variable names) */
        PyObject *co_cellvars;  # tuple of strings (cell variable names) */
        unsigned char *co_cell2arg; # Maps cell vars which are arguments. */
        PyObject *co_filename;	# unicode (where it was loaded from) */
        PyObject *co_name;		# unicode (name, for reference) */
        PyObject *co_lnotab;	# string (encoding addr<->lineno mapping) See
                                # Objects/lnotab_notes.txt for details. */
        void *co_zombieframe;   # for optimization only (see frameobject.c) */
        PyObject *co_weakreflist;   # to support weakrefs to code objects */
        void *co_extra;

cdef extern from "frameobject.h":
    ctypedef struct PyFrameObject:
        PyFrameObject *f_back
        PyCodeObject *f_code       # code segment
        PyObject *f_builtins       # builtin symbol table (PyDictObject)
        PyObject *f_globals        # global symbol table (PyDictObject) */
        PyObject *f_locals         # local symbol table (any mapping) */
        PyObject **f_valuestack   #
        PyObject **f_stacktop
        PyObject *f_trace         # Trace function */
        PyObject *f_exc_type
        PyObject *f_exc_value
        PyObject *f_exc_traceback
        PyObject *f_gen;

        int f_lasti;                #/* Last instruction if called */
        int f_lineno;               #/* Current line number */
        int f_iblock;               #/* index in f_blockstack */
        char f_executing;           #/* whether the frame is still executing */
        PyObject *f_localsplus[1];

cdef extern from "release_mem.h":
    void release_co_extra(void *)

cdef extern from "code.h":
    ctypedef void freefunc(void *)
    int _PyCode_GetExtra(PyObject *code, Py_ssize_t index, void **extra)
    int _PyCode_SetExtra(PyObject *code, Py_ssize_t index, void *extra)
    
# TODO: Things are in a different place for Python 3.11.
# cdef extern from "cpython/code.h":
#     ctypedef void freefunc(void *)
#     int _PyCode_GetExtra(PyObject *code, Py_ssize_t index, void **extra)
#     int _PyCode_SetExtra(PyObject *code, Py_ssize_t index, void *extra)

cdef extern from "Python.h":
    void Py_INCREF(object o)
    void Py_DECREF(object o)
    object PyImport_ImportModule(char *name)
    PyObject* PyObject_CallFunction(PyObject *callable, const char *format, ...)
    object PyObject_GetAttrString(object o, char *attr_name)

cdef extern from "pystate.h":
    # ctypedef PyObject* _PyFrameEvalFunction(PyThreadState* tstate, PyFrameObject *frame, int exc)
    # ctypedef PyObject* _PyFrameEvalFunction(PyFrameObject *frame, int exc)
    ctypedef PyObject* _PyFrameEvalFunction(...)

    ctypedef struct PyInterpreterState:
        PyInterpreterState *next
        PyInterpreterState *tstate_head

        PyObject *modules

        PyObject *modules_by_index
        PyObject *sysdict
        PyObject *builtins
        PyObject *importlib

        PyObject *codec_search_path
        PyObject *codec_search_cache
        PyObject *codec_error_registry
        int codecs_initialized
        int fscodec_initialized

        int dlopenflags

        PyObject *builtins_copy
        PyObject *import_func
        # Initialized to PyEval_EvalFrameDefault().
        _PyFrameEvalFunction eval_frame

    ctypedef struct PyThreadState:
        PyThreadState *prev
        PyThreadState *next
        PyInterpreterState *interp
        # ...

    PyThreadState *PyThreadState_Get()

cdef extern from "ceval.h":
    '''
#if PY_VERSION_HEX >= 0x03090000
PyObject * noop(PyFrameObject *frame, int exc) {
    return NULL;
}
#define CALL_EvalFrameDefault_38(a, b)    noop(a, b)
#define CALL_EvalFrameDefault_39(a, b, c)    _PyEval_EvalFrameDefault(a, b, c)
#else
PyObject * noop(PyThreadState* tstate, PyFrameObject *frame, int exc) {
    return NULL;
}
#define CALL_EvalFrameDefault_39(a, b, c)    noop(a, b, c)
#define CALL_EvalFrameDefault_38(a, b)    _PyEval_EvalFrameDefault(a, b)
#endif
    '''

    int _PyEval_RequestCodeExtraIndex(freefunc)
    PyFrameObject *PyEval_GetFrame()
    PyObject* PyEval_CallFunction(PyObject *callable, const char *format, ...)

    # PyObject* _PyEval_EvalFrameDefault(PyThreadState* tstate, PyFrameObject *frame, int exc)
    # PyObject* _PyEval_EvalFrameDefault(PyFrameObject *frame, int exc)
    PyObject* _PyEval_EvalFrameDefault(...)
    PyObject* CALL_EvalFrameDefault_38(PyFrameObject *frame, int exc)  # Actually a macro.
    PyObject* CALL_EvalFrameDefault_39(PyThreadState* tstate, PyFrameObject *frame, int exc)  # Actually a macro.
