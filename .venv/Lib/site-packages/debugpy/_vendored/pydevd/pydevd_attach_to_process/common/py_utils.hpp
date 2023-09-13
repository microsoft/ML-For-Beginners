#ifndef _PY_UTILS_HPP_
#define _PY_UTILS_HPP_

typedef int (Py_IsInitialized)();
typedef PyInterpreterState* (PyInterpreterState_Head)();
typedef enum { PyGILState_LOCKED, PyGILState_UNLOCKED } PyGILState_STATE;
typedef PyGILState_STATE(PyGILState_Ensure)();
typedef void (PyGILState_Release)(PyGILState_STATE);
typedef int (PyRun_SimpleString)(const char *command);
typedef PyThreadState* (PyInterpreterState_ThreadHead)(PyInterpreterState* interp);
typedef PyThreadState* (PyThreadState_Next)(PyThreadState *tstate);
typedef PyThreadState* (PyThreadState_Swap)(PyThreadState *tstate);
typedef PyThreadState* (_PyThreadState_UncheckedGet)();
typedef PyObject* (PyObject_CallFunctionObjArgs)(PyObject *callable, ...);    // call w/ varargs, last arg should be nullptr
typedef PyObject* (PyInt_FromLong)(long);
typedef PyObject* (PyErr_Occurred)();
typedef void (PyErr_Fetch)(PyObject **ptype, PyObject **pvalue, PyObject **ptraceback);
typedef void (PyErr_Restore)(PyObject *type, PyObject *value, PyObject *traceback);
typedef PyObject* (PyImport_ImportModule) (const char *name);
typedef PyObject* (PyImport_ImportModuleNoBlock) (const char *name);
typedef PyObject* (PyObject_GetAttrString)(PyObject *o, const char *attr_name);
typedef PyObject* (PyObject_HasAttrString)(PyObject *o, const char *attr_name);
typedef void* (PyThread_get_key_value)(int);
typedef int (PyThread_set_key_value)(int, void*);
typedef void (PyThread_delete_key_value)(int);
typedef int (PyObject_Not) (PyObject *o);
typedef PyObject* (PyDict_New)();
typedef PyObject* (PyUnicode_InternFromString)(const char *u);
typedef PyObject * (_PyObject_FastCallDict)(
    PyObject *callable, PyObject *const *args, Py_ssize_t nargs, PyObject *kwargs);
typedef int (PyTraceBack_Here)(PyFrameObject *frame);

typedef PyObject* PyTuple_New(Py_ssize_t len);
typedef PyObject* PyEval_CallObjectWithKeywords(PyObject *callable, PyObject *args, PyObject *kwargs);

typedef void (PyEval_SetTrace)(Py_tracefunc, PyObject *);
typedef int (*Py_tracefunc)(PyObject *, PyFrameObject *frame, int, PyObject *);
typedef int (_PyEval_SetTrace)(PyThreadState *tstate, Py_tracefunc func, PyObject *arg);

typedef PyObject* PyObject_Repr(PyObject *);
typedef const char* PyUnicode_AsUTF8(PyObject *unicode);

// holder to ensure we release the GIL even in error conditions
class GilHolder {
    PyGILState_STATE _gilState;
    PyGILState_Release* _release;
public:
    GilHolder(PyGILState_Ensure* acquire, PyGILState_Release* release) {
        _gilState = acquire();
        _release = release;
    }

    ~GilHolder() {
        _release(_gilState);
    }
};

#ifdef _WIN32

#define PRINT(msg) {std::cout << msg << std::endl << std::flush;}

#define DEFINE_PROC_NO_CHECK(func, funcType, funcNameStr, errorCode) \
                    funcType func=reinterpret_cast<funcType>(GetProcAddress(module, funcNameStr));

#define DEFINE_PROC(func, funcType, funcNameStr, errorCode) \
                    DEFINE_PROC_NO_CHECK(func, funcType, funcNameStr, errorCode); \
                    if(func == nullptr){std::cout << funcNameStr << " not found." << std::endl << std::flush; return errorCode;};

#else // LINUX -----------------------------------------------------------------

#define PRINT(msg) {printf(msg); printf("\n");}

#define CHECK_NULL(ptr, msg, errorCode) if(ptr == nullptr){printf(msg); return errorCode;}

#define DEFINE_PROC_NO_CHECK(func, funcType, funcNameStr, errorCode) \
                    funcType func; *(void**)(&func) = dlsym(module, funcNameStr);

#define DEFINE_PROC(func, funcType, funcNameStr, errorCode) \
                    DEFINE_PROC_NO_CHECK(func, funcType, funcNameStr, errorCode); \
                    if(func == nullptr){printf(funcNameStr); printf(" not found.\n"); return errorCode;};

#endif //_WIN32

#endif //_PY_UTILS_HPP_