#ifndef _REF_UTILS_HPP_
#define _REF_UTILS_HPP_


PyObject* GetPyObjectPointerNoDebugInfo(bool isDebug, PyObject* object) {
    if (object != nullptr && isDebug) {
        // debug builds have 2 extra pointers at the front that we don't care about
        return (PyObject*)((size_t*)object + 2);
    }
    return object;
}

void DecRef(PyObject* object, bool isDebug) {
    auto noDebug = GetPyObjectPointerNoDebugInfo(isDebug, object);

    if (noDebug != nullptr && --noDebug->ob_refcnt == 0) {
        ((PyTypeObject*)GetPyObjectPointerNoDebugInfo(isDebug, noDebug->ob_type))->tp_dealloc(object);
    }
}

void IncRef(PyObject* object) {
    object->ob_refcnt++;
}

class PyObjectHolder {
private:
    PyObject* _object;
public:
    bool _isDebug;

    PyObjectHolder(bool isDebug) {
        _object = nullptr;
        _isDebug = isDebug;
    }

    PyObjectHolder(bool isDebug, PyObject *object) {
        _object = object;
        _isDebug = isDebug;
    };

    PyObjectHolder(bool isDebug, PyObject *object, bool addRef) {
        _object = object;
        _isDebug = isDebug;
        if (_object != nullptr && addRef) {
            GetPyObjectPointerNoDebugInfo(_isDebug, _object)->ob_refcnt++;
        }
    };

    PyObject* ToPython() {
        return _object;
    }

    ~PyObjectHolder() {
        DecRef(_object, _isDebug);
    }

    PyObject* operator* () {
        return GetPyObjectPointerNoDebugInfo(_isDebug, _object);
    }
};


#endif //_REF_UTILS_HPP_