
#ifndef __PY_VERSION_HPP__
#define __PY_VERSION_HPP__


#include <cstring>

enum PythonVersion {
    PythonVersion_Unknown,
    PythonVersion_25 = 0x0205,
    PythonVersion_26 = 0x0206,
    PythonVersion_27 = 0x0207,
    PythonVersion_30 = 0x0300,
    PythonVersion_31 = 0x0301,
    PythonVersion_32 = 0x0302,
    PythonVersion_33 = 0x0303,
    PythonVersion_34 = 0x0304,
    PythonVersion_35 = 0x0305,
    PythonVersion_36 = 0x0306,
    PythonVersion_37 = 0x0307,
    PythonVersion_38 = 0x0308,
    PythonVersion_39 = 0x0309,
    PythonVersion_310 = 0x030A,
    PythonVersion_311 = 0x030B
};


#ifdef _WIN32

typedef const char* (GetVersionFunc)();

static PythonVersion GetPythonVersion(HMODULE hMod) {
    auto versionFunc = reinterpret_cast<GetVersionFunc*>(GetProcAddress(hMod, "Py_GetVersion"));
    if (versionFunc == nullptr) {
        return PythonVersion_Unknown;
    }
    const char* version = versionFunc();


#else // LINUX -----------------------------------------------------------------

typedef const char* (*GetVersionFunc) ();

static PythonVersion GetPythonVersion(void *module) {
    GetVersionFunc versionFunc;
    *(void**)(&versionFunc) = dlsym(module, "Py_GetVersion");
    if(versionFunc == nullptr) {
        return PythonVersion_Unknown;
    }
    const char* version = versionFunc();

#endif //_WIN32

    if (version != nullptr && strlen(version) >= 3 && version[1] == '.') {
        if (version[0] == '2') {
            switch (version[2]) {
            case '5': return PythonVersion_25;
            case '6': return PythonVersion_26;
            case '7': return PythonVersion_27;
            }
        }
        else if (version[0] == '3') {
            switch (version[2]) {
            case '0': return PythonVersion_30;
            case '1':
                if(strlen(version) >= 4){
                    if(version[3] == '0'){
                        return PythonVersion_310;
                    }
                    if(version[3] == '1'){
                        return PythonVersion_311;
                    }
                }
                return PythonVersion_Unknown; // we don't care about 3.1 anymore...
            
            case '2': return PythonVersion_32;
            case '3': return PythonVersion_33;
            case '4': return PythonVersion_34;
            case '5': return PythonVersion_35;
            case '6': return PythonVersion_36;
            case '7': return PythonVersion_37;
            case '8': return PythonVersion_38;
            case '9': return PythonVersion_39;
            }
        }
    }
    return PythonVersion_Unknown;
}

#endif