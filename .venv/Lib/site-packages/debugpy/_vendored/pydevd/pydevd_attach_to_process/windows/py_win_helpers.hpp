#ifndef _PY_WIN_HELPERS_HPP_
#define _PY_WIN_HELPERS_HPP_

bool IsPythonModule(HMODULE module, bool &isDebug) {
    wchar_t mod_name[MAX_PATH];
    isDebug = false;
    if (GetModuleBaseName(GetCurrentProcess(), module, mod_name, MAX_PATH)) {
        if (_wcsnicmp(mod_name, L"python", 6) == 0) {
            if (wcslen(mod_name) >= 10 && _wcsnicmp(mod_name + 8, L"_d", 2) == 0) {
                isDebug = true;
            }
            
            // Check if the module has Py_IsInitialized.
            DEFINE_PROC_NO_CHECK(isInit, Py_IsInitialized*, "Py_IsInitialized", 0);
            DEFINE_PROC_NO_CHECK(gilEnsure, PyGILState_Ensure*, "PyGILState_Ensure", 51);
            DEFINE_PROC_NO_CHECK(gilRelease, PyGILState_Release*, "PyGILState_Release", 51);
            if (isInit == nullptr || gilEnsure == nullptr || gilRelease == nullptr) {
                return false;
            }
            

            return true;
        }
    }
    return false;
}


struct ModuleInfo {
    HMODULE module;
    bool isDebug;
    int errorGettingModule; // 0 means ok, negative values some error (should never be positive).
};


ModuleInfo GetPythonModule() {
    HANDLE hProcess = GetCurrentProcess();
    ModuleInfo moduleInfo;
    moduleInfo.module = nullptr;
    moduleInfo.isDebug = false;
    moduleInfo.errorGettingModule = 0;
    
    DWORD modSize = sizeof(HMODULE) * 1024;
    HMODULE* hMods = (HMODULE*)_malloca(modSize);
    if (hMods == nullptr) {
        std::cout << "hmods not allocated! " << std::endl << std::flush;
        moduleInfo.errorGettingModule = -1;
        return moduleInfo;
    }

    DWORD modsNeeded;
    while (!EnumProcessModules(hProcess, hMods, modSize, &modsNeeded)) {
        // try again w/ more space...
        _freea(hMods);
        hMods = (HMODULE*)_malloca(modsNeeded);
        if (hMods == nullptr) {
            std::cout << "hmods not allocated (2)! " << std::endl << std::flush;
            moduleInfo.errorGettingModule = -2;
            return moduleInfo;
        }
        modSize = modsNeeded;
    }
    
    for (size_t i = 0; i < modsNeeded / sizeof(HMODULE); i++) {
        bool isDebug;
        if (IsPythonModule(hMods[i], isDebug)) {
            moduleInfo.isDebug = isDebug;
            moduleInfo.module = hMods[i]; 
            return moduleInfo;
        }
    }
    std::cout << "Unable to find python module. " << std::endl << std::flush;
    moduleInfo.errorGettingModule = -3;
    return moduleInfo;
}

#endif