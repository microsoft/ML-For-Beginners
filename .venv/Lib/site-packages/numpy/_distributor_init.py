
'''
Helper to preload windows dlls to prevent dll not found errors.
Once a DLL is preloaded, its namespace is made available to any
subsequent DLL. This file originated in the numpy-wheels repo,
and is created as part of the scripts that build the wheel.
'''
import os
import glob
if os.name == 'nt':
    # convention for storing / loading the DLL from
    # numpy/.libs/, if present
    try:
        from ctypes import WinDLL
        basedir = os.path.dirname(__file__)
    except:
        pass
    else:
        libs_dir = os.path.abspath(os.path.join(basedir, '.libs'))
        DLL_filenames = []
        if os.path.isdir(libs_dir):
            for filename in glob.glob(os.path.join(libs_dir,
                                                   '*openblas*dll')):
                # NOTE: would it change behavior to load ALL
                # DLLs at this path vs. the name restriction?
                WinDLL(os.path.abspath(filename))
                DLL_filenames.append(filename)
        if len(DLL_filenames) > 1:
            import warnings
            warnings.warn("loaded more than 1 DLL from .libs:"
                          "\n%s" % "\n".join(DLL_filenames),
                          stacklevel=1)
